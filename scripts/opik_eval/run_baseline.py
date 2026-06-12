#!/usr/bin/env python
"""Baseline reliability report for the current "firehose" path (DEVGRAPH R04 / F015h).

Reads recent CoScientist runs from Opik, computes per-trace reliability metrics
(empty responses, tool-not-found, runaway loops), aggregates them by model, and
writes a markdown + JSON report. This is the BASELINE that the experiments module
(F015) must beat — run it again after the module lands (roadmap step R19).

Usage:
    python scripts/opik_eval/run_baseline.py [--limit 40] [--out scripts/opik_eval/results]
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import statistics
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # allow sibling imports
from opik_client import get_client          # noqa: E402
from metrics import trace_metrics, RUNAWAY_LLM_CALLS, RUNAWAY_SECONDS  # noqa: E402


def _agg(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"traces": 0}
    llm = [r["n_llm"] for r in rows]
    return {
        "traces": n,
        "n_llm_median": statistics.median(llm),
        "n_llm_max": max(llm),
        "runaway_pct": round(100 * sum(r["runaway"] for r in rows) / n, 1),
        "errored_pct": round(100 * sum(r["errored"] for r in rows) / n, 1),
        "empty_llm_traces_pct": round(100 * sum(r["empty_llm"] > 0 for r in rows) / n, 1),
        "tool_not_found_total": sum(r["tool_not_found"] for r in rows),
        "duration_median_s": round(statistics.median(r["duration_s"] for r in rows), 1),
    }


def build_report(rows: list[dict], limit: int) -> str:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    L = []
    L.append(f"# Firehose baseline — reliability on the current path\n")
    L.append(f"Source: Opik, last {limit} traces · generated {datetime.date.today().isoformat()}\n")
    L.append(f"Runaway = ≥{RUNAWAY_LLM_CALLS} LLM calls or ≥{RUNAWAY_SECONDS:.0f}s.\n")

    L.append("## Aggregate by main_model\n")
    L.append("| model | traces | LLM-calls median | LLM-calls max | runaway % | errored % | empty-resp traces % | tool-not-found Σ | dur median s |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for model, mrows in sorted(by_model.items(), key=lambda kv: -len(kv[1])):
        a = _agg(mrows)
        L.append(f"| {model} | {a['traces']} | {a['n_llm_median']} | {a['n_llm_max']} | "
                 f"{a['runaway_pct']} | {a['errored_pct']} | {a['empty_llm_traces_pct']} | "
                 f"{a['tool_not_found_total']} | {a['duration_median_s']} |")
    overall = _agg(rows)
    L.append(f"| **ALL** | {overall['traces']} | {overall['n_llm_median']} | {overall['n_llm_max']} | "
             f"{overall['runaway_pct']} | {overall['errored_pct']} | {overall['empty_llm_traces_pct']} | "
             f"{overall['tool_not_found_total']} | {overall['duration_median_s']} |")

    # Worst runaways
    worst = sorted(rows, key=lambda r: -r["n_llm"])[:8]
    L.append("\n## Worst runaways (by LLM-call count)\n")
    L.append("| start | model | LLM calls | dur s | tool-not-found | query |")
    L.append("|---|---|---|---|---|---|")
    for r in worst:
        L.append(f"| {r['start']} | {r['model']} | {r['n_llm']} | {r['duration_s']} | "
                 f"{r['tool_not_found']} | {r['query'][:50]} |")

    # Hallucinated tool tally
    tally: Counter = Counter()
    for r in rows:
        tally.update(r["notfound_names"])
    if tally:
        L.append("\n## Hallucinated / unregistered tool names (tool-not-found)\n")
        for name, cnt in tally.most_common():
            L.append(f"- `{name}` × {cnt}")

    # Error-type tally
    errs = Counter(r["error_type"] for r in rows if r["errored"])
    if errs:
        L.append("\n## Trace-level error types\n")
        for et, cnt in errs.most_common():
            L.append(f"- {et} × {cnt}")

    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--sleep", type=float, default=0.4, help="pause between trace fetches (rate-limit friendly)")
    ap.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parent / "results"))
    args = ap.parse_args()

    client, project = get_client()
    print(f"[opik] project={project} · fetching last {args.limit} traces …", file=sys.stderr)
    traces = client.search_traces(project_name=project, max_results=args.limit)
    rows = []
    for i, t in enumerate(traces):
        rows.append(trace_metrics(client, t, project))
        print(f"\r[opik] {i + 1}/{len(traces)}", end="", file=sys.stderr)
        if args.sleep:
            time.sleep(args.sleep)
    print("", file=sys.stderr)

    report = build_report(rows, args.limit)
    print(report)

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.date.today().isoformat()
    (outdir / f"baseline_{stamp}.md").write_text(report)
    (outdir / f"baseline_{stamp}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print(f"[written] {outdir}/baseline_{stamp}.md (+ .json)", file=sys.stderr)


if __name__ == "__main__":
    main()
