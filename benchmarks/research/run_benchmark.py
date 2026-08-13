#!/usr/bin/env python3
"""Benchmark the research pipeline (HypothesesAgent → ResearchAgent →
BackgroundValidatorPlugin) against a list of research questions.

Companion to ``benchmarks/alembic/run_benchmark.py`` — same shape (task list →
isolated runs → structured metric extraction → markdown + JSON summary), but
here the "pipeline" is a full CoScientist run and the "artefact" we grade is
the Research Context Graph it produced, not a Docker image.

Each task gets its OWN (user_id, session_id) so its Research Context Graph is
completely isolated from every other task's (the graph store is a process-wide
registry keyed by that pair — see CoScientist/graph/research/store.py).

What this does NOT do yet (be honest about scope, don't oversell it):
  - No token/cost accounting per task (only wall-clock time).
  - Scoring is either `keyword_any` (cheap, crude substring check against the
    Conclusion text) or `manual` (no verdict — the synthesis is just recorded
    for a human to grade). Neither is a substitute for real domain grading;
    they're a first rung, not a finished evaluation methodology.
  - Runs are sequential by default (--parallel 1): unlike the alembic bench
    (isolated Docker containers), these are real, possibly-shared-state LLM
    sessions against a live API budget — don't fan them out wide by default.

Usage:
    python benchmarks/research/run_benchmark.py
    python benchmarks/research/run_benchmark.py --tasks-file my_tasks.jsonl --parallel 2
    python benchmarks/research/run_benchmark.py --ids paracetamol-cox --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

# benchmarks/research/run_benchmark.py → project root is 2 levels up
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_TASKS_FILE = Path(__file__).resolve().parent / "tasks.jsonl"
RUNS_DIR = Path(__file__).resolve().parent / "runs"

# How long (seconds) to wait, after the ADK run itself finishes, for the
# BackgroundValidatorPlugin's fire-and-forget judgments to land — it schedules
# an async task per unresolved hypothesis and the main run does NOT await it
# (see CoScientist/graph/research/validator.py), so reading the graph
# immediately after `manager.run()` returns can catch it mid-flight.
VALIDATOR_SETTLE_TIMEOUT = 60
VALIDATOR_POLL_INTERVAL = 2


# ══════════════════════════════════════════════════════════════════════════════
# Task loading
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Task:
    id: str
    question: str
    domain: str = ""
    check: dict = field(default_factory=lambda: {"type": "manual"})


def load_tasks(path: Path) -> list[Task]:
    tasks: list[Task] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            sys.exit(f"[bench] {path}:{lineno}: invalid JSON — {exc}")
        if "id" not in raw or "question" not in raw:
            sys.exit(f"[bench] {path}:{lineno}: task needs at least 'id' and 'question'")
        tasks.append(Task(
            id=raw["id"], question=raw["question"], domain=raw.get("domain", ""),
            check=raw.get("check") or {"type": "manual"},
        ))
    return tasks


# ══════════════════════════════════════════════════════════════════════════════
# Scoring
# ══════════════════════════════════════════════════════════════════════════════
def score_conclusions(synthesis_texts: list[str], check: dict) -> str:
    """One of: pass | fail | manual | no_conclusion.

    `manual` means "not automatically checkable" — the synthesis text is still
    saved in the record for a human to grade later; it is NOT a silent pass.
    """
    if not synthesis_texts:
        return "no_conclusion"
    check_type = (check or {}).get("type", "manual")
    if check_type == "keyword_any":
        expected = [str(k).lower() for k in check.get("expected", [])]
        blob = " ".join(synthesis_texts).lower()
        return "pass" if any(k in blob for k in expected) else "fail"
    return "manual"


# ══════════════════════════════════════════════════════════════════════════════
# One task run
# ══════════════════════════════════════════════════════════════════════════════
async def _wait_for_validator_settle(graph, timeout: float, poll: float) -> bool:
    """Poll until no Hypothesis is left `under_verification` (the
    BackgroundValidatorPlugin has judged everything it was going to), or
    `timeout` elapses. Returns True if it settled, False if it timed out —
    a False result means any 'stuck under_verification' count below is a
    measurement artefact (validator still working), not necessarily a real
    validator failure; a longer --validator-timeout disambiguates the two."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        counts = graph.overview().get("counts", {}).get("Hypothesis", {})
        if counts.get("under_verification", 0) == 0:
            return True
        await asyncio.sleep(poll)
    return graph.overview().get("counts", {}).get("Hypothesis", {}).get("under_verification", 0) == 0


async def run_one_task(task: Task, idx: int, total: int, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        print(f"[bench] ↑ start  {task.id}  ({idx}/{total})", flush=True)
        from CoScientist.main import CoScientistManager
        from CoScientist.graph.research.store import get_research_graph

        run_tag = uuid4().hex[:8]
        user_id = f"bench_{task.id}_{run_tag}"
        session_id = f"bench_{task.id}_{run_tag}"

        record: dict[str, Any] = {
            "id": task.id, "domain": task.domain, "question": task.question,
            "check": task.check, "elapsed_sec": None, "error": None,
            "hypotheses": {}, "hypotheses_detail": [], "conclusions": [], "score": "error",
            "validator_settled": None,
        }

        manager = CoScientistManager(user_id=user_id, session_id=session_id)
        started = time.monotonic()
        try:
            await manager.initialize()
            await manager.run(task.question, verbose=False)
        except Exception as exc:  # noqa: BLE001 — one task's crash must not kill the run
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["elapsed_sec"] = round(time.monotonic() - started, 1)
            print(f"[bench] ↓ done   {task.id}  ERROR: {record['error']}", flush=True)
            return record
        finally:
            try:
                await manager.close()
            except Exception:  # noqa: BLE001 — best-effort cleanup
                pass
        record["elapsed_sec"] = round(time.monotonic() - started, 1)

        try:
            graph = get_research_graph(user_id=user_id, session_id=session_id)
            record["validator_settled"] = await _wait_for_validator_settle(
                graph, VALIDATOR_SETTLE_TIMEOUT, VALIDATOR_POLL_INTERVAL
            )
            # full() is the raw serialized graph (public — used the same way by
            # validator.py) — one call gets every node's attrs AND status_history
            # in one shot, including the ValidatorAgent's verdict `reason`, which
            # overview()/get_context_slice() don't surface.
            raw = graph.full()
            nodes = raw.get("nodes", [])

            status_counts: dict[str, int] = {}
            hyp_detail = []
            for n in nodes:
                if n.get("type") != "Hypothesis":
                    continue
                status = n.get("status", "?")
                status_counts[status] = status_counts.get(status, 0) + 1
                reason = ""
                for entry in reversed(n.get("status_history") or []):
                    if entry.get("to") == status and entry.get("reason"):
                        reason = entry["reason"]
                        break
                hyp_detail.append({
                    "id": n.get("id"), "status": status,
                    "formulation": (n.get("attrs") or {}).get("formulation", ""),
                    "reason": reason,
                })
            record["hypotheses"] = status_counts
            record["hypotheses_detail"] = hyp_detail

            synthesis_texts = []
            for n in nodes:
                if n.get("type") != "Conclusion":
                    continue
                text = (n.get("attrs") or {}).get("synthesis", "")
                if text:
                    synthesis_texts.append(text)
                record["conclusions"].append({"id": n.get("id"), "synthesis": text})

            record["score"] = score_conclusions(synthesis_texts, task.check)
        except Exception as exc:  # noqa: BLE001 — grading failure shouldn't hide a real run
            record["error"] = f"grading failed: {type(exc).__name__}: {exc}"
            record["score"] = "error"

        verdict_summary = ", ".join(f"{k}={v}" for k, v in sorted(record["hypotheses"].items())) or "no hypotheses"
        print(f"[bench] ↓ done   {task.id}  ({record['elapsed_sec']:.0f}s, "
              f"verdicts: {verdict_summary}, content-score={record['score']})", flush=True)
        return record


# ══════════════════════════════════════════════════════════════════════════════
# Aggregation + report
# ══════════════════════════════════════════════════════════════════════════════
def aggregate_metrics(records: list[dict]) -> dict:
    n = len(records)
    with_hyp = sum(1 for r in records if r.get("hypotheses"))
    with_conclusion = sum(1 for r in records if r.get("conclusions"))
    verdict_totals: dict[str, int] = {}
    for r in records:
        for status, count in (r.get("hypotheses") or {}).items():
            verdict_totals[status] = verdict_totals.get(status, 0) + count
    score_totals: dict[str, int] = {}
    for r in records:
        score_totals[r["score"]] = score_totals.get(r["score"], 0) + 1
    stuck = sum(
        1 for r in records
        if (r.get("hypotheses") or {}).get("under_verification", 0) > 0
    )
    errors = sum(1 for r in records if r.get("error"))
    avg_elapsed = (
        sum(r["elapsed_sec"] for r in records if r.get("elapsed_sec") is not None) / n
        if n else 0.0
    )
    return {
        "tasks_total": n,
        "tasks_with_hypothesis": with_hyp,
        "tasks_with_conclusion": with_conclusion,
        "tasks_stuck_under_verification": stuck,
        "tasks_errored": errors,
        "hypothesis_verdict_totals": verdict_totals,
        "score_totals": score_totals,
        "avg_elapsed_sec": round(avg_elapsed, 1),
    }


def _row(r: dict) -> str:
    h = r.get("hypotheses") or {}
    h_str = ", ".join(f"{k}={v}" for k, v in sorted(h.items())) or "-"
    elapsed = f"{r['elapsed_sec']:.0f}s" if r.get("elapsed_sec") is not None else "-"
    err = (r.get("error") or "")[:60]
    return (f"| {r['id']} | {r.get('domain','')} | {elapsed} | {h_str} "
            f"| {len(r.get('conclusions') or [])} | {r['score']} | {err} |")


def write_summary(records: list[dict], out: Path) -> None:
    lines = [
        f"# Research pipeline benchmark — {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"Tasks run: {len(records)}",
        "",
        "**Two separate signals, don't conflate them:**",
        "- **Hypothesis statuses** (this column, and the per-task Verdicts section "
        "below) — the actual `confirmed`/`refuted`/`postponed` call the "
        "`ValidatorAgent` made, with its reason. THIS answers 'did the hypothesis "
        "get confirmed'.",
        "- **Score** — a crude, separate content check (`keyword_any`: did the "
        "expected term appear anywhere in the Conclusion text; `manual`: not "
        "auto-checkable, read it yourself; `no_conclusion`/`error`: self-explanatory). "
        "A `pass` does NOT mean the hypothesis was confirmed — it can `pass` while "
        "every hypothesis is `postponed`, as happened in the first smoke run.",
        "",
        "| Task | Domain | Time | Hypothesis statuses | #Conclusions | Score | Error |",
        "|---|---|---:|---|---:|---|---|",
    ]
    for r in sorted(records, key=lambda x: x["id"]):
        lines.append(_row(r))

    lines += ["", "## Per-task verdicts (the actual confirmed/refuted/postponed call)"]
    for r in sorted(records, key=lambda x: x["id"]):
        lines += ["", f"### {r['id']}", f"- Question: {r['question']}",
                  f"- Validator settled: {r.get('validator_settled')}", ""]
        detail = r.get("hypotheses_detail") or []
        if not detail:
            lines.append("- (no Hypothesis node produced)")
        for h in detail:
            mark = {"confirmed": "✅", "refuted": "❌", "postponed": "⏸️",
                    "under_verification": "⏳", "formulated": "•"}.get(h["status"], "•")
            lines.append(f"- {mark} **{h['id']}** [{h['status'].upper()}]: {h['formulation'][:220]}")
            if h.get("reason"):
                lines.append(f"    - *why*: {h['reason'][:300]}")

    lines += ["", "## Per-task conclusions (synthesis text)"]
    for r in sorted(records, key=lambda x: x["id"]):
        lines += ["", f"### {r['id']}"]
        if not r.get("conclusions"):
            lines.append("- (no Conclusion node produced)")
        for c in r.get("conclusions", []):
            lines.append(f"- **{c['id']}**: {c['synthesis'][:500]}")

    agg = aggregate_metrics(records)
    lines += ["", "## Aggregate", "",
              f"- tasks with ≥1 hypothesis: {agg['tasks_with_hypothesis']}/{agg['tasks_total']}",
              f"- tasks with ≥1 conclusion: {agg['tasks_with_conclusion']}/{agg['tasks_total']}",
              f"- tasks stuck (hypothesis left under_verification): "
              f"{agg['tasks_stuck_under_verification']}/{agg['tasks_total']}",
              f"- tasks errored: {agg['tasks_errored']}/{agg['tasks_total']}",
              f"- avg wall-clock time: {agg['avg_elapsed_sec']}s",
              "", "**Hypothesis verdict totals (across all tasks):**", ""]
    if agg["hypothesis_verdict_totals"]:
        lines += [f"- {k}: {v}" for k, v in sorted(agg["hypothesis_verdict_totals"].items())]
    else:
        lines.append("- (none)")
    lines += ["", "**Score totals:**", ""]
    lines += [f"- {k}: {v}" for k, v in sorted(agg["score_totals"].items())]

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks-file", type=Path, default=DEFAULT_TASKS_FILE,
                    help=f"JSONL task file (default: {DEFAULT_TASKS_FILE.relative_to(PROJECT_ROOT)}).")
    ap.add_argument("--ids", nargs="+", default=None,
                    help="Only run tasks with these ids (default: all in the file).")
    ap.add_argument("--limit", type=int, default=None, help="Only run the first N tasks.")
    ap.add_argument("--parallel", type=int, default=1,
                    help="Concurrent task runs (default 1 — these are real LLM-cost "
                         "runs, not isolated Docker containers; raise with care).")
    ap.add_argument("--validator-timeout", type=float, default=VALIDATOR_SETTLE_TIMEOUT,
                    help="Max seconds to wait per task for background validation to settle.")
    ap.add_argument("--output", type=Path, default=None, help="Markdown summary path.")
    ap.add_argument("--json-output", type=Path, default=None, help="JSON dump path.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Just print the loaded tasks and exit — no LLM calls.")
    return ap.parse_args()


async def main_async() -> None:
    ns = parse_args()
    global VALIDATOR_SETTLE_TIMEOUT
    VALIDATOR_SETTLE_TIMEOUT = ns.validator_timeout

    if not ns.tasks_file.exists():
        sys.exit(f"[bench] tasks file not found: {ns.tasks_file}")
    tasks = load_tasks(ns.tasks_file)
    if ns.ids:
        wanted = set(ns.ids)
        tasks = [t for t in tasks if t.id in wanted]
        missing = wanted - {t.id for t in tasks}
        if missing:
            sys.exit(f"[bench] unknown task ids: {sorted(missing)}")
    if ns.limit:
        tasks = tasks[:ns.limit]
    if not tasks:
        sys.exit("[bench] no tasks selected")

    print(f"[bench] {len(tasks)} task(s) from {ns.tasks_file}")
    if ns.dry_run:
        for t in tasks:
            print(f"  - {t.id} [{t.domain}] check={t.check.get('type')}: {t.question}")
        return

    from CoScientist.config import get_settings
    if not get_settings().research_graph.enabled:
        sys.exit("[bench] RESEARCH_GRAPH__ENABLED is off — this benchmark grades the "
                  "Research Context Graph and needs it on.")

    run_dir = RUNS_DIR / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_md = ns.output or run_dir / "summary.md"
    out_json = ns.json_output or run_dir / "summary.json"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    print(f"[bench] summary → {out_md}")
    print(f"[bench] json    → {out_json}")

    semaphore = asyncio.Semaphore(max(1, ns.parallel))
    records: list[dict] = []

    def flush() -> None:
        write_summary(records, out_md)
        out_json.write_text(
            json.dumps({"tasks": records, "aggregate": aggregate_metrics(records)},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    total = len(tasks)
    coros = [run_one_task(t, i + 1, total, semaphore) for i, t in enumerate(tasks)]
    try:
        for coro in asyncio.as_completed(coros):
            rec = await coro
            records.append(rec)
            flush()
    except KeyboardInterrupt:
        print("\n[bench] interrupted by user — partial results saved", file=sys.stderr)

    flush()
    print(f"\n[bench] done. summary → {out_md}")
    print(f"[bench]       json    → {out_json}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()