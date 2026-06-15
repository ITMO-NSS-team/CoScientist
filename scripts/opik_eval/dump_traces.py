#!/usr/bin/env python
"""Download Opik traces (+ their spans) since a date into a separate folder.

Server-side `start_time` filter (date_time column, ISO `...T00:00:00Z` required),
full content (`truncate=False`), one JSON file per trace plus an `index.jsonl`
summary rebuilt from disk.

Robust against Opik flakiness (it 429s AND 504s on heavy span queries):
  - span fetch retries on 429/5xx/JSON-decode transients with backoff;
  - each trace is isolated — a trace that keeps failing is written with
    spans=[] + a `spans_error`, and the dump moves on (no whole-run abort);
  - resumable — a re-run skips traces already on disk, so it finishes the rest.

Usage:
    python scripts/opik_eval/dump_traces.py --since 2026-06-12
    python scripts/opik_eval/dump_traces.py --since 2026-06-12 --force   # re-fetch all
    python scripts/opik_eval/dump_traces.py --since 2026-06-12 --no-spans
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from opik_client import get_client            # noqa: E402

REPO = _HERE.resolve().parents[1]
_TRANSIENT = {429, 500, 502, 503, 504}


def _iso(since: str) -> str:
    """Normalize a date/datetime to the ISO form Opik's OQL requires."""
    s = since.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return f"{s}T00:00:00Z"
    if s.endswith("Z") or "+" in s:
        return s
    return s + "Z"


def _slug(v) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(v))[:80] if v else "none"


def _query_head(trace_dump: dict) -> str:
    inp = trace_dump.get("input")
    try:
        if isinstance(inp, dict):
            parts = inp.get("parts")
            if parts and isinstance(parts, list):
                return str(parts[0].get("text", ""))[:120]
        return str(inp)[:120]
    except Exception:
        return ""


def fetch_spans(client, project: str, trace_id: str, max_spans: int, attempts: int = 6):
    """Fetch+dump spans for one trace, retrying transient 429/5xx/JSON errors.

    Returns (list_of_span_dicts, error_str_or_None). On terminal failure returns
    ([], "<reason>") so the caller can record it and keep going.
    """
    delay = 4
    for i in range(attempts):
        try:
            raw = client.search_spans(project_name=project, trace_id=trace_id,
                                      max_results=max_spans)
            return [s.model_dump() for s in raw], None
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            transient = (status in _TRANSIENT
                         or isinstance(exc, json.JSONDecodeError)
                         or any(w in str(exc).lower() for w in ("timeout", "gateway")))
            if i < attempts - 1 and transient:
                wait = delay
                if status == 429:
                    try:
                        reset = int((getattr(exc, "headers", {}) or {}).get("ratelimit-reset"))
                        wait = reset + 1
                    except Exception:
                        pass
                time.sleep(min(wait, 60))
                delay = min(delay * 2, 60)
                continue
            return [], f"{type(exc).__name__}:{status}"
    return [], "exhausted"


def rebuild_index(out: pathlib.Path) -> int:
    """(Re)build index.jsonl from every trace file on disk. Returns the count."""
    rows = []
    for fp in sorted(out.glob("*.json")):
        if fp.name == "index.json":
            continue
        try:
            d = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        td = d.get("trace", {})
        start = str(td.get("start_time") or "")[:19]
        rows.append({
            "trace_id": td.get("id"),
            "thread_id": td.get("thread_id"),
            "name": td.get("name"),
            "start_time": start,
            "duration_s": round(float(td["duration"]) / 1000, 1) if td.get("duration") else None,
            "span_count": td.get("span_count"),
            "spans_dumped": len(d.get("spans", [])),
            "spans_error": d.get("spans_error"),
            "model": (td.get("metadata") or {}).get("main_model"),
            "error": bool(td.get("error_info")),
            "query": _query_head(td),
            "file": fp.name,
        })
    rows.sort(key=lambda r: r["start_time"])
    (out / "index.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Dump Opik traces+spans since a date (resumable).")
    ap.add_argument("--since", default="2026-06-12", help="date (YYYY-MM-DD) or ISO datetime")
    ap.add_argument("--out", default=None, help="output dir (default opik_dump/traces_since_<since>)")
    ap.add_argument("--max", type=int, default=5000, help="max traces to pull")
    ap.add_argument("--limit", type=int, default=None, help="dump only the first N (testing)")
    ap.add_argument("--no-spans", action="store_true", help="trace bodies only, skip span fetch")
    ap.add_argument("--max-spans", type=int, default=2000, help="max spans per trace")
    ap.add_argument("--force", action="store_true", help="re-fetch traces already on disk")
    args = ap.parse_args()

    since_iso = _iso(args.since)
    out = pathlib.Path(args.out) if args.out else REPO / f"opik_dump/traces_since_{args.since}"
    out.mkdir(parents=True, exist_ok=True)

    client, project = get_client()
    print(f"=== dumping traces · project={project} · start_time >= {since_iso} -> {out} ===", flush=True)
    traces = client.search_traces(project_name=project,
                                  filter_string=f'start_time >= "{since_iso}"',
                                  max_results=args.max, truncate=False)
    traces = sorted(traces, key=lambda t: str(getattr(t, "start_time", "") or ""))
    if args.limit:
        traces = traces[: args.limit]
    print(f"found {len(traces)} traces", flush=True)

    n_done = n_skip = n_failed_spans = n_spans = 0
    for i, t in enumerate(traces, 1):
        td = t.model_dump()
        start = str(td.get("start_time") or "")[:19]
        fname = f"{start[:10]}__{_slug(td.get('thread_id'))}__{t.id}.json"
        fpath = out / fname
        if fpath.exists() and not args.force:
            n_skip += 1
            continue

        spans, spans_err = ([], None) if args.no_spans else fetch_spans(
            client, project, t.id, args.max_spans)
        n_spans += len(spans)
        if spans_err:
            n_failed_spans += 1
        rec = {"trace": td, "spans": spans}
        if spans_err:
            rec["spans_error"] = spans_err
        fpath.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str),
                         encoding="utf-8")
        n_done += 1
        if n_done % 10 == 0 or i == len(traces):
            print(f"  [{i}/{len(traces)}] {start} {td.get('thread_id')} "
                  f"spans={len(spans)}{' ERR=' + spans_err if spans_err else ''}", flush=True)

    total = rebuild_index(out)
    print(f"\n=== done: {total} traces on disk "
          f"(this run: {n_done} fetched, {n_skip} skipped, {n_failed_spans} span-fetch failures), "
          f"{n_spans} spans this run -> {out} (index.jsonl) ===", flush=True)
    if n_failed_spans:
        print(f"NOTE: {n_failed_spans} trace(s) saved with spans_error — re-run with --force "
              f"to retry just those (or accept the trace body without spans).", flush=True)


if __name__ == "__main__":
    main()
