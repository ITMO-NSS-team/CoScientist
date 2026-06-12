#!/usr/bin/env python
"""Reliable run -> Opik-trace correlation.

The fragile way (old ab_analyze): fetch the last N traces and filter `thread_id`
on the client — your trace silently drops off if >N other traces arrived since.

This module instead:
  1. Filters `thread_id` SERVER-SIDE via Opik OQL (`search_traces(filter_string=...)`),
     so a match is found regardless of how many other traces exist.
  2. Persists a durable manifest (`results/trace_manifest.jsonl`) mapping each
     run's `session_id` (== the trace's `thread_id`) to its real `trace_id`, plus
     the query/condition/model/timestamps. After that, pulling "the trace for a
     run" is a direct lookup by session_id OR by query text — at any later time.

Every CoScientist run sets a `session_id`; the ADK OpikTracer records it as the
trace `thread_id` (opik/integrations/adk/opik_tracer.py:173). So `session_id` is
the join key end to end.

CLI:
    python scripts/opik_eval/trace_locator.py ab_B2_00_225459      # exact session
    python scripts/opik_eval/trace_locator.py --prefix ab_B2_      # a run family
    python scripts/opik_eval/trace_locator.py --query "GSK-3beta"  # by manifest query text
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from opik_client import get_client            # noqa: E402
from metrics import search_spans_retry        # noqa: E402

MANIFEST_PATH = _HERE.resolve().parents[1] / "scripts/experiments/results/trace_manifest.jsonl"


# ----------------------------------------------------------------------------- OQL
def _oql(field: str, op: str, value: str) -> str:
    """Build one server-side OQL clause, rejecting values that would break quoting.

    Opik string literals are double-quoted; session_ids/prefixes are [A-Za-z0-9_-],
    so a literal `"` in the value means a bug or an injection — refuse it.
    """
    if '"' in value or "\\" in value:
        raise ValueError(f"unsafe OQL value: {value!r}")
    if op not in ("=", "contains", "starts_with", "ends_with"):
        raise ValueError(f"unsupported OQL op: {op}")
    return f'{field} {op} "{value}"'


def _newest_first(traces: list):
    return sorted(traces, key=lambda t: str(getattr(t, "start_time", "") or ""), reverse=True)


def resolve_traces(*, eq: str | None = None, prefix: str | None = None,
                   contains: str | None = None, client=None, project: str | None = None,
                   limit: int = 100, wait_for_at_least: int | None = None,
                   wait_for_timeout: int = 60) -> list:
    """Server-side `thread_id` filter -> traces (newest first).

    Exactly one of eq/prefix/contains. `wait_for_at_least` blocks until that many
    matching traces are indexed (use 1 right after a run, before the UI catches up).
    """
    if sum(x is not None for x in (eq, prefix, contains)) != 1:
        raise ValueError("pass exactly one of eq / prefix / contains")
    if client is None:
        client, project = get_client()
    if eq is not None:
        fs = _oql("thread_id", "=", eq)
    elif prefix is not None:
        fs = _oql("thread_id", "starts_with", prefix)
    else:
        fs = _oql("thread_id", "contains", contains)
    kw = {"project_name": project, "filter_string": fs, "max_results": limit}
    if wait_for_at_least is not None:
        kw["wait_for_at_least"] = wait_for_at_least
        kw["wait_for_timeout"] = wait_for_timeout
    return _newest_first(client.search_traces(**kw))


def find_trace(session_id: str, *, client=None, project: str | None = None,
               wait: bool = False, wait_timeout: int = 90):
    """Newest trace whose thread_id == session_id, or None."""
    hits = resolve_traces(eq=session_id, client=client, project=project,
                          wait_for_at_least=1 if wait else None, wait_for_timeout=wait_timeout)
    return hits[0] if hits else None


def get_spans(trace_id: str, *, client=None, project: str | None = None, max_results: int = 700):
    if client is None:
        client, project = get_client()
    return search_spans_retry(client, project, trace_id, max_results=max_results)


# ------------------------------------------------------------------------- manifest
def _trace_url(trace_id: str) -> str | None:
    """Best-effort deep link to the trace in the Opik UI (None if SDK can't build one)."""
    try:
        from opik import url_helpers  # type: ignore
        for fn in ("get_trace_url_by_id", "get_traces_url"):
            f = getattr(url_helpers, fn, None)
            if f:
                try:
                    return f(trace_id)
                except TypeError:
                    continue
    except Exception:
        pass
    return None


def record_run(session_id: str, *, query: str | None = None, condition: str | None = None,
               model: str | None = None, extra: dict | None = None, client=None,
               project: str | None = None, manifest: pathlib.Path = MANIFEST_PATH,
               wait_timeout: int = 90) -> dict | None:
    """Resolve the trace for a finished run and append it to the manifest.

    Best-effort: returns the entry on success, None if the trace can't be resolved
    (never raises — a tracing hiccup must not fail the experiment that called it).
    """
    try:
        if client is None:
            client, project = get_client()
        t = find_trace(session_id, client=client, project=project, wait=True,
                       wait_timeout=wait_timeout)
        if t is None:
            return None
        md = t.metadata or {}
        entry = {
            "session_id": session_id,
            "thread_id": getattr(t, "thread_id", session_id),
            "trace_id": t.id,
            "query": query,
            "condition": condition,
            "model": model or md.get("main_model"),
            "start_time": str(getattr(t, "start_time", ""))[:19],
            "duration_s": round(float(t.duration) / 1000, 1) if getattr(t, "duration", None) else None,
            "project": project,
            "url": _trace_url(t.id),
            "recorded_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        if extra:
            entry["extra"] = extra
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry
    except Exception as exc:  # never let trace bookkeeping break a run
        print(f"[trace_locator] record_run failed for {session_id}: {exc!r}", file=sys.stderr)
        return None


def read_manifest(manifest: pathlib.Path = MANIFEST_PATH) -> list[dict]:
    if not manifest.exists():
        return []
    out = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def lookup(*, session_id: str | None = None, query_substr: str | None = None,
           condition: str | None = None, manifest: pathlib.Path = MANIFEST_PATH) -> list[dict]:
    """Find recorded runs by session_id (exact), query substring, or condition.

    Manifest-first means you can get a trace_id with NO Opik round-trip; entries
    are returned most-recent-last (append order)."""
    rows = read_manifest(manifest)
    if session_id is not None:
        rows = [r for r in rows if r.get("session_id") == session_id]
    if query_substr is not None:
        q = query_substr.lower()
        rows = [r for r in rows if q in str(r.get("query") or "").lower()]
    if condition is not None:
        rows = [r for r in rows if r.get("condition") == condition]
    return rows


# ------------------------------------------------------------------------------ CLI
def _print_trace(t) -> None:
    md = t.metadata or {}
    dur = round(float(t.duration) / 1000, 1) if getattr(t, "duration", None) else "?"
    print(f"  {getattr(t, 'thread_id', '?')}  trace={t.id}  "
          f"start={str(getattr(t, 'start_time', ''))[:19]}  dur={dur}s  "
          f"model={(md.get('main_model') or '?').split('/')[-1]}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Resolve Opik traces for a run (server-side).")
    ap.add_argument("session", nargs="?", help="exact session_id / thread_id")
    ap.add_argument("--prefix", help="thread_id prefix (a run family, e.g. ab_B2_)")
    ap.add_argument("--contains", help="thread_id substring")
    ap.add_argument("--query", help="manifest lookup by query substring (no Opik call)")
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()

    if args.query:
        rows = lookup(query_substr=args.query)
        print(f"=== manifest: {len(rows)} run(s) matching query ~ {args.query!r} ===")
        for r in rows:
            print(f"  session={r['session_id']}  trace={r.get('trace_id')}  "
                  f"cond={r.get('condition')}  model={(r.get('model') or '?').split('/')[-1]}  "
                  f"start={r.get('start_time')}\n      query: {str(r.get('query'))[:100]}")
        return

    if args.session:
        traces = resolve_traces(eq=args.session, limit=args.limit)
        kind = f"thread_id == {args.session!r}"
    elif args.prefix:
        traces = resolve_traces(prefix=args.prefix, limit=args.limit)
        kind = f"thread_id starts_with {args.prefix!r}"
    elif args.contains:
        traces = resolve_traces(contains=args.contains, limit=args.limit)
        kind = f"thread_id contains {args.contains!r}"
    else:
        ap.error("give a session_id, --prefix, --contains, or --query")
        return

    print(f"=== {len(traces)} trace(s) · {kind} (server-side) ===")
    for t in traces:
        _print_trace(t)
    man = {r["session_id"]: r for r in read_manifest()}
    n_manifest = sum(1 for t in traces if getattr(t, "thread_id", None) in man)
    print(f"--- {n_manifest}/{len(traces)} also in manifest ({MANIFEST_PATH.name}) ---")


if __name__ == "__main__":
    main()
