#!/usr/bin/env python
"""Dump the FEDOT.MAS-invoking subtree of a real run from Opik (DEVGRAPH F015g evidence).

Finds a recent trace that called `fedot_tool` (the seam into FEDOT.MAS) and prints:
  - what was handed to FEDOT.MAS (the fedot_tool span INPUT = task_description + servers),
  - the FEDOT.MAS meta-agent's worker subtree (e.g. `invoke_agent molecule_generator`),
  - which tools that worker actually called and any "Tool 'X' not found" errors.

Usage: python scripts/opik_eval/dump_fedot_trace.py [scan_limit] [trace_id]
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from opik_client import get_client          # noqa: E402
from metrics import search_spans_retry      # noqa: E402


def short(x, n: int) -> str:
    if x is None:
        return ""
    s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)
    return s.replace("\n", " ")[:n]


def query_of(trace) -> str:
    try:
        if isinstance(trace.input, dict):
            return trace.input["parts"][0].get("text", "")
        return str(trace.input)
    except Exception:
        return str(trace.input)


def dump(client, project, trace) -> None:
    spans = search_spans_retry(client, project, trace.id, max_results=600)
    fedot = [s for s in spans if getattr(s, "name", None) == "fedot_tool"]
    if not fedot:
        print(f"(trace {trace.id} has no fedot_tool span)", file=sys.stderr)
        return
    md = trace.metadata or {}
    model = (md.get("main_model") or "?").split("/")[-1]
    print("=" * 90)
    print(f"TRACE {trace.id} | {str(trace.start_time)[:19]} | model={model} | spans={trace.span_count}")
    print(f"QUERY: {short(query_of(trace), 220)}")

    def children(pid):
        return [s for s in spans if getattr(s, "parent_span_id", None) == pid]

    def descendants(root_id):
        out, queue = [], list(children(root_id))
        while queue:
            s = queue.pop(0)
            out.append(s)
            queue.extend(children(s.id))
        return out

    for i, fs in enumerate(fedot, 1):
        print(f"\n##### fedot_tool call #{i} — what the agent hands to FEDOT.MAS #####")
        print("INPUT (task_description + servers passed):")
        print("  " + short(fs.input, 1800))
        print("OUTPUT:")
        print("  " + short(fs.output, 700))
        if getattr(fs, "error_info", None):
            print("  fedot_tool ERROR: " + short(fs.error_info, 400))

        desc = descendants(fs.id)
        if not desc:
            print("  (no child spans — FEDOT.MAS internals not nested under this span)")
            continue

        import re
        workers = [getattr(s, "name", None) for s in desc
                   if "invoke_agent" in (getattr(s, "name", "") or "")]
        tools_called = sorted({getattr(s, "name", None) for s in desc
                               if getattr(s, "type", None) == "tool"})
        n_llm = sum(1 for s in desc if getattr(s, "type", None) == "llm")
        print(f"\n  FEDOT.MAS subtree: {len(desc)} spans · {n_llm} llm · workers={workers}")
        print(f"  tools the workers actually called: {tools_called}")
        notfound = []
        for s in desc:
            ei = getattr(s, "error_info", None)
            if ei and "not found" in str(ei).lower():
                m = re.search(r"message=(\"|')(.*?)(\1)", str(ei), re.S)
                body = (m.group(2) if m else str(ei)).split("Possible causes")[0]
                notfound.append((getattr(s, "name", None), short(body, 320)))
        if notfound:
            print("  ⚠ TOOL-NOT-FOUND inside FEDOT.MAS (worker called a tool it wasn't given):")
            for nm, body in notfound:
                print(f"     @ {nm}: {body}")


def main() -> None:
    client, project = get_client()
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    want_id = sys.argv[2] if len(sys.argv) > 2 else None

    traces = client.search_traces(project_name=project, max_results=limit)
    if want_id:
        for t in traces:
            if t.id == want_id:
                dump(client, project, t)
                return
        print(f"trace {want_id} not in last {limit}", file=sys.stderr)
        return

    _KW = ("generate", "inhibitor", "molecul", "drug", "kras", "gsk", "btk", "stat3", "design")

    def is_molgen(t):
        return any(k in query_of(t).lower() for k in _KW)

    # prefer molecule-generation runs (the tool-not-found case), high span count first
    cands = [t for t in traces if (t.span_count or 0) >= 60]
    cands.sort(key=lambda t: (0 if is_molgen(t) else 1, -(t.span_count or 0)))
    print(f"scanning {len(cands)} high-span traces (molecule-gen first) for a fedot_tool call…",
          file=sys.stderr)
    for t in cands[:15]:
        spans = search_spans_retry(client, project, t.id, max_results=600)
        if any(getattr(s, "name", None) == "fedot_tool" for s in spans):
            dump(client, project, t)
            return
    print("no fedot_tool call found in scanned traces (try a larger scan_limit)", file=sys.stderr)


if __name__ == "__main__":
    main()
