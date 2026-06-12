#!/usr/bin/env python
"""Analyze A/B/C run behavior from Opik by thread_id prefix (ab_A_/ab_B_/ab_C_)."""
import json
import statistics
import sys
from collections import Counter

sys.path.insert(0, "scripts/opik_eval")
from opik_client import get_client          # noqa: E402
from metrics import search_spans_retry      # noqa: E402


def analyze(prefix: str, limit: int = 40):
    client, project = get_client()
    traces = client.search_traces(project_name=project, max_results=limit)
    sel = [t for t in traces if str(getattr(t, "thread_id", "") or "").startswith(prefix)]
    rows = []
    for t in sel:
        spans = search_spans_retry(client, project, t.id, max_results=700)
        llm = [s for s in spans if getattr(s, "type", None) == "llm"]
        tool = [s for s in spans if getattr(s, "type", None) == "tool"]
        names = Counter(getattr(s, "name", None) for s in tool)
        notfound = sum(1 for s in spans
                       if getattr(s, "error_info", None) and "not found" in str(s.error_info).lower())
        reached_fedot = names.get("fedot_tool", 0) > 0
        reached_gen = any("generate" in str(n or "") for n in names)
        calls = Counter(
            (getattr(s, "name", None),
             json.dumps(s.input, sort_keys=True, ensure_ascii=False)[:140] if isinstance(s.input, dict) else "")
            for s in tool)
        maxrep = max(calls.values()) if calls else 0
        dur = float(t.duration) / 1000 if t.duration else 0
        # top tool names (excluding the per-step critic which is 'general')
        top = ", ".join(f"{n}×{c}" for n, c in names.most_common(8))
        rows.append({"thread": t.thread_id, "n_llm": len(llm), "n_tool": len(tool),
                     "notfound": notfound, "fedot": reached_fedot, "gen": reached_gen,
                     "maxrep": maxrep, "dur": round(dur), "tools": top})
    return rows


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else "ab_A_"
    rows = analyze(prefix)
    if not rows:
        print(f"no traces for thread prefix {prefix}")
        return
    print(f"=== {prefix} · n={len(rows)} ===")
    print(f"median LLM calls : {statistics.median(r['n_llm'] for r in rows)}  "
          f"(max {max(r['n_llm'] for r in rows)})")
    print(f"median tool calls: {statistics.median(r['n_tool'] for r in rows)}")
    print(f"reached fedot_tool : {sum(r['fedot'] for r in rows)}/{len(rows)}")
    print(f"reached generation : {sum(r['gen'] for r in rows)}/{len(rows)}")
    print(f"tool-not-found Σ   : {sum(r['notfound'] for r in rows)}")
    print(f"max-repeated-call median: {statistics.median(r['maxrep'] for r in rows)}")
    print("--- per run ---")
    for r in rows:
        print(f"  {r['thread']}: llm={r['n_llm']} tool={r['n_tool']} fedot={int(r['fedot'])} "
              f"gen={int(r['gen'])} notfound={r['notfound']} maxrep={r['maxrep']} dur={r['dur']}s")
        print(f"      tools: {r['tools']}")


if __name__ == "__main__":
    main()
