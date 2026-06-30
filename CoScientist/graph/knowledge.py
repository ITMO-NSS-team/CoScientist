"""Knowledge graph = the VERIFICATION RECORD of the runs.

This is the auditable log of HOW answers were produced: for each query, the
agents involved and every tool call WITH its input and output — the chain of
connections you can check to verify the system ("it answered X about acetone
using tool T with input I → output O").

It is a focused projection of the execution graph: the goal subtrees only
(query → agents → tool calls with I/O → result), without the static agent roster.
The reusable distilled FACTS live in the separate MEMORY graph, which points
back to these nodes as its source.
"""
from __future__ import annotations

import collections
from typing import Any, Dict, List


def to_knowledge_graph(full: Dict[str, Any]) -> Dict[str, Any]:
    by_id = {n["id"]: n for n in full.get("nodes", [])}
    kids = collections.defaultdict(list)
    for n in full.get("nodes", []):
        for p in (n.get("parent_ids") or []):
            kids[p].append(n["id"])

    # Keep each goal and everything under it (the actual run activity); drop the
    # static roster (system + agent nodes) — it's not part of the audit trail.
    keep: set = set()
    for g in [n for n in full.get("nodes", []) if n.get("kind") == "goal"]:
        stack = [g["id"]]
        while stack:
            cur = stack.pop()
            if cur in keep:
                continue
            keep.add(cur)
            stack.extend(kids.get(cur, []))

    nodes = [by_id[i] for i in keep if i in by_id]
    edges = [e for e in full.get("edges", [])
             if e.get("src") in keep and e.get("dst") in keep]
    return {"run_id": full.get("run_id"), "view": "knowledge", "nodes": nodes, "edges": edges}
