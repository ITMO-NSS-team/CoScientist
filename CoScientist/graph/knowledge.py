"""Knowledge graph (interpretable): Question → Facts → their Sources.

Reads as "here are the facts this run established and, for each, the data source
(a tool/agent call, with its input and output) it came from" — NOT the mechanical
execution call-tree. Structural nodes (orchestrator/planner/agent nesting) are
intentionally dropped.

    Question
      ├─ Fact: "acetaldehyde is hepatotoxic"  ──from_source──▶ [tavily_extract] (in/out)
      ├─ Fact: "CYP2E1 → oxidative stress"     ──from_source──▶ [ResearchAgent] (in/out)
      └─ Fact: "fibrosis precedes cirrhosis"   ──from_source──▶ [tavily_search] (in/out)

Facts are the claims extracted for THIS run into the cross-run memory (relations
between domain entities). Each fact is attributed to the run's data source whose
output best matches it (content overlap), so you can click the source to audit
its exact input/output.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List


def _tokens(text: str) -> set:
    return {w for w in re.split(r"[^a-z0-9]+", (text or "").lower()) if len(w) > 2}


def _norm(s: str) -> str:
    return (s or "").rstrip(" …")[:80]


def _from_run(sources: List[str], gnorm: str) -> bool:
    return bool(gnorm) and any(
        gnorm.startswith(_norm(s)) or _norm(s).startswith(gnorm) for s in (sources or [])
    )


def to_knowledge_graph(full: Dict[str, Any]) -> Dict[str, Any]:
    by_id = {n["id"]: n for n in full.get("nodes", [])}

    # Data sources = tool/agent calls that actually produced output (with I/O).
    sources = [n for n in full.get("nodes", [])
               if n.get("kind") in ("tool_call", "agent_call") and n.get("output")]
    src_tokens = {s["id"]: _tokens(f"{s.get('label', '')} {s.get('output', '')}") for s in sources}

    try:
        from CoScientist.graph.memory_store import knowledge_memory as km
        ents, rels = km.entities, list(km.relations.values())
    except Exception:  # noqa: BLE001
        ents, rels = {}, []

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    seen: set = set()

    def node(nid: str, kind: str, label: str, **extra: Any) -> None:
        if nid in seen:
            return
        seen.add(nid)
        nodes.append({"id": nid, "kind": kind, "label": label or "", "status": "success", **extra})

    def edge(src: str, dst: str, etype: str) -> None:
        edges.append({"src": src, "dst": dst, "type": etype})

    for g in [n for n in full.get("nodes", []) if n.get("kind") == "goal"]:
        qid = "q:" + g["id"]
        query = g.get("label", "")
        node(qid, "question", "Q: " + query[:60], output=query)
        gnorm = _norm(query)

        # This run's facts = memory relations (claims) established for this query.
        for r in [r for r in rels if _from_run(r.get("sources"), gnorm)]:
            sn = ents.get(r["src"], {}).get("name", r["src"])
            dn = ents.get(r["dst"], {}).get("name", r["dst"])
            val = (r.get("attrs") or {}).get("value")
            claim = f"{sn} {r.get('type', '')} {dn}" + (f" = {val}" if val is not None else "")
            fid = f"fact:{r['src']}|{r.get('type', '')}|{r['dst']}"
            node(fid, "fact", claim, output=claim)
            edge(qid, fid, "has_finding")

            # Attribute the fact to the source whose output best matches it.
            ftok = _tokens(claim)
            best, best_score = None, 0
            for s in sources:
                score = len(ftok & src_tokens[s["id"]])
                if score > best_score:
                    best, best_score = s, score
            if best is not None and best_score >= 1:
                b = by_id[best["id"]]
                node(best["id"], b["kind"], b.get("label") or best["id"],
                     executor_agent=b.get("executor_agent"), input=b.get("input"),
                     output=b.get("output"), status=b.get("status", "success"))
                edge(fid, best["id"], "from_source")

    return {"run_id": full.get("run_id"), "view": "knowledge", "nodes": nodes, "edges": edges}
