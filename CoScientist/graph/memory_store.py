"""Per-user knowledge graph — persistent semantic *graph memory*.

While ``KnowledgeGraph`` (memory.py) records ONE session's execution trace, this
store accumulates the *semantic* layer ACROSS one user's sessions: the typed
domain entities and relations extracted from agent outputs (semantic.py). It is
what lets a later session reuse what that user established without exposing it
to another local user.

Design notes addressing the obvious failure modes:
  * **Dedup** — entities are keyed by a CANONICAL id derived from the name
    (lowercased, punctuation-folded), independent of the LLM's chosen ``type:``
    prefix, so "molecule:paracetamol" and "drug:Paracetamol" collapse into ONE
    node. Aliases (the raw keys/names seen) are kept for lookup.
  * **Reuse** — facts are retrieved two ways: ``relevant(query)`` (ranked search,
    for the prompt injection) and ``neighbors(entity)`` (1-hop traversal, so an
    agent can search then walk the graph).

One data-driven template (Node/Edge shape), persisted to JSON so it survives
restarts and grows over time.
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from CoScientist.graph.semantic import Extraction
from CoScientist.graph.session_scope import (
    DEFAULT_SESSION_KEY,
    SessionKey,
    safe_component,
    session_key,
)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")


def _canon(name: str = "", key: str = "") -> str:
    """Canonical entity id from the name (preferred) or the key's tail."""
    base = name or (key.split(":", 1)[-1] if ":" in (key or "") else key)
    return _slug(base) or _slug(key) or "entity"


def _tokens(text: str) -> set:
    return {w for w in re.split(r"[^a-z0-9]+", (text or "").lower()) if len(w) > 2}


class KnowledgeMemory:
    def __init__(self, path: Optional[str] = None) -> None:
        self.path = Path(path or os.getenv("KG_MEMORY_PATH", "graph_runs/knowledge_memory.json"))
        self.entities: Dict[str, Dict[str, Any]] = {}     # canon id -> entity record
        self.relations: Dict[str, Dict[str, Any]] = {}    # "src|type|dst" -> relation record
        self._lock = threading.Lock()
        self._load()

    # ── persistence ───────────────────────────────────────────────────────────
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.entities = {e["id"]: e for e in data.get("entities", []) if e.get("id")}
            self.relations = {self._rkey(r["src"], r["type"], r["dst"]): r
                              for r in data.get("relations", []) if r.get("src") and r.get("dst")}
        except Exception:  # noqa: BLE001
            self.entities, self.relations = {}, {}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            payload = {"entities": list(self.entities.values()), "relations": list(self.relations.values())}
            tmp.write_text(json.dumps(payload, ensure_ascii=False, default=str, indent=2), encoding="utf-8")
            os.replace(tmp, self.path)
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _rkey(src: str, type_: str, dst: str) -> str:
        return f"{src}|{type_}|{dst}"

    @staticmethod
    def _add_prov(rec: Dict[str, Any], prov: Dict[str, Any]) -> None:
        """Attach a provenance ref pointing into the knowledge graph (best-effort,
        deduped by the producing result node, capped)."""
        if not prov:
            return
        pl = rec.setdefault("provenance", [])
        key = prov.get("result_id") or prov.get("goal_id") or prov.get("goal")
        if any((p.get("result_id") or p.get("goal_id") or p.get("goal")) == key for p in pl):
            return
        pl.append(prov)
        del pl[:-20]

    def _prune_orphans(self) -> None:
        """Drop entities not connected by any relation — the memory graph must
        have no disconnected nodes (a fact only counts if it links to something)."""
        connected = set()
        for r in self.relations.values():
            connected.add(r["src"])
            connected.add(r["dst"])
        self.entities = {k: e for k, e in self.entities.items() if k in connected}

    # ── ingest (accumulate + dedup across runs) ────────────────────────────────
    def ingest(self, extraction: Extraction, source: str = "",
               refs: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
        added = {"entities": 0, "relations": 0}
        # provenance ref into the knowledge graph (which run/goal/result produced it)
        prov = dict(refs or {})
        if source:
            prov.setdefault("goal", source)
        with self._lock:
            keymap: Dict[str, str] = {}   # the extraction's raw key/name -> canonical id
            for e in extraction.entities:
                cid = _canon(e.name, e.key)
                keymap[e.key] = cid
                if e.name:
                    keymap[e.name] = cid
                cur = self.entities.get(cid)
                if cur is None:
                    cur = self.entities[cid] = {
                        "id": cid, "type": e.type, "name": e.name or cid,
                        "description": (e.description or "").strip(),
                        "attrs": dict(e.attrs or {}), "mentions": 1,
                        "aliases": sorted({a for a in (e.key, e.name) if a}),
                        "sources": [source] if source else [], "provenance": [],
                    }
                    added["entities"] += 1
                else:
                    cur["attrs"].update(e.attrs or {})
                    cur["mentions"] = cur.get("mentions", 1) + 1
                    if e.name and len(e.name) > len(cur.get("name", "")):
                        cur["name"] = e.name
                    if e.description and len(e.description) > len(cur.get("description", "")):
                        cur["description"] = e.description.strip()
                    cur["aliases"] = sorted(set(cur.get("aliases", [])) | {a for a in (e.key, e.name) if a})
                    if source and source not in cur["sources"]:
                        cur["sources"].append(source)
                self._add_prov(cur, prov)

            def resolve(ref: str) -> str:
                return keymap.get(ref) or _canon(key=ref)

            for r in extraction.relations:
                if not (r.src and r.dst):
                    continue
                src, dst = resolve(r.src), resolve(r.dst)
                if src == dst:
                    continue
                rk = self._rkey(src, r.type, dst)
                cur = self.relations.get(rk)
                if cur is None:
                    cur = self.relations[rk] = {"src": src, "dst": dst, "type": r.type,
                                                "attrs": dict(getattr(r, "attrs", {}) or {}),
                                                "count": 1, "sources": [source] if source else [],
                                                "provenance": []}
                    added["relations"] += 1
                else:
                    cur["attrs"].update(getattr(r, "attrs", {}) or {})
                    cur["count"] = cur.get("count", 1) + 1
                    if source and source not in cur["sources"]:
                        cur["sources"].append(source)
                self._add_prov(cur, prov)

            self._prune_orphans()
            self._save()
        return added

    # ── reads ───────────────────────────────────────────────────────────────
    def full(self) -> Dict[str, Any]:
        """Node/Edge-shaped dict so the same viz renders the memory graph.
        Entity labels carry a salient attr so the graph reads at a glance."""
        nodes = []
        for e in self.entities.values():
            attrs = e.get("attrs") or {}
            salient = next(iter(attrs.items()), None)
            label = e.get("name") or e["id"]
            if salient:
                label = f"{label} · {salient[0]}={salient[1]}"
            # provenance: pointers into the KNOWLEDGE graph (which run/goal/result
            # produced this fact) — so a memory fact is verifiable against the log.
            prov = e.get("provenance", [])
            prov_str = "; ".join(
                f"run={p.get('run')} goal={p.get('goal_id')} result={p.get('result_id')}"
                for p in prov[-3:]
            ) or str(e.get("sources", []))
            desc = e.get("description") or ""
            nodes.append({
                "id": e["id"], "run_id": "memory", "kind": "entity",
                "label": label, "status": "success",
                "semantic": {"type": e.get("type"), "entity": e["id"]},
                "input": attrs,
                "output": ((f"{desc}\n" if desc else "")
                           + f"type={e.get('type')} · mentions={e.get('mentions', 1)} · "
                           f"source (knowledge graph): {prov_str} · aliases={e.get('aliases', [])}"),
            })
        edges = []
        for r in self.relations.values():
            if r["src"] in self.entities and r["dst"] in self.entities:
                lbl = r.get("type", "")
                val = (r.get("attrs") or {}).get("value")
                edges.append({"src": r["src"], "dst": r["dst"],
                              "type": f"{lbl} ({val})" if val is not None else lbl})
        return {"run_id": "memory", "view": "memory", "nodes": nodes, "edges": edges}

    def _match(self, ref: str) -> Optional[Dict[str, Any]]:
        """Resolve a key/name/alias to an entity record."""
        cid = _canon(key=ref)
        if cid in self.entities:
            return self.entities[cid]
        cid2 = _canon(name=ref)
        if cid2 in self.entities:
            return self.entities[cid2]
        for e in self.entities.values():
            if ref in e.get("aliases", []) or ref.lower() == (e.get("name", "").lower()):
                return e
        return None

    def relevant(self, query: str, k: int = 8) -> List[Dict[str, Any]]:
        q = _tokens(query)
        if not q:
            return sorted(self.entities.values(), key=lambda e: -e.get("mentions", 1))[:k]
        scored = []
        for e in self.entities.values():
            blob = " ".join([e.get("name", ""), e.get("id", ""), e.get("type", ""),
                             " ".join(map(str, (e.get("attrs") or {}).values())),
                             " ".join(e.get("aliases", []))])
            score = len(q & _tokens(blob)) + 0.1 * e.get("mentions", 1)
            if score >= 1:
                scored.append((score, e))
        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:k]]

    def neighbors(self, ref: str) -> Dict[str, Any]:
        """The entity plus its 1-hop facts — for an agent to walk the graph."""
        e = self._match(ref)
        if e is None:
            return {"found": False, "query": ref}
        cid = e["id"]
        facts = []
        for r in self.relations.values():
            if r["src"] == cid:
                other = self.entities.get(r["dst"], {})
                facts.append({"fact": f"{e['name']} {r['type']} {other.get('name', r['dst'])}",
                              "direction": "out", "type": r["type"], "neighbor": r["dst"],
                              "attrs": r.get("attrs", {})})
            elif r["dst"] == cid:
                other = self.entities.get(r["src"], {})
                facts.append({"fact": f"{other.get('name', r['src'])} {r['type']} {e['name']}",
                              "direction": "in", "type": r["type"], "neighbor": r["src"],
                              "attrs": r.get("attrs", {})})
        return {"found": True,
                "entity": {k: e.get(k) for k in
                           ("id", "type", "name", "attrs", "mentions", "sources", "provenance")},
                "facts": facts}

    def relevant_summary(self, query: str = "", k: int = 8) -> str:
        ents = self.relevant(query, k)
        if not ents:
            return ""
        keys = {e["id"] for e in ents}
        lines = ["KNOWLEDGE MEMORY — facts established in PRIOR runs (build on these, don't redo):"]
        for e in ents:
            attrs = ", ".join(f"{kk}={vv}" for kk, vv in (e.get("attrs") or {}).items())
            desc = e.get("description") or ""
            lines.append(f"- {e.get('name')} [{e.get('type')}]"
                         + (f": {desc}" if desc else "")
                         + (f" ({attrs})" if attrs else ""))
        for r in [r for r in self.relations.values() if r["src"] in keys and r["dst"] in keys][:k]:
            sn = self.entities.get(r["src"], {}).get("name", r["src"])
            dn = self.entities.get(r["dst"], {}).get("name", r["dst"])
            val = (r.get("attrs") or {}).get("value")
            lines.append(f"  · {sn} {r.get('type')} {dn}" + (f" ({val})" if val is not None else ""))
        return "\n".join(lines)

    def known_types(self):
        """(entity types, relation types) currently in the graph — fed back to
        the extractor so it reuses them instead of minting near-duplicates."""
        ent = {e.get("type") for e in self.entities.values() if e.get("type")}
        rel = {r.get("type") for r in self.relations.values() if r.get("type")}
        return ent, rel

    def stats(self) -> Dict[str, int]:
        return {"entities": len(self.entities), "relations": len(self.relations),
                "entity_types": len(self.known_types()[0]), "relation_types": len(self.known_types()[1])}


# Legacy/default memory for standalone utilities and tests without ADK context.
knowledge_memory = KnowledgeMemory()
_knowledge_memories: Dict[SessionKey, KnowledgeMemory] = {
    DEFAULT_SESSION_KEY: knowledge_memory,
}
_registry_lock = threading.RLock()


def get_knowledge_memory(
    context: Any = None,
    *,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> KnowledgeMemory:
    """Return persistent semantic memory for one ADK user.

    ``session_id`` is accepted so every resolver has the same interface, but
    non-default sessions of the same user intentionally share this memory.
    """
    resolved = session_key(context, user_id=user_id, session_id=session_id)
    key = (
        DEFAULT_SESSION_KEY
        if resolved == DEFAULT_SESSION_KEY
        else (resolved[0], "__user_memory__")
    )
    with _registry_lock:
        memory = _knowledge_memories.get(key)
        if memory is None:
            configured = Path(
                os.getenv("KG_MEMORY_PATH", "graph_runs/knowledge_memory.json")
            )
            path = (
                configured.parent
                / "users"
                / safe_component(key[0])
                / "knowledge_memory.json"
            )
            memory = KnowledgeMemory(str(path))
            _knowledge_memories[key] = memory
        return memory
