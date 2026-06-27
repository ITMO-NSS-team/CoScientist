"""In-memory NetworkX store for execution graphs, one DiGraph per run_id.

MVP backend. Snapshots each run to JSON for persistence/replay (Fact 1). The
public methods are storage-engine agnostic so a Neo4j backend can replace this
without touching emitters/projection.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Dict, Optional

import networkx as nx

from CoScientist.graph.models import Edge, Node, StatusUpdate

logger = logging.getLogger(__name__)


class GraphStore:
    def __init__(self, snapshot_dir: Optional[str] = None) -> None:
        self._graphs: Dict[str, nx.DiGraph] = {}
        self._lock = threading.Lock()
        self._snapshot_dir = Path(snapshot_dir) if snapshot_dir else None
        if self._snapshot_dir:
            self._snapshot_dir.mkdir(parents=True, exist_ok=True)

    def _g(self, run_id: str) -> nx.DiGraph:
        g = self._graphs.get(run_id)
        if g is None:
            g = nx.DiGraph()
            self._graphs[run_id] = g
        return g

    # ── writes ──────────────────────────────────────────────────────────────
    def add_node(self, node: Node) -> None:
        with self._lock:
            g = self._g(node.run_id)
            data = node.model_dump()
            if g.has_node(node.id):
                # merge: keep existing fields unless the new payload sets them
                existing = g.nodes[node.id]
                for k, v in data.items():
                    if v is not None or k not in existing:
                        existing[k] = v
            else:
                g.add_node(node.id, **data)
            self._snapshot(node.run_id)

    def add_edge(self, edge: Edge) -> None:
        with self._lock:
            g = self._g(edge.run_id)
            # ensure endpoints exist as placeholders so an edge never dangles
            for nid in (edge.src, edge.dst):
                if not g.has_node(nid):
                    g.add_node(nid, id=nid, run_id=edge.run_id, kind="tool_call", status="running")
            g.add_edge(edge.src, edge.dst, type=edge.type)
            self._snapshot(edge.run_id)

    def set_status(self, node_id: str, upd: StatusUpdate) -> bool:
        with self._lock:
            g = self._g(upd.run_id)
            if not g.has_node(node_id):
                return False
            n = g.nodes[node_id]
            if upd.status is not None:
                n["status"] = upd.status
            if upd.output is not None:
                n["output"] = upd.output
            if upd.verdict is not None:
                n["verdict"] = upd.verdict
            if upd.t_end is not None:
                n["t_end"] = upd.t_end
            self._snapshot(upd.run_id)
            return True

    # ── reads ───────────────────────────────────────────────────────────────
    def _full_unlocked(self, run_id: str) -> dict:
        g = self._graphs.get(run_id)
        if g is None:
            return {"run_id": run_id, "nodes": [], "edges": []}
        nodes = [dict(g.nodes[n]) for n in g.nodes]
        edges = [{"src": u, "dst": v, "type": d.get("type")} for u, v, d in g.edges(data=True)]
        return {"run_id": run_id, "nodes": nodes, "edges": edges}

    def full(self, run_id: str) -> dict:
        with self._lock:
            return self._full_unlocked(run_id)

    def run_ids(self) -> list:
        with self._lock:
            return list(self._graphs.keys())

    # ── persistence ─────────────────────────────────────────────────────────
    def _snapshot(self, run_id: str) -> None:
        if not self._snapshot_dir:
            return
        try:
            path = self._snapshot_dir / f"{run_id}.json"
            tmp = path.with_suffix(".json.tmp")
            # NOTE: caller already holds self._lock — use the unlocked read to
            # avoid re-acquiring the non-reentrant lock (would deadlock).
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(self._full_unlocked(run_id), f, ensure_ascii=False, default=str)
            os.replace(tmp, path)
        except Exception as exc:  # noqa: BLE001 — persistence must never break writes
            logger.warning("Graph snapshot failed for %s: %s", run_id, exc)
