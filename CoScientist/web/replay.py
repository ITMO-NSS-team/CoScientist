"""Replay a recorded session into the live UI, for a demonstration.

A study that took six hours cannot be shown at a booth, and re-running it on
stage is not an option either. This replays what was actually recorded — the
agent events of one session and the growth of its research graph — into a fresh
session of the running server, at a chosen speed. What the screen shows is the
real run; only the clock is different.

Nothing here fabricates content. Events are the recorded events; the graph is
rebuilt from the final snapshot by inserting each node and edge at its own
recorded ``created_at``, so the shape at any moment is the shape the study
actually had at that moment.

Started through ``POST /api/demo/replay``; the caller gets the new session id and
watches it like any other.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("CoScientist.web.replay")

#: Marker put on every replayed event, so a recording is never mistaken for a
#: live run by anything reading the transcript later.
REPLAY_FLAG = "replayed_from"


def _load_events(path: Path) -> List[Dict[str, Any]]:
    """Recorded UI events, oldest first. Accepts a JSONL or a JSON array."""
    if not path.exists():
        raise FileNotFoundError(f"no event transcript at {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        events = json.loads(text)
    else:
        events = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [e for e in events if isinstance(e, dict)]


def _event_time(event: Dict[str, Any]) -> Optional[float]:
    """Seconds since epoch for an event, from whichever field carries its time."""
    for key in ("ts", "timestamp", "time", "created_at"):
        value = event.get(key)
        if value is None:
            continue
        if isinstance(value, (int, float)):
            return float(value)
        try:
            from datetime import datetime
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
        except ValueError:
            continue
    return None


def _graph_steps(graph: Dict[str, Any]) -> List[Tuple[float, str, Dict[str, Any]]]:
    """The graph's growth, as (time, kind, item) sorted by recorded creation.

    Edges inherit the later of their endpoints' times when they carry none, so an
    edge never appears before the nodes it connects.
    """
    nodes = graph.get("nodes") or []
    born = {n["id"]: float(n.get("created_at") or 0) for n in nodes if isinstance(n, dict)}
    steps: List[Tuple[float, str, Dict[str, Any]]] = [
        (born.get(n["id"], 0.0), "node", n) for n in nodes if isinstance(n, dict)
    ]
    for e in graph.get("edges") or []:
        if not isinstance(e, dict):
            continue
        t = e.get("created_at")
        t = float(t) if t else max(born.get(e.get("from"), 0.0), born.get(e.get("to"), 0.0))
        steps.append((t, "edge", e))
    steps.sort(key=lambda s: s[0])
    return steps


class ReplaySession:
    """One replay in flight. Cancelled by dropping the reference."""

    def __init__(self, runtime, user_id: str, session_id: str, *,
                 events: List[Dict[str, Any]], graph: Dict[str, Any],
                 speed: float, max_gap: float, source: str,
                 min_gap: float = 0.4, warmup: float = 8.0) -> None:
        self.runtime = runtime
        self.key = (user_id, session_id)
        self.events = events
        self.graph = graph
        self.speed = max(1.0, float(speed))
        self.max_gap = max(0.0, float(max_gap))
        # A floor, not only a cap. The recording is bursty -- dozens of events
        # share a second, then nothing for forty minutes -- and replaying that
        # faithfully dumps the burst in one frame. The floor spaces the burst
        # out without reordering it.
        self.min_gap = max(0.0, float(min_gap))
        self.warmup = max(0.0, float(warmup))
        self.source = source
        self.cancelled = False

    # ── timing ───────────────────────────────────────────────────────────────
    def _sleep_for(self, previous: Optional[float], current: Optional[float]) -> float:
        """Wall-clock pause before the next item, compressed and capped.

        The cap matters more than the ratio: a recorded run has forty-minute
        waits in it, and at any honest speed those are still dead air on stage.
        """
        if previous is None or current is None or current < previous:
            return self.min_gap
        return max(self.min_gap,
                   min(self.max_gap, (current - previous) / self.speed))

    # ── the run ──────────────────────────────────────────────────────────────
    async def run(self) -> None:
        started = time.time()
        try:
            if self.warmup:
                await asyncio.sleep(self.warmup)
            await self._emit({"type": "status", "status": "processing",
                              "message": f"Replaying a recorded session at "
                                         f"{self.speed:g}x. Every event below was "
                                         f"produced by the real run."})
            await asyncio.gather(self._play_events(), self._grow_graph())
            await self._emit({"type": "status", "status": "idle",
                              "message": "Replay complete."})
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — a demo must not take the server down
            logger.exception("replay failed")
            await self._emit({"type": "error", "message": f"replay failed: {exc}"})
        finally:
            logger.info("replay of %s finished in %.1fs", self.source,
                        time.time() - started)

    async def _emit(self, event: Dict[str, Any]) -> None:
        event = {**event, REPLAY_FLAG: self.source}
        self.runtime.record_event(self.key, event)

    #: The recorded commit payloads carry the study's own Russian text. Half
    #: translating a JSON blob reads worse than not showing it, so a payload
    #: with Cyrillic in it is displayed as its size instead. The bundle keeps
    #: the payload itself.
    _CYRILLIC = re.compile("[\u0410-\u044f\u0401\u0451]")

    @classmethod
    def _payload(cls, value: Any) -> Any:
        if isinstance(value, str) and cls._CYRILLIC.search(value):
            return f"[{len(value)} chars, not shown in the English recording]"
        return value

    @classmethod
    def _to_ui(cls, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Recorded agent event -> the shape the browser dispatches on.

        The transcript is written by the agent-event logger, which keys on
        ``kind``; the UI switches on ``type`` and would silently drop anything
        else. Nothing is invented here: author, text, tool name, arguments and
        result are the recorded ones.
        """
        if event.get("type"):                     # already a UI event
            return dict(event)
        kind = event.get("kind")
        author = event.get("agent") or "system"
        stamp = event.get("ts")
        if kind in ("text", "message", "agent_text", "thought", "final_response"):
            text = (event.get("text") or "").strip()
            if not text:
                return None
            return {"type": "agent_event", "author": author, "content": text,
                    "timestamp": stamp, "is_final": kind == "final_response"}
        if kind in ("user_input", "user", "user_message", "prompt"):
            text = (event.get("text") or event.get("message") or "").strip()
            return {"type": "user_message", "message": text,
                    "timestamp": stamp} if text else None
        if kind == "tool_error":
            return {"type": "tool_activity", "phase": "error", "author": author,
                    "tool": event.get("tool"),
                    "error": cls._payload(event.get("error")),
                    "call_id": event.get("call_id") or event.get("id"),
                    "timestamp": stamp}
        if kind in ("tool_call", "tool_start"):
            return {"type": "tool_activity", "phase": "call", "author": author,
                    "tool": event.get("tool"),
                    "args": cls._payload(event.get("args")),
                    "call_id": event.get("call_id") or event.get("id"),
                    "timestamp": stamp}
        if kind in ("tool_result", "tool_end"):
            failed = str(event.get("status") or "").lower() in ("error", "failed")
            out = {"type": "tool_activity", "author": author,
                   "tool": event.get("tool"),
                   "call_id": event.get("call_id") or event.get("id"),
                   "timestamp": stamp}
            if failed:
                out["phase"] = "error"
                out["error"] = cls._payload(event.get("result"))
            else:
                out["phase"] = "result"
                out["result"] = cls._payload(event.get("result"))
            return out
        return None

    @staticmethod
    def _pair_tool_calls(events: List[Dict[str, Any]]) -> None:
        """Give each call and its result a shared id, in place.

        The transcript carries no call id, and the tool panel pairs a response
        to its call by one. Matching the next result for the same agent and
        tool reproduces the pairing the run actually had.
        """
        pending: Dict[Tuple[str, str], str] = {}
        counter = 0
        for event in events:
            key = (str(event.get("agent")), str(event.get("tool")))
            kind = event.get("kind")
            if kind == "tool_call":
                counter += 1
                pending[key] = f"replay-{counter}"
                event["call_id"] = pending[key]
            elif kind in ("tool_result", "tool_end", "tool_error"):
                event["call_id"] = pending.pop(key, None)

    async def _play_events(self) -> None:
        self._pair_tool_calls(self.events)
        previous = None
        for event in self.events:
            if self.cancelled:
                return
            if event.get("kind") == "tool_start":   # duplicate of tool_call
                continue
            current = _event_time(event)
            await asyncio.sleep(self._sleep_for(previous, current))
            previous = current if current is not None else previous
            translated = self._to_ui(event)
            if translated is not None:
                await self._emit(translated)

    def _event_wall_clock(self) -> float:
        """Seconds the event stream will take, under the current pacing."""
        total, previous = 0.0, None
        for event in self.events:
            if event.get("kind") == "tool_start":
                continue
            current = _event_time(event)
            total += self._sleep_for(previous, current)
            previous = current if current is not None else previous
        return total

    async def _grow_graph(self) -> None:
        """Insert nodes and edges into the session's live graph, in recorded order."""
        if self.warmup:
            await asyncio.sleep(self.warmup)
        from CoScientist.graph.research.store import get_research_graph
        store = get_research_graph(user_id=self.key[0], session_id=self.key[1])
        steps = _graph_steps(self.graph)
        if not steps:
            return
        # The recorded graph was written in a handful of bursts, so replaying
        # its own intervals makes it appear almost at once. Spread the
        # insertions evenly over the projected length of the event stream, in
        # the recorded order, so the shape grows while the transcript runs.
        span = self._event_wall_clock()
        step_delay = max(self.min_gap, span / len(steps)) if span else self.min_gap
        for _when, kind, item in steps:
            if self.cancelled:
                return
            await asyncio.sleep(step_delay)
            try:
                self._insert(store, kind, item)
            except Exception:  # noqa: BLE001 — one bad item must not stop the show
                logger.debug("replay could not insert %s %s", kind, item.get("id"),
                             exc_info=True)

    @staticmethod
    def _insert(store, kind: str, item: Dict[str, Any]) -> None:
        """Write one recorded item straight into the graph.

        Two deliberate bypasses. The commit path is skipped because it re-checks
        role permissions and status transitions this content already passed when
        the study ran, and a replay is not entitled to re-assert them. And the
        write goes to the store's own graph rather than to ``full_graph()``,
        which hands out a copy — inserting there changes nothing anyone can see.
        """
        with store._lock:                       # noqa: SLF001 — demo playback
            graph = store._g                    # noqa: SLF001
            ReplaySession._write(graph, kind, item)
        save = getattr(store, "_save", None)
        if callable(save):
            save()

    @staticmethod
    def _write(graph, kind: str, item: Dict[str, Any]) -> None:
        if kind == "node":
            # The whole record goes in as node data, `id` included: the store
            # serialises a node from its attribute dict alone, so an id kept only
            # as the networkx key would vanish on the way to the renderer.
            graph.add_node(item["id"], **dict(item))
        else:
            u, v = item.get("from"), item.get("to")
            if u in graph.nodes and v in graph.nodes:
                graph.add_edge(u, v, key=item.get("type"), type=item.get("type"),
                               attrs=item.get("attrs") or {},
                               source=item.get("source"))


def load_recording(bundle: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Events and final graph of a recorded session.

    ``bundle`` is either a directory produced by ``scripts/collect_session.py``
    or a session directory under ``graph_runs/sessions``; the event transcript is
    looked up next to it under the web-state directory when the bundle has none.
    """
    root = Path(bundle)
    if not root.exists():
        raise FileNotFoundError(f"no recording at {root}")

    graph_path = next((p for p in (root / "graphs" / "research_active.json",
                                   root / "research_active.json")
                       if p.exists()), None)
    if graph_path is None:
        raise FileNotFoundError(f"no research_active.json under {root}")
    graph = json.loads(graph_path.read_text(encoding="utf-8"))

    candidates = list((root / "events").glob("*.jsonl")) if (root / "events").is_dir() else []
    candidates += list(root.glob("*.jsonl"))
    if not candidates:
        # A session directory under graph_runs/sessions holds the graph; its UI
        # transcript lives beside it under the web-state tree, keyed the same way.
        import os
        state = Path(os.getenv("WEB_STATE_DIR", "graph_runs/web_state"))
        guess = state / "sessions" / root.parent.name / f"{root.name}.jsonl"
        if guess.exists():
            candidates = [guess]
    events = _load_events(candidates[0]) if candidates else []
    return events, graph
