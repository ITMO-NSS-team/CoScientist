"""ADK plugin that grows the in-process knowledge graph from agent activity.

Attached to the in-process Runner (web / cli) next to the event logger. Unlike
``GraphEmitterPlugin`` (which POSTs to the A2A graph service over HTTP), this
writes straight into the shared ``knowledge_graph`` so any agent can read it
synchronously via the graph tools.

Shape of one query — a tree rooted at the orchestrator, with full agent
hierarchy (delegations AND sequential/parallel children):

    goal (the user query)
      └─ OrchestratorAgent
           ├─ tool_call: retrieve_tools          (a tool the orchestrator ran)
           ├─ agent_call: PlannerAgent           (a delegation)
           │     └─ tool_call: create_plan
           ├─ agent_call: TaskExecutorAgent      (a sequential composite)
           │     └─ agent_call: ExperimentAgent  (its child — hierarchy preserved)
           │           └─ tool_call: fedot_tool
           └─ result                             (final answer — out of the orchestrator)

Node labels: tool_call shows the TOOL name, agent_call shows the AGENT name (who
called it is clear from the parent). Best-effort: a graph failure never breaks a
run. Toggle with LOG_AGENT_EVENTS=0 (shared with the event logger).
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

from google.adk.plugins.base_plugin import BasePlugin

from CoScientist.graph.memory import ROOT_ID, knowledge_graph

_agent_names_cache: Optional[set] = None
_composite_parents_cache: Optional[dict] = None

# Fire-and-forget background tasks (semantic extraction, etc.). Held at module
# level so they are not garbage-collected mid-flight; never awaited by the run
# loop (full asynchrony). On the web's persistent event loop they complete on
# their own; in one-shot CLI they are best-effort.
_BG_TASKS: set = set()


def _spawn_background(coro) -> None:
    try:
        import asyncio
        task = asyncio.create_task(coro)
        _BG_TASKS.add(task)
        task.add_done_callback(_BG_TASKS.discard)
    except Exception:  # noqa: BLE001 — no running loop / scheduling failure
        pass


def _agent_names() -> set:
    global _agent_names_cache
    if _agent_names_cache is None:
        try:
            from CoScientist.assembly.schema import get_config
            _agent_names_cache = get_config().delegatable_names()
        except Exception:  # noqa: BLE001
            _agent_names_cache = set()
    return _agent_names_cache


def _composite_parents() -> dict:
    """child agent -> parent composite (from sequential/parallel `children`)."""
    global _composite_parents_cache
    if _composite_parents_cache is None:
        m: dict = {}
        try:
            from CoScientist.assembly.schema import get_config
            for name, a in get_config().agents.items():
                for ch in (getattr(a, "children", None) or []):
                    m[ch] = name
        except Exception:  # noqa: BLE001
            pass
        _composite_parents_cache = m
    return _composite_parents_cache


def _enabled() -> bool:
    value = os.getenv("LOG_AGENT_EVENTS") or os.getenv("A2A_LOG_EVENTS") or "1"
    return value not in ("0", "false", "False")


def _short(value: Any, limit: int = 300) -> str:
    s = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return s if len(s) <= limit else s[:limit] + "…"


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    parts = getattr(content, "parts", None) or []
    return "\n".join(p.text for p in parts if getattr(p, "text", None)).strip()


def _is_error(result: Any) -> bool:
    if isinstance(result, dict):
        if result.get("status") in ("error", "failed", "timeout"):
            return True
        if result.get("error"):
            return True
    return False


class GraphMemoryPlugin(BasePlugin):
    def __init__(self, name: str = "graph_memory") -> None:
        super().__init__(name=name)
        self._root_agent_name: Optional[str] = None
        self._goal_id = "goal:pending"
        self._goal_text = ""
        # Per-goal bookkeeping (reset on each top-level user message):
        #   agent name -> its single activity node id for this goal.
        self._agent_node: dict = {}
        #   function_call_id -> node id, so after_tool updates the right node.
        self._node_by_fcid: dict = {}

    @staticmethod
    def _ctx_agent(invocation_context) -> Optional[str]:
        agent = getattr(invocation_context, "agent", None)
        return getattr(agent, "name", None)

    def _root_name(self) -> str:
        return self._root_agent_name or "OrchestratorAgent"

    def _agent_node_for(self, agent: str, _seen: Optional[set] = None) -> str:
        """The activity node for an agent, creating the whole ancestor chain.

        The root agent hangs under the goal; a sequential/parallel child hangs
        under its composite parent's node (recursively), so the hierarchy of
        composite agents is preserved even though their children are not invoked
        as AgentTool delegations.
        """
        cached = self._agent_node.get(agent)
        if cached is not None:
            return cached

        if agent == self._root_name():
            nid = f"{self._goal_id}::agent:{agent}"
            knowledge_graph.add_node(
                id=nid, kind="agent_call", label=agent, executor_agent=agent,
                status="running", parent_ids=[self._goal_id], t_start=time.time(),
            )
            knowledge_graph.add_edge(self._goal_id, nid, type="caused_by")
            self._agent_node[agent] = nid
            return nid

        _seen = _seen or set()
        parent_agent = _composite_parents().get(agent)
        if parent_agent and parent_agent not in _seen:
            _seen.add(parent_agent)
            parent_node = self._agent_node_for(parent_agent, _seen)
        else:
            parent_node = self._agent_node_for(self._root_name())

        nid = f"{self._goal_id}::agent:{agent}"
        knowledge_graph.add_node(
            id=nid, kind="agent_call", label=agent, executor_agent=agent,
            status="running", parent_ids=[parent_node], t_start=time.time(),
        )
        knowledge_graph.add_edge(parent_node, nid, type="delegated_to")
        self._agent_node[agent] = nid
        return nid

    async def on_user_message_callback(self, *, invocation_context, user_message) -> Optional[Any]:
        if not _enabled():
            return None
        name = self._ctx_agent(invocation_context)
        if self._root_agent_name is None:
            self._root_agent_name = name
        if name != self._root_agent_name:
            return None
        inv = getattr(invocation_context, "invocation_id", "x")
        self._goal_id = f"goal:{inv}"
        self._goal_text = _content_text(user_message)
        self._agent_node = {}
        self._node_by_fcid = {}
        try:
            knowledge_graph.add_node(
                id=self._goal_id, kind="goal", label=_short(_content_text(user_message), 200),
                status="running", parent_ids=[ROOT_ID], t_start=time.time(),
            )
            knowledge_graph.add_edge(ROOT_ID, self._goal_id, type="caused_by")
        except Exception:  # noqa: BLE001
            pass
        return None

    async def before_tool_callback(self, *, tool, tool_args, tool_context) -> Optional[dict]:
        if not _enabled():
            return None
        try:
            agent = getattr(tool_context, "agent_name", None) or self._root_name()
            parent = self._agent_node_for(agent)
            fcid = getattr(tool_context, "function_call_id", None) or f"{tool.name}:{time.time()}"
            if tool.name in _agent_names():
                # Delegation: the called agent's activity node hangs under the
                # caller. Keep the request (the prompt the agent was called with)
                # as the node input so it shows on click.
                nid = f"{self._goal_id}::agent:{tool.name}"
                knowledge_graph.add_node(
                    id=nid, kind="agent_call", label=tool.name, executor_agent=tool.name,
                    status="running", parent_ids=[parent], input=_short(tool_args, 1000),
                    t_start=time.time(),
                )
                knowledge_graph.add_edge(parent, nid, type="delegated_to")
                self._agent_node[tool.name] = nid
            else:
                nid = f"{self._goal_id}::tool:{fcid}"
                knowledge_graph.add_node(
                    id=nid, kind="tool_call", label=tool.name, executor_agent=agent,
                    status="running", parent_ids=[parent], input=_short(tool_args), t_start=time.time(),
                )
                knowledge_graph.add_edge(parent, nid, type="caused_by")
            self._node_by_fcid[fcid] = nid
        except Exception:  # noqa: BLE001
            pass
        return None

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result) -> Optional[dict]:
        if not _enabled():
            return None
        try:
            nid = self._node_by_fcid.get(getattr(tool_context, "function_call_id", None))
            if nid:
                knowledge_graph.set_status(
                    nid, status="failed" if _is_error(result) else "success",
                    output=_short(result, 400), t_end=time.time(),
                )
        except Exception:  # noqa: BLE001
            pass
        return None

    async def on_tool_error_callback(self, *, tool, tool_args, tool_context, error) -> Optional[dict]:
        if not _enabled():
            return None
        try:
            nid = self._node_by_fcid.get(getattr(tool_context, "function_call_id", None))
            if nid:
                knowledge_graph.set_status(
                    nid, status="failed", output=_short(str(error), 300), t_end=time.time(),
                )
        except Exception:  # noqa: BLE001
            pass
        return None

    async def on_event_callback(self, *, invocation_context, event) -> Optional[Any]:
        if not _enabled():
            return None
        if self._ctx_agent(invocation_context) != self._root_agent_name:
            return None
        if not (getattr(event, "is_final_response", None) and event.is_final_response()):
            return None
        text = _content_text(getattr(event, "content", None))
        if not text:
            return None
        try:
            parent = self._agent_node_for(self._root_name())
            rid = f"result:{getattr(invocation_context, 'invocation_id', 'x')}"
            knowledge_graph.add_node(
                id=rid, kind="result", label=_short(text, 200),
                executor_agent=self._root_name(), status="success",
                parent_ids=[parent], output=_short(text, 600),
                t_start=time.time(), t_end=time.time(),
            )
            knowledge_graph.add_edge(parent, rid, type="produced")
            knowledge_graph.set_status(parent, status="success", t_end=time.time())
            knowledge_graph.set_status(self._goal_id, status="success", t_end=time.time())
        except Exception:  # noqa: BLE001
            pass

        # Semantic layer (Option B): extract domain entities/relations from the
        # answer and accumulate them in the cross-run knowledge memory. Off unless
        # KG_SEMANTIC_ENABLED=1; one small LLM call per query; never fatal.
        # FULLY ASYNC: the extraction LLM call is fired as a background task and
        # NOT awaited here, so the run loop is never blocked waiting on it (on the
        # web's persistent loop it lands moments after the answer). Mirrors the
        # fire-and-forget pattern in graph/research/validator.py.
        try:
            from CoScientist.graph.semantic import semantic_enabled
            if semantic_enabled():
                inv = getattr(invocation_context, "invocation_id", "x")
                _spawn_background(self._extract_semantic(text, inv))
        except Exception:  # noqa: BLE001
            pass
        return None

    async def _extract_semantic(self, text: str, inv: str) -> None:
        """Background: extract domain entities/relations from the final answer and
        accumulate them in the cross-run memory. Best-effort, never awaited by the
        run loop."""
        try:
            from CoScientist.graph.semantic import extract
            from CoScientist.graph.memory_store import knowledge_memory
            extraction = await extract(text, context=self._goal_text,
                                       known_types=knowledge_memory.known_types())
            knowledge_memory.ingest(
                extraction, source=_short(self._goal_text, 120),
                refs={"run": inv, "goal_id": self._goal_id, "result_id": f"result:{inv}"},
            )
        except Exception:  # noqa: BLE001
            pass
