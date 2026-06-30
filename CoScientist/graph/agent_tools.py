"""Read-only knowledge-graph tools available to EVERY agent.

Any agent can, at any point, inspect the shared graph: the history of what has
happened in the session and structured info about all the other agents. Wired
into agents via the ``graph`` tool key (see CoScientist/assembly/bindings.py).

Read-only on purpose — the graph grows automatically from agent activity
(GraphMemoryPlugin), so agents never have to maintain it by hand.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from google.adk.tools import BaseTool, FunctionTool
from google.adk.tools.base_toolset import BaseToolset
from google.adk.agents.readonly_context import ReadonlyContext

from CoScientist.graph.memory import knowledge_graph


def read_research_graph() -> Dict[str, Any]:
    """Read the shared research/knowledge graph of the whole session.

    Returns the system root, every agent (the roster) and every step taken so
    far (goals, delegations, tool calls, results) with their status. Use it to
    understand what has already happened and avoid repeating work.
    """
    full = knowledge_graph.full()
    nodes = [
        {
            "id": n.get("id"),
            "kind": n.get("kind"),
            "label": _short(n.get("label"), 160),
            "agent": n.get("executor_agent"),
            "status": n.get("status"),
            "output": _short(n.get("output"), 200),
        }
        for n in full.get("nodes", [])
    ]
    return {"nodes": nodes, "edges": full.get("edges", [])}


def get_graph_history(limit: int = 30) -> Dict[str, Any]:
    """Get the chronological history of steps taken in this session.

    Args:
        limit: max number of most-recent events to return.
    Returns:
        A dict with an ``events`` list (goals, delegations, tool calls, results).
    """
    return {"events": knowledge_graph.history(limit=limit)}


def get_agents_info() -> Dict[str, Any]:
    """Get structured info about all agents in the system (the graph root):
    name, description, role, tools and which agents each one can delegate to."""
    return {"agents": knowledge_graph.agents_info()}


def search_knowledge_memory(query: str, limit: int = 10) -> Dict[str, Any]:
    """Search the cross-run KNOWLEDGE MEMORY for facts established in EARLIER runs.

    Returns the domain entities (targets, molecules, metrics, papers, hypotheses,
    methods) most relevant to the query, with their attributes — so you can build
    on prior findings instead of recomputing them. Empty if nothing was learned yet.

    Args:
        query: what you're looking for (e.g. "GSK-3beta inhibitors docking").
        limit: max entities to return.
    """
    from CoScientist.graph.memory_store import knowledge_memory
    return {"entities": knowledge_memory.relevant(query, k=limit)}


def get_entity_neighbors(entity: str) -> Dict[str, Any]:
    """Walk the knowledge graph from one entity (search → then traverse).

    Given an entity name or key, returns it plus its 1-hop facts — e.g. which
    targets a molecule inhibits, which properties were measured (with values),
    related molecules. Use after search_knowledge_memory to explore connections.

    Args:
        entity: an entity name or key, e.g. "paracetamol" or "molecule:paracetamol".
    """
    from CoScientist.graph.memory_store import knowledge_memory
    return knowledge_memory.neighbors(entity)


def get_knowledge_memory() -> Dict[str, Any]:
    """Get the full cross-run knowledge memory (all entities + relations learned
    across previous runs). Prefer search_knowledge_memory + get_entity_neighbors."""
    from CoScientist.graph.memory_store import knowledge_memory
    return knowledge_memory.full()


class GraphReaderToolset(BaseToolset):
    """Exposes the read-only graph tools to an agent."""

    def __init__(self, prefix: Optional[str] = None) -> None:
        super().__init__(tool_name_prefix=prefix)

    async def get_tools(self, readonly_context: Optional[ReadonlyContext] = None) -> List[BaseTool]:
        return get_graph_tools()

    async def close(self) -> None:
        pass


def get_graph_tools() -> List[BaseTool]:
    return [
        FunctionTool(read_research_graph),
        FunctionTool(get_graph_history),
        FunctionTool(get_agents_info),
        FunctionTool(search_knowledge_memory),
        FunctionTool(get_entity_neighbors),
        FunctionTool(get_knowledge_memory),
    ]


def _short(value: Any, n: int = 200) -> str:
    if value is None:
        return ""
    s = value if isinstance(value, str) else str(value)
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n] + "…"


# Shared toolset instance (mirrors task_tracker_instance).
graph_reader_instance = GraphReaderToolset()
