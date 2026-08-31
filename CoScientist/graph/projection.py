"""Rule-based projections of the graph into agent context (Fact 2).

Deliberately rule-based (no LLM) so the same graph always yields the same
context — reproducibility is an evaluation metric. See docs/execution_graph.md.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

_MAX_ITEMS = 12
_LABEL = 200


def _short(s: Optional[str], n: int = _LABEL) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[:n] + "…"


def _node_line(n: dict) -> str:
    who = n.get("executor_agent") or n.get("kind", "?")
    label = _short(n.get("label"))
    out = n.get("output")
    tail = f" → {_short(out, 160)}" if out else ""
    return f"- [{who}] {label}{tail}"


def orchestrator_summary(full: dict) -> str:
    """Planning view: what already succeeded vs failed/rejected, so the
    orchestrator builds on the former and does NOT repeat the latter."""
    nodes = full.get("nodes", [])
    # Only delegations/decisions matter for planning — skip low-level tool calls.
    plany = [n for n in nodes if n.get("kind") in ("agent_call", "decision", "goal")]
    completed = [n for n in plany if n.get("status") == "success"]
    failed = [
        n for n in plany
        if n.get("status") == "failed" or (n.get("verdict") in ("reject", "wrong"))
    ]
    running = [n for n in plany if n.get("status") == "running"]

    if not (completed or failed or running):
        return ""

    blocks: List[str] = [
        "EXECUTION GRAPH — what has already happened in THIS run. Build on "
        "completed steps; do NOT repeat failed/rejected ones."
    ]
    if completed:
        blocks.append("Completed:\n" + "\n".join(_node_line(n) for n in completed[-_MAX_ITEMS:]))
    if failed:
        blocks.append(
            "Failed / rejected (do not retry as-is):\n"
            + "\n".join(_node_line(n) for n in failed[-_MAX_ITEMS:])
        )
    if running:
        blocks.append("In progress:\n" + "\n".join(_node_line(n) for n in running[-_MAX_ITEMS:]))
    return "\n\n".join(blocks)


def _index(full: dict) -> Dict[str, dict]:
    return {n["id"]: n for n in full.get("nodes", []) if "id" in n}


def local_view(full: dict, node_id: str) -> str:
    """Sub-agent view: the ancestral path (why this step exists) plus validated
    findings in scope. Pushed into the delegation envelope."""
    idx = _index(full)
    if node_id not in idx:
        return ""
    # walk parents up to the root
    chain: List[dict] = []
    seen = set()
    cur: Optional[dict] = idx.get(node_id)
    while cur is not None and cur["id"] not in seen:
        seen.add(cur["id"])
        chain.append(cur)
        parents = cur.get("parent_ids") or []
        cur = idx.get(parents[0]) if parents else None
    chain.reverse()
    if len(chain) <= 1:
        return ""
    path = "\n".join(f"{'  ' * i}↳ {_node_line(n)}" for i, n in enumerate(chain))
    return "REASONING PATH that led to this task (top → here):\n" + path


def turns(full: Dict[str, Any]) -> Dict[str, Any]:
    """The session's execution graph as a chronological list of turns.

    The call graph answers "what is connected to what". It cannot answer "what
    happened, in what order", because agent nodes are one per agent for the
    whole session, so every turn's calls hang off the same few nodes and the
    layout is force-directed with no time axis at all.

    This regroups the same records by the prompt that caused them and sorts each
    group by start time, which is the shape a trace viewer needs: one entry per
    user request, and under it every call with its agent, its arguments, its
    result and its duration.

    Turn membership comes from ``turn_id``. Snapshots recorded before that field
    existed are reconstructed two ways: ids of the old namespaced form
    ``goal:{inv}::tool:{call}`` carry the turn in their prefix, and anything left
    is assigned to the last request that started before it. Chronology is what a
    turn is, so recovering it from time is exact wherever the ids are silent.
    """
    nodes = {n["id"]: n for n in full.get("nodes", [])}
    order = {"goal": 0, "tool_call": 1, "result": 2}

    def tagged_turn(node: Dict[str, Any]) -> Optional[str]:
        if node.get("turn_id"):
            return node["turn_id"]
        nid = str(node.get("id", ""))
        for prefix in ("goal:", "result:"):
            if nid.startswith(prefix):
                # `goal:{inv}` and the namespaced `goal:{inv}::tool:{call}`.
                return nid[len(prefix):].split("::", 1)[0]
        return None

    members = [n for n in nodes.values() if n.get("kind") in order]
    members.sort(key=lambda n: (n.get("t_start") or 0.0, order[n["kind"]]))

    # Goals in the order they started, so an untagged call can be placed under
    # the one that was open when it ran.
    goals = [(g.get("t_start") or 0.0, tagged_turn(g) or g["id"])
             for g in members if g["kind"] == "goal"]

    def turn_of(node: Dict[str, Any]) -> str:
        tag = tagged_turn(node)
        if tag:
            return tag
        started = node.get("t_start") or 0.0
        current = None
        for goal_start, goal_turn in goals:
            if goal_start <= started:
                current = goal_turn
            else:
                break
        return current or "untagged"

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for node in members:
        grouped.setdefault(turn_of(node), []).append(node)

    out = []
    for turn_id, members in grouped.items():
        members.sort(key=lambda n: (n.get("t_start") or 0.0, order[n["kind"]]))
        goal = next((m for m in members if m["kind"] == "goal"), None)
        result = next((m for m in members if m["kind"] == "result"), None)
        calls = [m for m in members if m["kind"] == "tool_call"]
        started = min((m.get("t_start") or 0.0) for m in members)
        ended = max((m.get("t_end") or m.get("t_start") or 0.0) for m in members)
        out.append({
            "turn_id": turn_id,
            "prompt": (goal or {}).get("label", ""),
            "answer": (result or {}).get("output", ""),
            "status": (goal or {}).get("status", ""),
            "t_start": started,
            "t_end": ended,
            "duration": round(ended - started, 3) if ended and started else None,
            "calls": [{
                "id": c["id"],
                "agent": c.get("executor_agent"),
                "tool": c.get("label"),
                "status": c.get("status"),
                "input": c.get("input"),
                "output": c.get("output"),
                "t_start": c.get("t_start"),
                "t_end": c.get("t_end"),
                "duration": (round(c["t_end"] - c["t_start"], 3)
                             if c.get("t_end") and c.get("t_start") else None),
            } for c in calls],
        })
    out.sort(key=lambda t: t["t_start"])
    return {"turns": out, "count": len(out)}


def execution_tree(full: Dict[str, Any]) -> Dict[str, Any]:
    """The call graph as one tree per user request, ready to draw left to right.

    The stored graph is not shaped for reading. Older snapshots still carry the
    seeded roster — every configured agent wired to a hub by ``has_member``,
    whether or not it was ever called — and the hub itself adds a level above
    the request that carries no information. What a reader wants is the shape
    the run actually had:

        request -> agent -> its calls -> nested agent -> its calls -> answer

    So the roster and the hub are dropped, an agent survives only if something
    invoked it, and each node is given the depth it sits at in its own request.
    Depth is measured here rather than inferred by the viewer, because an agent
    node can be shared by several requests and inference would pin it to
    whichever one reached it first.
    """
    nodes = {n["id"]: dict(n) for n in full.get("nodes", [])}
    edges = [e for e in full.get("edges", []) if e.get("type") != "has_member"]

    roots = [n for n in nodes.values() if n.get("kind") == "system"]
    for root in roots:                       # the hub is a fixture, not an event
        nodes.pop(root["id"], None)
    edges = [e for e in edges
             if e.get("src") in nodes and e.get("dst") in nodes]

    # An agent that nothing called is roster, not history.
    called = {e["dst"] for e in edges}
    for node_id, node in list(nodes.items()):
        if node.get("kind") in ("agent", "agent_call") and node_id not in called:
            nodes.pop(node_id)
    edges = [e for e in edges if e["src"] in nodes and e["dst"] in nodes]

    children: Dict[str, List[str]] = {}
    for edge in edges:
        children.setdefault(edge["src"], []).append(edge["dst"])

    # Breadth-first from each request; a node keeps the shallowest depth found.
    level: Dict[str, int] = {}
    goals = sorted((n for n in nodes.values() if n.get("kind") == "goal"),
                   key=lambda n: n.get("t_start") or 0.0)
    frontier = [(g["id"], 0) for g in goals]
    while frontier:
        node_id, depth = frontier.pop(0)
        if node_id in level and level[node_id] <= depth:
            continue
        level[node_id] = depth
        for child in children.get(node_id, []):
            frontier.append((child, depth + 1))

    # Anything unreachable from a request still has to be placed somewhere.
    for node_id, node in nodes.items():
        level.setdefault(node_id, 0 if node.get("kind") == "goal" else 1)

    # The answer ends its request, so it belongs to the right of everything the
    # request did rather than beside the calls that produced it.
    deepest = max(level.values(), default=0)
    for node_id, node in nodes.items():
        if node.get("kind") == "result":
            level[node_id] = deepest + 1

    for node_id, node in nodes.items():
        node["level"] = level[node_id]

    ordered = sorted(nodes.values(),
                     key=lambda n: (n.get("t_start") or 0.0, n["level"]))
    _place_in_time(ordered, edges)
    return {"run_id": full.get("run_id"), "nodes": ordered, "edges": edges}


#: Horizontal pixels between two calls that began at the same moment, and the
#: most a single idle stretch is allowed to occupy. A run waits forty minutes
#: for a sandbox; drawn to scale that gap is the whole picture and the calls on
#: either side of it are a smudge, so long waits are compressed and the order
#: is what survives.
_MIN_STEP, _MAX_STEP, _PIXELS_PER_SECOND = 34, 260, 6.0
_ROW_HEIGHT = 120


def _place_in_time(ordered: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> None:
    """Give every node an x from when it ran and a y from who ran it.

    Depth answers "what followed from what", which is why it stays on the node,
    but it is the wrong horizontal axis: laying calls out by depth puts thirty
    of them in one column and says nothing about the order they happened in.
    Here x advances with the clock, so the picture stretches as the run goes on
    and reading left to right is reading forward in time.

    Rows keep it legible. The request and its answer bookend row zero; every
    agent gets a row of its own in the order it first appears, and a call sits
    in the row of the agent that made it.
    """
    if not ordered:
        return

    owner = {e["dst"]: e["src"] for e in edges}

    def name_of(node: Dict[str, Any]) -> str:
        return str(node.get("executor_agent") or node.get("label")
                   or node["id"].split("::")[-1])

    def agent_of(node: Dict[str, Any]) -> Optional[str]:
        """The agent whose row this node belongs in, by walking up its callers.

        Keyed by name, not by node: the same agent called in three requests is
        three nodes, and giving each its own row would spread one participant
        across the page instead of showing it as one lane of activity.
        """
        if node.get("kind") in ("agent", "agent_call"):
            return name_of(node)
        seen, current = set(), owner.get(node["id"])
        while current and current not in seen:
            seen.add(current)
            parent = by_id.get(current)
            if parent is None:
                return None
            if parent.get("kind") in ("agent", "agent_call"):
                return name_of(parent)
            current = owner.get(current)
        return None

    by_id = {n["id"]: n for n in ordered}
    rows: Dict[str, int] = {}
    for node in ordered:                      # first appearance decides the row
        if node.get("kind") in ("goal", "result"):
            continue
        agent = agent_of(node)
        if agent is not None and agent not in rows:
            rows[agent] = len(rows) + 1

    previous_start, x = None, 0.0
    for node in ordered:
        started = node.get("t_start")
        if previous_start is not None and started is not None:
            gap = max(0.0, started - previous_start) * _PIXELS_PER_SECOND
            x += min(_MAX_STEP, max(_MIN_STEP, gap))
        elif previous_start is not None:
            x += _MIN_STEP
        if started is not None:
            previous_start = started
        node["x"] = round(x)
        node["row"] = 0 if node.get("kind") in ("goal", "result") \
            else rows.get(agent_of(node) or "", 1)
        node["y"] = node["row"] * _ROW_HEIGHT
