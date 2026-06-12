"""Prompt construction for the experiment planner (F015a / R05)."""
from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple

# Server-grouped tool inventory: {server_name: [(tool_name, description), ...]}.
# This mirrors what the live tool-analysis tool (retrieve_tools / F015c) returns,
# so the planner can bind each tool to the server that provides it.
ToolInventory = Mapping[str, Sequence[Tuple[str, str]]]

_SCHEMA = """\
Return ONE JSON object, no prose, no markdown fences:
{
  "goal": "<one line: what the whole plan achieves>",
  "steps": [
    {
      "id": "s1",                         // unique, short
      "subtask": "<single concrete computational sub-task>",
      "tool_servers": [                   // tools GROUPED BY the server that provides them
        { "server": "<exact MCP server name from INVENTORY>",
          "tools": ["<exact tool name on that server>", ...] }
      ],
      "run_params": { "<param>": "<value or {artifact_id} of an upstream step>" },
      "expected_artifacts": [
        { "id": "<stable id>", "description": "<what it holds>", "kind": "molecules|score|table|figure|text|model|dataset|data" }
      ],
      "deps": ["<id of a prerequisite step>", ...],
      "provenance": { "hypothesis": "<optional>", "source": "<optional paper/repo>" }
    }
  ]
}"""

_RULES = """\
Rules:
- Decompose into the SMALLEST useful steps; one capability per step.
- In tool_servers, name the EXACT server AND the exact tools on it, taken from the
  INVENTORY (each server is listed with the tools it provides). A step that needs a
  capability the INVENTORY does NOT cover must still list it under the server you
  believe should provide it (or "server": "UNKNOWN") — it will be flagged as a gap to
  build later. Never call a tool you did not list under a server.
- Wire data flow with deps + artifacts: a step that uses a previous step's output
  must list that step in deps and reference its artifact id as {artifact_id} in run_params.
- Keep the graph acyclic. Give every produced artifact a stable id.
- Output JSON only."""


def format_inventory(tools: Optional[ToolInventory]) -> str:
    if not tools:
        return "(inventory not provided — name the server you believe provides each tool)"
    lines = []
    for server, items in tools.items():
        tool_list = ", ".join(name for name, _ in items)
        lines.append(f"- server `{server}` provides: {tool_list}")
        for name, desc in items:
            lines.append(f"    · {name}: {desc}")
    return "\n".join(lines)


def build_planner_messages(
    task: str,
    *,
    hypothesis: Optional[str] = None,
    literature: Optional[str] = None,
    tools: Optional[ToolInventory] = None,
) -> list[dict]:
    system = (
        "You are an experiment planner for a multi-agent scientific-discovery system. "
        "You turn a research task into a structured, verifiable, step-by-step JSON plan "
        "that an executor runs one step at a time, dispatching each step to the right MCP servers.\n\n"
        + _SCHEMA + "\n\n" + _RULES
        + "\n\nINVENTORY (available MCP servers and their tools):\n" + format_inventory(tools)
    )
    user_parts = [f"TASK:\n{task}"]
    if hypothesis:
        user_parts.append(f"\nHYPOTHESIS:\n{hypothesis}")
    if literature:
        user_parts.append(f"\nLITERATURE SUMMARY:\n{literature}")
    user_parts.append("\nReturn the plan as a single JSON object now.")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(user_parts)},
    ]
