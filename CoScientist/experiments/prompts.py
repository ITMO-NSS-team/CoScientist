"""Prompt construction for the experiment planner (F015a / R05)."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

# A tool inventory entry is (name, description).
ToolInventory = Sequence[Tuple[str, str]]

_SCHEMA = """\
Return ONE JSON object, no prose, no markdown fences:
{
  "goal": "<one line: what the whole plan achieves>",
  "steps": [
    {
      "id": "s1",                         // unique, short
      "subtask": "<single concrete computational sub-task>",
      "required_tools": ["<exact tool name from the INVENTORY>", ...],
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
- required_tools MUST be exact names taken from the INVENTORY below. If a needed
  capability is NOT in the inventory, still write the step and put the missing
  capability name in required_tools (it will be flagged as a gap later) — never invent
  a tool you would call directly without listing it.
- Wire data flow with deps + artifacts: a step that uses a previous step's output
  must list that step in deps and reference its artifact id as {artifact_id} in run_params.
- Keep the graph acyclic. Give every produced artifact a stable id.
- Output JSON only."""


def format_inventory(tools: Optional[ToolInventory]) -> str:
    if not tools:
        return "(inventory not provided — name the tools you would need by capability)"
    return "\n".join(f"- {name}: {desc}" for name, desc in tools)


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
        "that an executor runs one step at a time.\n\n"
        + _SCHEMA + "\n\n" + _RULES
        + "\n\nINVENTORY (available tools):\n" + format_inventory(tools)
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
