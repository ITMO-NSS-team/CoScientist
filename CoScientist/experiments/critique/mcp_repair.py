"""Fill empty fedot/react mcp_servers from inventory, or demote if uncovered.

Demote only when inventory is empty or no tool covers the task capability —
not Option A (unused inventory is critique-minor only).
"""
from __future__ import annotations

import copy
from typing import Any, Iterable

from CoScientist.experiments.capabilities.inventory import (
    index_inventory_tools,
    inventory_covers_capabilities,
    match_inventory_tool,
    request_capabilities,
)

_MCP_ROUTES = frozenset({"fedot_mas", "react_tools"})


def _task_hint_blob(task: dict[str, Any]) -> str:
    parts = [
        str(task.get("name") or ""),
        str(task.get("description") or ""),
        str(task.get("rationale") or ""),
    ]
    design = task.get("design") if isinstance(task.get("design"), dict) else {}
    parts.append(str(design.get("experiment_question") or ""))
    for art in design.get("analysis_artifacts") or []:
        if isinstance(art, dict):
            parts.append(str(art.get("path_or_tool") or ""))
            parts.append(str(art.get("name") or ""))
    for server in task.get("mcp_servers") or []:
        if not isinstance(server, dict):
            continue
        for tool in server.get("tools") or []:
            parts.append(str(tool.get("name") if isinstance(tool, dict) else tool or ""))
    return " ".join(parts)


def _has_mcp_tool(task: dict[str, Any]) -> bool:
    return any(
        isinstance(s, dict) and (s.get("tools") or [])
        for s in (task.get("mcp_servers") or [])
    )


def _bind_server(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": item["server_id"],
        "server_id": item["server_id"],
        "url": None,
        "tools": [{"name": item["tool"]}],
        "source": "registry",
        "health": "unknown",
    }


def _rewrite_artifacts_to_tool(task: dict[str, Any], tool_name: str) -> None:
    design = task.get("design")
    if not isinstance(design, dict):
        return
    for art in design.get("analysis_artifacts") or []:
        if isinstance(art, dict) and (art.get("prepare_via") == "mcp" or not art.get("path_or_tool")):
            art["prepare_via"] = "mcp"
            art["path_or_tool"] = tool_name


def _demote_to_coder(task: dict[str, Any], note: str) -> None:
    task["route"] = "coder"
    task["mcp_servers"] = []
    design = task.get("design") if isinstance(task.get("design"), dict) else None
    if design:
        for art in design.get("analysis_artifacts") or []:
            if isinstance(art, dict) and art.get("prepare_via") == "mcp":
                art["prepare_via"] = "coder"
    task["warnings"] = list(task.get("warnings") or []) + [note]


def _apply_bind(task: dict[str, Any], matched: dict[str, Any]) -> None:
    task["mcp_servers"] = [_bind_server(matched)]
    _rewrite_artifacts_to_tool(task, matched["tool"])
    warnings = [w for w in (task.get("warnings") or []) if "demoted_to_coder" not in str(w)]
    warnings.append(f"auto_bound_mcp:{matched['server_id']}/{matched['tool']}")
    task["warnings"] = warnings


def repair_plan_mcp_bindings(
    payload: Any,
    available_tools: Iterable[dict[str, Any]] = (),
) -> Any:
    """Fill empty fedot/react mcp_servers from inventory, or demote if uncovered."""
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        return payload

    by_tool = index_inventory_tools(available_tools)
    source_request = str(payload.get("source_request") or "")
    out = copy.deepcopy(payload)

    for task in out.get("tasks") or []:
        if not isinstance(task, dict):
            continue
        if str(task.get("route") or "") not in _MCP_ROUTES or _has_mcp_tool(task):
            continue

        blob = _task_hint_blob(task)
        task_needed = request_capabilities(blob)
        matched = match_inventory_tool(blob, by_tool, source_request=source_request) if by_tool else None
        if matched:
            _apply_bind(task, matched)
            continue
        if not by_tool:
            _demote_to_coder(task, "demoted_to_coder:empty_inventory")
        elif not task_needed:
            _demote_to_coder(task, "demoted_to_coder:no_task_capability_signal")
        elif inventory_covers_capabilities(by_tool, task_needed) and (
            matched := match_inventory_tool(blob, by_tool, source_request=source_request)
        ):
            _apply_bind(task, matched)
        else:
            _demote_to_coder(task, "demoted_to_coder:no_inventory_capability_cover")

    out.pop("warnings", None)
    return out


def repair_orphan_hypotheses(plan: "ExperimentPlan") -> "ExperimentPlan":
    """No-op: orphan coverage belongs to critique (no silent auto-link)."""
    return plan


__all__ = ["repair_plan_mcp_bindings", "repair_orphan_hypotheses"]
