"""Fill empty fedot/react mcp_servers from this-run inventory, or Alembic / coder."""
from __future__ import annotations

import copy
from typing import Any, Iterable

from CoScientist.experiments.capabilities.inventory import (
    index_inventory_tools,
    match_inventory_tool,
    match_named_family_capability,
    match_named_inventory_tool,
    match_slot_inventory_tool,
)
from CoScientist.experiments.critique.coverage import (
    operation_statement as _operation_statement,
    task_ask_blob as _task_ask_blob,
    task_coverage_blob as _coverage_blob,
)

_MCP_ROUTES = frozenset({"fedot_mas", "react_tools"})
_AGENT_ROUTES = frozenset({"research", "medical", "alembic_build"})


def _normalize_ops(operations: Iterable[Any]) -> list[dict[str, str]]:
    from CoScientist.context_init.operations import normalize_operation_rows

    return normalize_operation_rows(list(operations or []))


def _design_dict(task: dict[str, Any]) -> dict[str, Any]:
    design = task.get("design")
    if not isinstance(design, dict):
        design = {}
        task["design"] = design
    return design


def _task_operation_id(task: dict[str, Any]) -> str:
    design = task.get("design") if isinstance(task.get("design"), dict) else {}
    return str(design.get("operation_ref") or task.get("operation_ref") or "").strip().upper()


def _ops_by_id(ops: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {str(op["operation_id"]).strip().upper(): op for op in ops if op.get("operation_id")}


def _bound_tool_names(task: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for server in task.get("mcp_servers") or []:
        if not isinstance(server, dict):
            continue
        for tool in server.get("tools") or []:
            name = str(tool.get("name") if isinstance(tool, dict) else tool or "").strip()
            if name:
                names.append(name)
    return names


def _bound_tools_in_inventory(
    task: dict[str, Any],
    by_tool: dict[str, dict[str, Any]],
) -> bool:
    names = _bound_tool_names(task)
    return bool(names) and all(name in by_tool for name in names)


def _apply_evidence_route(task: dict[str, Any], statement: str) -> None:
    family = match_named_family_capability(statement) or {
        "family": "research",
        "tool": "search_papers",
    }
    _apply_family_route(task, family)


def _has_mcp_tool(task: dict[str, Any]) -> bool:
    return any(
        isinstance(s, dict) and (s.get("tools") or [])
        for s in (task.get("mcp_servers") or [])
    )


def _bound_tool_in_inventory(task: dict[str, Any], by_tool: dict[str, dict[str, Any]]) -> bool:
    for server in task.get("mcp_servers") or []:
        if not isinstance(server, dict):
            continue
        for tool in server.get("tools") or []:
            name = str(tool.get("name") if isinstance(tool, dict) else tool or "").strip()
            if name and name in by_tool:
                return True
    return False


def _bind_server(item: dict[str, Any]) -> dict[str, Any]:
    url = str(item.get("url") or "").strip()
    return {
        "name": item["server_id"],
        "server_id": item["server_id"],
        "url": url or None,
        "tools": [{"name": item["tool"], "input_schema": item.get("input_schema")}],
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


def _apply_family_route(task: dict[str, Any], matched: dict[str, Any]) -> None:
    family = str(matched.get("family") or "").strip()
    task["route"] = family
    task["mcp_servers"] = []
    task.pop("post_build_route", None)
    design = task.get("design") if isinstance(task.get("design"), dict) else None
    if design:
        for art in design.get("analysis_artifacts") or []:
            if isinstance(art, dict):
                art["prepare_via"] = family
                if matched.get("tool"):
                    art["path_or_tool"] = matched["tool"]
    task["warnings"] = list(task.get("warnings") or []) + [
        f"auto_rewrote_coder_to_{family}:{matched.get('tool')}"
    ]


def _promote_coder_when_inventory_covers(
    task: dict[str, Any],
    by_tool: dict[str, dict[str, Any]],
    *,
    route_fedot: bool,
    ops_index: dict[str, dict[str, str]] | None = None,
) -> bool:
    """Coder is last resort. Named research/medical tool → that route; else named MCP."""
    blob = _coverage_blob(task, ops_index or {})
    if not blob.strip():
        blob = _task_ask_blob(task)
    from CoScientist.context_init.operations import is_evidence_operation

    if ops_index and blob and is_evidence_operation(blob):
        _apply_evidence_route(task, blob)
        return True
    if family_hit := match_named_family_capability(blob):
        _apply_family_route(task, family_hit)
        return True
    if not by_tool:
        return False
    matched = match_named_inventory_tool(blob, by_tool)
    if matched is None:
        return False
    task["route"] = "fedot_mas" if route_fedot else "react_tools"
    _apply_bind(task, matched)
    task["warnings"] = list(task.get("warnings") or []) + ["auto_rewrote_coder_to_ready_mcp"]
    return True


def _best_repo_candidate(repo_candidates: Iterable[Any]) -> dict[str, Any] | None:
    """First entry with a URL — callers already return ask-first, fit_score-desc order."""
    for item in repo_candidates or []:
        if isinstance(item, dict) and str(item.get("url") or "").strip():
            return item
    return None


def _route_to_alembic(
    task: dict[str, Any], repo: dict[str, Any], *, post_build_route: str,
    source_request: str = "",
) -> None:
    from CoScientist.experiments.runtime.alembic_bridge import stamp_alembic_science_description

    task["route"] = "alembic_build"
    task["repo_url"] = repo.get("url")
    task["post_build_route"] = post_build_route
    task["mcp_servers"] = []
    stamp_alembic_science_description(
        task, repo_url=str(repo.get("url") or ""), source_request=source_request,
    )
    warnings = [w for w in (task.get("warnings") or []) if "demoted_to_coder" not in str(w)]
    warnings.append(f"auto_routed_alembic:{repo.get('url')}")
    task["warnings"] = warnings


def _clean_alembic_task(task: dict[str, Any], *, source_request: str = "") -> None:
    """Alembic-build tasks must not carry mcp_servers at plan time (contract)."""
    from CoScientist.experiments.runtime.alembic_bridge import stamp_alembic_science_description

    if task.get("mcp_servers"):
        task["mcp_servers"] = []
        task["warnings"] = list(task.get("warnings") or []) + ["auto_cleared_mcp_servers:alembic_build"]
    stamp_alembic_science_description(
        task, repo_url=str(task.get("repo_url") or ""), source_request=source_request,
    )


def _next_exp_id(tasks: list[Any]) -> str:
    used = {
        str(task.get("id") or "").strip().upper()
        for task in tasks if isinstance(task, dict)
    }
    n = 1
    while f"EXP-{n}" in used:
        n += 1
    return f"EXP-{n}"


def _hypothesis_for_index(payload: dict[str, Any], index: int) -> str:
    hyps = payload.get("hypotheses") if isinstance(payload.get("hypotheses"), list) else []
    ids = [
        str(item.get("hypothesis_id") or "").strip().upper()
        for item in hyps if isinstance(item, dict)
    ]
    want = f"H{index}"
    if want in ids:
        return want
    return ids[0] if ids else "H1"


def _stub_coder_task(task_id: str, op: dict[str, str], hypothesis_id: str) -> dict[str, Any]:
    oid = str(op.get("operation_id") or "OP-1")
    stmt = str(op.get("statement") or task_id)
    return {
        "id": task_id,
        "name": stmt[:80],
        "description": stmt,
        "rationale": f"Frame operation {oid} had no covering task; required coder.",
        "route": "coder",
        "design": {
            "hypothesis_ref": hypothesis_id,
            "operation_ref": oid,
            "experiment_question": stmt,
            "dataset": {"name": "operation inputs", "ref": None, "notes": None},
            "baselines": [],
            "metrics": [],
            "analysis_artifacts": [],
        },
        "mcp_servers": [],
        "input_data": [],
        "launch_params": {},
        "success_criteria": [{
            "criterion_id": f"{task_id}-C1",
            "description": "The operation produces its deliverable.",
            "kind": "execution",
            "verification": "Check the structured route result status.",
        }],
        "expected_artifacts": [{
            "name": f"{task_id.lower()}-result.json",
            "role": "data",
            "media_type": "application/json",
            "description": "Operation deliverable.",
        }],
        "est_duration_min": 5,
        "depends_on": [],
        "optional": False,
        "warnings": ["auto_added_operation_slot:required_coder"],
    }


def _ensure_operation_slots(
    payload: dict[str, Any],
    ops: list[dict[str, str]],
    *,
    max_plan_tasks: int,
) -> None:
    """Bind operation_ref and append required coder tasks for uncovered slots."""
    if not ops:
        return
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        return
    ops_index = _ops_by_id(ops)
    assigned: set[str] = set()
    unassigned: list[dict[str, Any]] = []
    for task in tasks:
        if not isinstance(task, dict) or task.get("optional"):
            continue
        ref = _task_operation_id(task)
        if ref in ops_index and ref not in assigned:
            _design_dict(task)["operation_ref"] = ops_index[ref]["operation_id"]
            assigned.add(ref)
        else:
            unassigned.append(task)
    uncovered = [op for op in ops if op["operation_id"].upper() not in assigned]
    for task, op in zip(unassigned, uncovered):
        _design_dict(task)["operation_ref"] = op["operation_id"]
        assigned.add(op["operation_id"].upper())
    still = [op for op in ops if op["operation_id"].upper() not in assigned]
    for op in still:
        if len(tasks) >= max_plan_tasks:
            break
        hid = _hypothesis_for_index(payload, len(tasks) + 1)
        tasks.append(_stub_coder_task(_next_exp_id(tasks), op, hid))
    duration = 0
    for task in tasks:
        if isinstance(task, dict):
            try:
                duration += int(task.get("est_duration_min") or 0)
            except (TypeError, ValueError):
                pass
    if duration:
        payload["total_est_duration_min"] = duration


def repair_plan_mcp_bindings(
    payload: Any,
    available_tools: Iterable[dict[str, Any]] = (),
    *,
    repo_candidates: Iterable[dict[str, Any]] = (),
    route_alembic: bool = False,
    route_fedot: bool = True,
    operations: Iterable[Any] = (),
    max_plan_tasks: int = 8,
) -> Any:
    """Fill mcp_servers from this-run inventory; alembic_build if empty + repo; else coder."""
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        return payload

    from CoScientist.context_init.operations import is_evidence_operation

    by_tool = index_inventory_tools(available_tools)
    source_request = str(payload.get("source_request") or "")
    out = copy.deepcopy(payload)
    ops = _normalize_ops(operations)
    ops_index = _ops_by_id(ops)
    _ensure_operation_slots(out, ops, max_plan_tasks=max_plan_tasks)
    top_repo = _best_repo_candidate(repo_candidates) if route_alembic else None
    alembic_assigned = any(
        isinstance(t, dict) and str(t.get("route") or "") == "alembic_build"
        for t in (out.get("tasks") or [])
    )
    post_alembic = "fedot_mas" if route_fedot else "react_tools"
    used_tools = {
        name
        for task in (out.get("tasks") or [])
        if isinstance(task, dict)
        for name in _bound_tool_names(task)
        if name in by_tool
    }

    def _bind_from_inventory(task: dict[str, Any]) -> bool:
        blob = _coverage_blob(task, ops_index)
        if ops and blob and is_evidence_operation(blob):
            _apply_evidence_route(task, blob)
            return True
        if not by_tool:
            return False
        if ops:
            matched = match_slot_inventory_tool(blob, by_tool) if blob else None
            if matched is None:
                unused = [
                    item for item in by_tool.values()
                    if item["tool"] not in used_tools
                ]
                unused.sort(key=lambda item: float(item.get("score") or 0), reverse=True)
                matched = unused[0] if unused else None
        else:
            matched = match_inventory_tool(
                blob, by_tool, source_request=source_request,
            )
        if matched is None:
            return False
        task["route"] = "fedot_mas" if route_fedot else "react_tools"
        _apply_bind(task, matched)
        used_tools.add(matched["tool"])
        return True

    for task in out.get("tasks") or []:
        if not isinstance(task, dict):
            continue

        route = str(task.get("route") or "")
        statement = _operation_statement(task, ops_index)
        if route in _AGENT_ROUTES:
            if route == "alembic_build":
                if by_tool and ops and _bind_from_inventory(task):
                    continue
                _clean_alembic_task(task, source_request=source_request)
            else:
                task["mcp_servers"] = []
                task.pop("post_build_route", None)
            continue
        if statement and is_evidence_operation(statement) and route != "research":
            _apply_evidence_route(task, statement)
            continue
        if route == "coder":
            if _promote_coder_when_inventory_covers(
                task, by_tool, route_fedot=route_fedot, ops_index=ops_index,
            ):
                for name in _bound_tool_names(task):
                    used_tools.add(name)
                continue
            if ops and _bind_from_inventory(task):
                continue
            allow_alembic = top_repo and not alembic_assigned and (
                not ops or not by_tool
            )
            if allow_alembic:
                _route_to_alembic(
                    task, top_repo, post_build_route=post_alembic,
                    source_request=source_request,
                )
                alembic_assigned = True
                continue
            continue
        if route in _MCP_ROUTES and _has_mcp_tool(task):
            if ops:
                if _bound_tools_in_inventory(task, by_tool):
                    continue
                task["mcp_servers"] = []
                task["warnings"] = list(task.get("warnings") or []) + [
                    "untrusted_planner_bind:not_in_inventory"
                ]
            else:
                leftover_bind = (not by_tool) or (not _bound_tool_in_inventory(task, by_tool))
                if leftover_bind and top_repo and not alembic_assigned:
                    _route_to_alembic(
                        task, top_repo, post_build_route=post_alembic,
                        source_request=source_request,
                    )
                    alembic_assigned = True
                continue
        if route not in _MCP_ROUTES or _has_mcp_tool(task):
            continue

        if _bind_from_inventory(task):
            continue

        post_build_route = route if (route != "fedot_mas" or route_fedot) else "react_tools"
        if top_repo and not by_tool:
            _route_to_alembic(task, top_repo, post_build_route=post_build_route, source_request=source_request)
            alembic_assigned = True
        else:
            _demote_to_coder(
                task,
                "demoted_to_coder:unnamed_operation" if ops else "demoted_to_coder:empty_inventory",
            )

    if by_tool and not any(
        isinstance(t, dict) and str(t.get("route") or "") in _MCP_ROUTES
        for t in (out.get("tasks") or [])
    ):
        for task in out.get("tasks") or []:
            if not isinstance(task, dict):
                continue
            if str(task.get("route") or "") not in {"research", "medical"}:
                continue
            blob = _coverage_blob(task, ops_index)
            if match_named_family_capability(blob):
                continue
            matched = match_named_inventory_tool(blob, by_tool)
            if matched is None:
                continue
            task["route"] = "fedot_mas" if route_fedot else "react_tools"
            _apply_bind(task, matched)
            task["warnings"] = list(task.get("warnings") or []) + [
                "auto_rewrote_unbound_evidence_to_ready_mcp"
            ]

    out.pop("warnings", None)
    return out


__all__ = [
    "repair_plan_mcp_bindings",
]
