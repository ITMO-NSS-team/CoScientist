"""Thin Alembic success adapter: inject MCP and reopen on post_build_route."""
from __future__ import annotations

import copy
import re
from typing import Any, MutableMapping
from urllib.parse import urlparse

from CoScientist.experiments.schemas import ExecutionRoute, ExperimentTask


def _server_id_from_url(repo_url: str | None, mcp_url: str) -> str:
    if repo_url:
        path = urlparse(repo_url).path.strip("/")
        name = path.split("/")[-1] if path else ""
        name = re.sub(r"[^A-Za-z0-9._-]", "-", name).strip("-")
        if name:
            return f"alembic-{name}"
    host = urlparse(mcp_url).hostname or "mcp"
    safe = re.sub(r"[^A-Za-z0-9._-]", "-", host).strip("-") or "mcp"
    return f"alembic-{safe}"


def _tool_refs_from_outputs(outputs: dict[str, Any]) -> list[dict[str, Any]]:
    raw = outputs.get("tools") or outputs.get("mcp_tools") or []
    tools: list[dict[str, Any]] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str) and (name := item.strip()):
                tools.append(
                    {
                        "name": name,
                        "description": f"Alembic-built tool {name}",
                        "input_schema": None,
                        "required_for_task": True,
                    }
                )
            elif isinstance(item, dict) and (name := str(item.get("name") or "").strip()):
                tools.append(
                    {
                        "name": name,
                        "description": str(item.get("description") or f"Alembic-built tool {name}"),
                        "input_schema": item.get("input_schema"),
                        "required_for_task": bool(item.get("required_for_task", True)),
                    }
                )
    if not tools:
        tools.append(
            {
                "name": "alembic_built_tool",
                "description": "Primary tool exposed by the Alembic-built MCP server.",
                "input_schema": None,
                "required_for_task": True,
            }
        )
    return tools


def extract_mcp_url(outputs: Any) -> str | None:
    """Pull mcp_url from route result outputs (nested shapes tolerated)."""
    if not isinstance(outputs, dict):
        return None
    for key in ("mcp_url", "url", "endpoint"):
        value = outputs.get(key)
        if isinstance(value, str) and value.strip().startswith("http"):
            return value.strip()
    nested = outputs.get("build") or outputs.get("result") or outputs.get("mcp")
    if isinstance(nested, dict):
        return extract_mcp_url(nested)
    return None


def apply_alembic_success(
    state: MutableMapping[str, Any],
    runtime: dict[str, Any],
    task_runtime: dict[str, Any],
    *,
    mcp_url: str,
    outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Inject Alembic MCP and reopen the task on post_build_route."""
    outputs = outputs if isinstance(outputs, dict) else {}
    task = ExperimentTask.model_validate(task_runtime["task"])
    if task.route != ExecutionRoute.ALEMBIC_BUILD:
        raise ValueError("apply_alembic_success requires planned route alembic_build")
    if not task.post_build_route:
        raise ValueError("apply_alembic_success requires post_build_route on the task")
    if not (mcp_url or "").strip().startswith("http"):
        raise ValueError(f"invalid mcp_url for alembic success: {mcp_url!r}")

    from CoScientist.experiments.schemas.models import MCPServerRef

    server = MCPServerRef.model_validate(
        {
            "name": _server_id_from_url(task.repo_url, mcp_url),
            "server_id": _server_id_from_url(task.repo_url, mcp_url),
            "url": mcp_url.strip(),
            "tools": _tool_refs_from_outputs(outputs),
            "source": "alembic",
            "health": "healthy",
        }
    )
    post_route = task.post_build_route
    updated = task.model_copy(
        update={
            "route": ExecutionRoute(post_route),
            "mcp_servers": [server],
            "post_build_route": None,
            "repo_url": task.repo_url,
        }
    )
    updated = ExperimentTask.model_validate(updated.model_dump(mode="json"))

    task_runtime["task"] = updated.model_dump(mode="json")
    task_runtime["current_route"] = post_route
    task_runtime["route_history"].append(
        {"route": post_route, "reason": "alembic_post_build", "mcp_url": mcp_url.strip()}
    )
    task_runtime["status"] = "ready"
    task_runtime["last_message"] = (
        f"Alembic MCP ready at {mcp_url.strip()}; continuing via {post_route}."
    )

    plan = runtime.get("plan")
    if isinstance(plan, dict):
        tasks = plan.get("tasks") or []
        for idx, item in enumerate(tasks):
            if isinstance(item, dict) and item.get("id") == updated.id:
                tasks[idx] = copy.deepcopy(task_runtime["task"])
                break
        plan["tasks"] = tasks

    server_json = server.model_dump(mode="json")
    deployed = list(state.get("deployed_mcps") or [])
    deployed.append(copy.deepcopy(server_json))
    state["deployed_mcps"] = deployed

    return {
        "post_build_pending": True,
        "post_build_route": post_route,
        "mcp_url": mcp_url.strip(),
        "server_id": server.server_id or server.name,
    }


__all__ = [
    "apply_alembic_success",
    "extract_mcp_url",
]
