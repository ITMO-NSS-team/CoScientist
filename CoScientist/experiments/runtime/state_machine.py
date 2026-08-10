"""Deterministic v0 task/attempt state machine (ADK session is the store)."""
from __future__ import annotations

import copy
import functools
import html
import logging
import mimetypes
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, MutableMapping
from uuid import uuid4

from CoScientist.config import get_settings
from CoScientist.config.settings import ExperimentsSettings
from CoScientist.experiments.schemas import (
    ArtifactRef,
    CriterionCheck,
    ExecutionRoute,
    ExperimentPlan,
    ExperimentTask,
    TaskResult,
    is_presigned_url,
    utc_now,
)
from CoScientist.experiments.schemas.models import artifact_name_from_location
from CoScientist.experiments.runtime.shared import (
    FABRICATION_MARKERS,
    MOLECULE_GENERATOR_TOOLS,
    artifact_name_key,
    audit,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_ARTIFACT_KEYS = ("mcp_artifacts", "fedot_artifacts", "coder_artifacts")
_AMEND_FIELDS = frozenset({
    "route",
    "mcp_servers",
    "repo_url",
    "post_build_route",
    "input_data",
    "launch_params",
    "warnings",
    "success_criteria",
})
_CLEAR_ACTIVE_KEYS = (
    "experiment_active_envelope", "filtered_tools", "deployed_mcps", "upstream_artifact_inputs"
)
# LLM often emits synonyms outside the closed TaskResult.status enum.
_RESULT_STATUS_ALIASES = {
    "error": "failure",
    "failed": "failure",
    "fail": "failure",
    "partial_success": "partial",
    "partially_successful": "partial",
    "incomplete": "partial",
    "ok": "success",
    "succeeded": "success",
}


def _result_text_blob(result: dict[str, Any]) -> str:
    parts: list[str] = [str(result.get("summary") or "")]
    for w in result.get("warnings") or []:
        parts.append(str(w))
    if result.get("error_message"):
        parts.append(str(result["error_message"]))
    for check in result.get("criteria_checks") or []:
        if isinstance(check, dict):
            parts.append(str(check.get("observed") or ""))
            parts.append(str(check.get("details") or ""))
    return "\n".join(parts)


def fabrication_signals(result: dict[str, Any]) -> list[str]:
    """Matched fabrication/simulation markers in a record_result payload."""
    blob = _result_text_blob(result)
    return sorted({m.group(0).lower() for m in FABRICATION_MARKERS.finditer(blob)})


def _downgrade_fabricated_success(result: dict[str, Any]) -> dict[str, Any]:
    """Force success→partial when the agent admits simulated/fabricated evidence."""
    if result.get("status") != "success":
        return result
    hits = fabrication_signals(result)
    if not hits:
        return result
    out = copy.deepcopy(result)
    out["status"] = "partial"
    warnings = list(out.get("warnings") or [])
    warnings.append(
        "downgraded_from_success: fabricated/simulated evidence detected "
        f"({', '.join(hits)})"
    )
    out["warnings"] = warnings
    return out


RUNTIME_KEY = "experiment_runtime"
ROUTE_AGENT_BY_ROUTE = {
    ExecutionRoute.FEDOT_MAS.value: "FedotAgent",
    ExecutionRoute.REACT_TOOLS.value: "ExperimentAgent",
    ExecutionRoute.CODER.value: "CoderAgent",
    ExecutionRoute.ALEMBIC_BUILD.value: "McpBuilderAgent",
}
# Defaults; prefer resolve_fallback_chains(settings) so EXPERIMENTS__FALLBACK_* apply.
FALLBACK_CHAINS = {
    ExecutionRoute.FEDOT_MAS.value: [
        ExecutionRoute.FEDOT_MAS.value,
        ExecutionRoute.REACT_TOOLS.value,
        ExecutionRoute.CODER.value,
    ],
    ExecutionRoute.REACT_TOOLS.value: [ExecutionRoute.REACT_TOOLS.value, ExecutionRoute.CODER.value],
    ExecutionRoute.CODER.value: [ExecutionRoute.CODER.value],
    ExecutionRoute.ALEMBIC_BUILD.value: [ExecutionRoute.ALEMBIC_BUILD.value, ExecutionRoute.CODER.value],
}
TERMINAL_TASK_STATES = frozenset({"done", "done_with_warnings", "failed", "skipped", "blocked"})
SUCCESS_DEPENDENCY_STATES = frozenset({"done", "done_with_warnings", "skipped"})


class ExperimentRuntimeError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code

    def as_dict(self) -> dict[str, Any]:
        return {"status": "error", "error_code": self.code, "message": str(self)}


def _settings(value: ExperimentsSettings | None) -> ExperimentsSettings:
    return value or get_settings().experiments


def resolve_fallback_chains(settings: ExperimentsSettings | None = None) -> dict[str, list[str]]:
    """Route fallback chains from settings (EXPERIMENTS__FALLBACK_*)."""
    cfg = _settings(settings)
    return {
        ExecutionRoute.FEDOT_MAS.value: list(cfg.fallback_fedot_mas),
        ExecutionRoute.REACT_TOOLS.value: list(cfg.fallback_react_tools),
        ExecutionRoute.CODER.value: list(cfg.fallback_coder),
        ExecutionRoute.ALEMBIC_BUILD.value: list(cfg.fallback_alembic_build),
    }


def _runtime(state: MutableMapping[str, Any]) -> dict[str, Any]:
    if not isinstance(runtime := state.get(RUNTIME_KEY), dict):
        raise ExperimentRuntimeError("runtime_missing", "No experiment runtime is active.")
    return runtime


def _task(runtime: dict[str, Any], task_id: str) -> dict[str, Any]:
    if not isinstance(task := (runtime.get("tasks") or {}).get(task_id), dict):
        raise ExperimentRuntimeError("task_not_found", f"Unknown experiment task {task_id!r}.")
    return task


_audit = functools.partial(audit, logger)


def _artifact_suffix(name: str) -> str:
    return Path(str(name)).suffix.lower()


def _artifact_extension_compatible(captured: str, expected: str) -> bool:
    """Suffixes agree, expected has no suffix, or both look tabular."""
    c_suf, e_suf = _artifact_suffix(captured), _artifact_suffix(expected)
    if not e_suf or not c_suf or c_suf == e_suf:
        return True
    tabular = {".csv", ".tsv", ".txt"}
    return c_suf in tabular and e_suf in tabular


def _match_expected_artifact(
    name: str,
    expected: list[Any],
    *,
    role: str | None = None,
    claimed_names: set[str] | None = None,
) -> Any | None:
    claimed = claimed_names or set()
    pool = [item for item in expected if item.name not in claimed]
    if (exact := next((item for item in pool if item.name == name), None)) is not None:
        return exact
    if key := artifact_name_key(name):
        soft = [item for item in pool if artifact_name_key(item.name) == key]
        if len(soft) == 1:
            return soft[0]
    # UUID tool filenames vs semantic plan names: unique role(+ext) remaining expected.
    role_key = (role or "data").strip().lower()
    same_role = [
        item for item in pool
        if str(getattr(item, "role", "data") or "data").strip().lower() == role_key
    ]
    compatible = [item for item in same_role if _artifact_extension_compatible(name, item.name)]
    if len(compatible) == 1:
        return compatible[0]
    if role_key == "data" and len(same_role) == 1:
        return same_role[0]
    return None


def _publish_active_tasks(state: MutableMapping[str, Any], runtime: dict[str, Any]) -> None:
    state["active_tasks"] = [
        {
            "id": task_id,
            "title": tr["task"]["name"],
            "description": tr["task"]["description"],
            "assignee": "ExperimentModuleAgent",
            "route": tr["current_route"],
            "status": tr["status"],
            "notes": tr.get("last_message", ""),
        }
        for task_id in runtime["task_order"]
        for tr in (runtime["tasks"][task_id],)
    ]


def _refresh_readiness(runtime: dict[str, Any]) -> None:
    tasks = runtime["tasks"]
    for task_id in runtime["task_order"]:
        task = tasks[task_id]
        if task["status"] != "pending":
            continue
        deps = [tasks[dep]["status"] for dep in task["task"]["depends_on"]]
        if any(status in {"failed", "blocked"} for status in deps):
            task["status"], task["last_message"] = "blocked", "A required dependency failed."
        elif all(status in SUCCESS_DEPENDENCY_STATES for status in deps):
            task["status"] = "ready"


def _clear_active(state: MutableMapping[str, Any], runtime: dict[str, Any]) -> None:
    runtime["active_task_id"] = runtime["active_attempt_id"] = None
    for key in _CLEAR_ACTIVE_KEYS:
        state[key] = None


def _finish_if_terminal(runtime: dict[str, Any]) -> None:
    if all(runtime["tasks"][tid]["status"] in TERMINAL_TASK_STATES for tid in runtime["task_order"]):
        runtime["phase"] = "reporting"


def _sync_after_mutation(
    state: MutableMapping[str, Any], runtime: dict[str, Any], *, clear_active: bool = False
) -> None:
    if clear_active:
        _clear_active(state, runtime)
    _refresh_readiness(runtime)
    _finish_if_terminal(runtime)
    _publish_active_tasks(state, runtime)


def initialize_runtime(
    state: MutableMapping[str, Any],
    plan: ExperimentPlan,
    *,
    critique: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create task-scoped runtime for one reviewed plan draft."""
    tasks = {
        task.id: {
            "status": "pending",
            "planned_route": task.route.value,
            "current_route": task.route.value,
            "route_history": [{"route": task.route.value, "reason": "planned"}],
            "task": task.model_dump(mode="json"),
            "attempts": {},
            "attempt_order": [],
            "last_message": "",
        }
        for task in plan.tasks
    }
    runtime = {
        "run_id": plan.experiment_run_id,
        "plan_id": plan.plan_id,
        "phase": "awaiting_review",
        "approved": False,
        "plan": plan.model_dump(mode="json"),
        "critique": critique,
        "active_task_id": None,
        "active_attempt_id": None,
        "task_order": [task.id for task in plan.tasks],
        "tasks": tasks,
        "results": [],
        "result_review_feedback": None,
    }
    _refresh_readiness(runtime)
    state[RUNTIME_KEY] = runtime
    state["experiment_plan"] = runtime["plan"]
    state["experiment_task_results"] = []
    state["experiment_artifacts_manifest"] = []
    state["experiment_summary"] = None
    _publish_active_tasks(state, runtime)
    return runtime


def approve_plan(state: MutableMapping[str, Any]) -> dict[str, Any]:
    runtime = _runtime(state)
    if runtime["phase"] != "awaiting_review":
        raise ExperimentRuntimeError("invalid_phase", f"Plan approval requires awaiting_review, got {runtime['phase']!r}.")
    if (runtime.get("critique") or {}).get("verdict") != "approve":
        raise ExperimentRuntimeError("critique_revise", "Plan cannot be approved while deterministic critique requires revision.")
    runtime["approved"] = True
    runtime["phase"] = "execution"
    _refresh_readiness(runtime)
    _publish_active_tasks(state, runtime)
    return {"status": "success", "phase": "execution", "plan_id": runtime["plan_id"]}


def get_experiment_plan(state: MutableMapping[str, Any]) -> dict[str, Any]:
    runtime = _runtime(state)
    return {
        "status": "success",
        "phase": runtime["phase"],
        "approved": runtime["approved"],
        "plan": copy.deepcopy(runtime["plan"]),
        "tasks": copy.deepcopy(runtime["tasks"]),
    }


def generate_presigned_s3_url(bucket: str, s3_key: str, expiration: int) -> str:
    """Fresh input URL without persisting it in the approved plan."""
    app = get_settings()
    if not app.s3.use_s3:
        raise ExperimentRuntimeError("s3_unavailable", f"Cannot resolve required S3 input s3://{bucket}/{s3_key}: S3 is disabled.")
    import boto3
    client = boto3.client(
        "s3",
        endpoint_url=app.s3.endpoint_url,
        aws_access_key_id=app.s3.access_key,
        aws_secret_access_key=app.s3.secret_key,
    )
    return client.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": s3_key}, ExpiresIn=expiration)


def _route_timeout(settings: ExperimentsSettings, route: str) -> float:
    return {
        ExecutionRoute.FEDOT_MAS.value: settings.fedot_timeout_s,
        ExecutionRoute.REACT_TOOLS.value: settings.react_timeout_s,
        ExecutionRoute.CODER.value: settings.coder_timeout_s,
        ExecutionRoute.ALEMBIC_BUILD.value: settings.coder_timeout_s,
    }[route]


def _route_enabled(route: str, settings: ExperimentsSettings) -> bool:
    if route == ExecutionRoute.FEDOT_MAS.value:
        return settings.route_fedot
    if route == ExecutionRoute.ALEMBIC_BUILD.value:
        return settings.route_alembic
    return route in {ExecutionRoute.REACT_TOOLS.value, ExecutionRoute.CODER.value}


def _find_artifact(
    runtime: dict[str, Any],
    artifact_ref: str,
    *,
    source_task_id: str | None = None,
) -> dict[str, Any]:
    """Resolve prior-task artifact by ART-* id or expected name (latest wins)."""
    want = str(artifact_ref or "").strip()
    if not want:
        raise ExperimentRuntimeError("artifact_not_found", "Required artifact ref is empty.")

    matches: list[dict[str, Any]] = []
    for result in runtime.get("results") or []:
        if source_task_id and result.get("task_id") != source_task_id:
            continue
        if result.get("status") not in {"success", "partial"}:
            continue
        for artifact in result.get("artifacts") or []:
            if not isinstance(artifact, dict):
                continue
            if artifact.get("artifact_id") == want:
                return artifact
            name = str(artifact.get("name") or "")
            if name == want or Path(name).name == Path(want).name:
                matches.append(artifact)
            elif artifact_name_key(name) and artifact_name_key(name) == artifact_name_key(want):
                matches.append(artifact)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return matches[-1]
    raise ExperimentRuntimeError("artifact_not_found", f"Required artifact {want!r} does not exist.")


def _resolve_attempt_id(runtime: dict[str, Any], task_id: str, attempt_id: str) -> str:
    """Accept verbatim ids; repair common LLM truncations of the active ATT-*."""
    active_task = runtime.get("active_task_id")
    active_attempt = runtime.get("active_attempt_id")
    if active_task == task_id and active_attempt == attempt_id:
        return attempt_id
    # Near-miss: executor often drops the last hex char of ATT-<uuid.hex>.
    if (
        active_task == task_id
        and isinstance(active_attempt, str)
        and isinstance(attempt_id, str)
        and active_attempt.startswith("ATT-")
        and attempt_id.startswith("ATT-")
        and (
            active_attempt.startswith(attempt_id)
            or attempt_id.startswith(active_attempt)
            or (
                len(active_attempt) == len(attempt_id)
                and sum(a != b for a, b in zip(active_attempt, attempt_id)) == 1
            )
        )
    ):
        _audit(
            f"EXPERIMENT_ATTEMPT_ID_REPAIRED task_id={task_id} "
            f"provided={attempt_id} active={active_attempt}"
        )
        return active_attempt
    raise ExperimentRuntimeError(
        "attempt_mismatch",
        "task_id/attempt_id do not match the active attempt."
        + (f" active_attempt_id={active_attempt!r}" if active_attempt else ""),
    )


def _resolve_inputs(
    runtime: dict[str, Any],
    task: ExperimentTask,
    *,
    route: str,
    settings: ExperimentsSettings,
    presign: Callable[[str, str, int], str],
) -> list[dict[str, Any]]:
    expiration = max(60, int(_route_timeout(settings, route) + 60))
    expires_at = (utc_now() + timedelta(seconds=expiration)).isoformat()
    resolved: list[dict[str, Any]] = []
    for data_ref in task.input_data:
        item = data_ref.model_dump(mode="json")
        try:
            if data_ref.kind == "s3":
                item["resolved_url"], item["expires_at"] = presign(str(data_ref.bucket), str(data_ref.s3_key), expiration), expires_at
            elif data_ref.kind == "task_artifact":
                artifact = _find_artifact(
                    runtime,
                    str(data_ref.source_artifact_id),
                    source_task_id=str(data_ref.source_task_id) if data_ref.source_task_id else None,
                )
                if artifact.get("bucket") and artifact.get("s3_key"):
                    item["resolved_url"], item["expires_at"] = presign(artifact["bucket"], artifact["s3_key"], expiration), expires_at
                elif artifact.get("workspace_path"):
                    item["resolved_workspace_path"] = artifact["workspace_path"]
                elif artifact.get("external_url"):
                    item["resolved_url"] = artifact["external_url"]
            elif data_ref.kind == "url":
                item["resolved_url"] = str(data_ref.url)
            elif data_ref.kind == "workspace":
                item["resolved_workspace_path"] = data_ref.workspace_path
        except Exception as exc:
            if data_ref.required:
                if isinstance(exc, ExperimentRuntimeError):
                    raise
                raise ExperimentRuntimeError("input_resolution_failed", f"Could not resolve required input {data_ref.data_id!r}: {exc}") from exc
            item["resolution_warning"] = str(exc)
        resolved.append(item)
    return resolved


def _uses_molecule_generator(task: ExperimentTask) -> bool:
    return any(tool.name in MOLECULE_GENERATOR_TOOLS for server in task.mcp_servers for tool in server.tools)


def force_managed_s3_launch_params(launch_params: dict[str, Any] | None, *, require: bool) -> dict[str, Any]:
    """Ensure molecule generators persist a managed S3 artifact for v0 lineage."""
    params = copy.deepcopy(launch_params or {})
    if require:
        params["upload_results_to_s3"] = True
        params.setdefault("output_s3_prefix", "generated")
    return params


def _scope_tools(task: ExperimentTask) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    filtered = [
        {"tool": tool.name, "server_id": server.server_id, "description": tool.description, "input_schema": tool.input_schema}
        for server in task.mcp_servers if server.server_id for tool in server.tools
    ]
    deployed = [
        {
            "name": server.name,
            "url": str(server.url),
            "description": "; ".join(tool.description for tool in server.tools),
            "tools": [tool.model_dump(mode="json") for tool in server.tools],
        }
        for server in task.mcp_servers if server.url
    ]
    return filtered, deployed


def start_task(
    state: MutableMapping[str, Any],
    task_id: str,
    *,
    settings: ExperimentsSettings | None = None,
    presign: Callable[[str, str, int], str] = generate_presigned_s3_url,
) -> dict[str, Any]:
    cfg = _settings(settings)
    runtime = _runtime(state)
    task_runtime = _task(runtime, task_id)

    if task_runtime["status"] in TERMINAL_TASK_STATES:
        raise ExperimentRuntimeError("task_terminal", f"Task {task_id} is already terminal.")
    if runtime["phase"] != "execution" or not runtime["approved"]:
        raise ExperimentRuntimeError("plan_not_approved", "Only an approved plan in execution may start tasks.")
    if runtime.get("active_attempt_id"):
        raise ExperimentRuntimeError("task_already_running", "v0 permits only one running task at a time.")

    _refresh_readiness(runtime)
    if task_runtime["status"] != "ready":
        raise ExperimentRuntimeError("task_not_ready", f"Task {task_id} must be ready, got {task_runtime['status']!r}.")
    route = task_runtime["current_route"]
    if _attempts_for_route(task_runtime, route) >= cfg.task_max_attempts:
        raise ExperimentRuntimeError(
            "attempt_budget_exhausted",
            f"Task {task_id} exhausted its {cfg.task_max_attempts} attempts on route {route!r}.",
        )

    if route == ExecutionRoute.FEDOT_MAS.value and not cfg.route_fedot:
        route = ExecutionRoute.REACT_TOOLS.value
        task_runtime["current_route"] = route
        task_runtime["route_history"].append({"route": route, "reason": "EXPERIMENTS__ROUTE_FEDOT kill-switch"})
    if not _route_enabled(route, cfg):
        raise ExperimentRuntimeError("route_disabled", f"Route {route!r} is disabled for Experiment Module v0.")

    task_model = ExperimentTask.model_validate(task_runtime["task"])
    if route == ExecutionRoute.CODER.value and task_runtime["planned_route"] == ExecutionRoute.CODER.value and task_model.mcp_servers and not cfg.route_coder_mcp:
        raise ExperimentRuntimeError("route_disabled", "Direct MCP-to-Coder mode is disabled.")

    if _uses_molecule_generator(task_model):
        task_model = task_model.model_copy(update={"launch_params": force_managed_s3_launch_params(task_model.launch_params, require=True)})
        task_runtime["task"] = task_model.model_dump(mode="json")

    attempt_no = len(task_runtime["attempt_order"]) + 1
    attempt_id = f"ATT-{uuid4().hex}"
    filtered_tools, deployed_mcps = _scope_tools(task_model)
    if route == ExecutionRoute.CODER.value and not cfg.route_coder_mcp:
        filtered_tools, deployed_mcps = [], []
    resolved_inputs = _resolve_inputs(runtime, task_model, route=route, settings=cfg, presign=presign)
    from CoScientist.tools.fedot_artifact_handoff import seed_upstream_from_resolved_inputs

    upstream_bindings = seed_upstream_from_resolved_inputs(
        state, resolved_inputs, filtered_tools
    )
    started_at = utc_now().isoformat()
    attempt = {
        "attempt_id": attempt_id,
        "attempt_no": attempt_no,
        "status": "running",
        "route": route,
        "route_returned": False,
        "started_at": started_at,
        "artifact_cursor": {key: len(state.get(key) or []) for key in _ARTIFACT_KEYS} | {"workspace_started_at": started_at},
        "tool_scope": {"filtered_tools": copy.deepcopy(filtered_tools), "deployed_mcps": copy.deepcopy(deployed_mcps)},
    }
    task_runtime["attempts"][attempt_id] = attempt
    task_runtime["attempt_order"].append(attempt_id)
    task_runtime["status"] = "running"
    runtime["active_task_id"] = task_id
    runtime["active_attempt_id"] = attempt_id

    envelope = {
        "plan_id": runtime["plan_id"],
        "experiment_run_id": runtime["run_id"],
        "task_id": task_id,
        "attempt_id": attempt_id,
        "attempt_no": attempt_no,
        "route": route,
        "route_agent": ROUTE_AGENT_BY_ROUTE[route],
        "task": task_model.model_dump(mode="json"),
        "resolved_inputs": resolved_inputs,
        "upstream_bindings": upstream_bindings,
    }
    state["experiment_active_envelope"] = envelope
    state["filtered_tools"] = filtered_tools
    state["deployed_mcps"] = deployed_mcps
    _publish_active_tasks(state, runtime)
    _audit(f"EXPERIMENT_TASK_STARTED task_id={task_id} attempt_id={attempt_id} route={route}")
    return {"status": "success", **copy.deepcopy(envelope)}


def active_attempt(state: MutableMapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    runtime = _runtime(state)
    task_id, attempt_id = runtime.get("active_task_id"), runtime.get("active_attempt_id")
    if not task_id or not attempt_id:
        raise ExperimentRuntimeError("attempt_missing", "No route attempt is active.")
    task_runtime = _task(runtime, task_id)
    if not isinstance(attempt := task_runtime["attempts"].get(attempt_id), dict):
        raise ExperimentRuntimeError("attempt_missing", "Active attempt is missing.")
    return runtime, task_runtime, attempt


def mark_route_returned(state: MutableMapping[str, Any], route_agent: str) -> None:
    runtime, _, attempt = active_attempt(state)
    if route_agent != (expected := ROUTE_AGENT_BY_ROUTE.get(attempt["route"])):
        raise ExperimentRuntimeError("route_mismatch", f"Attempt expects {expected}, not {route_agent}.")
    attempt["route_returned"] = True
    attempt["route_agent"] = route_agent
    runtime["last_route_agent"] = route_agent


def _captured_delta(state: MutableMapping[str, Any], attempt: dict[str, Any]) -> list[dict[str, Any]]:
    cursor = attempt["artifact_cursor"]
    return [
        copy.deepcopy(item)
        for key in _ARTIFACT_KEYS
        for item in (state.get(key) or [])[int(cursor.get(key, 0)):]
        if isinstance(item, dict)
    ]


def _materialize_signed_artifact(url: str, *, task_id: str, attempt_id: str, name: str) -> str | None:
    """Copy URL-only signed output into the report workspace."""
    try:
        import requests
        folder = Path(get_settings().code_exec.workspace_root) / "experiment_artifacts" / task_id / attempt_id
        folder.mkdir(parents=True, exist_ok=True)
        destination = folder / (Path(name).name or f"artifact-{uuid4().hex}")
        response = requests.get(html.unescape(url), timeout=30)
        response.raise_for_status()
        destination.write_bytes(response.content)
        return str(destination)
    except Exception:
        return None


def _normalise_artifacts(
    raw_artifacts: list[dict[str, Any]],
    *,
    runtime: dict[str, Any],
    task_runtime: dict[str, Any],
    attempt: dict[str, Any],
) -> tuple[list[ArtifactRef], list[str]]:
    task = ExperimentTask.model_validate(task_runtime["task"])
    expected = task.expected_artifacts
    artifacts, warnings = [], []
    seen: set[tuple[Any, ...]] = set()
    claimed_names: set[str] = set()

    for raw in raw_artifacts:
        name = artifact_name_from_location(raw)
        role_hint = raw.get("role") or "data"
        match = _match_expected_artifact(
            name, expected, role=str(role_hint), claimed_names=claimed_names
        )
        role = match.role if match else role_hint
        if match is not None:
            name = match.name
            claimed_names.add(match.name)

        bucket = raw.get("bucket") or raw.get("bucket_name")
        s3_key = raw.get("s3_key") or raw.get("results_s3_key")
        workspace_path = raw.get("workspace_path")
        external_url = raw.get("external_url") or raw.get("url")
        durability = raw.get("durability")

        if s3_key and not bucket:
            bucket = get_settings().s3.bucket_name

        if bucket and s3_key:
            external_url, durability = None, "managed"
            location_key = ("s3", bucket, s3_key)
        elif workspace_path:
            external_url, durability = None, durability or "workspace"
            location_key = ("workspace", workspace_path)
        elif external_url and is_presigned_url(external_url):
            if not (materialized := _materialize_signed_artifact(str(external_url), task_id=task.id, attempt_id=attempt["attempt_id"], name=name)):
                warnings.append(f"Could not materialize signed artifact {name!r}.")
                continue
            workspace_path, external_url, durability = materialized, None, "workspace"
            location_key = ("workspace", workspace_path)
        elif external_url:
            durability = durability or "transient"
            location_key = ("external", str(external_url))
        else:
            warnings.append(f"Captured artifact {name!r} has no canonical location.")
            continue

        if location_key in seen:
            continue
        seen.add(location_key)
        artifacts.append(
            ArtifactRef(
                artifact_id=raw.get("artifact_id") or f"ART-{uuid4().hex}",
                plan_id=runtime["plan_id"],
                task_id=task.id,
                attempt_id=attempt["attempt_id"],
                role=role,
                name=name,
                bucket=bucket if s3_key else None,
                s3_key=s3_key,
                workspace_path=workspace_path,
                external_url=external_url,
                media_type=raw.get("media_type") or mimetypes.guess_type(name)[0],
                size_bytes=raw.get("size_bytes"),
                checksum_sha256=raw.get("checksum_sha256"),
                producer_route=attempt["route"],
                producer_tool=raw.get("producer_tool") or raw.get("tool"),
                derived_from=raw.get("derived_from") or [],
                created_at=utc_now(),
                durability=durability,
            )
        )
    return artifacts, warnings


def _artifact_exists(artifact: ArtifactRef) -> bool:
    return bool(artifact.bucket and artifact.s3_key) or (Path(artifact.workspace_path).is_file() if artifact.workspace_path else bool(artifact.external_url))


def _artifact_matches(artifact: ArtifactRef, expected: Any) -> bool:
    return artifact.name == expected.name or artifact_name_key(artifact.name) == artifact_name_key(expected.name)


def _evidence_expected_artifacts(task: ExperimentTask, *, route: str) -> list:
    """Artifacts gated on this attempt (alembic build: MCP evidence only)."""
    required = [item for item in task.expected_artifacts if item.required]
    if route != ExecutionRoute.ALEMBIC_BUILD.value:
        return required
    return [
        item for item in required
        if item.role == "mcp_server" or item.name in {"mcp_endpoint", "mcp_url"}
    ]


def _evidence_success_criteria(task: ExperimentTask, *, route: str) -> list:
    """Criteria gated on this attempt (alembic build: execution only)."""
    required = [item for item in task.success_criteria if item.required]
    if route != ExecutionRoute.ALEMBIC_BUILD.value:
        return required
    return [item for item in required if item.kind == "execution"]


def _required_artifacts_present(
    task: ExperimentTask,
    artifacts: list[ArtifactRef],
    *,
    route: str | None = None,
) -> tuple[bool, list[str]]:
    missing, claimed = [], set()
    expected_items = (
        _evidence_expected_artifacts(task, route=route)
        if route is not None
        else [item for item in task.expected_artifacts if item.required]
    )
    for expected in expected_items:
        hit = next(
            (
                a
                for a in artifacts
                if a.artifact_id not in claimed and _artifact_exists(a) and _artifact_matches(a, expected)
            ),
            None,
        )
        if hit is None:
            missing.append(expected.name)
        else:
            claimed.add(hit.artifact_id)
    return not missing, missing


def _criteria_valid(
    task: ExperimentTask,
    checks: list[CriterionCheck],
    *,
    route: str | None = None,
) -> tuple[bool, list[str]]:
    by_id = {check.criterion_id: check for check in checks}
    if unknown := set(by_id) - {c.criterion_id for c in task.success_criteria}:
        raise ExperimentRuntimeError("criterion_unknown", f"Unknown criterion checks: {sorted(unknown)}.")
    required = (
        _evidence_success_criteria(task, route=route)
        if route is not None
        else [c for c in task.success_criteria if c.required]
    )
    failed = [
        c.criterion_id
        for c in required
        if c.criterion_id not in by_id or by_id[c.criterion_id].passed is not True
    ]
    return not failed, failed


def _next_fallback(
    task_runtime: dict[str, Any],
    settings: ExperimentsSettings | None = None,
) -> str | None:
    chain = resolve_fallback_chains(settings)[task_runtime["planned_route"]]
    if (index := chain.index(task_runtime["current_route"]) if task_runtime["current_route"] in chain else -1) < 0:
        return None
    used = {entry["route"] for entry in task_runtime["route_history"]}
    return next((r for r in chain[index + 1 :] if r not in used), None)


def _attempts_for_route(task_runtime: dict[str, Any], route: str) -> int:
    """Count attempts already spent on one route (task_max_attempts is per-route)."""
    attempts = task_runtime.get("attempts") or {}
    return sum(
        1
        for aid in task_runtime.get("attempt_order") or []
        if str((attempts.get(aid) or {}).get("route") or "") == route
    )


def _store_result(
    state: MutableMapping[str, Any],
    runtime: dict[str, Any],
    result: TaskResult,
) -> dict[str, Any]:
    result_json = result.model_dump(mode="json")
    runtime["results"].append(result_json)
    state["experiment_task_results"] = copy.deepcopy(runtime["results"])
    # Lazy import: review → runtime at module load; avoid cycle.
    from CoScientist.experiments.review import build_experiment_artifacts_manifest

    state["experiment_artifacts_manifest"] = build_experiment_artifacts_manifest(state)
    return result_json


def record_result(
    state: MutableMapping[str, Any],
    task_id: str,
    attempt_id: str,
    result: dict[str, Any],
    *,
    settings: ExperimentsSettings | None = None,
) -> dict[str, Any]:
    cfg = _settings(settings)
    runtime, task_runtime, attempt = active_attempt(state)
    attempt_id = _resolve_attempt_id(runtime, task_id, attempt_id)
    if attempt["status"] != "running":
        raise ExperimentRuntimeError("attempt_terminal", "Attempt is already terminal.")
    if not attempt.get("route_returned"):
        raise ExperimentRuntimeError("route_not_returned", "The route agent must return before record_result.")

    # Coerce common LLM synonyms before the closed-enum check.
    raw_status = str(result.get("status") or "").strip().lower().replace("-", "_")
    if raw_status in _RESULT_STATUS_ALIASES:
        coerced = _RESULT_STATUS_ALIASES[raw_status]
        patch: dict[str, Any] = {"status": coerced}
        if coerced == "failure":
            patch["retryable"] = bool(result.get("retryable", True))
        result = {**result, **patch}

    if (status := result.get("status")) not in {"success", "partial", "failure"}:
        raise ExperimentRuntimeError("result_status", "Result status must be success, partial, or failure.")

    result = _downgrade_fabricated_success(result)
    status = result["status"]

    task = ExperimentTask.model_validate(task_runtime["task"])
    checks = [CriterionCheck.model_validate(item) for item in (result.get("criteria_checks") or [])]
    raw_artifacts = _captured_delta(state, attempt)
    raw_artifacts.extend(copy.deepcopy(item) for item in (result.get("artifacts") or []) if isinstance(item, dict))
    outputs = result.get("outputs") or {}
    if isinstance(outputs, dict) and outputs:
        from CoScientist.experiments.runtime.inline_artifacts import materialize_outputs_as_artifacts
        raw_artifacts.extend(
            materialize_outputs_as_artifacts(
                task_id=task_id,
                attempt_id=attempt_id,
                expected_artifacts=[item.model_dump(mode="json") for item in task.expected_artifacts],
                outputs=outputs,
                existing=raw_artifacts,
            )
        )

    artifacts, artifact_warnings = _normalise_artifacts(raw_artifacts, runtime=runtime, task_runtime=task_runtime, attempt=attempt)
    attempt_route = str(attempt.get("route") or task_runtime.get("current_route") or "")
    artifacts_ok, missing_artifacts = _required_artifacts_present(task, artifacts, route=attempt_route)
    criteria_ok, failed_criteria = _criteria_valid(task, checks, route=attempt_route)
    if status in {"success", "partial"} and (not criteria_ok or not artifacts_ok):
        raise ExperimentRuntimeError(
            "result_incomplete",
            f"A successful/partial result is missing required evidence: criteria={failed_criteria}, artifacts={missing_artifacts}.",
        )

    task_result = TaskResult.model_validate({
        "schema_version": "task-result/0.1",
        "result_id": result.get("result_id") or f"RES-{uuid4().hex}",
        "plan_id": runtime["plan_id"],
        "task_id": task_id,
        "attempt_id": attempt_id,
        "attempt_no": attempt["attempt_no"],
        "status": status,
        "planned_route": task_runtime["planned_route"],
        "route_used": attempt["route"],
        "started_at": attempt["started_at"],
        "finished_at": utc_now(),
        "summary": result.get("summary") or f"{task.name}: {status}",
        "outputs": outputs if isinstance(outputs, dict) else {},
        "artifacts": artifacts,
        "criteria_checks": checks,
        "error_code": result.get("error_code"),
        "error_message": result.get("error_message"),
        "retryable": bool(result.get("retryable", False)),
        "warnings": [*(result.get("warnings") or []), *artifact_warnings],
    })
    result_json = _store_result(state, runtime, task_result)

    attempt["status"] = status
    attempt["result_id"] = task_result.result_id
    task_runtime["last_message"] = task_result.summary
    post_build: dict[str, Any] | None = None
    if status == "success":
        if attempt["route"] == ExecutionRoute.ALEMBIC_BUILD.value:
            from CoScientist.experiments.runtime.alembic_bridge import (
                apply_alembic_success,
                extract_mcp_url,
            )

            mcp_url = extract_mcp_url(outputs if isinstance(outputs, dict) else {})
            if not mcp_url:
                raise ExperimentRuntimeError(
                    "alembic_mcp_url_missing",
                    "Alembic success requires outputs.mcp_url before post_build_route can continue.",
                )
            if not task.post_build_route:
                raise ExperimentRuntimeError(
                    "alembic_post_build_missing",
                    "Alembic success requires post_build_route on the task.",
                )
            post_build = apply_alembic_success(
                state,
                runtime,
                task_runtime,
                mcp_url=mcp_url,
                outputs=outputs if isinstance(outputs, dict) else {},
            )
        else:
            task_runtime["status"] = "done"
    elif status == "partial":
        task_runtime["status"] = "done_with_warnings"
    else:
        route = str(task_runtime.get("current_route") or "")
        attempts_left = _attempts_for_route(task_runtime, route) < cfg.task_max_attempts
        next_fb = _next_fallback(task_runtime)
        # Same-route retries first; else next route in resolve_fallback_chains().
        if task_result.retryable and attempts_left:
            task_runtime["status"] = "retry_pending"
        elif next_fb is not None:
            task_runtime["status"] = "fallback_pending"
        else:
            task_runtime["status"] = "failed"

    _sync_after_mutation(state, runtime, clear_active=True)
    if post_build:
        # clear_active nulls deployed_mcps; restore Alembic servers for post_build start_task.
        state["deployed_mcps"] = copy.deepcopy(
            (task_runtime.get("task") or {}).get("mcp_servers") or []
        )
    managed = sum(1 for a in result_json["artifacts"] if a.get("bucket") and a.get("s3_key"))
    _audit(
        f"EXPERIMENT_RECORD_RESULT_SUCCESS task_id={task_id} attempt_id={attempt_id} "
        f"result_status={status} phase={runtime['phase']} artifacts={len(result_json['artifacts'])} "
        f"managed_artifacts={managed}"
        + (f" post_build_route={post_build.get('post_build_route')}" if post_build else "")
    )
    response = {"status": "success", "task_result": result_json, "phase": runtime["phase"]}
    if post_build:
        response["post_build"] = post_build
    return response


def retry_task(
    state: MutableMapping[str, Any],
    task_id: str,
    *,
    settings: ExperimentsSettings | None = None,
) -> dict[str, Any]:
    cfg = _settings(settings)
    runtime = _runtime(state)
    task_runtime = _task(runtime, task_id)
    if task_runtime["status"] != "retry_pending":
        raise ExperimentRuntimeError("retry_not_allowed", "retry_task requires a retryable failed attempt.")
    route = str(task_runtime.get("current_route") or "")
    if _attempts_for_route(task_runtime, route) >= cfg.task_max_attempts:
        raise ExperimentRuntimeError("attempt_budget_exhausted", f"Retry budget exhausted on route {route!r}.")
    task_runtime["status"] = "ready"
    task_runtime["last_message"] = "Retry approved; start_task will create a new attempt."
    _publish_active_tasks(state, runtime)
    return {"status": "success", "task_id": task_id, "route": task_runtime["current_route"]}


def fallback_task(
    state: MutableMapping[str, Any],
    task_id: str,
    reason: str,
    *,
    settings: ExperimentsSettings | None = None,
) -> dict[str, Any]:
    cfg = _settings(settings)
    runtime = _runtime(state)
    task_runtime = _task(runtime, task_id)
    if task_runtime["status"] != "fallback_pending":
        raise ExperimentRuntimeError("fallback_not_allowed", "fallback_task requires fallback_pending state.")
    if (route := _next_fallback(task_runtime)) is None:
        raise ExperimentRuntimeError("fallback_exhausted", "No acyclic fallback route remains.")
    if not _route_enabled(route, cfg):
        raise ExperimentRuntimeError("route_disabled", f"Fallback route {route!r} is disabled.")
    task_runtime["current_route"] = route
    task_runtime["route_history"].append({"route": route, "reason": reason})
    task_runtime["status"] = "ready"
    task_runtime["last_message"] = f"Fallback to {route}: {reason}"
    _publish_active_tasks(state, runtime)
    return {
        "status": "success",
        "task_id": task_id,
        "route": route,
        "next_action": "start_task",
        "must_start_task_id": task_id,
        "message": f"Fallback ready on {route}. Call start_task({task_id!r}) next — same task only.",
    }


def skip_task(
    state: MutableMapping[str, Any], task_id: str, reason: str
) -> dict[str, Any]:
    runtime = _runtime(state)
    task_runtime = _task(runtime, task_id)
    task = ExperimentTask.model_validate(task_runtime["task"])
    if not task.optional:
        raise ExperimentRuntimeError("skip_required", "Only optional v0 tasks may be skipped without human amendment.")
    if task_runtime["status"] not in {"pending", "ready"}:
        raise ExperimentRuntimeError("skip_not_allowed", f"Cannot skip task in {task_runtime['status']!r} state.")
    attempt_id = f"ATT-{uuid4().hex}"
    now = utc_now()
    result = TaskResult(
        schema_version="task-result/0.1",
        result_id=f"RES-{uuid4().hex}",
        plan_id=runtime["plan_id"],
        task_id=task_id,
        attempt_id=attempt_id,
        attempt_no=len(task_runtime["attempt_order"]) + 1,
        status="skipped",
        planned_route=task_runtime["planned_route"],
        route_used=task_runtime["current_route"],
        started_at=now,
        finished_at=now,
        summary=reason,
        criteria_checks=[],
    )
    task_runtime["attempt_order"].append(attempt_id)
    task_runtime["attempts"][attempt_id] = {
        "attempt_id": attempt_id,
        "attempt_no": result.attempt_no,
        "status": "skipped",
        "route": task_runtime["current_route"],
        "route_returned": False,
        "started_at": now.isoformat(),
        "result_id": result.result_id,
    }
    task_runtime["status"] = "skipped"
    task_runtime["last_message"] = reason
    result_json = _store_result(state, runtime, result)
    _sync_after_mutation(state, runtime)
    return {"status": "success", "task_result": result_json}


def amend_task(
    state: MutableMapping[str, Any],
    task_id: str,
    patch: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    runtime = _runtime(state)
    task_runtime = _task(runtime, task_id)
    if task_runtime["status"] not in {"pending", "ready"}:
        raise ExperimentRuntimeError("amend_not_allowed", "Only pending/ready v0 tasks may be amended.")
    if unknown := set(patch) - _AMEND_FIELDS:
        raise ExperimentRuntimeError("amend_fields", f"Unsupported amendment fields: {sorted(unknown)}.")
    amended = copy.deepcopy(task_runtime["task"])
    amended.update(copy.deepcopy(patch))
    task = ExperimentTask.model_validate(amended)
    task_runtime["task"] = task.model_dump(mode="json")
    task_runtime["current_route"] = task.route.value
    task_runtime["planned_route"] = task.route.value
    task_runtime["route_history"].append({"route": task.route.value, "reason": f"amend: {reason}"})
    requires_review = "success_criteria" in patch
    if requires_review:
        runtime["approved"] = False
        runtime["phase"] = "awaiting_review"
    task_runtime["last_message"] = f"Amended: {reason}"
    _publish_active_tasks(state, runtime)
    return {
        "status": "success",
        "task_id": task_id,
        "requires_review": requires_review,
        "phase": runtime["phase"],
    }


def mark_result_review(
    state: MutableMapping[str, Any],
    *,
    approved: bool,
    feedback: str | None = None,
) -> dict[str, Any]:
    runtime = _runtime(state)
    if runtime["phase"] not in {"reporting", "awaiting_result_review"}:
        raise ExperimentRuntimeError("invalid_phase", "Result review requires a reported experiment.")
    if approved:
        runtime["phase"] = "completed"
    else:
        runtime["phase"] = "replan_requested"
        runtime["result_review_feedback"] = feedback or "Result redesign requested."
    return {"status": "success", "phase": runtime["phase"]}


__all__ = [
    "ExperimentRuntimeError",
    "FALLBACK_CHAINS",
    "ROUTE_AGENT_BY_ROUTE",
    "RUNTIME_KEY",
    "active_attempt",
    "amend_task",
    "approve_plan",
    "fallback_task",
    "force_managed_s3_launch_params",
    "fabrication_signals",
    "generate_presigned_s3_url",
    "get_experiment_plan",
    "initialize_runtime",
    "mark_result_review",
    "mark_route_returned",
    "record_result",
    "resolve_fallback_chains",
    "retry_task",
    "skip_task",
    "start_task",
]
