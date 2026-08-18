"""AgentTool / control-tool callbacks: one route + mandatory record_result."""
from __future__ import annotations

import copy
import json
import logging
from typing import Any, Optional

from google.adk.models import LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from .state_machine import (
    ROUTE_AGENT_BY_ROUTE,
    ExperimentRuntimeError,
    active_attempt,
    mark_route_returned,
)
from .shared import GATE_ROUTED_STATE_KEY, audit, schema_offers_s3_upload, session_inventory_rows

logger = logging.getLogger(__name__)
ROUTE_AGENT_NAMES = frozenset(ROUTE_AGENT_BY_ROUTE.values())
ROUTE_ALREADY_RETURNED_MESSAGE = (
    "Route already returned for this attempt. Call record_result, retry_task, "
    "fallback_task, skip_task, or amend_task."
)
# Close/read tools while route_returned + no result. retry/fallback need stored
# failure — refuse here so after_model can force record_result.
_PENDING_RECORD_ALLOWED = frozenset(
    {"record_result", "skip_task", "amend_task", "get_experiment_plan"}
)
RECORD_REQUIRED_MESSAGE = (
    "Route already returned for this attempt but record_result was not called. "
    "Call record_result with the same task_id and attempt_id from start_task "
    "(or skip_task / amend_task). Do not call retry_task/fallback_task until "
    "after record_result closes this attempt. Do not start another task or "
    "finish in prose."
)

def _stringify_agent_tool_request(args: dict[str, Any]) -> None:
    """ADK AgentTool requires ``request`` as a string (``Part.text``)."""
    if "request" not in args or isinstance(args["request"], str):
        return
    val = args["request"]
    args["request"] = (
        json.dumps(val, ensure_ascii=False) if isinstance(val, (dict, list))
        else str(val) if val is not None else val
    )

# Process-local pin so nested McpBuilder tool callbacks see the EM repo even
# if the sub-agent session copy is stale. Cleared when the executor delegates.
_EM_ALEMBIC_PIN: dict[str, Any] = {}


def _set_em_alembic_pin(
    *,
    repo_url: str,
    run_id: str | None = None,
    task_id: str | None = None,
    attempt_id: str | None = None,
) -> None:
    _EM_ALEMBIC_PIN.clear()
    _EM_ALEMBIC_PIN.update({
        "repo_url": repo_url,
        "run_id": run_id,
        "task_id": task_id,
        "attempt_id": attempt_id,
    })


def _em_alembic_attempt(
    state: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    """Active alembic_build attempt, or None."""
    try:
        runtime, task_runtime, attempt = active_attempt(state)
    except ExperimentRuntimeError:
        return None
    if str(attempt.get("route") or "") != "alembic_build":
        return None
    return runtime, task_runtime, attempt


def _inject_alembic_repo_url(
    args: dict[str, Any],
    task: dict[str, Any],
    *,
    runtime: dict[str, Any] | None = None,
    attempt: dict[str, Any] | None = None,
) -> None:
    """Force ``task.repo_url`` onto the McpBuilder request."""
    repo_url = task.get("repo_url")
    if not repo_url:
        return
    raw = args.get("request")
    payload: dict[str, Any] = {}
    if isinstance(raw, dict):
        payload = dict(raw)
    elif isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            payload = parsed
    payload["repo_url"] = repo_url
    args["request"] = payload
    _set_em_alembic_pin(
        repo_url=str(repo_url),
        run_id=str((runtime or {}).get("run_id") or "") or None,
        task_id=str((runtime or {}).get("active_task_id") or "") or None,
        attempt_id=str((attempt or {}).get("attempt_id") or "") or None,
    )


def pin_alembic_build_args(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext,
) -> dict[str, Any] | None:
    """before_tool on McpBuilder: pin ``repo_url`` from the EM task."""
    if getattr(tool, "name", "") != "build_mcp_server":
        return None
    state = tool_context.state
    ctx = _em_alembic_attempt(state)
    pin = dict(_EM_ALEMBIC_PIN)
    repo_url = ""
    run_id = task_id = attempt_id = None
    if ctx is not None:
        runtime, task_runtime, attempt = ctx
        repo_url = str((task_runtime.get("task") or {}).get("repo_url") or "")
        run_id = runtime.get("run_id")
        task_id = runtime.get("active_task_id")
        attempt_id = attempt.get("attempt_id")
    if not repo_url:
        repo_url = str(pin.get("repo_url") or "")
        run_id = run_id or pin.get("run_id")
        task_id = task_id or pin.get("task_id")
        attempt_id = attempt_id or pin.get("attempt_id")
    if not repo_url:
        return None
    args["repo_url"] = repo_url
    args["force_rebuild"] = False
    if run_id:
        args["run_id"] = str(run_id)
    if task_id:
        args["task_id"] = str(task_id)
        args["idempotency_key"] = f"{run_id or ''}:{task_id}:{repo_url.rstrip('/').lower()}"
    if attempt_id:
        args["attempt_id"] = str(attempt_id)
    audit(logger, f"EXPERIMENT_ALEMBIC_PIN repo_url={repo_url} task_id={task_id}")
    return None


def await_alembic_job_if_experiment(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext, tool_response: Any,
) -> Any:
    """after_tool on McpBuilder: block until the EM build is done or failed."""
    if getattr(tool, "name", "") != "build_mcp_server":
        return None
    if not isinstance(tool_response, dict):
        return None
    state = tool_context.state
    ctx = _em_alembic_attempt(state)
    if ctx is None and not _EM_ALEMBIC_PIN.get("repo_url"):
        return None
    job_id = str(tool_response.get("job_id") or "").strip()
    if not job_id:
        return None
    if ctx is not None:
        _, _, attempt = ctx
        attempt["alembic_job_id"] = job_id
    if tool_response.get("status") in {"done", "failed", "error"}:
        from CoScientist.tools.alembic_tools import enrich_snapshot_with_tools

        snap = enrich_snapshot_with_tools(dict(tool_response))
        if ctx is not None:
            ctx[2]["alembic_snapshot"] = copy.deepcopy(snap)
        _EM_ALEMBIC_PIN["snapshot"] = copy.deepcopy(snap)
        audit(
            logger,
            f"EXPERIMENT_ALEMBIC_WAIT_DONE job_id={job_id} status={snap.get('status')} "
            f"mcp_url={snap.get('mcp_url') or ''} reused=1",
            stdout=(
                f"EXPERIMENT_ALEMBIC_WAIT_DONE job_id={job_id} status={snap.get('status')} "
                f"mcp_url={snap.get('mcp_url') or ''}"
            ),
        )
        return snap
    from CoScientist.config import get_settings
    from CoScientist.tools.alembic_tools import wait_mcp_build

    cfg = get_settings().experiments
    audit(logger, f"EXPERIMENT_ALEMBIC_WAIT job_id={job_id} timeout_s={cfg.alembic_timeout_s}")
    snap = wait_mcp_build(
        job_id, timeout_s=cfg.alembic_timeout_s, poll_s=cfg.alembic_poll_s,
    )
    # alembic_timeout_s is a heartbeat, not a protocol exit: returning
    # status=running lets FORCE_RECORD loop against alembic_build_running.
    while snap.get("status") == "running":
        audit(
            logger,
            f"EXPERIMENT_ALEMBIC_WAIT_EXTEND job_id={job_id} "
            f"timeout_s={cfg.alembic_timeout_s}",
        )
        snap = wait_mcp_build(
            job_id, timeout_s=cfg.alembic_timeout_s, poll_s=cfg.alembic_poll_s,
        )
    from CoScientist.tools.alembic_tools import enrich_snapshot_with_tools

    snap = enrich_snapshot_with_tools(snap if isinstance(snap, dict) else {})
    if ctx is not None:
        ctx[2]["alembic_job_id"] = job_id
        ctx[2]["alembic_snapshot"] = copy.deepcopy(snap)
    _EM_ALEMBIC_PIN["snapshot"] = copy.deepcopy(snap)
    audit(
        logger,
        f"EXPERIMENT_ALEMBIC_WAIT_DONE job_id={job_id} status={snap.get('status')} "
        f"mcp_url={snap.get('mcp_url') or ''} timed_out={bool(snap.get('wait_timed_out'))}",
        stdout=(
            f"EXPERIMENT_ALEMBIC_WAIT_DONE job_id={job_id} status={snap.get('status')} "
            f"mcp_url={snap.get('mcp_url') or ''}"
        ),
    )
    return snap

def _pending_record_attempt(
    state: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    """Active attempt with route returned and no stored result yet."""
    try:
        runtime, task_runtime, attempt = active_attempt(state)
    except ExperimentRuntimeError:
        return None
    # status stays "running" until record_result; result_id is the close signal.
    if (
        not attempt.get("route_returned")
        or attempt.get("result_id")
        or attempt.get("status") not in {None, "running"}
    ):
        return None
    return runtime, task_runtime, attempt

def guard_route_agent_tool(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext,
) -> dict[str, Any] | None:
    """Refuse second/mismatched AgentTool, or control calls before record."""
    tool_name = getattr(tool, "name", "") or ""
    state = tool_context.state
    pending = _pending_record_attempt(state)
    if tool_name in ROUTE_AGENT_NAMES:
        try:
            _, task_runtime, attempt = active_attempt(state)
        except ExperimentRuntimeError as exc:
            return exc.as_dict()
        if tool_name != (expected := ROUTE_AGENT_BY_ROUTE.get(attempt["route"])):
            return {
                "status": "refused", "error_code": "route_mismatch",
                "message": f"Active attempt requires {expected}, not {tool_name}.",
            }
        if attempt.get("route_returned"):
            return {
                "status": "refused", "error_code": "route_already_returned",
                "message": ROUTE_ALREADY_RETURNED_MESSAGE,
            }
        if tool_name == "McpBuilderAgent":
            _inject_alembic_repo_url(
                args, task_runtime.get("task") or {},
                runtime=tool_context.state.get("experiment_runtime") or {},
                attempt=attempt,
            )
        elif tool_name in {"FedotAgent", "ExperimentAgent"}:
            from CoScientist.experiments.runtime.alembic_bridge import (
                pin_alembic_post_build_request,
            )

            if pin_alembic_post_build_request(
                args, task_runtime,
                runtime=tool_context.state.get("experiment_runtime") or {},
                state=state,
            ):
                audit(
                    logger,
                    "EXPERIMENT_ALEMBIC_POST_BUILD_PIN "
                    f"agent={tool_name} task_id={task_runtime.get('task', {}).get('id')}",
                )
        _stringify_agent_tool_request(args)
        return None
    if pending is not None and tool_name and tool_name not in _PENDING_RECORD_ALLOWED:
        runtime, _, attempt = pending
        return {
            "status": "refused", "error_code": "record_result_required",
            "message": RECORD_REQUIRED_MESSAGE,
            "must_record_task_id": runtime.get("active_task_id"),
            "must_record_attempt_id": attempt.get("attempt_id"),
            "next_action": "record_result",
        }
    return None

def _schema_from_tool(tool: BaseTool, tool_context: ToolContext) -> dict[str, Any]:
    for attr in ("input_schema", "schema"):
        val = getattr(tool, attr, None)
        if isinstance(val, dict):
            return val
    name = getattr(tool, "name", "")
    for item in tool_context.state.get("filtered_tools") or []:
        if isinstance(item, dict) and item.get("tool") == name and isinstance(item.get("input_schema"), dict):
            return item["input_schema"]
    return {}


def force_schema_s3_upload(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext,
) -> dict[str, Any] | None:
    """If the tool schema offers upload_results_to_s3, force it on during EM runs."""
    if not tool_context.state.get("experiment_runtime"):
        return None
    if schema_offers_s3_upload(_schema_from_tool(tool, tool_context)):
        args["upload_results_to_s3"] = True
        args.setdefault("output_s3_prefix", "generated")
    return None


def force_molecule_generator_s3_upload(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext,
) -> dict[str, Any] | None:
    """Backward-compatible alias — schema-driven, not a named-tool list."""
    return force_schema_s3_upload(tool, args, tool_context)

def on_route_agent_returned(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext, tool_response: Any,
) -> None:
    """Close the route slot after a successful or failed agent response."""
    tool_name = getattr(tool, "name", "")
    if tool_name not in ROUTE_AGENT_NAMES:
        return
    try:
        _, _, attempt = active_attempt(tool_context.state)
        if tool_name != ROUTE_AGENT_BY_ROUTE.get(attempt["route"]) or attempt.get("route_returned"):
            return
        if tool_name == "CoderAgent":
            from CoScientist.experiments.runtime.coder_artifacts import promote_coder_workspace_artifacts
            promote_coder_workspace_artifacts(tool_context.state)
        stored = tool_response
        snap = attempt.get("alembic_snapshot") or _EM_ALEMBIC_PIN.get("snapshot")
        if tool_name == "McpBuilderAgent" and isinstance(snap, dict):
            stored = copy.deepcopy(snap)
            if not attempt.get("alembic_snapshot"):
                attempt["alembic_snapshot"] = stored
            if snap.get("job_id"):
                attempt["alembic_job_id"] = snap["job_id"]
        mark_route_returned(tool_context.state, tool_name)
        tool_context.state["experiment_last_route_response"] = copy.deepcopy(stored)
    except ExperimentRuntimeError:
        return

def _force_call(name: str, args: dict[str, Any], role: str = "model") -> LlmResponse:
    """LlmResponse that replaces the model turn with one forced function call."""
    return LlmResponse(content=types.Content(
        role=role,
        parts=[types.Part.from_function_call(name=name, args=args)],
    ))

def _llm_has_pending_close_call(llm_response: LlmResponse) -> bool:
    content = getattr(llm_response, "content", None)
    parts = getattr(content, "parts", None) if content is not None else None
    return any(
        getattr(getattr(p, "function_call", None), "name", None) in _PENDING_RECORD_ALLOWED
        for p in (parts or [])
    )

def _summary_from_last_route(state: dict[str, Any]) -> str:
    fallback = "Route returned; executor omitted record_result — auto-closing attempt."
    last = state.get("experiment_last_route_response")
    if last is None:
        return fallback
    if isinstance(last, dict):
        for key in ("summary", "message", "status"):
            if last.get(key):
                return str(last.get(key))[:1500]
        return str(last)[:1500]
    text = str(last).strip()
    return text[:1500] if text else fallback

def _auto_record_result_payload(
    state: dict[str, Any], task_runtime: dict[str, Any], attempt: dict[str, Any],
) -> dict[str, Any]:
    """Best-effort TaskResult so the control loop cannot skip record_result."""
    from CoScientist.experiments.runtime.artifacts import captured_delta
    criteria = (task_runtime.get("task") or {}).get("success_criteria") or []
    summary = _summary_from_last_route(state)
    last = state.get("experiment_last_route_response")
    snap = last if isinstance(last, dict) else {}
    if not snap and isinstance(attempt.get("alembic_snapshot"), dict):
        snap = attempt["alembic_snapshot"]
    mcp_url = ""
    if str(attempt.get("route") or "") == "alembic_build":
        from CoScientist.experiments.runtime.alembic_bridge import harvest_alembic_mcp_url

        task = task_runtime.get("task") if isinstance(task_runtime.get("task"), dict) else {}
        mcp_url = harvest_alembic_mcp_url(
            snap,
            last,
            summary,
            repo_url=str(task.get("repo_url") or "").strip() or None,
        )
        if mcp_url.startswith("http"):
            checks = [
                {"criterion_id": cid, "passed": True, "details": f"mcp_url={mcp_url}"}
                for item in (criteria if isinstance(criteria, list) else [])
                if isinstance(item, dict) and (cid := str(item.get("criterion_id") or "").strip())
            ]
            return {
                "status": "success",
                "summary": f"Alembic MCP ready at {mcp_url}",
                "outputs": {
                    "mcp_url": mcp_url,
                    "mcp_endpoint": mcp_url,
                    "tools": snap.get("tools") or [],
                    "image": snap.get("image"),
                    "container": snap.get("container"),
                    "job_id": snap.get("job_id") or attempt.get("alembic_job_id"),
                },
                "criteria_checks": checks,
                "retryable": False,
                "warnings": ["auto_recorded_alembic_success"],
            }
    # Alembic evidence is mcp_url, not leftover captures. Success-without-URL
    # raises alembic_mcp_url_missing and leaves the attempt running.
    has_artifacts = (
        bool(captured_delta(state, attempt))
        and str(attempt.get("route") or "") != "alembic_build"
    )
    detail = (
        "Auto-recorded: route returned and executor omitted "
        "record_result; evidence taken from route capture."
        if has_artifacts else
        "Auto-recorded failure: route returned with no "
        "captured artifacts and executor omitted record_result."
    )
    checks = [
        {"criterion_id": cid, "passed": has_artifacts, "details": detail}
        for item in (criteria if isinstance(criteria, list) else [])
        if isinstance(item, dict) and (cid := str(item.get("criterion_id") or "").strip())
    ]
    base: dict[str, Any] = {
        "summary": summary, "criteria_checks": checks, "outputs": {},
        "warnings": ["auto_recorded_omitted_record_result"],
    }
    if has_artifacts:
        return {**base, "status": "success", "retryable": False}
    return {
        **base, "status": "failure", "error_code": "route_failed_or_empty",
        "error_message": (
            "Auto-recorded: route returned without captured artifacts; "
            "executor omitted record_result."
        ),
        "retryable": True,
    }

def _alembic_job_still_running(attempt: dict[str, Any]) -> bool:
    if str(attempt.get("route") or "") != "alembic_build":
        return False
    job_id = str(attempt.get("alembic_job_id") or "").strip()
    if not job_id:
        return False
    from CoScientist.tools.alembic_tools import peek_mcp_build

    return peek_mcp_build(job_id).get("status") == "running"


def enforce_pending_record_result(
    callback_context: Any, llm_response: LlmResponse,
) -> LlmResponse | None:
    """after_model: force record_result when route returned but model skips close."""
    state = callback_context.state
    pending = _pending_record_attempt(state)
    if pending is None or _llm_has_pending_close_call(llm_response):
        return None
    _, task_runtime, attempt = pending
    if _alembic_job_still_running(attempt):
        return None
    runtime = state.get("experiment_runtime") or {}
    task_id = str(runtime.get("active_task_id") or "")
    attempt_id = str(attempt.get("attempt_id") or runtime.get("active_attempt_id") or "")
    if not task_id or not attempt_id:
        return None
    payload = _auto_record_result_payload(state, task_runtime, attempt)
    audit(
        logger,
        f"EXPERIMENT_FORCE_RECORD_RESULT task_id={task_id} attempt_id={attempt_id} "
        f"status={payload.get('status')} retryable={payload.get('retryable')}",
    )
    return _force_call(
        "record_result",
        {"task_id": task_id, "attempt_id": attempt_id, "result": payload},
    )


def _llm_has_any_function_call(llm_response: LlmResponse) -> bool:
    return bool(_llm_function_names(llm_response))


def _llm_function_names(llm_response: LlmResponse) -> list[str]:
    content = getattr(llm_response, "content", None)
    names: list[str] = []
    for part in getattr(content, "parts", None) or []:
        name = getattr(getattr(part, "function_call", None), "name", None)
        if name:
            names.append(str(name))
    return names


def _pending_route_agent(state: Any) -> tuple[str, dict[str, Any]] | None:
    """Active attempt waiting for its route AgentTool — not a control tool."""
    try:
        runtime, _, attempt = active_attempt(state)
    except ExperimentRuntimeError:
        return None
    if attempt.get("route_returned") or attempt.get("result_id"):
        return None
    name = ROUTE_AGENT_BY_ROUTE.get(str(attempt.get("route") or ""))
    if not name:
        return None
    envelope = state.get("experiment_active_envelope") if hasattr(state, "get") else None
    if isinstance(envelope, dict) and envelope:
        request = json.dumps(envelope, ensure_ascii=False)
    else:
        request = json.dumps(
            {
                "task_id": runtime.get("active_task_id"),
                "attempt_id": runtime.get("active_attempt_id"),
            },
            ensure_ascii=False,
        )
    return name, {"request": request}


def _running_task_id(runtime: dict[str, Any]) -> str | None:
    tasks = runtime.get("tasks") or {}
    for tid in runtime.get("task_order") or []:
        tr = tasks.get(tid) if isinstance(tasks.get(tid), dict) else None
        if tr and str(tr.get("status") or "") == "running":
            return str(tid)
    return None


def _next_control_action(runtime: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """Pick the next deterministic control call while phase=execution."""
    # v0 is single-flight: never nudge start/retry/fallback while a task runs.
    if _running_task_id(runtime) is not None:
        return None
    tasks = runtime.get("tasks") or {}
    for tid in runtime.get("task_order") or []:
        tr = tasks.get(tid) if isinstance(tasks.get(tid), dict) else None
        if not tr:
            continue
        status = str(tr.get("status") or "")
        if status == "ready":
            return "start_task", {"task_id": tid}
        if status == "retry_pending":
            return "retry_task", {"task_id": tid}
        if status == "fallback_pending":
            return "fallback_task", {"task_id": tid}
    return None


def enforce_continue_until_reporting(
    callback_context: Any, llm_response: LlmResponse,
) -> LlmResponse | None:
    """after_model: block prose-only exit while ready/retry/fallback work remains."""
    state = callback_context.state
    runtime = state.get("experiment_runtime") or {}
    if runtime.get("phase") != "execution":
        return None
    if pending_route := _pending_route_agent(state):
        name, args = pending_route
        if name in _llm_function_names(llm_response):
            return None
        audit(logger, f"EXPERIMENT_FORCE_ROUTE_AGENT action={name}")
        return _force_call(name, args)
    if _pending_record_attempt(state) is not None:
        return None
    if _llm_has_any_function_call(llm_response):
        return None
    action = _next_control_action(runtime)
    if action is None:
        return None
    name, args = action
    audit(logger, f"EXPERIMENT_FORCE_CONTINUE action={name} args={args}")
    return _force_call(name, args)


_CONTROL_TRANSITION_TOOLS = frozenset(
    {"retry_task", "fallback_task", "start_task", "skip_task", "amend_task"}
)


def rewrite_mismatched_control_action(
    callback_context: Any, llm_response: LlmResponse,
) -> LlmResponse | None:
    """after_model: rewrite wrong retry/fallback/start to the next control action."""
    state = callback_context.state
    runtime = state.get("experiment_runtime") or {}
    phase = str(runtime.get("phase") or "")
    content = getattr(llm_response, "content", None)
    parts = list(getattr(content, "parts", None) or [])
    control_fcs: list[tuple[int, str, dict[str, Any]]] = []
    for i, part in enumerate(parts):
        fc = getattr(part, "function_call", None)
        name = getattr(fc, "name", None) if fc is not None else None
        if name in _CONTROL_TRANSITION_TOOLS:
            control_fcs.append((i, name, dict(getattr(fc, "args", None) or {})))

    if phase == "execution" and _pending_record_attempt(state) is None:
        if pending_route := _pending_route_agent(state):
            name, args = pending_route
            if name not in _llm_function_names(llm_response):
                called = {n for _, n, _ in control_fcs}
                called.update(
                    n for n in _llm_function_names(llm_response)
                    if n in ROUTE_AGENT_NAMES or n in _CONTROL_TRANSITION_TOOLS
                    or n == "get_experiment_plan"
                )
                if called:
                    audit(
                        logger,
                        f"EXPERIMENT_REWRITE_CONTROL from={sorted(called)} to={name} "
                        f"reason=pending_route_agent",
                        stdout=f"EXPERIMENT_REWRITE_CONTROL to={name} reason=pending_route_agent",
                    )
                    return _force_call(
                        name, args, role=getattr(content, "role", None) or "model",
                    )

    if not control_fcs:
        return None

    def _suppress(reason: str) -> LlmResponse:
        audit(
            logger,
            f"EXPERIMENT_REWRITE_CONTROL suppress reason={reason} "
            f"from={[n for _, n, _ in control_fcs]} phase={phase}",
            stdout=f"EXPERIMENT_REWRITE_CONTROL suppress reason={reason} phase={phase}",
        )
        return _force_call(
            "get_experiment_plan", {}, role=getattr(content, "role", None) or "model",
        )

    # Outside execution, start/retry/fallback/skip hard-error — park on get_experiment_plan.
    if phase != "execution":
        return _suppress(f"phase_{phase or 'none'}")

    if _pending_record_attempt(state) is not None:
        return None

    running = _running_task_id(runtime)
    if running is not None:
        bad = [
            (n, a) for _, n, a in control_fcs
            if n in {"start_task", "retry_task", "fallback_task", "skip_task"}
        ]
        if bad:
            return _suppress(f"while_running:{running}")
        return None

    expected = _next_control_action(runtime)
    if expected is None:
        # No legal transition — suppress orphan retry/fallback/skip/start.
        return _suppress("no_pending_transition")

    exp_name, exp_args = expected
    # skip_task on the task that would otherwise start is a legal attempt to
    # leave the queue; rewriting it back to start_task is what turned
    # missing-artifact into an infinite start loop.
    for _, name, args in control_fcs:
        if name != "skip_task" or exp_name != "start_task":
            continue
        if str(args.get("task_id") or "") == str(exp_args.get("task_id") or ""):
            return None
    # If any control call already matches the expected transition, leave alone.
    for _, name, args in control_fcs:
        if name == exp_name and str(args.get("task_id") or "") == str(exp_args.get("task_id") or ""):
            return None
    # Build corrected args (fallback needs a reason).
    fixed_args = dict(exp_args)
    if exp_name == "fallback_task" and not str(fixed_args.get("reason") or "").strip():
        wrong = ",".join(sorted({n for _, n, _ in control_fcs}))
        fixed_args["reason"] = (
            f"Auto-corrected control action (model called {wrong}; "
            f"runtime requires {exp_name})."
        )
    audit(
        logger,
        f"EXPERIMENT_REWRITE_CONTROL from={[n for _, n, _ in control_fcs]} "
        f"to={exp_name} args={fixed_args}",
        stdout=f"EXPERIMENT_REWRITE_CONTROL to={exp_name} task_id={fixed_args.get('task_id')}",
    )
    return _force_call(
        exp_name, fixed_args, role=getattr(content, "role", None) or "model",
    )


# ── EM inventory feasibility (early NO_MATCHING_TOOL) ───────────────────────
# After ToolPreparer, decide whether a structurally-gated ask (see
# GATE_ROUTED_STATE_KEY) can be treated as compute. Explicit orchestrator
# choices of the module are always trusted and never checked here — only asks
# that arrived because enforce_experiment_module_first rewrote a Research call
# are eligible for an early NO_MATCHING_TOOL, so unrelated retrieved tools
# cannot drag a literature ask into Hypotheses→Plan→Coder.
NO_MATCHING_TOOL_STATE_KEY = "experiment_no_matching_tool"
_NO_MATCHING_TOOL_TOKEN = "NO_MATCHING_TOOL"


def assess_experiment_inventory_feasibility(callback_context: Any) -> None:
    """after_agent(ToolPreparer): set/clear an early NO_MATCHING_TOOL verdict."""
    from CoScientist.experiments.capabilities.inventory import index_inventory_tools
    from CoScientist.config import get_settings

    state = callback_context.state
    getter = getattr(state, "get", None)
    gate_routed = bool(getter(GATE_ROUTED_STATE_KEY)) if callable(getter) else False
    if hasattr(state, "__setitem__"):
        state[GATE_ROUTED_STATE_KEY] = None
    by_tool = index_inventory_tools(session_inventory_rows(state))
    alembic_on = False
    try:
        alembic_on = bool(get_settings().experiments.route_alembic)
    except Exception:  # noqa: BLE001
        alembic_on = False

    if (not gate_routed) or by_tool or alembic_on:
        if hasattr(state, "__setitem__"):
            state[NO_MATCHING_TOOL_STATE_KEY] = None
        audit(
            logger,
            f"EXPERIMENT_FEASIBILITY_OK gate_routed={gate_routed} inventory={len(by_tool)}",
            stdout=f"EXPERIMENT_FEASIBILITY_OK gate_routed={gate_routed} inventory={len(by_tool)}",
        )
        return

    message = (
        f"{_NO_MATCHING_TOOL_TOKEN}: inventory has no tool relevant to this request "
        f"(retrieved={len(by_tool)} tool(s)). Recommend ResearchAgent with the original ask."
    )
    state[NO_MATCHING_TOOL_STATE_KEY] = message
    audit(
        logger,
        f"EXPERIMENT_NO_MATCHING_TOOL early inventory={len(by_tool)}",
        stdout=f"EXPERIMENT_NO_MATCHING_TOOL early inventory={len(by_tool)}",
    )


def skip_when_experiment_not_feasible(callback_context: Any) -> Optional[types.Content]:
    """before_agent: short-circuit EM children after an early NO_MATCHING_TOOL."""
    state = callback_context.state
    message = state.get(NO_MATCHING_TOOL_STATE_KEY) if hasattr(state, "get") else None
    if not isinstance(message, str) or not message.strip():
        return None
    # Surface on common EM output keys so the sequential module's last skip
    # still leaves a readable summary for the orchestrator / aggregator.
    state["experiment_execution_summary"] = message
    state["experiment_summary"] = message
    state["hypotheses"] = message
    audit(logger, "EXPERIMENT_SKIP_NOT_FEASIBLE")
    return types.Content(role="model", parts=[types.Part(text=message)])


def skip_when_experiment_stage_complete(callback_context: Any) -> Optional[types.Content]:
    """before_agent: do not start a second ExperimentPlan after result HITL accepted."""
    state = callback_context.state
    getter = getattr(state, "get", None)
    if not callable(getter):
        return None
    runtime = getter("experiment_runtime")
    if not isinstance(runtime, dict) or runtime.get("phase") != "completed":
        return None
    summary = getter("experiment_summary")
    if not isinstance(summary, str) or not summary.strip():
        summary = getter("experiment_execution_summary")
    if not isinstance(summary, str) or not summary.strip():
        summary = (
            "Experiment stage already completed for this session; "
            "not starting a second plan on the same ask."
        )
    audit(logger, "EXPERIMENT_SKIP_STAGE_COMPLETE")
    return types.Content(role="model", parts=[types.Part(text=summary)])


def pin_fedot_alembic_task(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext,
) -> dict[str, Any] | None:
    """before_tool on FedotAgent: replace scripty fedot_tool briefs after Alembic."""
    if getattr(tool, "name", "") != "fedot_tool":
        return None
    from CoScientist.experiments.runtime.alembic_bridge import (
        alembic_post_build_context,
        compose_alembic_fedot_task,
    )

    ctx = alembic_post_build_context(tool_context.state)
    if not ctx:
        return None
    original = str(args.get("task_description") or "")
    args["task_description"] = compose_alembic_fedot_task(ctx, original)
    audit(logger, f"EXPERIMENT_ALEMBIC_FEDOT_PIN mcp_url={ctx.get('mcp_url')}")
    return None


__all__ = [
    "ROUTE_ALREADY_RETURNED_MESSAGE",
    "RECORD_REQUIRED_MESSAGE",
    "NO_MATCHING_TOOL_STATE_KEY",
    "assess_experiment_inventory_feasibility",
    "await_alembic_job_if_experiment",
    "force_molecule_generator_s3_upload",
    "guard_route_agent_tool",
    "on_route_agent_returned",
    "pin_alembic_build_args",
    "pin_fedot_alembic_task",
    "enforce_pending_record_result",
    "enforce_continue_until_reporting",
    "rewrite_mismatched_control_action",
    "skip_when_experiment_not_feasible",
    "skip_when_experiment_stage_complete",
]
