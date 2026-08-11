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
from .shared import GATE_ROUTED_STATE_KEY, MOLECULE_GENERATOR_TOOLS, audit

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
        _stringify_agent_tool_request(args)
        try:
            _, _, attempt = active_attempt(state)
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

def force_molecule_generator_s3_upload(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext,
) -> dict[str, Any] | None:
    """Force generate_* MCP calls to upload managed S3 artifacts in experiment runs."""
    if tool_context.state.get("experiment_runtime") and getattr(tool, "name", "") in MOLECULE_GENERATOR_TOOLS:
        args["upload_results_to_s3"] = True
        args.setdefault("output_s3_prefix", "generated")
    return None

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
        mark_route_returned(tool_context.state, tool_name)
        tool_context.state["experiment_last_route_response"] = copy.deepcopy(tool_response)
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
    """Best-effort TaskResult so the control loop cannot skip record_result.
    Captured artifacts → optimistic success (record_result may still downgrade).
    No captures → failure + retryable so retry/fallback pending is set.
    """
    from .state_machine import _captured_delta
    criteria = (task_runtime.get("task") or {}).get("success_criteria") or []
    summary = _summary_from_last_route(state)
    has_artifacts = bool(_captured_delta(state, attempt))
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

def enforce_pending_record_result(
    callback_context: Any, llm_response: LlmResponse,
) -> LlmResponse | None:
    """after_model: force record_result when route returned but model skips close."""
    state = callback_context.state
    pending = _pending_record_attempt(state)
    if pending is None or _llm_has_pending_close_call(llm_response):
        return None
    _, task_runtime, attempt = pending
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
    content = getattr(llm_response, "content", None)
    parts = getattr(content, "parts", None) if content is not None else None
    return any(getattr(p, "function_call", None) is not None for p in (parts or []))


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
    """after_model: rewrite wrong retry/fallback/start to the deterministic next action.

    Prevents spurious ``status=error`` (e.g. retry_not_allowed while
    fallback_pending, or task_already_running from parallel start_task)
    that otherwise trip strict smoke markers even when the executor later recovers.
    """
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
        if any(n in {"start_task", "retry_task", "fallback_task", "skip_task", "amend_task"}
               for _, n, _ in control_fcs):
            return _suppress(f"phase_{phase or 'none'}")
        return None

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
        if any(n in {"start_task", "retry_task", "fallback_task", "skip_task", "amend_task"}
               for _, n, _ in control_fcs):
            return _suppress("no_pending_transition")
        return None

    exp_name, exp_args = expected
    # If any control call already matches the expected transition, leave alone.
    for _, name, args in control_fcs:
        if name != exp_name:
            continue
        if exp_name == "fallback_task":
            if str(args.get("task_id") or "") == str(exp_args.get("task_id") or ""):
                return None
        elif str(args.get("task_id") or "") == str(exp_args.get("task_id") or ""):
            return None
        elif not exp_args:
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


def _inventory_rows(state: Any) -> list[dict[str, Any]]:
    from CoScientist.experiments.context.builder import (
        DISCOVERED_CAPABILITIES_KEY,
        RETRIEVED_CAPABILITIES_KEY,
    )

    rows: list[dict[str, Any]] = []
    getter = getattr(state, "get", None)
    if not callable(getter):
        return rows
    for key in (RETRIEVED_CAPABILITIES_KEY, DISCOVERED_CAPABILITIES_KEY, "filtered_tools"):
        blob = getter(key)
        if isinstance(blob, list):
            rows.extend(item for item in blob if isinstance(item, dict))
    return rows


def assess_experiment_inventory_feasibility(callback_context: Any) -> None:
    """after_agent(ToolPreparer): set/clear an early NO_MATCHING_TOOL verdict.

    Only applies when this EM entry was a structural reroute (gate-routed);
    an explicit orchestrator call to the module is never second-guessed.
    """
    from CoScientist.experiments.capabilities.inventory import (
        ask_has_computational_signal,
        index_inventory_tools,
    )

    state = callback_context.state
    getter = getattr(state, "get", None)
    request = str(getter("experiment_source_request") or "").strip() if callable(getter) else ""
    gate_routed = bool(getter(GATE_ROUTED_STATE_KEY)) if callable(getter) else False
    if hasattr(state, "__setitem__"):
        # Consume: relevant only for the one ask that triggered the reroute.
        state[GATE_ROUTED_STATE_KEY] = None
    by_tool = index_inventory_tools(_inventory_rows(state))

    if not gate_routed or ask_has_computational_signal(request):
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
        f"(no computational capability signal in the ask; retrieved={len(by_tool)} "
        "tool(s) were unrelated). Recommend ResearchAgent with the original ask."
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


__all__ = [
    "ROUTE_AGENT_NAMES",
    "ROUTE_ALREADY_RETURNED_MESSAGE",
    "RECORD_REQUIRED_MESSAGE",
    "NO_MATCHING_TOOL_STATE_KEY",
    "assess_experiment_inventory_feasibility",
    "force_molecule_generator_s3_upload",
    "guard_route_agent_tool",
    "on_route_agent_returned",
    "enforce_pending_record_result",
    "enforce_continue_until_reporting",
    "rewrite_mismatched_control_action",
    "skip_when_experiment_not_feasible",
]
