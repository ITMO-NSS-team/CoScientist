"""Bridge B — deterministic DAG-executor (roadmap R10, the F015a/ReWOO vision).

After submit_plan validates a plan, the orchestrator STOPS driving step-by-step and this
executor takes over: it walks plan.topological_order(), substitutes {artifact_id}
placeholders in each step's run_params with the real outputs of upstream steps, dispatches
the step to the right sub-agent, and records the produced artifacts. The plan literally IS
the execution — no per-step LLM improvisation, so no runaways and explicit data flow
(ReWOO: ~5x fewer tokens than step-by-step ReAct).

`dispatch(step, resolved_params) -> result` is supplied by the caller (it calls the real
sub-agents in prod, or stubs in the experiment). This module is model-free and unit-testable.
"""
from __future__ import annotations

import re
from typing import Any, Awaitable, Callable

from CoScientist.experiments.plan import ExperimentPlan, ExperimentStep
from CoScientist.experiments.bridge import STEP_KIND_TO_AGENT

_PLACEHOLDER = re.compile(r"\{([A-Za-z0-9_]+)\}")


def _resolve(value: Any, artifacts: dict[str, Any]) -> Any:
    """Recursively substitute {artifact_id} in strings with produced artifact values."""
    if isinstance(value, str):
        return _PLACEHOLDER.sub(lambda m: str(artifacts.get(m.group(1), m.group(0))), value)
    if isinstance(value, dict):
        return {k: _resolve(v, artifacts) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v, artifacts) for v in value]
    return value


def agent_for(step: ExperimentStep) -> str:
    return STEP_KIND_TO_AGENT.get(step.kind, "TaskExecutorAgent")


async def execute_plan(
    plan: ExperimentPlan,
    dispatch: Callable[[ExperimentStep, str, dict], Awaitable[Any]],
    *,
    artifacts: dict[str, Any] | None = None,
) -> dict:
    """Execute the plan deterministically in topological order.

    dispatch(step, agent_name, resolved_params) -> result. The result is stored under each
    of the step's expected_artifact ids so downstream {artifact_id} placeholders resolve.

    Returns {"trace": [...per step...], "artifacts": {...}, "completed": bool}.
    """
    arts: dict[str, Any] = dict(artifacts or {})
    trace: list[dict] = []
    completed = True
    for step in plan.topological_order():
        agent = agent_for(step)
        params = _resolve(step.run_params, arts)
        try:
            result = await dispatch(step, agent, params)
            ok = True
        except Exception as e:  # a failed step aborts the DAG (downstream deps unmet)
            result = f"ERROR: {type(e).__name__}: {str(e)[:160]}"
            ok = False
        for a in step.expected_artifacts:
            arts[a.id] = result
        trace.append({"step": step.id, "kind": step.kind, "agent": agent,
                      "params": params, "result": result, "ok": ok})
        if not ok:
            completed = False
            break
    return {"trace": trace, "artifacts": arts, "completed": completed}
