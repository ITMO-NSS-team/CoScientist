"""Bridge A — "plan-as-contract" (the lightweight plan->delegation bridge).

PlanReAct keeps driving (the orchestrator decides each delegation itself), but the
delegation-gate consults the submitted plan (state['experiment_plan']) and CHECKS each
proposed delegation against it: is the target agent one the plan expects NEXT (a step
whose deps are satisfied and that isn't done yet)? An off-plan delegation is flagged so
the critic can nudge the orchestrator back onto the plan.

This is a SOFT contract: it constrains/nudges, it does not seize control (cf. the hard
DAG-executor in executor.py = Bridge B). It is enough to MEASURE whether an (edited) plan
actually governs the delegations.

Step->agent mapping uses ExperimentStep.kind (the executor uses the same map).
"""
from __future__ import annotations

from typing import Iterable

from CoScientist.experiments.plan import ExperimentPlan

# Which sub-agent runs a step of each kind (the orchestrator's roster).
STEP_KIND_TO_AGENT = {
    "compute": "TaskExecutorAgent",
    "research": "ResearchAgent",
    "hypothesize": "HypothesesAgent",
    "code_exec": "CoderAgent",
}


def ready_step_ids(plan: ExperimentPlan, done_ids: Iterable[str]) -> list[str]:
    """Steps not yet done whose deps are ALL done — the ones runnable next."""
    done = set(done_ids)
    return [s.id for s in plan.steps if s.id not in done and all(d in done for d in s.deps)]


def expected_agents(plan: ExperimentPlan, done_ids: Iterable[str]) -> set[str]:
    """Agents the plan expects for the steps runnable next."""
    by_id = {s.id: s for s in plan.steps}
    return {STEP_KIND_TO_AGENT.get(by_id[i].kind, "TaskExecutorAgent")
            for i in ready_step_ids(plan, done_ids)}


def check_conformance(agent_name: str, plan: ExperimentPlan, done_ids: Iterable[str]) -> tuple[bool, str]:
    """Is delegating to `agent_name` consistent with the plan's next ready step(s)?

    Returns (ok, reason). ok=True when the agent matches a ready step, or when the plan
    is already complete / has no ready step (don't block — let PlanReAct finalize).
    """
    exp = expected_agents(plan, done_ids)
    if not exp:
        return True, "plan complete or no ready step — allow"
    if agent_name in exp:
        return True, f"on-plan: {agent_name} matches a ready step"
    return False, f"off-plan: proposed {agent_name}, plan expects one of {sorted(exp)} next"
