"""Unit tests for the two plan->delegation bridges + submit_plan (deterministic, no LLM).

Bridge A (bridge.py)  — conformance check: on-plan vs off-plan delegations.
Bridge B (executor.py) — deterministic topological execution + {artifact_id} substitution.
submit_plan (submit_plan_tool.py) — validate + gate + HITL-EDIT re-validation.
"""
import asyncio

import pytest

from CoScientist.experiments.plan import ExperimentPlan
from CoScientist.experiments.bridge import check_conformance, expected_agents
from CoScientist.experiments.executor import execute_plan
from CoScientist.experiments.submit_plan_tool import run_submit_plan
from CoScientist.hitl.models import HITLResponse, HITLAction

INV = {"generative-models-mcp": ["generate_case_mols"], "chemical-mcp-server": ["calculate_docking"]}

# s1 generate -> s2 dock(deps s1) -> s3 filter(deps s2)
_STEPS = [
    {"id": "s1", "subtask": "generate", "kind": "compute",
     "tool_servers": [{"server": "generative-models-mcp", "tools": ["generate_case_mols"]}],
     "expected_artifacts": [{"id": "mols", "description": "generated molecules"}]},
    {"id": "s2", "subtask": "dock", "kind": "compute", "deps": ["s1"],
     "tool_servers": [{"server": "chemical-mcp-server", "tools": ["calculate_docking"]}],
     "run_params": {"input": "{mols}"},
     "expected_artifacts": [{"id": "scored", "description": "docked"}]},
    {"id": "s3", "subtask": "filter selective", "kind": "code_exec", "deps": ["s2"],
     "run_params": {"data": "{scored}"},
     "expected_artifacts": [{"id": "final", "description": "filtered"}]},
]


def _plan():
    return ExperimentPlan(goal="selective inhibitors", steps=_STEPS)


# ── Bridge B: deterministic executor ─────────────────────────────────────────
def test_executor_topological_order_and_substitution():
    seen = []

    async def dispatch(step, agent, params):
        seen.append((step.id, agent, params))
        return f"R_{step.id}"

    out = asyncio.run(execute_plan(_plan(), dispatch))
    assert out["completed"] is True
    assert [t["step"] for t in out["trace"]] == ["s1", "s2", "s3"]          # topological
    assert seen[1][2] == {"input": "R_s1"}                                   # {mols} -> s1's result
    assert seen[2][2] == {"data": "R_s2"}                                    # {scored} -> s2's result
    assert seen[0][1] == "TaskExecutorAgent" and seen[2][1] == "CoderAgent"  # kind->agent


def test_executor_aborts_on_failed_step():
    async def dispatch(step, agent, params):
        if step.id == "s2":
            raise RuntimeError("docking unavailable")
        return f"R_{step.id}"

    out = asyncio.run(execute_plan(_plan(), dispatch))
    assert out["completed"] is False
    assert [t["step"] for t in out["trace"]] == ["s1", "s2"]   # s3 never runs (dep unmet)


# ── Bridge A: conformance ────────────────────────────────────────────────────
def test_conformance_on_and_off_plan():
    p = _plan()
    assert expected_agents(p, []) == {"TaskExecutorAgent"}
    assert check_conformance("TaskExecutorAgent", p, [])[0] is True
    assert check_conformance("CoderAgent", p, [])[0] is False          # too early for s3
    assert check_conformance("ResearchAgent", p, [])[0] is False       # not in the plan at all
    # after s1+s2 done, s3 (code_exec) is ready -> CoderAgent is on-plan
    assert expected_agents(p, ["s1", "s2"]) == {"CoderAgent"}
    assert check_conformance("CoderAgent", p, ["s1", "s2"])[0] is True


# ── submit_plan tool core ────────────────────────────────────────────────────
def test_submit_plan_accepts_valid_and_stores_state():
    state = {}
    out = asyncio.run(run_submit_plan({"goal": "g", "steps": _STEPS}, INV, state=state))
    assert out["accepted"] is True and out["order"] == ["s1", "s2", "s3"]
    assert state["experiment_plan"]["steps"][0]["id"] == "s1"


def test_submit_plan_rejects_empty_compute_step():
    bad = {"goal": "g", "steps": [{"id": "s1", "subtask": "clinical trial sim", "kind": "compute"}]}
    out = asyncio.run(run_submit_plan(bad, INV))
    assert out["accepted"] is False and out["gate_code"] == "reject:empty_compute_step"


class _ScriptedHITL:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    async def handle_request(self, request):
        r = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return r


def test_submit_plan_hitl_edit_revalidation_catches_bad_edit():
    # Human EDIT returns malformed JSON -> must be re-validated and rejected, then APPROVE.
    handler = _ScriptedHITL([
        HITLResponse(action=HITLAction.EDIT, approved=False, instructions="{not valid json"),
        HITLResponse(action=HITLAction.APPROVE, approved=True),
    ])
    state = {}
    out = asyncio.run(run_submit_plan({"goal": "g", "steps": _STEPS}, INV,
                                      hitl_handler=handler, state=state))
    assert out["accepted"] is True
    assert out["edit_revalidation_caught"] is True   # the bad edit was caught, not blindly accepted
    assert handler.calls >= 2
