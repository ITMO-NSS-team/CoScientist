"""Unit tests for the deterministic plan gate (R12 / F015b).

Proves the gate closes the wave-1 'gap' hole: an out-of-inventory tool and a
tool-less compute step both fail deterministically (no LLM), while valid plans
and legitimately tool-free research steps pass.
"""
from CoScientist.experiments.plan import ExperimentPlan
from CoScientist.experiments.gate import deterministic_gate

INV = {
    "generative-models-mcp": ["generate_case_mols", "generate_mols"],
    "chemical-mcp-server": [("calculate_docking", "dock"), ("get_rdkit_properties", "props")],
}


def _plan(steps):
    return ExperimentPlan(goal="g", steps=steps)


def test_pass_valid_plan():
    p = _plan([{"id": "s1", "subtask": "generate",
                "tool_servers": [{"server": "generative-models-mcp", "tools": ["generate_case_mols"]}]}])
    r = deterministic_gate(p, INV)
    assert r.ok and r.code == "pass"


def test_reject_empty_compute_step():
    # The wave-1 'gap' hazard: a compute step for a capability with no tool.
    p = _plan([{"id": "s1", "subtask": "run a full clinical-trial simulation", "tool_servers": []}])
    r = deterministic_gate(p, INV)
    assert not r.ok and r.code == "reject:empty_compute_step" and "s1" in r.offending


def test_reject_unresolvable_tool():
    p = _plan([{"id": "s1", "subtask": "x",
                "tool_servers": [{"server": "chemical-mcp-server", "tools": ["fetch_protein_activities"]}]}])
    r = deterministic_gate(p, INV)
    assert not r.ok and r.code == "reject:unresolvable_tool"


def test_reject_unknown_server():
    p = _plan([{"id": "s1", "subtask": "x",
                "tool_servers": [{"server": "chemical-mcp", "tools": ["calculate_docking"]}]}])
    r = deterministic_gate(p, INV)
    assert not r.ok and r.code == "reject:unknown_server"


def test_research_step_allowed_toolless():
    # A non-compute step legitimately carries no tool_server.
    p = _plan([{"id": "s1", "subtask": "literature review", "kind": "research", "tool_servers": []}])
    r = deterministic_gate(p, INV)
    assert r.ok
