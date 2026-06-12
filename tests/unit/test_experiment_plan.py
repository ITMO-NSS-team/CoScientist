"""Unit tests for the experiment-plan schema (F015a / R05). No network."""
import pytest
from pydantic import ValidationError

from CoScientist.experiments.plan import (
    Artifact,
    ExperimentPlan,
    ExperimentStep,
    ServerTools,
)


def _step(sid, deps=None, arts=None, tool_servers=None):
    # tool_servers: dict {server: [tool, ...]}
    ts = [ServerTools(server=srv, tools=tools) for srv, tools in (tool_servers or {}).items()]
    return ExperimentStep(
        id=sid,
        subtask=f"do {sid}",
        deps=deps or [],
        tool_servers=ts,
        expected_artifacts=[Artifact(id=a, description="x", kind="data") for a in (arts or [])],
    )


def test_valid_dag_topological_order():
    plan = ExperimentPlan(goal="g", steps=[
        _step("s3", deps=["s1", "s2"]),
        _step("s1"),
        _step("s2", deps=["s1"]),
    ])
    order = [s.id for s in plan.topological_order()]
    assert order.index("s1") < order.index("s2") < order.index("s3")


def test_duplicate_ids_rejected():
    with pytest.raises(ValidationError):
        ExperimentPlan(goal="g", steps=[_step("s1"), _step("s1")])


def test_dangling_dependency_rejected():
    with pytest.raises(ValidationError):
        ExperimentPlan(goal="g", steps=[_step("s1", deps=["nope"])])


def test_self_dependency_rejected():
    with pytest.raises(ValidationError):
        ExperimentPlan(goal="g", steps=[_step("s1", deps=["s1"])])


def test_cycle_rejected():
    with pytest.raises(ValidationError) as ei:
        ExperimentPlan(goal="g", steps=[_step("a", deps=["b"]), _step("b", deps=["a"])])
    assert "cycle" in str(ei.value).lower()


def test_empty_plan_rejected():
    with pytest.raises(ValidationError):
        ExperimentPlan(goal="g", steps=[])


def test_artifacts_servers_and_tools():
    plan = ExperimentPlan(goal="g", steps=[
        _step("s1", arts=["m1"], tool_servers={"gen-mcp": ["generate_mols"]}),
        _step("s2", deps=["s1"], arts=["m2"],
              tool_servers={"chem-mcp": ["calculate_docking"], "gen-mcp": ["generate_mols"]}),
    ])
    assert plan.artifact_ids() == {"m1", "m2"}
    # de-duplicated, order-preserving
    assert plan.required_servers() == ["gen-mcp", "chem-mcp"]
    assert plan.required_tool_names() == ["generate_mols", "calculate_docking"]


def test_json_roundtrip():
    plan = ExperimentPlan(goal="g", steps=[_step("s1", arts=["m1"])])
    again = ExperimentPlan.model_validate_json(plan.model_dump_json())
    assert again.steps[0].expected_artifacts[0].id == "m1"
