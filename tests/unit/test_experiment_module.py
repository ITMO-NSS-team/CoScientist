"""Experiment Module v0 contracts, state machine and normative acceptance."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from CoScientist.assembly.schema import load_config, resolve_config_path
from CoScientist.config.settings import ExperimentsSettings, Settings
from CoScientist.experiments.critique import critique_plan
from CoScientist.experiments.runtime import (
    ExperimentRuntimeError,
    RECORD_REQUIRED_MESSAGE,
    amend_task,
    approve_plan,
    enforce_continue_until_reporting,
    enforce_pending_record_result,
    fallback_task,
    guard_route_agent_tool,
    initialize_runtime,
    mark_route_returned,
    on_route_agent_returned,
    record_result,
    retry_task,
    skip_task,
    start_task,
)
from CoScientist.experiments.schemas import ExperimentPlan, ExperimentTask
from CoScientist.hitl.models import HITLAction, HITLRequest
from CoScientist.web.handler import WebHITLHandler


NOW = datetime(2026, 7, 31, 18, 0, tzinfo=timezone.utc).isoformat()


@pytest.fixture(autouse=True)
def _experiment_module_defaults(monkeypatch):
    """Keep unit tests independent of ambient EXPERIMENTS__* kill-switches."""
    monkeypatch.setenv("EXPERIMENTS__ROUTE_FEDOT", "true")
    monkeypatch.delenv("EXPERIMENTS__ROUTE_CODER_MCP", raising=False)
    monkeypatch.delenv("EXPERIMENTS__ROUTE_ALEMBIC", raising=False)
    from CoScientist.config import get_settings

    # get_settings() returns an import-time singleton that may already have
    # absorbed EXPERIMENTS__ROUTE_FEDOT=false from the ambient shell.
    experiments = ExperimentsSettings(
        route_fedot=True,
        route_coder_mcp=False,
        route_alembic=False,
    )
    monkeypatch.setattr(get_settings(), "experiments", experiments, raising=False)
    yield


def _design(hypothesis_ref: str = "H1", *, question: str | None = None) -> dict:
    return {
        "hypothesis_ref": hypothesis_ref,
        "experiment_question": question
        or f"Does the ready MCP produce evidence for {hypothesis_ref}?",
        "dataset": {
            "name": "ready_mcp_inputs",
            "ref": None,
            "notes": "Synthetic fixture inputs for unit tests.",
        },
        "baselines": [
            {
                "name": "no-tool control",
                "kind": "method",
                "ref": None,
            }
        ],
        "metrics": [
            {
                "name": "artifact_present",
                "direction": "maximize",
                "threshold": 1,
                "test": "descriptive",
            }
        ],
        "analysis_artifacts": [
            {
                "name": "metrics_table.json",
                "role": "metrics_table",
                "prepare_via": "mcp",
                "path_or_tool": "metrics_table.json",
            }
        ],
    }


def _server(tool: str = "estimate_property") -> dict:
    return {
        "name": "chem-ready",
        "server_id": "srv-chem",
        "tools": [
            {
                "name": tool,
                "description": "Compute the requested chemical property.",
                "input_schema": {
                    "type": "object",
                    "properties": {"smiles": {"type": "string"}},
                    "required": ["smiles"],
                },
            }
        ],
        "source": "registry",
    }


def _task(
    task_id: str,
    *,
    route: str = "fedot_mas",
    tool: str = "estimate_property",
    depends_on: list[str] | None = None,
    optional: bool = False,
    artifact_name: str | None = None,
    hypothesis_ref: str = "H1",
    design: dict | None = None,
) -> dict:
    artifact_name = artifact_name or f"{task_id.lower()}-result.csv"
    return {
        "id": task_id,
        "name": f"Chemical computation {task_id}",
        "description": "Compute a bounded chemical property using the ready MCP.",
        "rationale": "Produces direct computational evidence for the hypothesis.",
        "route": route,
        "design": design or _design(hypothesis_ref),
        "mcp_servers": [_server(tool)] if route in {"fedot_mas", "react_tools"} else [],
        "input_data": [],
        "launch_params": {"smiles": "CCO"},
        "success_criteria": [
            {
                "criterion_id": f"{task_id}-C1",
                "description": "The MCP execution completes.",
                "kind": "execution",
                "verification": "Check the structured route result status.",
            }
        ],
        "expected_artifacts": [
            {
                "name": artifact_name,
                "role": "data",
                "media_type": "text/csv",
                "description": "Managed result table.",
            }
        ],
        "est_duration_min": 1,
        "depends_on": depends_on or [],
        "optional": optional,
    }


def _plan(*tasks: dict, hypotheses: list[dict] | None = None) -> ExperimentPlan:
    refs = {
        str(task.get("design", {}).get("hypothesis_ref") or "H1").upper()
        for task in tasks
    }
    default_hypotheses = [
        {
            "hypothesis_id": hid,
            "statement": f"Fixture statement for {hid}.",
        }
        for hid in sorted(refs)
    ]
    return ExperimentPlan.model_validate(
        {
            "schema_version": "experiment-plan/1.0",
            "plan_id": "PLAN-acceptance",
            "experiment_run_id": "EXRUN-acceptance",
            "revision": 1,
            "source_request": "Run a bounded chemistry MCP experiment.",
            "goal": "Produce managed computational evidence.",
            "hypothesis": "The ready chemistry MCP can compute the requested property.",
            "hypotheses": hypotheses if hypotheses is not None else default_hypotheses,
            "methods": ["Exact ready-MCP execution"],
            "context_digest": "Ready chemistry MCP is registered.",
            "context_refs": ["TOOL-srv-chem"],
            "tasks": list(tasks),
            "risks": [],
            "assumptions": [],
            "total_est_duration_min": sum(task["est_duration_min"] for task in tasks),
            "created_at": NOW,
        }
    )


def _inventory() -> list[dict]:
    return [
        {
            "tool": "estimate_property",
            "server_id": "srv-chem",
            "description": "Compute the requested chemical property.",
            "input_schema": {},
        }
    ]


def _approved_state(plan: ExperimentPlan) -> dict:
    state: dict = {}
    critique = critique_plan(
        plan,
        settings=ExperimentsSettings(),
        available_tools=_inventory(),
    )
    assert critique.verdict == "approve"
    initialize_runtime(state, plan, critique=critique.model_dump(mode="json"))
    approve_plan(state)
    return state


def _tool_context(state: dict) -> SimpleNamespace:
    return SimpleNamespace(state=state)


def _route_return(state: dict, agent: str) -> None:
    tool = SimpleNamespace(name=agent)
    assert guard_route_agent_tool(tool, {}, _tool_context(state)) is None
    on_route_agent_returned(tool, {}, _tool_context(state), {"status": "success"})


def _success_result(task_id: str) -> dict:
    return {
        "status": "success",
        "summary": f"{task_id} completed with a managed result.",
        "criteria_checks": [
            {
                "criterion_id": f"{task_id}-C1",
                "passed": True,
                "details": "Structured route result reports success.",
            }
        ],
    }


def test_experiments_settings_defaults_and_nested_env(monkeypatch):
    defaults = ExperimentsSettings()
    assert defaults.route_fedot is True
    assert defaults.route_coder_mcp is False
    assert defaults.route_alembic is False
    assert defaults.require_task_design is True
    assert defaults.task_max_attempts == 2
    assert defaults.max_plan_tasks == 8

    monkeypatch.setenv("EXPERIMENTS__ROUTE_FEDOT", "false")
    monkeypatch.setenv("EXPERIMENTS__MAX_PLAN_TASKS", "6")
    configured = Settings(_env_file=None)
    assert configured.experiments.route_fedot is False
    assert configured.experiments.max_plan_tasks == 6


def test_experiment_profile_is_isolated_and_preserves_a2a_contract():
    from CoScientist.assembly import build_system
    from CoScientist.experiments.review import ExperimentReviewSessionAgent

    config = load_config(resolve_config_path("experiments"))
    assert config.root.name == "OrchestratorAgent"
    orch = config.agent("OrchestratorAgent")
    assert orch.cls == "llm"
    assert orch.children == []
    assert "ExperimentModuleAgent" in orch.subordinates
    assert "TaskExecutorAgent" not in orch.subordinates
    assert "redirect_research_to_experiment_module" in orch.callbacks.after_model
    assert "coalesce_experiment_module_calls" in orch.callbacks.after_model
    assert "inject_upstream_artifacts" not in orch.callbacks.before_agent

    module = config.agent("ExperimentModuleAgent")
    assert module.a2a.key == "task_execution"
    assert module.a2a.port == 8004
    assert module.children == [
        "ToolPreparerAgent",
        "HypothesesAgent",
        "ExperimentPlannerAgent",
        "ExperimentExecutorAgent",
        "ExperimentResultReviewAgent",
    ]
    hyp = config.agent("HypothesesAgent")
    assert hyp.prompt == "hypotheses"
    assert hyp.model == "openai/glm-4.7"
    assert hyp.tools == ["graph", "research_graph"]
    assert "commit_experiment_hypotheses" in hyp.callbacks.after_agent
    assert "seed_hypotheses_from_em_request" in hyp.callbacks.before_model
    assert "enforce_hypothesis_research_commit" in hyp.callbacks.after_model
    assert "normalize_em_hypothesis_commit" in hyp.callbacks.after_model
    assert hyp.callbacks.after_model.index("enforce_hypothesis_research_commit") < (
        hyp.callbacks.after_model.index("normalize_em_hypothesis_commit")
    )
    assert "capture_hypotheses_after_research_commit" in hyp.callbacks.after_tool
    # Root must be bootstrapped before inject_research_context renders the
    # {research_context?} placeholder, so a fresh graph never reads EMPTY.
    assert hyp.callbacks.before_agent.index("bootstrap_research_question_if_empty") < (
        hyp.callbacks.before_agent.index("inject_research_context")
    )
    assert "CoderAgent" in orch.subordinates
    assert "McpBuilderAgent" in config.agents
    assert "McpBuilderAgent" in config.agent("ExperimentExecutorAgent").subordinates
    fedot_before = config.agent("FedotAgent").callbacks.before_agent
    assert "refuse_when_fedot_deliverable" in fedot_before
    assert "inject_upstream_artifacts" not in fedot_before
    assert "refuse_when_fedot_deliverable" in config.agent("CoderAgent").callbacks.before_agent
    assert config.agent("ExperimentExecutorAgent").callbacks.before_tool == [
        "guard_experiment_route"
    ]
    assert config.agent("ExperimentExecutorAgent").callbacks.after_tool == [
        "mark_experiment_route_returned"
    ]
    retriever = config.agent("ToolRetrieverAgent")
    assert "persist_experiment_em_request" in retriever.callbacks.before_agent
    assert "persist_experiment_retrieved_capabilities" in retriever.callbacks.after_agent

    reranker = config.agent("ToolReranker")
    assert reranker.callbacks.after_model == ["sanitize_json_output"]
    assert "collect_reranked_tools" in reranker.callbacks.after_agent
    assert "collect_reranked_tools_from_model" not in reranker.callbacks.after_model

    planner = config.agent("ExperimentPlannerAgent")
    assert planner.model == "openai/glm-4.7"
    assert planner.include_contents == "none"
    assert "skip_retriever_context" in planner.callbacks.before_model
    # ExperimentPlan is enforced by sanitize_json_output + deterministic critique.
    assert planner.output_schema is None

    system = build_system(config)
    for name in ("ExperimentPlannerAgent", "ExperimentResultReviewAgent"):
        review_agent = system.agent(name)
        assert isinstance(review_agent, ExperimentReviewSessionAgent)
        assert review_agent.hitl_handler is not None  # fail-closed even headless
    assert system.agent("ExperimentPlannerAgent").include_contents == "none"


def test_experiment_plan_json_schema_is_openai_structured_output_safe():
    schema = ExperimentPlan.model_json_schema()
    bad: list[str] = []

    def walk(obj: object, path: str = "") -> None:
        if isinstance(obj, dict):
            if obj.get("additionalProperties") is True:
                bad.append(path or "$")
            for key, value in obj.items():
                walk(value, f"{path}.{key}" if path else str(key))
        elif isinstance(obj, list):
            for index, value in enumerate(obj):
                walk(value, f"{path}[{index}]")

    walk(schema)
    assert bad == []

    task = ExperimentTask.model_validate(
        {
            **_task("EXP-1"),
            "launch_params": '{"case":"cancer","num":5}',
            "mcp_servers": [
                {
                    "name": "generative",
                    "server_id": "srv-gen",
                    "source": "registry",
                    "health": "unknown",
                    "tools": [
                        {
                            "name": "generate_case_mols",
                            "description": "Generate molecules",
                            "input_schema": '{"type":"object"}',
                            "required_for_task": True,
                        }
                    ],
                }
            ],
        }
    )
    assert task.launch_params == {"case": "cancer", "num": 5}
    assert task.mcp_servers[0].tools[0].input_schema == {"type": "object"}


def test_experiment_retrieval_budget_stops_repeated_llm_tool_calls():
    from google.adk.models import LlmResponse
    from google.genai import types

    from CoScientist.experiments.context import (
        enforce_experiment_retrieval_budget,
        reset_experiment_retrieval_budget,
    )

    context = SimpleNamespace(state={"retrieval_queries": ["prior request"]})
    reset_experiment_retrieval_budget(context)
    repeated_call = LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="retrieve_tools",
                        args={"query": "KRAS G12C candidates"},
                    )
                )
            ],
        )
    )

    assert enforce_experiment_retrieval_budget(context, repeated_call) is None
    # Budget is 4 request-local retrieve_tools calls counted after the baseline.
    context.state["retrieval_queries"].extend(
        ["query one", "query two", "query three", "query four"]
    )
    stopped = enforce_experiment_retrieval_budget(context, repeated_call)

    assert stopped is not None
    assert "EXPERIMENT_RETRIEVAL_BUDGET_EXHAUSTED" in stopped.content.parts[0].text
    assert context.state["experiment_retrieval_budget_exhausted"] is True


def test_four_experiment_contracts_are_registered():
    from CoScientist.assembly.registry import REGISTRY
    from CoScientist.experiments.schemas import (
        ExperimentTask,
        PlanCritique,
        TaskResult,
    )

    assert REGISTRY.output_schema("experiment_plan") is ExperimentPlan
    assert REGISTRY.output_schema("experiment_task") is ExperimentTask
    assert REGISTRY.output_schema("task_result") is TaskResult
    assert REGISTRY.output_schema("plan_critique") is PlanCritique


def test_contracts_reject_presigned_urls_and_non_dag_plans():
    task = _task("EXP-1")
    task["input_data"] = [
        {
            "data_id": "input",
            "kind": "url",
            "description": "bad canonical URL",
            "url": "https://s3.example/x.csv?X-Amz-Signature=secret",
        }
    ]
    with pytest.raises(ValidationError, match="signing"):
        _plan(task)

    first = _task("EXP-1", depends_on=["EXP-2"])
    second = _task("EXP-2", depends_on=["EXP-1"])
    with pytest.raises(ValidationError, match="DAG"):
        _plan(first, second)


def test_deterministic_critique_blocks_disabled_and_unknown_routes():
    plan = _plan(_task("EXP-1"))
    disabled = critique_plan(
        plan,
        settings=ExperimentsSettings(route_fedot=False),
        available_tools=_inventory(),
    )
    assert disabled.verdict == "revise"
    assert any(issue.category == "feasibility" for issue in disabled.issues)

    unknown = critique_plan(
        plan,
        settings=ExperimentsSettings(),
        available_tools=[],
    )
    assert unknown.verdict == "revise"
    assert any("absent from the capability inventory" in issue.message for issue in unknown.issues)


def test_mcp_health_rejects_non_enum_values():
    task = _task("EXP-1")
    task["mcp_servers"][0]["health"] = "active"
    with pytest.raises(ValidationError):
        _plan(task)

    task["mcp_servers"][0]["health"] = "healthy"
    assert _plan(task).tasks[0].mcp_servers[0].health == "healthy"


def test_dataref_requires_canonical_fields():
    from CoScientist.experiments.schemas import DataRef

    with pytest.raises(ValidationError):
        DataRef.model_validate({"description": "no kind or location"})

    # producer string + artifact alias → task_artifact (GLM/weak planner shape).
    ref_alias = DataRef.model_validate({"artifact": "activity_data", "producer": "EXP-2"})
    assert ref_alias.kind == "task_artifact"
    assert ref_alias.source_task_id == "EXP-2"
    assert ref_alias.source_artifact_id == "activity_data"

    ref = DataRef.model_validate(
        {
            "data_id": "activity_data",
            "kind": "task_artifact",
            "description": "Activity table from EXP-2",
            "source_task_id": "EXP-2",
            "source_artifact_id": "activity_data",
        }
    )
    assert ref.kind == "task_artifact"
    assert ref.source_task_id == "EXP-2"


def test_criterion_check_coerces_null_evidence_and_details():
    from CoScientist.experiments.schemas.models import CriterionCheck

    check = CriterionCheck.model_validate(
        {
            "criterion_id": "C1",
            "passed": True,
            "evidence_artifact_ids": None,
            "details": None,
        }
    )
    assert check.evidence_artifact_ids == []
    assert check.details == "n/a"


def test_glm_hypothesis_and_dataref_extras_are_coerced():
    """Weak GLM planners emit falsifiable/tests/summary and input_data.ref/name."""
    from CoScientist.experiments.schemas import DataRef, HypothesisSpec

    h = HypothesisSpec.model_validate(
        {
            "hypothesis_id": "H1",
            "summary": "Metabolites can be clustered by similarity.",
            "falsifiable": "Clusters do not separate toxicity.",
            "falsification_criteria": "No clear clusters.",
            "tests": "Run chemical_space_clustering.",
            "status": "pending",
            "type": "structural",
            "test_strategy": "Cluster chemical space.",
        }
    )
    assert h.statement == "Metabolites can be clustered by similarity."
    assert not hasattr(h, "type") or "type" not in h.model_dump()

    ref = DataRef.model_validate(
        {
            "ref": "T2/clustering_results.json",
            "name": "clustering_results.json",
        }
    )
    assert ref.kind == "task_artifact"
    assert ref.source_task_id == "EXP-2"
    assert ref.source_artifact_id == "clustering_results.json"
    assert ref.data_id == "clustering_results.json"

    # producer/ref as bare task ids (no /artifact) must still coerce.
    bare = DataRef.model_validate(
        {"kind": "task_artifact", "ref": "EXP-T1", "producer": "EXP-T1"}
    )
    assert bare.source_task_id == "EXP-1"
    assert bare.source_artifact_id == "artifact"

    # GLM also puts ExpectedArtifact/task fields on DataRef + EXP-Tn ids.
    ref2 = DataRef.model_validate(
        {
            "kind": "task_artifact",
            "source_task_id": "EXP-T1",
            "role": "data",
            "depends_on": ["T1"],
            "path_or_tool": "dataset_overview.json",
            "location": "ignored_when_path_set.json",
        }
    )
    assert ref2.source_task_id == "EXP-1"
    assert ref2.source_artifact_id == "dataset_overview.json"
    assert ref2.data_id == "dataset_overview.json"

    t2 = _task("EXP-T2", depends_on=[])
    t2["input_data"] = [
        {
            "kind": "workspace",
            "workspace_path": "dataset_overview.json",
            "data_id": "overview",
            "description": "Overview file",
            "depends_on": ["EXP-T1"],
            "uri": "dataset_overview.json",
        }
    ]
    plan = ExperimentPlan.model_validate(
        {
            **_plan(_task("EXP-1"), t2).model_dump(mode="json"),
            "tasks": [_task("EXP-1"), t2],
            "total_est_duration_min": 2,
        }
    )
    assert plan.tasks[1].id == "EXP-2"
    assert plan.tasks[1].depends_on == ["EXP-1"]
    assert plan.tasks[1].input_data[0].workspace_path == "dataset_overview.json"


def test_depends_on_normalizes_t_ids():
    task = _task("EXP-2", depends_on=["T1", "TASK-3"])
    plan = _plan(task, _task("EXP-1"), _task("EXP-3"))
    # rebuild with proper deps
    t2 = _task("EXP-2", depends_on=["T1"])
    t1 = _task("EXP-1")
    plan = ExperimentPlan.model_validate(
        {
            **_plan(t1, t2).model_dump(mode="json"),
            "tasks": [t1, t2],
            "total_est_duration_min": 2,
        }
    )
    assert plan.tasks[1].depends_on == ["EXP-1"]


def test_criterion_rejects_invented_kinds_without_alias_map():
    from CoScientist.experiments.schemas import SuccessCriterion

    with pytest.raises(ValidationError):
        SuccessCriterion.model_validate(
            {
                "criterion_id": "C2",
                "description": "Output mentions candidates",
                "kind": "output_contains",
                "verification": "Check textual output.",
            }
        )
    with pytest.raises(ValidationError):
        SuccessCriterion.model_validate(
            {
                "criterion_id": "C4",
                "description": "SMILES are valid",
                "kind": "output_valid_smiles",
                "verification": "Validate SMILES strings.",
            }
        )

    ok = SuccessCriterion.model_validate(
        {
            "criterion_id": "C1",
            "description": "Artifact produced",
            "kind": "artifact_exists",
            "verification": "Check artifact.",
        }
    )
    assert ok.kind == "artifact_exists"
    assert ok.operator is None

    # Non-threshold criteria silently drop stray metric/operator/target.
    cleaned = SuccessCriterion.model_validate(
        {
            "criterion_id": "C5",
            "description": "Artifact produced",
            "kind": "artifact_exists",
            "operator": "==",
            "verification": "Check artifact.",
        }
    )
    assert cleaned.operator is None
    assert cleaned.metric is None
    assert cleaned.target is None


def test_plan_rejects_non_string_hypothesis_and_coerces_risk_objects():
    task = _task("EXP-1")
    plan_data = _plan(task).model_dump(mode="json")
    plan_data["hypothesis"] = ["H1. Conflict criteria.", "H2. SA vs affinity."]
    with pytest.raises(ValidationError):
        ExperimentPlan.model_validate(plan_data)

    plan_data = _plan(task).model_dump(mode="json")
    plan_data["risks"] = [
        {
            "risk_id": "R1",
            "description": "Too few candidates pass filters.",
            "mitigation": "Relax thresholds.",
        }
    ]
    plan = ExperimentPlan.model_validate(plan_data)
    assert plan.risks == ["Too few candidates pass filters. Mitigation: Relax thresholds."]

    plan_data = _plan(task).model_dump(mode="json")
    plan_data["hypothesis"] = "H1 and H2 are in tension."
    plan_data["risks"] = ["Too few candidates pass filters."]
    plan = ExperimentPlan.model_validate(plan_data)
    assert plan.hypothesis == "H1 and H2 are in tension."
    assert plan.risks == ["Too few candidates pass filters."]


def test_task_level_risks_assumptions_popped_and_hoisted():
    """Plan-only fields on tasks must not burn extra_forbidden revise budget."""
    base = _plan(_task("EXP-1", route="fedot_mas")).model_dump(mode="json")
    task = _task("EXP-1", route="fedot_mas")
    task["risks"] = []
    task["assumptions"] = []
    plan = ExperimentPlan.model_validate({**base, "tasks": [task], "risks": [], "assumptions": []})
    assert plan.risks == []
    assert plan.assumptions == []

    task2 = _task("EXP-1", route="fedot_mas")
    task2["risks"] = ["MCP may time out"]
    task2["assumptions"] = ["Inventory covers generation"]
    plan2 = ExperimentPlan.model_validate(
        {**base, "tasks": [task2], "risks": [], "assumptions": []}
    )
    assert plan2.risks == ["MCP may time out"]
    assert plan2.assumptions == ["Inventory covers generation"]


def test_mcp_empty_expected_artifacts_not_required_report():
    """fedot/react with empty arts get a neutral data placeholder, not *_report."""
    for route in ("fedot_mas", "react_tools"):
        payload = _task("EXP-1", route=route)
        payload["expected_artifacts"] = []
        task = ExperimentTask.model_validate(payload)
        assert len(task.expected_artifacts) >= 1
        primary = task.expected_artifacts[0]
        assert primary.role == "data"
        assert primary.required is True
        assert not str(primary.name).endswith("_report")
        assert primary.media_type != "text/markdown"


def test_plan_duration_aligns_when_nested_under_design():
    """Planner often nests est_duration_min under design; do not burn revise budget."""
    base = _plan(_task("EXP-1")).model_dump(mode="json")
    t1 = _task("EXP-1", route="coder")
    t1.pop("est_duration_min", None)
    t1["design"]["est_duration_min"] = 30
    t2 = _task("EXP-2", route="fedot_mas")
    t2.pop("est_duration_min", None)
    t2["design"]["est_duration_min"] = 45
    plan = ExperimentPlan.model_validate(
        {
            **base,
            "tasks": [t1, t2],
            "total_est_duration_min": 450,
            "hypotheses": [
                {"hypothesis_id": "H1", "statement": "Fixture statement for H1."},
            ],
        }
    )
    assert plan.tasks[0].est_duration_min == 30
    assert plan.tasks[1].est_duration_min == 45
    assert plan.total_est_duration_min == 75


def test_bare_task_payload_fails_strict_plan_schema():
    from CoScientist.experiments.critique import (
        PlanValidationError,
        validate_and_critique_plan,
    )

    bare = _task("EXP-1")
    with pytest.raises(PlanValidationError):
        validate_and_critique_plan(
            bare,
            settings=ExperimentsSettings(),
            available_tools=_inventory(),
            experiment_run_id="EXRUN-acceptance",
            source_request="Run a bounded chemistry MCP experiment.",
        )


def test_completeness_critique_rejects_when_request_explicitly_requires_unused_tools():
    overview = _task("EXP-1", tool="dataset_overview")
    overview["mcp_servers"][0]["server_id"] = "srv-heracleum"
    overview["mcp_servers"][0]["tools"][0] = {
        "name": "dataset_overview",
        "description": "Overview of the reconstructed metabolite dataset.",
        "input_schema": {},
    }
    plan = ExperimentPlan.model_validate(
        {
            **_plan(overview).model_dump(mode="json"),
            "source_request": (
                "Use dataset_overview, then call chemical_space_cluster, "
                "run predict_ld50, and invoke predict_general_toxicity on the panel."
            ),
        }
    )
    inventory = [
        {
            "tool": "dataset_overview",
            "server_id": "srv-heracleum",
            "description": "Overview of the reconstructed metabolite dataset.",
        },
        {
            "tool": "chemical_space_cluster",
            "server_id": "srv-heracleum",
            "description": "Cluster metabolites by molecular similarity.",
        },
        {
            "tool": "predict_ld50",
            "server_id": "srv-heracleum",
            "description": "Impute mouse LD50 across administration routes.",
        },
        {
            "tool": "predict_general_toxicity",
            "server_id": "srv-heracleum",
            "description": "Hepatotoxicity, DILI, cardiotoxicity, carcinogenicity.",
        },
    ]
    critique = critique_plan(
        plan,
        settings=ExperimentsSettings(),
        available_tools=inventory,
    )
    assert critique.verdict == "revise"
    assert any(issue.category == "completeness" for issue in critique.issues)
    assert any("chemical_space_cluster" in issue.message for issue in critique.issues)


def test_completeness_critique_ignores_incidental_tool_name_mentions():
    """Same-domain tool names in prose must not force revise (Option A)."""
    plan = ExperimentPlan.model_validate(
        {
            **_plan(_task("EXP-1")).model_dump(mode="json"),
            "source_request": (
                "Run a statistical comparison of cleaned vs raw splits. "
                "The registry may contain chemical_space_cluster as a nearby "
                "chemistry tool, but do not stretch it onto this coder analysis."
            ),
        }
    )
    inventory = _inventory() + [
        {
            "tool": "chemical_space_cluster",
            "server_id": "srv-chem",
            "description": "Cluster metabolites by molecular similarity.",
        },
    ]
    critique = critique_plan(
        plan,
        settings=ExperimentsSettings(),
        available_tools=inventory,
        hypothesis_refs=[{"hypothesis_id": "H1", "statement": "Fixture"}],
    )
    assert critique.verdict == "approve"
    assert not any(
        issue.severity in {"blocker", "major"} and "chemical_space_cluster" in issue.message
        for issue in critique.issues
    )


def test_completeness_critique_keeps_single_capability_plans():
    plan = _plan(_task("EXP-1"))
    critique = critique_plan(
        plan,
        settings=ExperimentsSettings(),
        available_tools=_inventory(),
    )
    assert critique.verdict == "approve"
    assert not any(issue.category == "completeness" for issue in critique.issues)


def test_completeness_critique_allows_named_tool_alternatives():
    plan = ExperimentPlan.model_validate(
        {
            **_plan(_task("EXP-1", tool="generate_case_mols")).model_dump(mode="json"),
            "source_request": (
                "Generate candidates with generate_case_mols for alzheimer; "
                "otherwise generate_mols."
            ),
        }
    )
    inventory = [
        {
            "tool": "generate_case_mols",
            "server_id": "srv-gen",
            "description": "Case-conditioned molecule generation.",
        },
        {
            "tool": "generate_mols",
            "server_id": "srv-gen",
            "description": "Generic molecule generation.",
        },
    ]
    plan.tasks[0].mcp_servers[0].server_id = "srv-gen"
    critique = critique_plan(
        plan,
        settings=ExperimentsSettings(),
        available_tools=inventory,
        preferred_tools=inventory,
    )
    assert critique.verdict == "approve"
    assert not any(issue.category == "completeness" for issue in critique.issues)


def test_scenario_a_ready_chemistry_mcp_defaults_to_one_fedot_attempt_and_artifact():
    """§11.6 A: ready MCP -> FEDOT -> one guarded call -> managed artifact."""
    plan = _plan(_task("EXP-1"))
    assert plan.tasks[0].route.value == "fedot_mas"
    state = _approved_state(plan)

    started = start_task(state, "EXP-1")
    assert started["route"] == "fedot_mas"
    assert started["route_agent"] == "FedotAgent"
    attempt_id = started["attempt_id"]
    assert state["experiment_runtime"]["tasks"]["EXP-1"]["attempts"][attempt_id][
        "route_returned"
    ] is False

    _route_return(state, "FedotAgent")
    state["fedot_artifacts"] = [
        {
            "name": "exp-1-result.csv",
            "bucket": "managed-experiments",
            "s3_key": "experiments/run/plan/EXP-1/attempt/result.csv",
            "tool": "estimate_property",
        }
    ]
    stored = record_result(state, "EXP-1", attempt_id, _success_result("EXP-1"))
    result = stored["task_result"]

    assert result["status"] == "success"
    assert result["route_used"] == "fedot_mas"
    assert len(result["artifacts"]) == 1
    artifact = result["artifacts"][0]
    assert artifact["bucket"] == "managed-experiments"
    assert artifact["plan_id"] == plan.plan_id
    assert artifact["task_id"] == "EXP-1"
    assert artifact["attempt_id"] == attempt_id
    assert state["experiment_runtime"]["phase"] == "reporting"


def test_guard_coerces_nested_agent_tool_request_to_string():
    state = _approved_state(_plan(_task("EXP-1")))
    start_task(state, "EXP-1")
    args = {
        "request": {
            "task_id": "EXP-1",
            "attempt_id": "ATT-x",
            "launch_params": {"case": "cancer", "num": 10},
        }
    }
    assert (
        guard_route_agent_tool(
            SimpleNamespace(name="FedotAgent"), args, _tool_context(state)
        )
        is None
    )
    assert isinstance(args["request"], str)
    assert '"case": "cancer"' in args["request"] or '"case":"cancer"' in args["request"]


def test_guard_refuses_start_task_until_record_result():
    state = _approved_state(_plan(_task("EXP-1"), _task("EXP-2", depends_on=["EXP-1"])))
    first = start_task(state, "EXP-1")
    _route_return(state, "FedotAgent")

    refused = guard_route_agent_tool(
        SimpleNamespace(name="start_task"),
        {"task_id": "EXP-2"},
        _tool_context(state),
    )
    assert refused["status"] == "refused"
    assert refused["error_code"] == "record_result_required"
    assert refused["must_record_attempt_id"] == first["attempt_id"]
    assert RECORD_REQUIRED_MESSAGE in refused["message"]

    # Closing tools remain allowed.
    assert (
        guard_route_agent_tool(
            SimpleNamespace(name="record_result"),
            {},
            _tool_context(state),
        )
        is None
    )


def test_enforce_pending_record_result_injects_function_call():
    from google.adk.models import LlmResponse
    from google.genai import types

    state = _approved_state(_plan(_task("EXP-1")))
    started = start_task(state, "EXP-1")
    _route_return(state, "FedotAgent")
    state["experiment_last_route_response"] = "Fedot failed: model restriction."

    prose = LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part(text="All done, here is the experiment summary.")],
        )
    )
    # No captured artifacts → force failure + retryable (not false success).
    forced = enforce_pending_record_result(SimpleNamespace(state=state), prose)
    assert forced is not None
    fc = forced.content.parts[0].function_call
    assert fc.name == "record_result"
    assert fc.args["task_id"] == "EXP-1"
    assert fc.args["attempt_id"] == started["attempt_id"]
    assert fc.args["result"]["status"] == "failure"
    assert fc.args["result"]["retryable"] is True
    assert fc.args["result"]["error_code"] == "route_failed_or_empty"

    # Already calling record_result → leave response alone.
    closing = LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part.from_function_call(
                    name="record_result",
                    args={"task_id": "EXP-1", "attempt_id": started["attempt_id"], "result": {}},
                )
            ],
        )
    )
    assert enforce_pending_record_result(SimpleNamespace(state=state), closing) is None

    # With captured artifacts → optimistic success (tools may still downgrade).
    state2 = _approved_state(_plan(_task("EXP-1")))
    started2 = start_task(state2, "EXP-1")
    _route_return(state2, "FedotAgent")
    state2["fedot_artifacts"] = [
        {
            "name": "candidates.csv",
            "bucket": "managed-experiments",
            "s3_key": "experiments/EXP-1/candidates.csv",
        }
    ]
    state2["experiment_last_route_response"] = "Fedot finished with managed table."
    forced_ok = enforce_pending_record_result(SimpleNamespace(state=state2), prose)
    assert forced_ok is not None
    assert forced_ok.content.parts[0].function_call.args["result"]["status"] == "success"
    assert forced_ok.content.parts[0].function_call.args["attempt_id"] == started2["attempt_id"]

    # retry_task before record is refused so FORCE_RECORD can close the attempt.
    state3 = _approved_state(_plan(_task("EXP-1")))
    start_task(state3, "EXP-1")
    _route_return(state3, "FedotAgent")
    refused_retry = guard_route_agent_tool(
        SimpleNamespace(name="retry_task"),
        {"task_id": "EXP-1"},
        _tool_context(state3),
    )
    assert refused_retry is not None
    assert refused_retry["error_code"] == "record_result_required"
    retry_llm = LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part.from_function_call(
                    name="retry_task",
                    args={"task_id": "EXP-1"},
                )
            ],
        )
    )
    forced_over_retry = enforce_pending_record_result(SimpleNamespace(state=state3), retry_llm)
    assert forced_over_retry is not None
    assert forced_over_retry.content.parts[0].function_call.name == "record_result"
    assert forced_over_retry.content.parts[0].function_call.args["result"]["retryable"] is True


def test_enforce_continue_until_reporting_starts_next_ready_task():
    from google.adk.models import LlmResponse
    from google.genai import types

    plan = _plan(_task("EXP-1"), _task("EXP-2", depends_on=["EXP-1"]))
    state = _approved_state(plan)
    started = start_task(state, "EXP-1")
    _route_return(state, "FedotAgent")
    state["fedot_artifacts"] = [
        {
            "name": "exp-1-result.csv",
            "bucket": "managed-experiments",
            "s3_key": "experiments/EXP-1/result.csv",
        }
    ]
    record_result(state, "EXP-1", started["attempt_id"], _success_result("EXP-1"))
    assert state["experiment_runtime"]["phase"] == "execution"
    assert state["experiment_runtime"]["tasks"]["EXP-2"]["status"] == "ready"

    prose = LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part(text="EXP-1 succeeded; experiment complete.")],
        )
    )
    forced = enforce_continue_until_reporting(SimpleNamespace(state=state), prose)
    assert forced is not None
    fc = forced.content.parts[0].function_call
    assert fc.name == "start_task"
    assert fc.args["task_id"] == "EXP-2"

    # Tool call already present → do not override.
    with_tool = LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part.from_function_call(
                    name="get_experiment_plan",
                    args={},
                )
            ],
        )
    )
    assert enforce_continue_until_reporting(SimpleNamespace(state=state), with_tool) is None


def test_rewrite_mismatched_control_action_fixes_retry_when_fallback_pending():
    from google.adk.models import LlmResponse
    from google.genai import types

    from CoScientist.experiments.runtime import rewrite_mismatched_control_action

    state = _approved_state(_plan(_task("EXP-1")))
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    record_result(
        state,
        "EXP-1",
        started["attempt_id"],
        {
            "status": "failure",
            "summary": "Transient FEDOT timeout.",
            "criteria_checks": [],
            "error_code": "timeout",
            "error_message": "timeout",
            "retryable": True,
        },
    )
    retry_task(state, "EXP-1")
    retried = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    record_result(
        state,
        "EXP-1",
        retried["attempt_id"],
        {
            "status": "failure",
            "summary": "FEDOT still failing after retry.",
            "criteria_checks": [],
            "error_code": "timeout",
            "error_message": "timeout",
            "retryable": True,
        },
    )
    assert state["experiment_runtime"]["tasks"]["EXP-1"]["status"] == "fallback_pending"

    wrong = LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part.from_function_call(name="retry_task", args={"task_id": "EXP-1"})],
        )
    )
    fixed = rewrite_mismatched_control_action(SimpleNamespace(state=state), wrong)
    assert fixed is not None
    assert fixed.content.parts[0].function_call.name == "fallback_task"
    assert fixed.content.parts[0].function_call.args["task_id"] == "EXP-1"
    assert fixed.content.parts[0].function_call.args.get("reason")


def test_rewrite_mismatched_control_action_suppresses_start_while_running():
    from google.adk.models import LlmResponse
    from google.genai import types

    from CoScientist.experiments.runtime import rewrite_mismatched_control_action

    plan = _plan(_task("EXP-1"), _task("EXP-2", depends_on=[]))
    state = _approved_state(plan)
    start_task(state, "EXP-1")
    assert state["experiment_runtime"]["tasks"]["EXP-1"]["status"] == "running"
    # EXP-2 may be ready concurrently; model must not start it while EXP-1 runs.
    state["experiment_runtime"]["tasks"]["EXP-2"]["status"] = "ready"

    wrong = LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part.from_function_call(name="start_task", args={"task_id": "EXP-2"})],
        )
    )
    fixed = rewrite_mismatched_control_action(SimpleNamespace(state=state), wrong)
    assert fixed is not None
    assert fixed.content.parts[0].function_call.name == "get_experiment_plan"


def test_rewrite_mismatched_control_action_suppresses_orphan_outside_execution():
    from google.adk.models import LlmResponse
    from google.genai import types

    from CoScientist.experiments.runtime import rewrite_mismatched_control_action

    state = _approved_state(_plan(_task("EXP-1")))
    state["experiment_runtime"]["phase"] = "reporting"
    state["experiment_runtime"]["tasks"]["EXP-1"]["status"] = "failed"
    wrong = LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part.from_function_call(
                    name="fallback_task",
                    args={"task_id": "EXP-1", "reason": "try again"},
                )
            ],
        )
    )
    fixed = rewrite_mismatched_control_action(SimpleNamespace(state=state), wrong)
    assert fixed is not None
    assert fixed.content.parts[0].function_call.name == "get_experiment_plan"


def test_normalize_em_hypothesis_commit_shrinks_and_stashes():
    from google.adk.models import LlmResponse
    from google.genai import types

    from CoScientist.experiments.hypotheses import (
        commit_experiment_hypotheses,
        normalize_em_hypothesis_commit,
    )

    fat_nodes = []
    for i in range(5):
        fat_nodes.append(
            {
                "type": "Hypothesis",
                "ref": f"h{i}",
                "attrs": {"formulation": f"Hypothesis about target {i} works.", "status": "formulated"},
            }
        )
        fat_nodes.append(
            {
                "type": "VerificationMethod",
                "ref": f"vm{i}",
                "attrs": {"description": "x" * 200, "protocol_steps": ["a", "b", "c"]},
            }
        )
    state: dict = {}
    resp = LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part.from_function_call(
                    name="research_commit",
                    args={"nodes": fat_nodes, "edges": []},
                )
            ],
        )
    )
    out = normalize_em_hypothesis_commit(SimpleNamespace(state=state), resp)
    assert out is not None
    fc = out.content.parts[0].function_call
    assert fc.name == "research_commit"
    assert len(fc.args["nodes"]) == 3
    assert all(n["type"] == "Hypothesis" for n in fc.args["nodes"])
    assert len(state["_em_hypotheses_from_fc"]) == 3

    # even without output_key text, commit uses FC stash
    commit_experiment_hypotheses(SimpleNamespace(state=state, user_content=None))
    assert len(state["hypothesis_refs"]) == 3
    assert state["hypothesis_refs"][0]["hypothesis_id"] == "H1"


def test_seed_hypotheses_instructs_small_commits():
    from google.adk.models import LlmRequest
    from google.genai import types

    from CoScientist.experiments.hypotheses import seed_hypotheses_from_em_request

    state = {"experiment_source_request": "Design BTK and KRAS inhibitors."}
    req = LlmRequest(contents=[types.Content(role="user", parts=[types.Part(text="noise")])])
    seed_hypotheses_from_em_request(SimpleNamespace(state=state, user_content=None), req)
    text = req.contents[0].parts[0].text
    assert "Hypothesis nodes ONLY" in text
    assert "At most" in text


def test_seed_hypotheses_does_not_instruct_creating_research_question():
    """HypothesesAgent has no research_init tool and no ResearchQuestion in its
    ACL (schema.AGENT_PERMISSIONS) — the seed prompt must not tell it to try."""
    from google.adk.models import LlmRequest
    from google.genai import types

    from CoScientist.experiments.hypotheses import seed_hypotheses_from_em_request

    state = {"experiment_source_request": "Design BTK and KRAS inhibitors."}
    req = LlmRequest(contents=[types.Content(role="user", parts=[types.Part(text="noise")])])
    seed_hypotheses_from_em_request(SimpleNamespace(state=state, user_content=None), req)
    text = req.contents[0].parts[0].text
    assert "Create a ResearchQuestion root" not in text
    assert "do NOT try to create the ResearchQuestion yourself" in text


def test_enforce_hypothesis_research_commit_forces_from_prose(monkeypatch):
    """Thought/prose-only exit → rewrite into a Hypothesis research_commit."""
    from google.adk.models import LlmResponse
    from google.genai import types

    from CoScientist.experiments.hypotheses import enforce_hypothesis_research_commit

    _patch_research_graph(monkeypatch, [])
    state: dict = {}
    prose = (
        "Hypothesis 1: ATP-competitive binders fit the GSK-3β pocket better.\n"
        "Hypothesis 2: Selective hinge motifs reduce off-target kinase hits.\n"
    )
    resp = LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=prose)])
    )
    out = enforce_hypothesis_research_commit(
        SimpleNamespace(state=state, user_content=None), resp
    )
    assert out is not None
    fc = out.content.parts[0].function_call
    assert fc.name == "research_commit"
    assert len(fc.args["nodes"]) == 2
    assert all(n["type"] == "Hypothesis" for n in fc.args["nodes"])
    assert state["_em_hypotheses_commit_forced"] is True
    assert len(state["_em_hypotheses_from_fc"]) == 2


def test_enforce_hypothesis_research_commit_forces_from_numbered_thinking(monkeypatch):
    from google.adk.models import LlmResponse
    from google.genai import types

    from CoScientist.experiments.hypotheses import enforce_hypothesis_research_commit

    _patch_research_graph(monkeypatch, [])
    thinking = (
        "Now I need to generate hypotheses:\n"
        "1. **ATP-competitive binding hypothesis**: Molecules with high structural "
        "similarity to known ATP-competitive GSK-3β inhibitors will show superior "
        "inhibitory activity due to optimal fitting in the ATP-binding pocket.\n"
        "2. **Selectivity kinase hypothesis**: Molecules designed with selective "
        "hinge-binding motifs targeting the unique Leu132 residue in GSK-3β will "
        "achieve higher selectivity and reduced off-target effects.\n"
    )
    resp = LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part(text=thinking, thought=True)],
        )
    )
    out = enforce_hypothesis_research_commit(
        SimpleNamespace(state={}, user_content=None),
        resp,
    )
    assert out is not None
    nodes = out.content.parts[0].function_call.args["nodes"]
    assert len(nodes) == 2
    assert "ATP-binding pocket" in nodes[0]["attrs"]["formulation"]


def test_enforce_hypothesis_research_commit_skips_when_tool_call_present(monkeypatch):
    from google.adk.models import LlmResponse
    from google.genai import types

    from CoScientist.experiments.hypotheses import enforce_hypothesis_research_commit

    _patch_research_graph(monkeypatch, [])
    resp = LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part.from_function_call(name="research_overview", args={})],
        )
    )
    out = enforce_hypothesis_research_commit(
        SimpleNamespace(state={}, user_content=None), resp
    )
    assert out is None


def test_enforce_hypothesis_research_commit_skips_when_already_have_refs(monkeypatch):
    from google.adk.models import LlmResponse
    from google.genai import types

    from CoScientist.experiments.hypotheses import enforce_hypothesis_research_commit

    _patch_research_graph(monkeypatch, [])
    state = {
        "_em_hypotheses_from_fc": [
            {"hypothesis_id": "H1", "statement": "Already stashed hypothesis."}
        ]
    }
    prose = "Hypothesis 1: Should not force again because stash exists.\n"
    resp = LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=prose)])
    )
    assert enforce_hypothesis_research_commit(
        SimpleNamespace(state=state, user_content=None), resp
    ) is None


def test_enforce_hypothesis_research_commit_only_once(monkeypatch):
    from google.adk.models import LlmResponse
    from google.genai import types

    from CoScientist.experiments.hypotheses import enforce_hypothesis_research_commit

    _patch_research_graph(monkeypatch, [])
    state: dict = {}
    prose = (
        "Hypothesis 1: First recoverable draft statement is long enough.\n"
        "Hypothesis 2: Second recoverable draft statement is long enough.\n"
    )
    resp = LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=prose)])
    )
    first = enforce_hypothesis_research_commit(
        SimpleNamespace(state=state, user_content=None), resp
    )
    assert first is not None
    # Clear stash to isolate the once-flag behavior.
    state["_em_hypotheses_from_fc"] = []
    second = enforce_hypothesis_research_commit(
        SimpleNamespace(state=state, user_content=None), resp
    )
    assert second is None


def test_scenario_b_two_sequential_fedot_tasks_and_duplicate_route_refused():
    """§11.6 B: no session hard-stop; second call in one attempt is refused."""
    plan = _plan(
        _task("EXP-1"),
        _task("EXP-2", depends_on=["EXP-1"]),
    )
    state = _approved_state(plan)
    state["fedot_artifacts"] = [
        {
            "name": "old.csv",
            "bucket": "managed-experiments",
            "s3_key": "old/other-attempt.csv",
        }
    ]

    first = start_task(state, "EXP-1")
    _route_return(state, "FedotAgent")
    duplicate = guard_route_agent_tool(
        SimpleNamespace(name="FedotAgent"), {}, _tool_context(state)
    )
    assert duplicate["status"] == "refused"
    assert duplicate["error_code"] == "route_already_returned"
    state["fedot_artifacts"].append(
        {
            "name": "exp-1-result.csv",
            "bucket": "managed-experiments",
            "s3_key": "experiments/EXP-1/result.csv",
        }
    )
    record_result(state, "EXP-1", first["attempt_id"], _success_result("EXP-1"))

    # A legacy session flag must not prevent a distinct ready attempt.
    state["fedot_deliverable_ready"] = True
    second = start_task(state, "EXP-2")
    assert second["route_agent"] == "FedotAgent"
    _route_return(state, "FedotAgent")
    state["fedot_artifacts"].append(
        {
            "name": "exp-2-result.csv",
            "bucket": "managed-experiments",
            "s3_key": "experiments/EXP-2/result.csv",
        }
    )
    record_result(state, "EXP-2", second["attempt_id"], _success_result("EXP-2"))

    results = state["experiment_task_results"]
    assert [result["task_id"] for result in results] == ["EXP-1", "EXP-2"]
    assert results[0]["artifacts"][0]["s3_key"].endswith("EXP-1/result.csv")
    assert results[1]["artifacts"][0]["s3_key"].endswith("EXP-2/result.csv")
    assert state["experiment_runtime"]["phase"] == "reporting"


def test_record_result_coerces_error_status_to_failure():
    state = _approved_state(_plan(_task("EXP-1")))
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    stored = record_result(
        state,
        "EXP-1",
        started["attempt_id"],
        {
            "status": "error",
            "summary": "No relevant data found.",
            "criteria_checks": [],
            "error_code": "no_data",
            "error_message": "empty",
            "retryable": True,
        },
    )
    assert stored["status"] == "success"
    assert stored["task_result"]["status"] == "failure"
    assert state["experiment_runtime"]["tasks"]["EXP-1"]["status"] == "retry_pending"


def test_record_result_coerces_partial_success_alias_to_partial():
    """LLM often emits partial_success; closed enum only allows partial."""
    state = _approved_state(_plan(_task("EXP-1")))
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    state["fedot_artifacts"] = [
        {
            "name": "exp-1-result.csv",
            "bucket": "managed-experiments",
            "s3_key": "experiments/run/plan/EXP-1/attempt/result.csv",
            "tool": "estimate_property",
        }
    ]
    payload = _success_result("EXP-1")
    payload["status"] = "partial_success"
    payload["summary"] = "KRAS docking ok; HRAS/NRAS timed out."
    stored = record_result(state, "EXP-1", started["attempt_id"], payload)
    assert stored["status"] == "success"
    assert stored["task_result"]["status"] == "partial"
    assert state["experiment_runtime"]["tasks"]["EXP-1"]["status"] == "done_with_warnings"


def test_start_task_generates_fresh_transient_s3_links():
    task = _task("EXP-1", route="coder")
    task["input_data"] = [
        {
            "data_id": "input-csv",
            "kind": "s3",
            "description": "Managed source data",
            "bucket": "inputs",
            "s3_key": "data/source.csv",
        }
    ]
    plan = _plan(task)
    state = _approved_state(plan)
    calls = []

    def presign(bucket: str, key: str, expiration: int) -> str:
        calls.append((bucket, key, expiration))
        return f"https://s3.local/{key}?X-Amz-Signature=fresh-{len(calls)}"

    started = start_task(state, "EXP-1", presign=presign)
    assert calls and calls[0][2] > ExperimentsSettings().coder_timeout_s
    assert "X-Amz-Signature=fresh-1" in started["resolved_inputs"][0]["resolved_url"]
    assert "resolved_url" not in state["experiment_runtime"]["plan"]["tasks"][0][
        "input_data"
    ][0]


def test_retry_fallback_skip_and_amend_transitions():
    # retry
    state = _approved_state(_plan(_task("EXP-1")))
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    record_result(
        state,
        "EXP-1",
        started["attempt_id"],
        {
            "status": "failure",
            "summary": "Transient FEDOT timeout.",
            "criteria_checks": [],
            "error_code": "timeout",
            "error_message": "timeout",
            "retryable": True,
        },
    )
    assert state["experiment_runtime"]["tasks"]["EXP-1"]["status"] == "retry_pending"
    retry_task(state, "EXP-1")
    retried = start_task(state, "EXP-1")
    assert retried["attempt_id"] != started["attempt_id"]

    # After per-route retry budget is spent, still fall back to the next route.
    mark_route_returned(state, "FedotAgent")
    record_result(
        state,
        "EXP-1",
        retried["attempt_id"],
        {
            "status": "failure",
            "summary": "FEDOT still failing after retry.",
            "criteria_checks": [],
            "error_code": "timeout",
            "error_message": "timeout",
            "retryable": True,
        },
    )
    assert state["experiment_runtime"]["tasks"]["EXP-1"]["status"] == "fallback_pending"
    fb = fallback_task(state, "EXP-1", "FEDOT retries exhausted")
    assert fb["route"] == "react_tools"
    assert fb.get("must_start_task_id") == "EXP-1"
    assert start_task(state, "EXP-1")["route_agent"] == "ExperimentAgent"

    # fallback (non-retryable → immediate next route: react_tools)
    state = _approved_state(_plan(_task("EXP-1")))
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    record_result(
        state,
        "EXP-1",
        started["attempt_id"],
        {
            "status": "failure",
            "summary": "No runnable FEDOT server.",
            "criteria_checks": [],
            "error_code": "route_unavailable",
            "error_message": "server unavailable",
            "retryable": False,
        },
    )
    assert state["experiment_runtime"]["tasks"]["EXP-1"]["status"] == "fallback_pending"
    fallback = fallback_task(state, "EXP-1", "FEDOT route unavailable")
    assert fallback["route"] == "react_tools"
    assert start_task(state, "EXP-1")["route_agent"] == "ExperimentAgent"

    # skip optional
    optional = _task("EXP-1", route="coder", optional=True)
    state = _approved_state(_plan(optional))
    skipped = skip_task(state, "EXP-1", "Optional comparison omitted.")
    assert skipped["task_result"]["status"] == "skipped"

    # amend an unstarted task; criteria changes force review.
    state = _approved_state(_plan(_task("EXP-1", route="coder")))
    amended = amend_task(
        state,
        "EXP-1",
        {
            "success_criteria": [
                {
                    "criterion_id": "EXP-1-C1",
                    "description": "The script exits successfully.",
                    "kind": "execution",
                    "verification": "Check exit code 0.",
                }
            ]
        },
        "Clarify deterministic verification.",
    )
    assert amended["requires_review"] is True
    assert state["experiment_runtime"]["phase"] == "awaiting_review"


def test_react_to_coder_fallback_completes_with_real_workspace_artifact(tmp_path):
    """The v0 demo fallback reaches Coder without enabling MCP-in-Coder mode."""
    state = _approved_state(_plan(_task("EXP-1", route="react_tools")))
    first = start_task(state, "EXP-1")
    assert first["route_agent"] == "ExperimentAgent"
    mark_route_returned(state, "ExperimentAgent")
    record_result(
        state,
        "EXP-1",
        first["attempt_id"],
        {
            "status": "failure",
            "summary": "Ready MCP returned no usable output.",
            "criteria_checks": [],
            "error_code": "empty_result",
            "error_message": "empty result",
            "retryable": False,
        },
    )
    fallback_task(state, "EXP-1", "ReAct MCP result was empty")
    second = start_task(state, "EXP-1")
    assert second["route_agent"] == "CoderAgent"
    assert state["filtered_tools"] == []
    assert state["deployed_mcps"] == []

    artifact_path = tmp_path / "exp-1-result.csv"
    artifact_path.write_text("property,value\nmw,46.07\n", encoding="utf-8")
    mark_route_returned(state, "CoderAgent")
    stored = record_result(
        state,
        "EXP-1",
        second["attempt_id"],
        {
            **_success_result("EXP-1"),
            "artifacts": [
                {
                    "name": "exp-1-result.csv",
                    "workspace_path": str(artifact_path),
                    "durability": "workspace",
                    "tool": "execute_bash",
                }
            ],
        },
    )
    assert stored["task_result"]["route_used"] == "coder"
    assert stored["task_result"]["artifacts"][0]["workspace_path"] == str(
        artifact_path
    )
    assert state["experiment_runtime"]["phase"] == "reporting"


def test_terminal_and_incomplete_results_are_rejected():
    state = _approved_state(_plan(_task("EXP-1")))
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    with pytest.raises(ExperimentRuntimeError, match="missing required evidence"):
        record_result(
            state,
            "EXP-1",
            started["attempt_id"],
            {
                "status": "success",
                "summary": "Claimed success without evidence.",
                "criteria_checks": [],
            },
        )

    state["fedot_artifacts"] = [
        {
            "name": "exp-1-result.csv",
            "bucket": "managed-experiments",
            "s3_key": "experiments/EXP-1/result.csv",
        }
    ]
    record_result(state, "EXP-1", started["attempt_id"], _success_result("EXP-1"))
    with pytest.raises(ExperimentRuntimeError, match="terminal"):
        start_task(state, "EXP-1")


def test_record_result_requires_canonical_result_keys():
    state = _approved_state(_plan(_task("EXP-1")))
    started = start_task(state, "EXP-1")
    state["fedot_artifacts"] = [
        {
            "name": "exp-1-result.csv",
            "bucket": "managed-experiments",
            "s3_key": "experiments/EXP-1/result.csv",
        }
    ]
    mark_route_returned(state, "FedotAgent")

    with pytest.raises(ValidationError):
        record_result(
            state,
            "EXP-1",
            started["attempt_id"],
            {
                "status": "success",
                "summary": "Alias keys must not be accepted.",
                "actual_outputs": {"rows": 1},
                "criteria_checks": [
                    {
                        "criterion_id": "EXP-1-C1",
                        "met": True,
                        "evidence": {"status": "success"},
                        "message": "alias keys",
                    }
                ],
            },
        )

    stored = record_result(
        state,
        "EXP-1",
        started["attempt_id"],
        {
            "status": "success",
            "summary": "Canonical keys only.",
            "outputs": {"rows": 1},
            "criteria_checks": [
                {
                    "criterion_id": "EXP-1-C1",
                    "passed": True,
                    "observed": {"status": "success"},
                    "details": "The route returned a structured success result.",
                }
            ],
        },
    )

    result = stored["task_result"]
    assert result["outputs"] == {"rows": 1}
    assert result["criteria_checks"] == [
        {
            "criterion_id": "EXP-1-C1",
            "passed": True,
            "observed": {"status": "success"},
            "evidence_artifact_ids": [],
            "details": "The route returned a structured success result.",
        }
    ]
    assert stored["phase"] == "reporting"


def test_record_result_repairs_truncated_attempt_id():
    """LLM executors often drop the last hex char of ATT-<uuid>; repair near-miss."""
    state = _approved_state(_plan(_task("EXP-1")))
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    full_id = started["attempt_id"]
    truncated = full_id[:-1]
    assert truncated != full_id

    state["fedot_artifacts"] = [
        {
            "name": "exp-1-result.csv",
            "bucket": "managed-experiments",
            "s3_key": "experiments/EXP-1/result.csv",
        }
    ]
    stored = record_result(
        state,
        "EXP-1",
        truncated,
        _success_result("EXP-1"),
    )
    assert stored["task_result"]["status"] == "success"
    assert stored["task_result"]["attempt_id"] == full_id


def test_record_result_rejects_unrelated_attempt_id():
    state = _approved_state(_plan(_task("EXP-1")))
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    with pytest.raises(ExperimentRuntimeError, match="do not match the active attempt"):
        record_result(
            state,
            "EXP-1",
            "ATT-deadbeefdeadbeefdeadbeefdeadbeef",
            _success_result("EXP-1"),
        )
    assert started["attempt_id"] == state["experiment_runtime"]["active_attempt_id"]


def test_start_task_resolves_task_artifact_by_name(tmp_path, monkeypatch):
    """Planner stores source_artifact_id as the filename; runtime ART-* is unknown at plan time."""
    from CoScientist.config import get_settings

    monkeypatch.setattr(get_settings().code_exec, "workspace_root", str(tmp_path))

    upstream = _task("EXP-1", artifact_name="diagnosis_findings.json")
    downstream = _task("EXP-2", route="coder", depends_on=["EXP-1"], artifact_name="literature.json")
    downstream["mcp_servers"] = []
    downstream["input_data"] = [
        {
            "data_id": "findings",
            "kind": "task_artifact",
            "description": "Findings from EXP-1",
            "source_task_id": "EXP-1",
            "source_artifact_id": "diagnosis_findings.json",
        }
    ]
    state = _approved_state(_plan(upstream, downstream))
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    folder = tmp_path / "experiment_artifacts" / "EXP-1" / started["attempt_id"]
    folder.mkdir(parents=True)
    path = folder / "diagnosis_findings.json"
    path.write_text('{"ok": true}', encoding="utf-8")
    state["fedot_artifacts"] = [
        {
            "name": "diagnosis_findings.json",
            "workspace_path": str(path),
            "media_type": "application/json",
        }
    ]
    record_result(state, "EXP-1", started["attempt_id"], _success_result("EXP-1"))

    started2 = start_task(state, "EXP-2")
    assert started2["status"] == "success"
    resolved = started2["resolved_inputs"]
    assert resolved[0]["resolved_workspace_path"] == str(path)


def test_control_tool_downgrades_incomplete_success_to_terminal_failure():
    from CoScientist.experiments.runtime.tools import ExperimentControlToolset

    state = _approved_state(_plan(_task("EXP-1")))
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")

    stored = ExperimentControlToolset().record_result(
        "EXP-1",
        started["attempt_id"],
        {
            "status": "partial",
            "summary": "Route returned text but no required artifact.",
            "criteria_checks": [
                {
                    "criterion_id": "EXP-1-C1",
                    "passed": False,
                    "details": "Required execution evidence was absent.",
                }
            ],
            "retryable": False,
        },
        SimpleNamespace(state=state),
    )

    assert stored["status"] == "success"
    assert stored["downgraded_from"] == "partial"
    assert stored["task_result"]["status"] == "failure"
    assert stored["task_result"]["error_code"] == "result_incomplete"
    assert stored["task_result"]["retryable"] is True
    assert state["experiment_runtime"]["active_attempt_id"] is None
    assert state["experiment_runtime"]["tasks"]["EXP-1"]["status"] == "retry_pending"


def test_record_result_downgrades_fabricated_success_to_partial():
    state = _approved_state(_plan(_task("EXP-1")))
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    state["fedot_artifacts"] = [
        {
            "name": "exp-1-result.csv",
            "bucket": "managed-experiments",
            "s3_key": "experiments/run/plan/EXP-1/attempt/result.csv",
            "tool": "estimate_property",
        }
    ]
    payload = _success_result("EXP-1")
    payload["summary"] = "Completed with simulated PubMed hits and hardcoded metabolite list."
    payload["warnings"] = ["Literature data was simulated via a hardcoded list"]

    stored = record_result(state, "EXP-1", started["attempt_id"], payload)
    result = stored["task_result"]
    assert result["status"] == "partial"
    assert state["experiment_runtime"]["tasks"]["EXP-1"]["status"] == "done_with_warnings"
    assert any("downgraded_from_success" in w for w in result["warnings"])


def test_repair_plan_mcp_bindings_fills_or_demotes():
    from CoScientist.experiments.critique.mcp_repair import repair_plan_mcp_bindings

    inventory = [
        {
            "tool": "chemical_space_clustering",
            "server_id": "bfc62a287aaf7b5a",
            "description": "Cluster metabolites in chemical space",
        },
        {
            "tool": "predict_ld50",
            "server_id": "bfc62a287aaf7b5a",
            "description": "Predict LD50",
        },
    ]
    payload = {
        "source_request": "Cluster metabolites and report chemical space structure.",
        "tasks": [
            {
                "id": "EXP-1",
                "name": "Clustering metabolites",
                "description": "Cluster by molecular similarity",
                "route": "fedot_mas",
                "design": {
                    "analysis_artifacts": [
                        {
                            "name": "clustering_report.pdf",
                            "prepare_via": "mcp",
                            "path_or_tool": "clustering_report_generator",
                        }
                    ]
                },
                "mcp_servers": [],
            },
            {
                "id": "EXP-2",
                "name": "Obscure step",
                "description": "Something with no matching inventory tool",
                "route": "fedot_mas",
                "design": {
                    "analysis_artifacts": [
                        {"name": "x.json", "prepare_via": "mcp", "path_or_tool": "totally_fake_tool"}
                    ]
                },
                "mcp_servers": [],
            },
        ]
    }
    repaired = repair_plan_mcp_bindings(payload, inventory)
    assert repaired["tasks"][0]["mcp_servers"][0]["tools"][0]["name"] == "chemical_space_clustering"
    assert repaired["tasks"][0]["design"]["analysis_artifacts"][0]["path_or_tool"] == "chemical_space_clustering"
    # Obscure step has no capability cover in inventory → demote allowed.
    assert repaired["tasks"][1]["route"] == "coder"
    assert repaired["tasks"][1]["mcp_servers"] == []
    assert any("demoted_to_coder" in w for w in repaired["tasks"][1]["warnings"])


def test_start_task_seeds_upstream_from_resolved_inputs(tmp_path, monkeypatch):
    """Consumer start_task materializes producer CSV into upstream_bindings."""
    from CoScientist.config import get_settings

    monkeypatch.setattr(get_settings().code_exec, "workspace_root", str(tmp_path))

    upstream = _task("EXP-1", artifact_name="generated_molecules.csv")
    downstream = _task(
        "EXP-2",
        route="react_tools",
        tool="estimate_property",
        depends_on=["EXP-1"],
        artifact_name="toxicity.json",
    )
    downstream["input_data"] = [
        {
            "data_id": "mols",
            "kind": "task_artifact",
            "description": "Generated molecules",
            "source_task_id": "EXP-1",
            "source_artifact_id": "generated_molecules.csv",
        }
    ]
    state = _approved_state(_plan(upstream, downstream))
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    folder = tmp_path / "experiment_artifacts" / "EXP-1" / started["attempt_id"]
    folder.mkdir(parents=True)
    path = folder / "generated_molecules.csv"
    path.write_text("smiles,score\nCCO,0.9\nCCN,0.8\n", encoding="utf-8")
    state["fedot_artifacts"] = [
        {
            "name": "generated_molecules.csv",
            "workspace_path": str(path),
            "media_type": "text/csv",
        }
    ]
    # Ambient wrong table must lose to EM lineage.
    state["fedot_artifact_tables"] = [
        {
            "columns": ["smiles"],
            "rows": [{"smiles": "c1ccccc1"}],
            "format": "csv",
            "url": "https://example.invalid/benzene.csv",
        }
    ]
    record_result(state, "EXP-1", started["attempt_id"], _success_result("EXP-1"))

    started2 = start_task(state, "EXP-2")
    assert started2["status"] == "success"
    assert started2["resolved_inputs"][0]["resolved_workspace_path"] == str(path)
    bindings = started2.get("upstream_bindings") or {}
    assert "smiles" in bindings
    assert "CCO" in bindings["smiles"]
    assert "c1ccccc1" not in bindings["smiles"]
    assert state.get("upstream_artifact_inputs")
    tables = state.get("fedot_artifact_tables") or []
    assert tables and any(
        row.get("smiles") == "CCO"
        for t in tables
        for row in (t.get("rows") or [])
        if isinstance(row, dict)
    )


def test_repair_binds_generate_case_mols_not_docking_or_demote():
    """MADD-L shaped: empty mcp + affinity wording + Generate molecules → generation tool."""
    from CoScientist.experiments.critique.mcp_repair import repair_plan_mcp_bindings

    inventory = [
        {
            "tool": "calculate_docking",
            "server_id": "srv-dock",
            "description": "Calculate docking score / affinity for a molecule.",
        },
        {
            "tool": "generate_case_mols",
            "server_id": "srv-gen",
            "description": "GAN molecule generation for disease cases.",
        },
    ]
    payload = {
        "source_request": (
            "Suggest several molecules that have high docking affinity with KRAS G12C protein. "
            "Molecules should possess common drug-like properties. "
            "Generate highly potent non-covalent BTK inhibitors."
        ),
        "tasks": [
            {
                "id": "EXP-1",
                "name": "Molecule Generation with High Docking Affinity",
                "description": (
                    "Identify and generate molecules that exhibit high docking affinity "
                    "for KRAS G12C and have drug-like properties."
                ),
                "route": "fedot_mas",
                "design": {
                    "experiment_question": (
                        "What molecules possess high docking affinity for KRAS G12C "
                        "protein and meet drug-like properties?"
                    ),
                    "analysis_artifacts": [
                        {
                            "name": "generated_molecules.json",
                            "prepare_via": "mcp",
                            "path_or_tool": "generated_molecules.json",
                        }
                    ],
                },
                "mcp_servers": [],
            }
        ],
    }
    repaired = repair_plan_mcp_bindings(payload, inventory)
    task = repaired["tasks"][0]
    assert task["route"] == "fedot_mas"
    assert task["mcp_servers"][0]["tools"][0]["name"] == "generate_case_mols"
    assert task["design"]["analysis_artifacts"][0]["path_or_tool"] == "generate_case_mols"
    assert not any("demoted_to_coder" in str(w) for w in (task.get("warnings") or []))


def test_repair_never_demotes_when_inventory_covers_request():
    """Live inventory covering the ask must bind, never demote to coder."""
    from CoScientist.experiments.critique.mcp_repair import repair_plan_mcp_bindings

    inventory = [
        {
            "tool": "generate_case_mols",
            "server_id": "srv-gen",
            "description": "Generate case molecules",
        }
    ]
    payload = {
        "source_request": "Generate small molecules for alzheimer anti-inflammatory case.",
        "tasks": [
            {
                "id": "EXP-1",
                "name": "Generate candidates",
                "description": "Generate molecules for the case",
                "route": "fedot_mas",
                "design": {
                    "experiment_question": "Which generated molecules look drug-like?",
                    "analysis_artifacts": [
                        {
                            "name": "out.csv",
                            "prepare_via": "mcp",
                            "path_or_tool": "candidates.csv",
                        }
                    ],
                },
                "mcp_servers": [],
            }
        ],
    }
    repaired = repair_plan_mcp_bindings(payload, inventory)
    assert repaired["tasks"][0]["route"] == "fedot_mas"
    assert repaired["tasks"][0]["mcp_servers"][0]["tools"][0]["name"] == "generate_case_mols"
    assert not any(
        "demoted_to_coder" in str(w) for w in (repaired["tasks"][0].get("warnings") or [])
    )


def test_repair_demotes_only_when_inventory_empty():
    from CoScientist.experiments.critique.mcp_repair import repair_plan_mcp_bindings

    payload = {
        "source_request": "Generate molecules with high docking affinity.",
        "tasks": [
            {
                "id": "EXP-1",
                "name": "Generate",
                "description": "Generate molecules",
                "route": "fedot_mas",
                "design": {
                    "analysis_artifacts": [
                        {"name": "out.json", "prepare_via": "mcp", "path_or_tool": "x"}
                    ]
                },
                "mcp_servers": [],
            }
        ],
    }
    repaired = repair_plan_mcp_bindings(payload, [])
    assert repaired["tasks"][0]["route"] == "coder"
    assert any("empty_inventory" in w for w in repaired["tasks"][0]["warnings"])

def test_repair_orphan_hypotheses_links_via_also_tests():
    """Orphans are left for critique — no silent auto-link onto the first task."""
    from CoScientist.experiments.critique.mcp_repair import repair_orphan_hypotheses

    plan = _plan(
        _task("EXP-1"),
        hypotheses=[
            {"hypothesis_id": "H1", "statement": "Primary claim."},
            {"hypothesis_id": "H2", "statement": "Secondary claim."},
        ],
    )
    fixed = repair_orphan_hypotheses(plan)
    assert fixed.tasks[0].design.also_tests == plan.tasks[0].design.also_tests
    assert not any("auto_linked_hypotheses" in w for w in (fixed.tasks[0].warnings or []))

    # No context hypothesis_refs → plan.hypotheses orphans are critique majors.
    critique = critique_plan(
        plan,
        settings=ExperimentsSettings(),
        available_tools=_inventory(),
    )
    assert critique.verdict == "revise"
    assert any("not linked from tasks" in i.message for i in critique.issues)


def test_planner_and_coder_prompts_cover_multi_h_and_anti_fabrication():
    from unittest.mock import MagicMock

    from CoScientist.experiments.prompts import templates as exp_prompts

    ctx = MagicMock()
    ctx.render_tools.return_value = ""
    ctx.render_agents.return_value = ""
    planner = exp_prompts.experiment_planner(ctx)
    coder = exp_prompts.experiment_coder_route(ctx)
    executor = exp_prompts.experiment_executor(ctx)
    retriever = exp_prompts.experiment_tool_retriever(ctx)
    assert "hypothesis_refs" in planner and "AUTHORITATIVE" in planner
    assert "Do NOT invent extra hypotheses" in planner
    assert "HypothesesAgent" in planner
    assert "Empty mcp_servers is invalid" in planner or "bind or" in planner
    assert "risks/assumptions only at plan root" in planner
    assert "Mandatory markdown/HTML reports are forbidden" in planner
    assert "role=data" in planner
    assert "different-family" in planner
    assert "Cover every distinct facet" in retriever
    assert "one non-optional" in planner and "distinct target" in planner
    assert "ANTI-FABRICATION" in coder
    assert "hardcoded" in coder.lower()
    assert "simulated/hardcoded" in executor.lower() or "fabricated" in executor.lower()
    assert "phase is still" in executor and "reporting" in executor


def test_start_task_forces_s3_upload_for_molecule_generators():
    task = _task("EXP-1", tool="generate_case_mols")
    task["launch_params"] = {
        "case": "cancer",
        "num": 10,
        "upload_results_to_s3": False,
        "return_inline_results": False,
    }
    plan = _plan(task)
    state: dict = {}
    initialize_runtime(
        state,
        plan,
        critique={
            "schema_version": "plan-critique/0.1",
            "critique_id": "CRIT-test",
            "plan_id": plan.plan_id,
            "verdict": "approve",
            "issues": [],
            "checked_at": NOW,
        },
    )
    approve_plan(state)
    started = start_task(state, "EXP-1")
    assert started["task"]["launch_params"]["upload_results_to_s3"] is True
    assert (
        state["experiment_active_envelope"]["task"]["launch_params"][
            "upload_results_to_s3"
        ]
        is True
    )


def test_guard_does_not_mutate_route_request_payload():
    state = _approved_state(_plan(_task("EXP-1")))
    start_task(state, "EXP-1")
    payload = {
        "case": "cancer",
        "num": 10,
        "upload_results_to_s3": False,
        "output_s3_prefix": "generated",
    }
    args = {"request": json.dumps(payload)}
    assert (
        guard_route_agent_tool(
            SimpleNamespace(name="FedotAgent"), args, _tool_context(state)
        )
        is None
    )
    assert json.loads(args["request"]) == payload


def test_inline_fedot_csv_is_materialized_as_expected_workspace_artifact(
    tmp_path, monkeypatch
):
    from CoScientist.config import get_settings
    from CoScientist.experiments.runtime.inline_artifacts import (
        materialize_inline_result,
    )

    monkeypatch.setattr(get_settings().code_exec, "workspace_root", str(tmp_path))
    state = _approved_state(_plan(_task("EXP-1")))
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")

    artifacts = materialize_inline_result(
        state,
        {
            "metabolite_analysis_output": (
                "```csv\n"
                '"Name","Cluster","LD50"\n'
                '"Bergapten","E","597.6"\n'
                "```"
            )
        },
    )

    assert len(artifacts) == 1
    artifact_path = artifacts[0]["workspace_path"]
    assert artifact_path.endswith("exp-1-result.csv")
    assert '"Bergapten","E","597.6"' in open(
        artifact_path, encoding="utf-8"
    ).read()

    stored = record_result(
        state,
        "EXP-1",
        started["attempt_id"],
        _success_result("EXP-1"),
    )
    assert stored["task_result"]["status"] == "success"
    assert stored["task_result"]["artifacts"][0]["workspace_path"] == artifact_path


def test_record_result_requires_explicit_criteria_checks(tmp_path, monkeypatch):
    """Executors must supply criteria_checks; no auto-attest from artifacts."""
    from CoScientist.config import get_settings

    monkeypatch.setattr(get_settings().code_exec, "workspace_root", str(tmp_path))
    task = _task("EXP-2", artifact_name="antitarget_ranking")
    task["success_criteria"] = [
        {
            "criterion_id": "C2",
            "description": "Antitarget ranking artifact exists.",
            "kind": "artifact_exists",
            "verification": "Confirm antitarget_ranking is present.",
        }
    ]
    task["expected_artifacts"] = [
        {
            "name": "antitarget_ranking",
            "role": "data",
            "media_type": "application/json",
            "description": "Ranked antitargets.",
        }
    ]
    state = _approved_state(_plan(task))
    started = start_task(state, "EXP-2")
    mark_route_returned(state, "FedotAgent")

    with pytest.raises(ExperimentRuntimeError, match="missing required evidence"):
        record_result(
            state,
            "EXP-2",
            started["attempt_id"],
            {
                "status": "success",
                "summary": "Antitarget ranking generated.",
                "outputs": {
                    "antitarget_ranking": [
                        {"protein": "KCNH2", "median_pLD50": 3.97},
                    ]
                },
            },
        )

    stored = record_result(
        state,
        "EXP-2",
        started["attempt_id"],
        {
            "status": "success",
            "summary": "Antitarget ranking generated.",
            "outputs": {
                "antitarget_ranking": [
                    {"protein": "KCNH2", "median_pLD50": 3.97},
                ]
            },
            "criteria_checks": [
                {
                    "criterion_id": "C2",
                    "passed": True,
                    "observed": "artifact present",
                    "details": "antitarget_ranking captured",
                }
            ],
        },
    )
    result = stored["task_result"]
    assert result["status"] == "success"
    assert result["criteria_checks"][0]["criterion_id"] == "C2"


def test_record_result_does_not_auto_pass_threshold_without_checks(tmp_path, monkeypatch):
    from CoScientist.config import get_settings

    monkeypatch.setattr(get_settings().code_exec, "workspace_root", str(tmp_path))
    task = _task("EXP-1")
    task["success_criteria"] = [
        {
            "criterion_id": "C-metric",
            "description": "LD50 below threshold.",
            "kind": "threshold",
            "metric": "ld50",
            "operator": "<=",
            "target": 100,
            "verification": "Compare predicted LD50 to target.",
        }
    ]
    state = _approved_state(_plan(task))
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    state["fedot_artifacts"] = [
        {
            "name": "exp-1-result.csv",
            "workspace_path": str(tmp_path / "exp-1-result.csv"),
            "media_type": "text/csv",
        }
    ]
    (tmp_path / "exp-1-result.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(ExperimentRuntimeError, match="missing required evidence"):
        record_result(
            state,
            "EXP-1",
            started["attempt_id"],
            {
                "status": "success",
                "summary": "Model ran.",
                "outputs": {"exp-1-result.csv": "a,b\n1,2\n"},
            },
        )


def test_fedot_tool_skips_legacy_hard_stop_for_experiment_runtime(monkeypatch):
    import CoScientist.tools.fedotmas_tools as module

    calls = []

    def hard_stop(state):
        calls.append(state)
        return True

    class FakePostgres:
        def __init__(self, settings):
            pass

        async def initialize(self):
            return None

        async def close(self):
            return None

        async def get_server(self, server_id):
            return None

    monkeypatch.setattr(module, "should_hard_stop_fedot", hard_stop)
    monkeypatch.setattr(module, "PostgresClient", FakePostgres)

    context = SimpleNamespace(
        state={
            "experiment_runtime": {"run_id": "EXRUN-1"},
            "fedot_deliverable_ready": True,
            "filtered_tools": [],
        }
    )
    result = asyncio.run(module.fedot_toolset.fedot_tool("task", context))
    assert calls == []
    assert result["status"] == "error"  # reached normal no-server validation

    legacy = SimpleNamespace(state={"fedot_deliverable_ready": True})
    legacy_result = asyncio.run(module.fedot_toolset.fedot_tool("task", legacy))
    assert len(calls) == 1
    assert legacy_result["already_delivered"] is True


def test_web_hitl_timeout_is_fail_closed_only_for_experiment_review():
    async def scenario():
        handler = WebHITLHandler()
        experiment = await handler.handle_request(
            HITLRequest(
                agent_name="ExperimentPlannerAgent",
                action_type=HITLAction.APPROVE,
                message="Approve experiment plan",
                context={"experiment_review_kind": "plan"},
                timeout_seconds=0.001,
            )
        )
        assert experiment.approved is False
        assert experiment.timed_out is True

        legacy = await handler.handle_request(
            HITLRequest(
                agent_name="CoderAgent",
                action_type=HITLAction.APPROVE,
                message="Approve outward action",
                timeout_seconds=0.001,
            )
        )
        assert legacy.approved is True
        assert legacy.timed_out is False

    asyncio.run(scenario())


def test_expected_artifact_role_is_closed_enum_and_image_artifacts_need_tool_mime():
    from CoScientist.experiments.schemas import ExpectedArtifact

    with pytest.raises(ValidationError):
        ExpectedArtifact.model_validate(
            {
                "name": "overview_plot",
                "role": "visualization",
                "description": "Invented figure.",
            }
        )

    task = _task("EXP-1", tool="dataset_overview")
    task["mcp_servers"][0]["tools"][0] = {
        "name": "dataset_overview",
        "description": "Return tabular summary statistics for the dataset.",
        "input_schema": {},
    }
    task["expected_artifacts"] = [
        {
            "name": "dataset_overview",
            "role": "data",
            "media_type": "application/json",
            "description": "Overview JSON.",
        },
        {
            "name": "dataset_overview_plot",
            "role": "plot",
            "media_type": "image/png",
            "required": True,
            "description": "Invented plot the tool cannot produce.",
        },
    ]
    plan = _plan(task)
    inventory = [
        {
            "tool": "dataset_overview",
            "server_id": "srv-chem",
            "description": "Return tabular summary statistics for the dataset.",
        }
    ]
    critique = critique_plan(
        plan,
        settings=ExperimentsSettings(),
        available_tools=inventory,
    )
    # Soft advisory: media/role mismatch must not force plan revision loops.
    assert critique.verdict == "approve"
    assert any(
        issue.severity == "minor"
        and ("image/plot" in issue.message or "image/*" in issue.message)
        for issue in critique.issues
    )


def test_coder_workspace_artifacts_are_promoted_into_lineage(tmp_path, monkeypatch):
    from CoScientist.config import get_settings

    monkeypatch.setattr(get_settings().code_exec, "workspace_root", str(tmp_path))
    task = _task("EXP-1", route="coder", artifact_name="concentrations.csv")
    task["expected_artifacts"] = [
        {
            "name": "concentrations.csv",
            "role": "data",
            "media_type": "text/csv",
            "description": "ODE concentration time series.",
        }
    ]
    state = _approved_state(_plan(task))
    started = start_task(state, "EXP-1")

    sandbox = tmp_path / "ws_session_test"
    sandbox.mkdir()
    (sandbox / "concentrations.csv").write_text("t,A,B,C\n0,1,0,0\n", encoding="utf-8")
    state["coder_workspace_id"] = "ws_session_test"

    on_route_agent_returned(
        SimpleNamespace(name="CoderAgent"),
        {},
        SimpleNamespace(state=state),
        {"status": "success"},
    )
    assert state["coder_artifacts"]
    assert Path(state["coder_artifacts"][0]["workspace_path"]).is_file()

    stored = record_result(
        state,
        "EXP-1",
        started["attempt_id"],
        _success_result("EXP-1"),
    )
    assert stored["task_result"]["status"] == "success"
    assert stored["task_result"]["artifacts"][0]["name"] == "concentrations.csv"


def test_soft_artifact_name_match_accepts_stem_variants(tmp_path, monkeypatch):
    from CoScientist.config import get_settings

    monkeypatch.setattr(get_settings().code_exec, "workspace_root", str(tmp_path))
    task = _task("EXP-1", artifact_name="dataset_overview")
    task["expected_artifacts"] = [
        {
            "name": "dataset_overview",
            "role": "data",
            "media_type": "application/json",
            "description": "Overview payload.",
        }
    ]
    state = _approved_state(_plan(task))
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")

    folder = tmp_path / "experiment_artifacts" / "EXP-1" / started["attempt_id"]
    folder.mkdir(parents=True)
    path = folder / "dataset_overview.json"
    path.write_text('{"n": 3}', encoding="utf-8")
    state["fedot_artifacts"] = [
        {
            "name": "dataset_overview.json",
            "workspace_path": str(path),
            "media_type": "application/json",
            "producer_tool": "dataset_overview",
        }
    ]

    stored = record_result(
        state,
        "EXP-1",
        started["attempt_id"],
        _success_result("EXP-1"),
    )
    assert stored["task_result"]["status"] == "success"
    assert stored["task_result"]["artifacts"][0]["name"] == "dataset_overview"


def test_uuid_s3_csv_binds_to_semantic_expected_name(tmp_path, monkeypatch):
    """MCP generators return UUID filenames; plan expects alzheimer_candidates.csv."""
    from CoScientist.config import get_settings

    monkeypatch.setattr(get_settings().code_exec, "workspace_root", str(tmp_path))
    monkeypatch.setattr(get_settings().s3, "bucket_name", "molecule-generative-mcp")
    task = _task("EXP-1", tool="generate_case_mols", artifact_name="alzheimer_candidates.csv")
    plan = _plan(task)
    state: dict = {}
    initialize_runtime(
        state,
        plan,
        critique={
            "schema_version": "plan-critique/0.1",
            "critique_id": "CRIT-test",
            "plan_id": plan.plan_id,
            "verdict": "approve",
            "issues": [],
            "checked_at": NOW,
        },
    )
    approve_plan(state)
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")

    state["fedot_artifacts"] = [
        {
            "url": "http://10.32.1.114:9000/molecule-generative-mcp/generated/alzheimer/35dcc1e32f934178935c4e9cc2415f49.csv",
            "s3_key": "generated/alzheimer/35dcc1e32f934178935c4e9cc2415f49.csv",
            "bucket": "molecule-generative-mcp",
            "tool": "generate_case_mols",
        }
    ]

    stored = record_result(
        state,
        "EXP-1",
        started["attempt_id"],
        _success_result("EXP-1"),
    )
    assert stored["task_result"]["status"] == "success"
    assert stored["task_result"]["artifacts"][0]["name"] == "alzheimer_candidates.csv"
    assert stored["task_result"]["artifacts"][0]["durability"] == "managed"
    assert not any("URL-only" in w for w in stored["task_result"].get("warnings") or [])


def test_managed_data_satisfies_mistyped_required_data_name(tmp_path, monkeypatch):
    """Managed MCP CSV satisfies fantasy required data name (e.g. *.json) — R2 class."""
    from CoScientist.config import get_settings

    monkeypatch.setattr(get_settings().code_exec, "workspace_root", str(tmp_path))
    monkeypatch.setattr(get_settings().s3, "bucket_name", "molecule-generative-mcp")
    task = _task("EXP-1", tool="generate_case_mols", artifact_name="generated_molecules.json")
    task["expected_artifacts"] = [
        {
            "name": "generated_molecules.json",
            "role": "data",
            "media_type": "application/json",
            "required": True,
            "description": "Planner fantasy name; tool returns CSV.",
        }
    ]
    plan = _plan(task)
    state: dict = {}
    initialize_runtime(
        state,
        plan,
        critique={
            "schema_version": "plan-critique/0.1",
            "critique_id": "CRIT-test",
            "plan_id": plan.plan_id,
            "verdict": "approve",
            "issues": [],
            "checked_at": NOW,
        },
    )
    approve_plan(state)
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    state["fedot_artifacts"] = [
        {
            "url": "http://10.32.1.114:9000/molecule-generative-mcp/generated/alzheimer/aabbccdd11223344.csv",
            "s3_key": "generated/alzheimer/aabbccdd11223344.csv",
            "bucket": "molecule-generative-mcp",
            "tool": "generate_case_mols",
            "role": "data",
        }
    ]
    stored = record_result(
        state,
        "EXP-1",
        started["attempt_id"],
        _success_result("EXP-1"),
    )
    assert stored["task_result"]["status"] == "success"
    assert stored["task_result"]["artifacts"][0]["name"] == "generated_molecules.json"
    assert stored["task_result"]["artifacts"][0]["durability"] == "managed"
    assert stored["task_result"]["artifacts"][0]["s3_key"].endswith(".csv")


def test_discovered_capabilities_survive_filtered_tools_clear_and_revise():
    """Attempt clears must not erase critique inventory; revise keeps run id."""
    from CoScientist.experiments.context import (
        DISCOVERED_CAPABILITIES_KEY,
        RETRIEVED_CAPABILITIES_KEY,
        build_experiment_context,
        skip_executor_without_runtime,
        snapshot_experiment_discovered_capabilities,
        stash_experiment_retrieved_capabilities,
    )
    from CoScientist.experiments.critique import critique_plan

    tool = {
        "tool": "estimate_property",
        "server_id": "srv-chem",
        "description": "Compute a property.",
        "input_schema": {"type": "object"},
        "score": 0.9,
    }
    extra = {
        "tool": "calculate_docking",
        "server_id": "srv-dock",
        "description": "Dock a molecule.",
        "input_schema": {"type": "object"},
        "score": 0.4,
    }
    state: dict = {
        "accumulated_tools": [tool, extra],
        "filtered_tools": [tool],
        "experiment_source_request": "Estimate a chemical property with ready MCP tools.",
    }
    ctx = SimpleNamespace(state=state, user_content=None)
    stash_experiment_retrieved_capabilities(ctx)
    snapshot_experiment_discovered_capabilities(ctx)
    build_experiment_context(ctx)
    run_id = state["experiment_context"]["experiment_run_id"]
    assert state[RETRIEVED_CAPABILITIES_KEY][1]["tool"] == "calculate_docking"
    assert {c["tool"] for c in state["experiment_context"]["available_mcp_capabilities"]} == {
        "estimate_property",
        "calculate_docking",
    }
    assert {c["tool"] for c in state["experiment_context"]["critique_mcp_capabilities"]} >= {
        "estimate_property",
        "calculate_docking",
    }
    prompt_ctx = state["experiment_planner_context"]
    assert isinstance(prompt_ctx, str)
    assert "estimate_property" in prompt_ctx
    assert "calculate_docking" in prompt_ctx
    # Prompt projection must not triple-dump preferred/critique inventories.
    assert prompt_ctx.count("available_mcp_capabilities") == 1
    assert "critique_mcp_capabilities" not in prompt_ctx
    assert '"input_schema"' not in prompt_ctx

    # Simulate post-attempt clear used by the runtime.
    state["filtered_tools"] = []
    state["experiment_active_envelope"] = None
    state["experiment_plan_critique"] = {
        "verdict": "revise",
        "issues": [{"severity": "blocker", "message": "fix"}],
    }

    build_experiment_context(ctx)
    caps = state["experiment_context"]["critique_mcp_capabilities"]
    assert {c["server_id"] for c in caps} >= {"srv-chem", "srv-dock"}
    assert state["experiment_context"]["experiment_run_id"] == run_id
    assert state[DISCOVERED_CAPABILITIES_KEY]

    plan = ExperimentPlan.model_validate(_plan(_task("EXP-1")))
    approved = critique_plan(
        plan,
        settings=ExperimentsSettings(route_fedot=True),
        available_tools=caps,
    )
    assert approved.verdict == "approve"
    assert skip_executor_without_runtime(ctx) is not None
    state["experiment_plan_critique"] = approved.model_dump(mode="json")
    initialize_runtime(state, plan, critique=approved.model_dump(mode="json"))
    approve_plan(state)
    assert skip_executor_without_runtime(ctx) is None


def test_first_planner_entry_keeps_stashed_retrieval_without_prior_request():
    """ToolPreparer→Planner handoff: empty prev_request must not wipe RETRIEVED."""
    from CoScientist.experiments.context import (
        RETRIEVED_CAPABILITIES_KEY,
        build_experiment_context,
    )

    full = {
        "tool": "generate_case_mols",
        "server_id": "d36e3d994404e957",
        "description": "Generate case molecules.",
        "input_schema": {"type": "object", "properties": {"case": {}}},
        "score": 0.8,
    }
    kept = {
        "tool": "fetch_activity_data",
        "server_id": "bfd3f80438ba403b",
        "description": "Fetch activity.",
        "input_schema": {"type": "object"},
        "score": 0.9,
    }
    # After ToolRetriever+Reranker: full set stashed, accumulated cleared, keep-set in filtered.
    state: dict = {
        "accumulated_tools": [],
        "filtered_tools": [kept],
        RETRIEVED_CAPABILITIES_KEY: [full, kept],
        # No experiment_source_request yet — first planner entry.
    }
    user = SimpleNamespace(parts=[SimpleNamespace(text="Generate GSK-3beta inhibitors with high activit")])
    ctx = SimpleNamespace(state=state, user_content=user)
    build_experiment_context(ctx)
    tools = {c["tool"] for c in state["experiment_context"]["available_mcp_capabilities"]}
    assert tools == {"generate_case_mols", "fetch_activity_data"}
    assert state[RETRIEVED_CAPABILITIES_KEY][0]["tool"] == "generate_case_mols"
    assert "generate_case_mols" in state["experiment_planner_context"]


def test_stash_before_rerank_clear_preserves_full_retrieval():
    from CoScientist.experiments.context import (
        RETRIEVED_CAPABILITIES_KEY,
        stash_experiment_retrieved_capabilities,
    )

    tools = [
        {
            "tool": "generate_case_mols",
            "server_id": "d36e3d994404e957",
            "description": "gen",
            "input_schema": {},
        },
        {
            "tool": "calculate_docking",
            "server_id": "bfd3f80438ba403b",
            "description": "dock",
            "input_schema": {},
        },
    ]
    state: dict = {"accumulated_tools": tools}
    ctx = SimpleNamespace(state=state, user_content=None)
    stash_experiment_retrieved_capabilities(ctx)
    state["accumulated_tools"] = []  # rerank clear
    state["filtered_tools"] = [tools[0]]
    assert {c["tool"] for c in state[RETRIEVED_CAPABILITIES_KEY]} == {
        "generate_case_mols",
        "calculate_docking",
    }


def test_design_field_coercions_are_llm_tolerant():
    from CoScientist.experiments.schemas import TaskDesign

    design = TaskDesign.model_validate(
        {
            "hypothesis_ref": ["H2", "H3"],
            "experiment_question": "Is SA related to affinity?",
            "dataset": {
                "name": "pubchem",
                "ref": "https://pubchem.ncbi.nlm.nih.gov/",
                "notes": None,
            },
            "baselines": [{"name": "null model", "kind": "method", "ref": None}],
            "metrics": [{"name": "corr", "direction": "compare", "threshold": None, "test": "spearman"}],
            "analysis_artifacts": [
                {"name": "plot.png", "role": "plot", "prepare_via": "coder", "path_or_tool": "plot.png"},
                {"name": "table.csv", "role": "data", "prepare_via": "coder", "path_or_tool": "table.csv"},
            ],
        }
    )
    assert design.hypothesis_ref == "H2"
    assert design.also_tests == ["H3"]
    assert design.covered_hypothesis_ids() == {"H2", "H3"}
    assert design.dataset.ref is not None
    assert design.dataset.ref.kind == "url"
    assert design.analysis_artifacts[0].role == "report"
    assert design.analysis_artifacts[1].role == "metrics_table"

    messy = TaskDesign.model_validate(
        {
            "hypothesis_ref": "H1",
            "experiment_question": "Q",
            "dataset": {
                "name": "x",
                "ref": {
                    "data_id": "pubchem_source",
                    "kind": "external",
                    "source": "url",
                    "path_or_uri": "https://pubchem.ncbi.nlm.nih.gov/",
                },
            },
            "baselines": [{"name": "b", "kind": "method"}],
            "metrics": [{"name": "m", "direction": "maximize"}],
            "analysis_artifacts": [{"name": "a.py", "role": "code", "prepare_via": "coder"}],
        }
    )
    assert messy.dataset.ref is not None
    assert messy.dataset.ref.kind == "url"


def test_task_design_is_required_on_experiment_plan_1_0():
    task = _task("EXP-1")
    del task["design"]
    with pytest.raises(ValidationError):
        ExperimentPlan.model_validate(
            {
                **_plan(_task("EXP-1")).model_dump(mode="json"),
                "tasks": [task],
            }
        )


def test_critique_requires_inventory_tools_for_requested_capabilities():
    """Option A: unused thematic inventory is minor/advisory, not a revise blocker."""
    plan = ExperimentPlan.model_validate(
        {
            **_plan(_task("EXP-1", route="coder")).model_dump(mode="json"),
            "source_request": (
                "Generate candidates, dock them to the target (docking score), "
                "and filter by predicted toxicity endpoints."
            ),
        }
    )
    inventory = [
        {
            "tool": "calculate_docking",
            "server_id": "srv-dock",
            "description": "Calculate docking score for a molecule given SMILES and pdb_id.",
        },
        {
            "tool": "predict_general_toxicity",
            "server_id": "srv-tox",
            "description": "Hepatotoxicity / DILI / cardiotoxicity / carcinogenicity profile.",
        },
        {
            "tool": "reproduce_figure8_examples",
            "server_id": "srv-paper",
            "description": "Inverse-docking profiles for the six characterised paper molecules.",
        },
    ]
    critique = critique_plan(
        plan,
        settings=ExperimentsSettings(),
        available_tools=inventory,
        preferred_tools=inventory,
        hypothesis_refs=[{"hypothesis_id": "H1", "statement": "Fixture"}],
    )
    # Thematic unused-family hints must not alone force revise (inventory ≠ checklist).
    assert critique.verdict == "approve"
    assert not any(issue.severity in {"blocker", "major"} for issue in critique.issues)
    unused_minors = [
        i for i in critique.issues if i.severity == "minor" and "inventory tools cover" in i.message
    ]
    assert unused_minors
    messages = " ".join(i.message for i in unused_minors)
    assert "calculate_docking" in messages
    assert "predict_general_toxicity" in messages
    # Paper-demo tool should not be forced merely because docking is requested.
    assert "reproduce_figure8_examples" not in messages

    # Named explicit tool requirements still block approve (major).
    named = ExperimentPlan.model_validate(
        {
            **_plan(_task("EXP-1", route="coder")).model_dump(mode="json"),
            "source_request": (
                "Use calculate_docking on the candidates, then call predict_general_toxicity."
            ),
        }
    )
    named_critique = critique_plan(
        named,
        settings=ExperimentsSettings(),
        available_tools=inventory,
        preferred_tools=inventory,
        hypothesis_refs=[{"hypothesis_id": "H1", "statement": "Fixture"}],
    )
    assert named_critique.verdict == "revise"
    assert any(
        i.severity == "major" and "explicitly requires" in i.message for i in named_critique.issues
    )

    fixed_task = _task("EXP-2", route="fedot_mas", tool="calculate_docking", hypothesis_ref="H1")
    fixed_task["mcp_servers"][0]["server_id"] = "srv-dock"
    fixed_task["mcp_servers"][0]["tools"][0] = {
        "name": "calculate_docking",
        "description": "Calculate docking score for a molecule given SMILES and pdb_id.",
        "input_schema": {},
    }
    tox = _task("EXP-3", route="fedot_mas", tool="predict_general_toxicity", hypothesis_ref="H1")
    tox["mcp_servers"][0]["server_id"] = "srv-tox"
    tox["mcp_servers"][0]["tools"][0] = {
        "name": "predict_general_toxicity",
        "description": "Hepatotoxicity / DILI / cardiotoxicity / carcinogenicity profile.",
        "input_schema": {},
    }
    ok = ExperimentPlan.model_validate(
        {
            **_plan(_task("EXP-1", route="coder"), fixed_task, tox).model_dump(mode="json"),
            "source_request": (
                "Generate candidates, dock them to the target (docking score), "
                "and filter by predicted toxicity endpoints."
            ),
        }
    )
    approved = critique_plan(
        ok,
        settings=ExperimentsSettings(),
        available_tools=inventory,
        preferred_tools=inventory,
        hypothesis_refs=[{"hypothesis_id": "H1", "statement": "Fixture"}],
    )
    assert approved.verdict == "approve"
    assert not any("inventory tools cover" in i.message for i in approved.issues)


def test_experiment_task_does_not_default_missing_route_to_coder():
    from pydantic import ValidationError
    from CoScientist.experiments.schemas.models import ExperimentTask

    with pytest.raises(ValidationError):
        ExperimentTask.model_validate(
            {
                "id": "EXP-1",
                "name": "Data collection",
                "description": "Collect activity data",
                "rationale": "Need dataset",
                "design": {
                    "hypothesis_ref": "H1",
                    "experiment_question": "What activity data exists?",
                    "dataset": {"name": "activity"},
                    "baselines": [{"name": "none", "kind": "method"}],
                    "metrics": [{"name": "n", "direction": "maximize"}],
                    "analysis_artifacts": [
                        {"name": "a.py", "role": "code", "prepare_via": "coder"}
                    ],
                },
                "success_criteria": [
                    {
                        "criterion_id": "C1",
                        "description": "csv exists",
                        "kind": "artifact_exists",
                        "verification": "file",
                    }
                ],
                "expected_artifacts": [
                    {"name": "data.csv", "role": "data", "description": "table"}
                ],
                "est_duration_min": 20,
            }
        )


def test_extract_hypothesis_refs_from_neutral_multih_text():
    from CoScientist.experiments.context import extract_hypothesis_refs

    text = """
    Research ask.
    Hypotheses:
    • H1. Feature A anti-correlates with feature B under joint filters.
    • H2. Metric M decreases as complexity rises.
    H3: Protocol noise dominates sample-size gains.
    H4 - Ortholog scores are predictable from the primary target.
    H5. First cleaning pass captures most of the lift.
    """
    refs = extract_hypothesis_refs(text)
    assert [r["hypothesis_id"] for r in refs] == ["H1", "H2", "H3", "H4", "H5"]
    assert "anti-correlates" in refs[0]["statement"]
    assert "Ortholog" in refs[3]["statement"]

    merged = extract_hypothesis_refs(
        "No labels here.",
        legacy_hypotheses=[{"id": "H9", "statement": "Legacy only claim."}],
    )
    assert merged == [{"hypothesis_id": "H9", "statement": "Legacy only claim."}]


def test_extract_hypothesis_refs_from_system_hypothesis_n_prose():
    from CoScientist.experiments.context import extract_hypothesis_refs
    from CoScientist.experiments.hypotheses import commit_experiment_hypotheses
    from types import SimpleNamespace

    text = """
I have formulated one hypothesis for each.

**Hypothesis 1 (Parkinson's):**
*Statement:* Generative models can produce 10 novel BBB-permeable molecules.
*VerificationMethod:* Execute generate_case_mols with case=parkinson.

**Hypothesis 2 (Dyslipidemia):**
*Statement:* Case-conditioned generation yields molecules with bioavailability >= 80%.
*VerificationMethod:* generate_case_mols case=dyslipidemia.

Hypothesis 3 (KRAS):
Statement: Docking of candidate KRAS G12C inhibitors produces scored poses for ranking.
VerificationMethod: calculate_docking upload to S3.
"""
    refs = extract_hypothesis_refs(text)
    assert [r["hypothesis_id"] for r in refs] == ["H1", "H2", "H3"]
    assert "BBB-permeable" in refs[0]["statement"]
    assert "bioavailability" in refs[1]["statement"]
    assert "Docking" in refs[2]["statement"]

    state = {"hypotheses": text}
    commit_experiment_hypotheses(SimpleNamespace(state=state, user_content=None))
    assert [r["hypothesis_id"] for r in state["hypothesis_refs"]] == ["H1", "H2", "H3"]


def test_critique_requires_hypothesis_coverage_and_blocks_empty_inventory_mcp():
    coder_task = _task("EXP-1", route="coder", hypothesis_ref="H1")
    plan = _plan(
        coder_task,
        hypotheses=[
            {"hypothesis_id": "H1", "statement": "Claim one."},
            {"hypothesis_id": "H2", "statement": "Claim two."},
        ],
    )
    uncovered = critique_plan(
        plan,
        settings=ExperimentsSettings(),
        available_tools=[],
        hypothesis_refs=[
            {"hypothesis_id": "H1", "statement": "Claim one."},
            {"hypothesis_id": "H2", "statement": "Claim two."},
        ],
    )
    assert uncovered.verdict == "revise"
    assert any("H2" in issue.message for issue in uncovered.issues)

    mcp_no_inventory = _plan(_task("EXP-1", route="fedot_mas"))
    blocked = critique_plan(
        mcp_no_inventory,
        settings=ExperimentsSettings(),
        available_tools=[],
        hypothesis_refs=[{"hypothesis_id": "H1", "statement": "Fixture"}],
    )
    assert blocked.verdict == "revise"
    assert any("inventory is empty" in issue.message for issue in blocked.issues)

    ok = critique_plan(
        _plan(_task("EXP-1", route="coder")),
        settings=ExperimentsSettings(),
        available_tools=[],
        hypothesis_refs=[{"hypothesis_id": "H1", "statement": "Fixture"}],
    )
    assert ok.verdict == "approve"


def test_render_experiment_plan_includes_design_matrix():
    from CoScientist.experiments.review import render_experiment_plan

    text = render_experiment_plan(_plan(_task("EXP-1", route="coder")))
    assert "Design matrix" in text
    assert "`H1`" in text
    assert "Baselines:" in text
    assert "Metrics:" in text
    assert "Analysis artifacts:" in text
    assert "`coder`" in text


def test_scientific_check_is_optional_on_task_result():
    from CoScientist.experiments.schemas import TaskResult

    base = {
        "schema_version": "task-result/0.1",
        "result_id": "RES-1",
        "plan_id": "PLAN-acceptance",
        "task_id": "EXP-1",
        "attempt_id": "ATT-1",
        "attempt_no": 1,
        "status": "success",
        "planned_route": "coder",
        "route_used": "coder",
        "started_at": datetime(2026, 7, 31, 18, 0, tzinfo=timezone.utc),
        "finished_at": datetime(2026, 7, 31, 18, 1, tzinfo=timezone.utc),
        "summary": "ok",
        "criteria_checks": [
            {
                "criterion_id": "EXP-1-C1",
                "passed": True,
                "details": "ok",
            }
        ],
    }
    plain = TaskResult.model_validate(base)
    assert plain.scientific_check is None
    with_check = TaskResult.model_validate(
        {
            **base,
            "scientific_check": {
                "hypothesis_ref": "H1",
                "status": "inconclusive",
                "details": "Need more samples.",
            },
        }
    )
    assert with_check.scientific_check is not None
    assert with_check.scientific_check.status == "inconclusive"


def _alembic_task(task_id: str = "EXP-1", *, hypothesis_ref: str = "H1") -> dict:
    task = _task(task_id, route="coder", hypothesis_ref=hypothesis_ref)
    task.update(
        {
            "route": "alembic_build",
            "repo_url": "https://github.com/whitead/synspace",
            "post_build_route": "react_tools",
            "mcp_servers": [],
            "expected_artifacts": [
                {
                    "name": "mcp_endpoint",
                    "role": "mcp_server",
                    "description": "Served Alembic MCP URL",
                    "required": True,
                }
            ],
            "success_criteria": [
                {
                    "criterion_id": f"{task_id}-C1",
                    "description": "Alembic build exposes an MCP endpoint.",
                    "kind": "execution",
                    "verification": "outputs.mcp_url is an http(s) URL.",
                }
            ],
        }
    )
    return task


def test_alembic_task_requires_repo_and_post_build_route():
    bare = _alembic_task()
    del bare["repo_url"]
    with pytest.raises(ValidationError):
        ExperimentTask.model_validate(bare)
    missing_post = _alembic_task()
    missing_post["post_build_route"] = None
    with pytest.raises(ValidationError):
        ExperimentTask.model_validate(missing_post)
    with pytest.raises(ValidationError):
        ExperimentTask.model_validate(
            {**_task("EXP-1", route="coder"), "post_build_route": "fedot_mas"}
        )


def test_start_task_rejects_alembic_when_route_disabled():
    plan = _plan(_alembic_task())
    critique = critique_plan(
        plan,
        settings=ExperimentsSettings(route_alembic=False),
        available_tools=_inventory(),
        hypothesis_refs=[{"hypothesis_id": "H1", "statement": "Fixture"}],
    )
    assert critique.verdict == "revise"
    assert any("alembic_build" in i.message for i in critique.issues)

    # Bypass critique to assert runtime gate.
    state: dict = {}
    initialize_runtime(
        state,
        plan,
        critique={"verdict": "approve", "issues": [], "summary": "forced"},
    )
    approve_plan(state)
    with pytest.raises(ExperimentRuntimeError) as exc:
        start_task(state, "EXP-1", settings=ExperimentsSettings(route_alembic=False))
    assert exc.value.code == "route_disabled"


def test_alembic_success_reopens_task_on_post_build_route():
    plan = _plan(_alembic_task())
    state: dict = {}
    initialize_runtime(
        state,
        plan,
        critique={"verdict": "approve", "issues": [], "summary": "forced"},
    )
    approve_plan(state)
    settings = ExperimentsSettings(route_alembic=True)
    started = start_task(state, "EXP-1", settings=settings)
    assert started["route_agent"] == "McpBuilderAgent"
    mark_route_returned(state, "McpBuilderAgent")
    recorded = record_result(
        state,
        "EXP-1",
        started["attempt_id"],
        {
            "status": "success",
            "summary": "Built MCP",
            "outputs": {
                "mcp_url": "http://127.0.0.1:9000/mcp",
                "mcp_endpoint": "http://127.0.0.1:9000/mcp",
                "tools": ["synspace_score"],
            },
            "criteria_checks": [
                {
                    "criterion_id": "EXP-1-C1",
                    "passed": True,
                    "details": "mcp_url present",
                }
            ],
        },
        settings=settings,
    )
    assert recorded["status"] == "success"
    assert recorded["post_build"]["post_build_route"] == "react_tools"
    runtime = state["experiment_runtime"]
    task_runtime = runtime["tasks"]["EXP-1"]
    assert task_runtime["status"] == "ready"
    assert task_runtime["current_route"] == "react_tools"
    assert task_runtime["task"]["mcp_servers"][0]["source"] == "alembic"
    assert task_runtime["task"]["mcp_servers"][0]["url"] == "http://127.0.0.1:9000/mcp"
    assert task_runtime["task"]["mcp_servers"][0]["tools"][0]["name"] == "synspace_score"
    assert state["deployed_mcps"][0]["url"] == "http://127.0.0.1:9000/mcp"

    second = start_task(state, "EXP-1", settings=settings)
    assert second["route_agent"] == "ExperimentAgent"
    assert second["route"] == "react_tools"


def test_alembic_success_defers_scientific_evidence_to_post_build():
    """One task may list MCP + science artifacts; build attempt only owes MCP."""
    task = _alembic_task()
    task["expected_artifacts"] = [
        {
            "name": "mcp_endpoint",
            "role": "mcp_server",
            "description": "Served Alembic MCP URL",
            "required": True,
        },
        {
            "name": "candidates.csv",
            "role": "data",
            "media_type": "text/csv",
            "description": "Scientific table",
            "required": True,
        },
    ]
    task["success_criteria"] = [
        {
            "criterion_id": "EXP-1-C1",
            "description": "MCP URL ready",
            "kind": "execution",
            "verification": "outputs.mcp_url is an http(s) URL.",
            "required": True,
        },
        {
            "criterion_id": "EXP-1-C2",
            "description": "candidates.csv exists",
            "kind": "artifact_exists",
            "verification": "Confirm output file presence",
            "required": True,
        },
    ]
    plan = _plan(task)
    state: dict = {}
    initialize_runtime(
        state,
        plan,
        critique={"verdict": "approve", "issues": [], "summary": "forced"},
    )
    approve_plan(state)
    settings = ExperimentsSettings(route_alembic=True)
    started = start_task(state, "EXP-1", settings=settings)
    mark_route_returned(state, "McpBuilderAgent")
    recorded = record_result(
        state,
        "EXP-1",
        started["attempt_id"],
        {
            "status": "success",
            "summary": "Built MCP",
            "outputs": {
                "mcp_url": "http://127.0.0.1:9000/mcp",
                "mcp_endpoint": "http://127.0.0.1:9000/mcp",
            },
            "criteria_checks": [
                {
                    "criterion_id": "EXP-1-C1",
                    "passed": True,
                    "details": "mcp_url present",
                }
            ],
        },
        settings=settings,
    )
    assert recorded["status"] == "success"
    assert recorded["post_build"]["post_build_route"] == "react_tools"
    assert state["experiment_runtime"]["tasks"]["EXP-1"]["status"] == "ready"


def test_critique_approves_alembic_when_enabled_with_repo_and_post_build():
    plan = _plan(_alembic_task())
    critique = critique_plan(
        plan,
        settings=ExperimentsSettings(route_alembic=True),
        available_tools=_inventory(),
        hypothesis_refs=[{"hypothesis_id": "H1", "statement": "Fixture"}],
        repo_candidates=[{"url": "https://github.com/whitead/synspace", "repo_name": "synspace"}],
    )
    assert critique.verdict == "approve"


def test_critique_blocks_alembic_repo_not_in_candidates():
    plan = _plan(_alembic_task())
    critique = critique_plan(
        plan,
        settings=ExperimentsSettings(route_alembic=True),
        available_tools=_inventory(),
        hypothesis_refs=[{"hypothesis_id": "H1", "statement": "Fixture"}],
        repo_candidates=[{"url": "https://github.com/other/unrelated"}],
    )
    assert critique.verdict == "revise"
    assert any("not in" in i.message and "repo_candidates" in i.message for i in critique.issues)


def test_critique_blocks_alembic_with_premature_mcp_servers():
    task = _alembic_task()
    task["mcp_servers"] = [
        {
            "name": "synspace",
            "server_id": "synspace",
            "url": "http://example.invalid/mcp",
            "source": "alembic",
            "tools": [{"name": "generate", "description": "x"}],
        }
    ]
    plan = _plan(task)
    critique = critique_plan(
        plan,
        settings=ExperimentsSettings(route_alembic=True),
        available_tools=_inventory(),
        hypothesis_refs=[{"hypothesis_id": "H1", "statement": "Fixture"}],
        repo_candidates=[{"url": "https://github.com/whitead/synspace"}],
    )
    assert critique.verdict == "revise"
    assert any("mcp_servers empty" in i.message for i in critique.issues)


def test_extract_repo_candidates_from_ask():
    from CoScientist.experiments.context import extract_repo_candidates

    refs = extract_repo_candidates(
        "Use https://github.com/whitead/synspace and also "
        "https://github.com/encode/httpx for nothing."
    )
    urls = [r["url"] for r in refs]
    assert "https://github.com/whitead/synspace" in urls
    assert "https://github.com/encode/httpx" in urls
    assert refs[0]["repo_name"] == "synspace"



def test_mcp_server_ref_coerces_singular_tool_field():
    from CoScientist.experiments.schemas.models import MCPServerRef

    server = MCPServerRef.model_validate(
        {
            "source": "registry",
            "server_id": "srv-x",
            "tool": "calculate_docking",
            "health": "unknown",
        }
    )
    assert server.tools[0].name == "calculate_docking"


def test_mcp_server_ref_coerces_composite_server_id_tool_key():
    from CoScientist.experiments.schemas.models import MCPServerRef

    server = MCPServerRef.model_validate(
        {
            "source": "registry",
            "name": "molgen",
            "server_id": "d36e3d994404e957/generate_case_mols",
            "tools": ["generate_case_mols"],
        }
    )
    assert server.server_id == "d36e3d994404e957"
    assert server.tools[0].name == "generate_case_mols"


def test_baseline_kind_aliases_and_task_shape_coercion():
    from CoScientist.experiments.schemas.models import ExperimentTask

    task = ExperimentTask.model_validate(
        {
            "id": "TASK-1",
            "name": "PubChem fetch",
            "route": "coder",
            "design": {
                "hypothesis_ref": "H3",
                "experiment_question": "Is PubChem activity data heterogeneous?",
                "dataset": {"name": "PubChem bioassay"},
                "baselines": [{"name": "literature review", "kind": "report"}],
                "metrics": [{"name": "n_compounds", "direction": "maximize"}],
                "analysis_artifacts": [
                    {"name": "fetch.py", "role": "code", "prepare_via": "coder"}
                ],
                "est_duration_min": 45,
            },
        }
    )
    assert task.id == "EXP-1"
    assert task.description == "PubChem fetch"
    assert task.est_duration_min == 45
    assert task.design.baselines[0].kind == "prior_result"
    assert task.success_criteria[0].kind == "artifact_exists"
    assert task.expected_artifacts


def test_plan_methods_filled_when_empty():
    from datetime import datetime, timezone
    from CoScientist.experiments.schemas.models import ExperimentPlan

    plan = ExperimentPlan.model_validate(
        {
            "schema_version": "experiment-plan/1.0",
            "plan_id": "p1",
            "experiment_run_id": "EXRUN-1",
            "revision": 1,
            "source_request": "do 4.2",
            "goal": "multitarget design",
            "methods": [],
            "context_digest": "digest",
            "tasks": [
                {
                    "id": "EXP-1",
                    "name": "Docking filter",
                    "description": "Filter by docking",
                    "rationale": "Need affinity",
                    "route": "coder",
                    "design": {
                        "hypothesis_ref": "H1",
                        "experiment_question": "Docking threshold?",
                        "dataset": {"name": "candidates"},
                        "baselines": [{"name": "random", "kind": "method"}],
                        "metrics": [{"name": "hit_rate", "direction": "maximize"}],
                        "analysis_artifacts": [
                            {"name": "dock.py", "role": "code", "prepare_via": "coder"}
                        ],
                    },
                    "success_criteria": [
                        {
                            "criterion_id": "C1",
                            "description": "report exists",
                            "kind": "artifact_exists",
                            "verification": "file present",
                        }
                    ],
                    "expected_artifacts": [
                        {
                            "name": "dock_report",
                            "role": "report",
                            "description": "docking report",
                        }
                    ],
                    "est_duration_min": 20,
                }
            ],
            "total_est_duration_min": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    assert plan.methods
    assert plan.total_est_duration_min == 20


def test_lenient_planner_preserves_unspecified_for_critique():
    """Strict mode keeps unspecified* so completeness majors can fire."""
    from CoScientist.experiments.schemas.models import (
        ExperimentTask,
        reset_lenient_planner,
        set_lenient_planner,
    )

    payload = {
        "id": "EXP-1",
        "name": "Tox screen",
        "route": "coder",
        "design": {
            "hypothesis_ref": "H1",
            "experiment_question": "Is toxicity acceptable?",
            "dataset": {"name": "panel"},
            "baselines": [{"name": "unspecified", "kind": "method"}],
            "metrics": [{"name": "unspecified_metric", "direction": "compare"}],
            "analysis_artifacts": [
                {"name": "tox.py", "role": "code", "prepare_via": "coder"}
            ],
        },
        "success_criteria": [
            {
                "criterion_id": "C1",
                "description": "report exists",
                "kind": "artifact_exists",
                "verification": "file present",
            }
        ],
        "expected_artifacts": [
            {"name": "tox_report", "role": "report", "description": "tox report"}
        ],
        "est_duration_min": 10,
    }

    token = set_lenient_planner(True)
    try:
        invent = ExperimentTask.model_validate(payload)
        assert invent.design.baselines[0].name == "comparative reference method"
        assert invent.design.metrics[0].name == "primary_outcome"
    finally:
        reset_lenient_planner(token)

    token = set_lenient_planner(False)
    try:
        strict = ExperimentTask.model_validate(payload)
        assert strict.design.baselines[0].name.startswith("unspecified")
        assert strict.design.metrics[0].name.startswith("unspecified")
        plan = _plan(
            {
                **_task("EXP-1", route="coder"),
                "design": strict.design.model_dump(mode="json"),
            }
        )
    finally:
        reset_lenient_planner(token)

    critique = critique_plan(
        plan,
        settings=ExperimentsSettings(lenient_planner=False),
        available_tools=_inventory(),
        hypothesis_refs=[{"hypothesis_id": "H1", "statement": "Fixture"}],
    )
    assert critique.verdict == "revise"
    assert any("unspecified" in i.message for i in critique.issues)


def test_commit_experiment_hypotheses_normalizes_and_fallbacks():
    from types import SimpleNamespace

    from CoScientist.experiments.hypotheses import commit_experiment_hypotheses

    state: dict = {
        "hypotheses": json.dumps(
            [
                {"hypothesis_id": "H1", "statement": "KRAS binders exist."},
                {"hypothesis_id": "H2", "statement": "BTK modulators exist."},
            ]
        )
    }
    commit_experiment_hypotheses(SimpleNamespace(state=state, user_content=None))
    assert [r["hypothesis_id"] for r in state["hypothesis_refs"]] == ["H1", "H2"]
    assert state["hypotheses"] == state["experiment_hypotheses"] == state["hypothesis_refs"]

    empty: dict = {"hypotheses": "not-json", "experiment_source_request": "Make BBB antioxidant."}
    commit_experiment_hypotheses(SimpleNamespace(state=empty, user_content=None))
    assert empty["hypothesis_refs"][0]["hypothesis_id"] == "H1"
    assert "BBB antioxidant" in empty["hypothesis_refs"][0]["statement"]


def test_persist_and_seed_hypotheses_use_em_ask_not_tool_prep_noise():
    from types import SimpleNamespace

    from google.adk.models import LlmRequest
    from google.genai import types

    from CoScientist.experiments.hypotheses import (
        persist_experiment_em_request,
        seed_hypotheses_from_em_request,
    )

    ask = (
        "Complete as ONE stage:\n"
        "1. BTK non-covalent modulators for MS\n"
        "2. KRAS G12C inhibitors for lung cancer"
    )
    state: dict = {}
    user = types.Content(role="user", parts=[types.Part(text=ask)])
    persist_experiment_em_request(SimpleNamespace(state=state, user_content=user))
    assert state["experiment_source_request"] == ask

    # Noise must not overwrite a good ask.
    noise = types.Content(
        role="user",
        parts=[types.Part(text='{"mcp_scores":[{"index":0,"score":true}]}')],
    )
    persist_experiment_em_request(SimpleNamespace(state=state, user_content=noise))
    assert state["experiment_source_request"] == ask

    req = LlmRequest(
        contents=[
            types.Content(
                role="user",
                parts=[types.Part(text="[FullSetToolReranker] tools sufficient")],
            )
        ]
    )
    seed_hypotheses_from_em_request(SimpleNamespace(state=state, user_content=noise), req)
    blob = req.contents[0].parts[0].text
    assert "BTK" in blob and "KRAS" in blob
    assert "FullSetToolReranker" not in blob
    assert "mcp_scores" not in blob
    assert state["_em_hypotheses_seeded"] is True

    # Second before_model must not wipe tool-turn history.
    req2 = LlmRequest(
        contents=[
            types.Content(role="user", parts=[types.Part(text="ASK:\nkept")]),
            types.Content(role="model", parts=[types.Part(text="overview done")]),
        ]
    )
    seed_hypotheses_from_em_request(SimpleNamespace(state=state, user_content=noise), req2)
    assert len(req2.contents) == 2
    assert req2.contents[1].parts[0].text == "overview done"


def test_commit_experiment_hypotheses_prefers_graph_style_nodes():
    from types import SimpleNamespace

    from CoScientist.experiments.hypotheses import commit_experiment_hypotheses

    raw = json.dumps(
        {
            "nodes": [
                {
                    "type": "Hypothesis",
                    "ref": "h_pd",
                    "attrs": {"formulation": "PD dopamine modulators exist."},
                },
                {
                    "type": "Hypothesis",
                    "ref": "h_lipid",
                    "attrs": {"formulation": "Lipid clearance molecules exist."},
                },
                {"type": "VerificationMethod", "ref": "vm1", "attrs": {"method_type": "computational"}},
            ]
        }
    )
    state = {"hypotheses": raw}
    commit_experiment_hypotheses(SimpleNamespace(state=state, user_content=None))
    assert [r["hypothesis_id"] for r in state["hypothesis_refs"]] == ["H1", "H2"]
    assert "dopamine" in state["hypothesis_refs"][0]["statement"]
    assert "Lipid" in state["hypothesis_refs"][1]["statement"]


def test_critique_blocks_invented_hypothesis_beyond_refs():
    task = _task("EXP-1", hypothesis_ref="H1")
    plan = _plan(
        task,
        hypotheses=[
            {"hypothesis_id": "H1", "statement": "Fixture H1."},
            {"hypothesis_id": "H9", "statement": "Invented extra."},
        ],
    )
    # H9 is in plan but not covered — also invent beyond refs
    critique = critique_plan(
        plan,
        settings=ExperimentsSettings(),
        available_tools=_inventory(),
        hypothesis_refs=[{"hypothesis_id": "H1", "statement": "Fixture H1."}],
    )
    assert critique.verdict == "revise"
    assert any("invents ids" in i.message for i in critique.issues)


def test_resolve_fallback_chains_from_settings():
    from CoScientist.experiments.runtime.state_machine import (
        FALLBACK_CHAINS,
        resolve_fallback_chains,
    )

    default = resolve_fallback_chains(ExperimentsSettings())
    assert default["fedot_mas"] == FALLBACK_CHAINS["fedot_mas"] == [
        "fedot_mas",
        "react_tools",
        "coder",
    ]
    custom = resolve_fallback_chains(
        ExperimentsSettings(fallback_fedot_mas=["fedot_mas", "coder"])
    )
    assert custom["fedot_mas"] == ["fedot_mas", "coder"]


def test_critique_blocks_placeholder_urls_in_input_data():
    task = _task("EXP-1")
    task["input_data"] = [
        {
            "data_id": "ld50",
            "kind": "url",
            "description": "Public LD50 table",
            "url": "https://example.com/public_ld50_data.csv",
            "required": True,
        }
    ]
    plan = _plan(task)
    critique = critique_plan(
        plan,
        settings=ExperimentsSettings(),
        available_tools=_inventory(),
    )
    assert critique.verdict == "revise"
    assert any(
        i.severity == "blocker" and "placeholder" in i.message.lower()
        for i in critique.issues
    )


def test_critique_blocks_fake_s3_artifacts_in_dataset_notes():
    task = _task("EXP-1", route="coder")
    task["design"] = _design("H1")
    task["design"]["dataset"]["notes"] = "Load from s3://artifacts/fake_run/data.csv"
    plan = _plan(task)
    critique = critique_plan(
        plan,
        settings=ExperimentsSettings(),
        available_tools=_inventory(),
    )
    assert critique.verdict == "revise"
    assert any("s3://artifacts" in i.message for i in critique.issues)


def test_render_experiment_results_prefers_http_then_s3_and_sets_manifest():
    from CoScientist.experiments.review import (
        build_experiment_artifacts_manifest,
        render_experiment_results,
    )

    state = {
        "experiment_task_results": [
            {
                "task_id": "EXP-1",
                "status": "done",
                "summary": "ok",
                "route_used": "fedot_mas",
                "artifacts": [
                    {
                        "artifact_id": "ART-1",
                        "name": "candidates.csv",
                        "bucket": None,
                        "s3_key": None,
                        "workspace_path": None,
                        "external_url": "https://storage.example-cdn.test/runs/a/candidates.csv",
                        "media_type": "text/csv",
                    },
                    {
                        "artifact_id": "ART-2",
                        "name": "metrics.json",
                        "bucket": "bkt",
                        "s3_key": "runs/a/metrics.json",
                        "workspace_path": None,
                        "external_url": None,
                        "media_type": "application/json",
                    },
                ],
            }
        ]
    }
    text = render_experiment_results(state)
    assert "Canonical artifact locations" in text
    assert "https://storage.example-cdn.test/runs/a/candidates.csv" in text
    assert "s3://bkt/runs/a/metrics.json" in text
    assert "S3://artifacts" not in text
    manifest = state["experiment_artifacts_manifest"]
    assert len(manifest) == 2
    assert manifest == build_experiment_artifacts_manifest(state)
    assert manifest[1]["location"] == "s3://bkt/runs/a/metrics.json"


class _FakeResearchGraph:
    """Minimal stand-in for the research graph store used by hypotheses.py."""

    def __init__(self, hypothesis_statements: list[str]):
        self._nodes = [
            (f"H{i}", {"type": "Hypothesis", "attrs": {"formulation": s}})
            for i, s in enumerate(hypothesis_statements, start=1)
        ]

    def nodes(self, data: bool = True):
        return list(self._nodes)


def _patch_research_graph(monkeypatch, statements: list[str]) -> None:
    import CoScientist.graph.research.store as research_store

    graph = _FakeResearchGraph(statements)
    monkeypatch.setattr(
        research_store,
        "get_research_graph",
        lambda ctx: SimpleNamespace(full_graph=lambda: graph),
    )


def test_commit_experiment_hypotheses_source_priority(monkeypatch):
    """Max-by-length among non-empty sources; length beats source reliability.

    Restores the pre-Task-7 heuristic: the LONGEST candidate list wins
    (3 prose refs beat 1 graph node; 3 struct refs beat 1 FC ref). Empty
    sources are ignored; if all are empty, a separate test covers H1 fallback.
    """
    from CoScientist.experiments.hypotheses import commit_experiment_hypotheses

    # Case 1: prose regex parse (3 refs) beats the research graph (1 node).
    _patch_research_graph(monkeypatch, ["Graph-committed hypothesis."])
    prose = (
        "Hypothesis 1: Prose candidate one works.\n"
        "Hypothesis 2: Prose candidate two works.\n"
        "Hypothesis 3: Prose candidate three works.\n"
    )
    state: dict = {"hypotheses": prose}
    commit_experiment_hypotheses(SimpleNamespace(state=state, user_content=None))
    statements = [r["statement"] for r in state["hypothesis_refs"]]
    assert statements == [
        "Prose candidate one works.",
        "Prose candidate two works.",
        "Prose candidate three works.",
    ]

    # Case 2: structured output (3 refs) beats the FC stash (1 ref).
    _patch_research_graph(monkeypatch, [])  # graph empty for this case
    state2: dict = {
        "hypotheses": json.dumps(
            [
                {"hypothesis_id": "H1", "statement": "Struct one."},
                {"hypothesis_id": "H2", "statement": "Struct two."},
                {"hypothesis_id": "H3", "statement": "Struct three."},
            ]
        ),
        "_em_hypotheses_from_fc": [
            {"hypothesis_id": "H1", "statement": "FC-committed hypothesis."}
        ],
    }
    commit_experiment_hypotheses(SimpleNamespace(state=state2, user_content=None))
    statements2 = [r["statement"] for r in state2["hypothesis_refs"]]
    assert statements2 == ["Struct one.", "Struct two.", "Struct three."]


def test_commit_experiment_hypotheses_prefers_graph_over_text(monkeypatch):
    """When lengths tie at 1, max() keeps the first equal candidate (graph)."""
    from CoScientist.experiments.hypotheses import commit_experiment_hypotheses

    _patch_research_graph(monkeypatch, ["Only graph hypothesis."])
    prose = "Hypothesis 1: Prose filler number 1 works."
    state: dict = {"hypotheses": prose}
    commit_experiment_hypotheses(SimpleNamespace(state=state, user_content=None))
    refs = state["hypothesis_refs"]
    assert len(refs) == 1
    assert refs[0]["hypothesis_id"] == "H1"
    assert refs[0]["statement"] == "Only graph hypothesis."


def test_commit_experiment_hypotheses_prefers_fc_over_struct(monkeypatch):
    """Equal-length FC and struct: max() keeps FC (earlier in the candidate list)."""
    from CoScientist.experiments.hypotheses import commit_experiment_hypotheses

    _patch_research_graph(monkeypatch, [])
    state: dict = {
        "hypotheses": json.dumps(
            [
                {"hypothesis_id": "H1", "statement": "Struct alpha."},
                {"hypothesis_id": "H2", "statement": "Struct beta."},
            ]
        ),
        "_em_hypotheses_from_fc": [
            {"hypothesis_id": "H1", "statement": "FC alpha."},
            {"hypothesis_id": "H2", "statement": "FC beta."},
        ],
    }
    commit_experiment_hypotheses(SimpleNamespace(state=state, user_content=None))
    statements = [r["statement"] for r in state["hypothesis_refs"]]
    assert statements == ["FC alpha.", "FC beta."]


def test_commit_experiment_hypotheses_prefers_longer_text_over_graph(monkeypatch):
    """Explicit max-by-length: 5 prose refs beat 1 graph node."""
    from CoScientist.experiments.hypotheses import commit_experiment_hypotheses

    _patch_research_graph(monkeypatch, ["Only graph hypothesis."])
    prose = "\n".join(
        f"Hypothesis {i}: Prose filler number {i} works." for i in range(1, 6)
    )
    state: dict = {"hypotheses": prose}
    commit_experiment_hypotheses(SimpleNamespace(state=state, user_content=None))
    refs = state["hypothesis_refs"]
    assert len(refs) == 5
    assert refs[0]["statement"] == "Prose filler number 1 works."
    assert refs[-1]["statement"] == "Prose filler number 5 works."


def test_commit_experiment_hypotheses_all_sources_empty_falls_back_to_h1(monkeypatch):
    from CoScientist.experiments.hypotheses import commit_experiment_hypotheses

    _patch_research_graph(monkeypatch, [])
    state: dict = {
        "hypotheses": "no hypothesis markers here",
        "experiment_source_request": "Design a BBB-permeable antioxidant molecule.",
    }
    commit_experiment_hypotheses(SimpleNamespace(state=state, user_content=None))
    refs = state["hypothesis_refs"]
    assert len(refs) == 1
    assert refs[0]["hypothesis_id"] == "H1"
    assert "BBB-permeable antioxidant" in refs[0]["statement"]


class _FakeInitGraph:
    """Stand-in research graph store for bootstrap_research_question_if_empty."""

    def __init__(self, empty: bool):
        self._empty = empty
        self.init_calls: list[dict] = []

    def is_empty(self) -> bool:
        return self._empty

    def init_research(self, source: str, question: str):
        self.init_calls.append({"source": source, "question": question})
        return {"ok": True, "root_id": "Q1"}


def test_bootstrap_research_question_if_empty_seeds_root(monkeypatch):
    """B: deterministic root bootstrap — no longer relies on OrchestratorAgent
    catching every empty-graph turn before delegating to HypothesesAgent."""
    import CoScientist.graph.research.store as research_store

    from CoScientist.experiments.hypotheses import bootstrap_research_question_if_empty

    graph = _FakeInitGraph(empty=True)
    monkeypatch.setattr(research_store, "get_research_graph", lambda ctx: graph)
    state = {"experiment_source_request": "Design a BBB-permeable antioxidant molecule."}
    bootstrap_research_question_if_empty(SimpleNamespace(state=state, user_content=None))
    assert len(graph.init_calls) == 1
    assert graph.init_calls[0]["question"] == "Design a BBB-permeable antioxidant molecule."


def test_bootstrap_research_question_if_empty_noop_when_graph_has_content(monkeypatch):
    """Must never archive/replace an already-seeded graph (e.g. Orchestrator
    already called research_init this turn)."""
    import CoScientist.graph.research.store as research_store

    from CoScientist.experiments.hypotheses import bootstrap_research_question_if_empty

    graph = _FakeInitGraph(empty=False)
    monkeypatch.setattr(research_store, "get_research_graph", lambda ctx: graph)
    state = {"experiment_source_request": "Design a BBB-permeable antioxidant molecule."}
    bootstrap_research_question_if_empty(SimpleNamespace(state=state, user_content=None))
    assert graph.init_calls == []


def test_bootstrap_research_question_if_empty_survives_store_error(monkeypatch):
    """Bootstrap must never break the run — a broken store is a no-op, not a crash."""
    import CoScientist.graph.research.store as research_store

    from CoScientist.experiments.hypotheses import bootstrap_research_question_if_empty

    def _boom(ctx):
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(research_store, "get_research_graph", _boom)
    state = {"experiment_source_request": "Design a BBB-permeable antioxidant molecule."}
    bootstrap_research_question_if_empty(SimpleNamespace(state=state, user_content=None))  # no raise


def test_capture_hypotheses_after_research_commit_logs_failure(caplog):
    """D: a rejected research_commit (e.g. the ResearchQuestion-ACL error or the
    resulting malformed empty-commit retry) must leave an audit trail instead
    of silently falling through to the H1 fallback with no diagnosis."""
    from CoScientist.experiments.hypotheses import capture_hypotheses_after_research_commit

    state: dict = {}
    tool_response = {
        "ok": False,
        "message": "",
        "committed": {},
        "errors": ["empty commit — provide nodes, edges and/or status_updates"],
    }
    with caplog.at_level("WARNING", logger="CoScientist.experiments.hypotheses"):
        capture_hypotheses_after_research_commit(
            SimpleNamespace(name="research_commit"),
            {},
            SimpleNamespace(state=state),
            tool_response,
        )
    assert "EXPERIMENT_HYPOTHESES_COMMIT_FAILED" in caplog.text
    assert "empty commit" in caplog.text
    assert "_em_hypotheses_from_fc" not in state


def test_capture_hypotheses_after_research_commit_silent_on_non_hypothesis_ok_commit(caplog):
    """A successful commit with no Hypothesis nodes (e.g. VerificationMethod-only)
    is a normal case, not a failure — must not be logged as one."""
    from CoScientist.experiments.hypotheses import capture_hypotheses_after_research_commit

    state: dict = {}
    tool_response = {"ok": True, "committed": {"nodes": []}, "errors": []}
    with caplog.at_level("WARNING", logger="CoScientist.experiments.hypotheses"):
        capture_hypotheses_after_research_commit(
            SimpleNamespace(name="research_commit"),
            {"nodes": [{"type": "VerificationMethod", "ref": "vm0", "attrs": {}}]},
            SimpleNamespace(state=state),
            tool_response,
        )
    assert "EXPERIMENT_HYPOTHESES_COMMIT_FAILED" not in caplog.text
    assert state == {}
