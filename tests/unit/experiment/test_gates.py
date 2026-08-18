"""Orchestrator gates: coalesce, module-first, feasibility, retrieval budget."""
from __future__ import annotations

from types import SimpleNamespace

from .helpers import (
    _research_call_response,
)

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
    # Budget is 5 request-local retrieve_tools calls counted after the baseline.
    context.state["retrieval_queries"].extend(
        ["query one", "query two", "query three", "query four", "query five"]
    )
    stopped = enforce_experiment_retrieval_budget(context, repeated_call)

    assert stopped is not None
    assert "EXPERIMENT_RETRIEVAL_BUDGET_EXHAUSTED" in stopped.content.parts[0].text
    assert context.state["experiment_retrieval_budget_exhausted"] is True


def test_coalesce_merges_parallel_experiment_module_calls():
    from google.adk.models import LlmResponse
    from google.genai import types

    from CoScientist.experiments.runtime.coalesce import (
        coalesce_experiment_module_calls,
    )

    response = LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="ExperimentModuleAgent",
                        args={"request": "Generate KRAS G12C inhibitors."},
                    )
                ),
                types.Part(
                    function_call=types.FunctionCall(
                        name="ExperimentModuleAgent",
                        args={"request": "Dock the generated molecules."},
                    )
                ),
            ],
        )
    )
    ctx = SimpleNamespace(state={}, user_content=None, agent_name="OrchestratorAgent")

    assert coalesce_experiment_module_calls(ctx, response) is None
    parts = response.content.parts
    assert len(parts) == 1
    merged = parts[0].function_call.args["request"]
    assert "Generate KRAS G12C inhibitors." in merged
    assert "Dock the generated molecules." in merged


def test_module_first_gate_reroutes_research_before_module_ran():
    from CoScientist.experiments.runtime.coalesce import (
        enforce_experiment_module_first,
    )
    from CoScientist.experiments.runtime.shared import GATE_ROUTED_STATE_KEY

    state: dict = {}
    response = _research_call_response("Dock aspirin against COX-2 and report affinity.")
    ctx = SimpleNamespace(state=state, user_content=None, agent_name="OrchestratorAgent")

    assert enforce_experiment_module_first(ctx, response) is None
    fc = response.content.parts[0].function_call
    # State-keyed rewrite: the module gets the first shot with the same brief.
    assert fc.name == "ExperimentModuleAgent"
    assert fc.args["request"] == "Dock aspirin against COX-2 and report affinity."
    # Flagged as a structural reroute so the early feasibility gate may apply.
    assert state[GATE_ROUTED_STATE_KEY] is True


def test_module_first_gate_prefers_root_goal_over_reworded_brief():
    from CoScientist.experiments.runtime.coalesce import (
        enforce_experiment_module_first,
    )

    # Orchestrator declared a compute goal via research_init, then delegated a
    # reworded "find literature" brief. The gate must reroute to the module with
    # the real goal, not the literature rewording (otherwise EM plans a lit search).
    response = _research_call_response(
        "Find literature on non-covalent BTK inhibitors and their BBB permeability."
    )
    ctx = SimpleNamespace(
        state={
            "orchestrator_root_goal": (
                "Generate highly potent non-covalent BTK inhibitors with "
                "increased blood-brain barrier permeability."
            )
        },
        user_content=None,
        agent_name="OrchestratorAgent",
    )

    assert enforce_experiment_module_first(ctx, response) is None
    fc = response.content.parts[0].function_call
    assert fc.name == "ExperimentModuleAgent"
    assert fc.args["request"].startswith("Generate highly potent non-covalent BTK")
    assert "Find literature" not in fc.args["request"]


def test_module_first_gate_falls_back_to_brief_without_root_goal():
    from CoScientist.experiments.runtime.coalesce import (
        enforce_experiment_module_first,
    )

    response = _research_call_response("Dock aspirin against COX-2 and report affinity.")
    ctx = SimpleNamespace(state={}, user_content=None, agent_name="OrchestratorAgent")

    assert enforce_experiment_module_first(ctx, response) is None
    fc = response.content.parts[0].function_call
    assert fc.name == "ExperimentModuleAgent"
    assert fc.args["request"] == "Dock aspirin against COX-2 and report affinity."


def test_module_first_gate_no_op_after_module_attempted():
    from CoScientist.experiments.runtime.coalesce import (
        enforce_experiment_module_first,
    )

    response = _research_call_response("Find review papers on COX-2 selectivity.")
    # Module already started without NO_MATCHING_TOOL → no parallel orch Research.
    ctx = SimpleNamespace(
        state={"experiment_source_request": "prior compute ask"},
        user_content=None,
        agent_name="OrchestratorAgent",
    )

    assert enforce_experiment_module_first(ctx, response) is None
    names = [
        getattr(getattr(p, "function_call", None), "name", None)
        for p in response.content.parts
    ]
    assert "ResearchAgent" not in names


def test_module_first_gate_leaves_explicit_module_call_untouched():
    from google.adk.models import LlmResponse
    from google.genai import types

    from CoScientist.experiments.runtime.coalesce import (
        enforce_experiment_module_first,
    )
    from CoScientist.experiments.runtime.shared import GATE_ROUTED_STATE_KEY

    # Same-turn Research + EM is Dual EM — keep the module, drop Research.
    response = LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="ExperimentModuleAgent",
                        args={"request": "Generate candidates."},
                    )
                ),
                types.Part(
                    function_call=types.FunctionCall(
                        name="ResearchAgent",
                        args={"request": "Background on the target."},
                    )
                ),
            ],
        )
    )
    state: dict = {}
    ctx = SimpleNamespace(state=state, user_content=None, agent_name="OrchestratorAgent")

    assert enforce_experiment_module_first(ctx, response) is None
    names = [p.function_call.name for p in response.content.parts]
    assert names == ["ExperimentModuleAgent"]
    # Explicit module call stays trusted (not a structural Research rewrite).
    assert GATE_ROUTED_STATE_KEY not in state


def test_early_feasibility_skips_check_for_explicit_module_call():
    """An orchestrator-chosen EM call (not gate-routed) is never second-guessed,
    even for a lit-only ask with junk inventory — only the structural reroute
    path (enforce_experiment_module_first) is eligible for NO_MATCHING_TOOL."""
    from CoScientist.experiments.context.builder import RETRIEVED_CAPABILITIES_KEY
    from CoScientist.experiments.runtime.guards import (
        NO_MATCHING_TOOL_STATE_KEY,
        assess_experiment_inventory_feasibility,
    )

    state = {
        "experiment_source_request": (
            "Сделай обзор литературы по роли гиппокампа в консолидации памяти."
        ),
        RETRIEVED_CAPABILITIES_KEY: [
            {
                "tool": "generate_case_mols",
                "server_id": "d36e",
                "description": "Generate case molecules with a GAN.",
            },
        ],
    }
    assess_experiment_inventory_feasibility(
        SimpleNamespace(state=state, agent_name="ToolPreparerAgent")
    )
    assert state.get(NO_MATCHING_TOOL_STATE_KEY) in (None, "")


def test_early_feasibility_empty_inventory_is_no_matching_when_gate_routed():
    from CoScientist.experiments.runtime.guards import (
        NO_MATCHING_TOOL_STATE_KEY,
        assess_experiment_inventory_feasibility,
        skip_when_experiment_not_feasible,
    )
    from CoScientist.experiments.runtime.shared import GATE_ROUTED_STATE_KEY

    state = {
        "experiment_source_request": "Dock aspirin against COX-2 and report affinity.",
        "experiment_retrieved_capabilities": [],
        GATE_ROUTED_STATE_KEY: True,
    }
    assess_experiment_inventory_feasibility(
        SimpleNamespace(state=state, agent_name="ToolPreparerAgent")
    )
    msg = state[NO_MATCHING_TOOL_STATE_KEY]
    assert isinstance(msg, str) and msg.startswith("NO_MATCHING_TOOL:")
    skipped = skip_when_experiment_not_feasible(
        SimpleNamespace(state=state, agent_name="HypothesesAgent")
    )
    assert skipped is not None


def test_module_first_gate_reroutes_first_shot_mcp_builder():
    from google.adk.models import LlmResponse
    from google.genai import types

    from CoScientist.experiments.runtime.coalesce import enforce_experiment_module_first
    from CoScientist.experiments.runtime.shared import GATE_ROUTED_STATE_KEY

    compute = LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="McpBuilderAgent",
                        args={
                            "request": (
                                "Implement a small computational experiment: use the "
                                "PubChemPy Python package to fetch compound records."
                            )
                        },
                    )
                )
            ],
        )
    )
    state: dict = {}
    ctx = SimpleNamespace(state=state, user_content=None, agent_name="OrchestratorAgent")
    assert enforce_experiment_module_first(ctx, compute) is None
    assert compute.content.parts[0].function_call.name == "ExperimentModuleAgent"
    assert state[GATE_ROUTED_STATE_KEY] is True

    wrap = LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="McpBuilderAgent",
                        args={
                            "request": (
                                "Turn https://github.com/mcs07/PubChemPy into a "
                                "reusable MCP server we can register."
                            )
                        },
                    )
                )
            ],
        )
    )
    ctx2 = SimpleNamespace(state={}, user_content=None, agent_name="OrchestratorAgent")
    assert enforce_experiment_module_first(ctx2, wrap) is None
    assert wrap.content.parts[0].function_call.name == "ExperimentModuleAgent"
    assert ctx2.state[GATE_ROUTED_STATE_KEY] is True
