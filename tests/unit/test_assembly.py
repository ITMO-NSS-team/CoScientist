"""Structural tests for the YAML-driven agent assembly.

These build the real system from CoScientist/agents/system.yaml (no LLM calls)
and assert the invariants the assembler is supposed to guarantee — above all
that prompts and wiring cannot drift apart.

Run from the repo root:  pytest tests/unit/test_assembly.py -q
"""
import copy

import pytest
from dotenv import load_dotenv

load_dotenv()

from CoScientist.assembly import build_system, load_config  # noqa: E402
from CoScientist.assembly.prompting import PromptContext  # noqa: E402
from CoScientist.assembly.registry import REGISTRY  # noqa: E402
from CoScientist.assembly.schema import SystemConfig, get_config  # noqa: E402


@pytest.fixture(scope="module")
def config():
    return get_config()


@pytest.fixture(scope="module")
def system(config):
    return build_system(config)


# ── config validation ────────────────────────────────────────────────────────

def test_config_loads_and_has_one_root(config):
    assert config.root.name == "InitAgent"
    order = config.build_order()
    assert order.index("ToolRetrieverAgent") < order.index("LocalToolsExtractorAgent")
    assert set(order) == set(config.agents)


def test_every_referenced_name_is_registered(config):
    for agent in config.agents.values():
        for tool in agent.tools:
            REGISTRY.tool(tool)  # raises on unknown
        for kind, names in agent.callbacks.items():
            for name in names:
                entry = REGISTRY.callback(name)
                assert entry.kind == kind, f"{agent.name}: {name} listed under {kind}"
        if agent.prompt:
            REGISTRY.prompt(agent.prompt)
        if agent.output_schema:
            REGISTRY.output_schema(agent.output_schema)
        if agent.planner:
            REGISTRY.planner(agent.planner)
        if agent.cls.startswith("custom:"):
            REGISTRY.agent_class(agent.cls.split(":", 1)[1])


def test_unknown_agent_reference_rejected(config):
    raw = copy.deepcopy(config.model_dump(by_alias=True))
    raw["agents"]["OrchestratorAgent"]["subordinates"].append("NoSuchAgent")
    with pytest.raises(Exception, match="NoSuchAgent"):
        SystemConfig.model_validate(raw)


def test_dependency_cycle_rejected(config):
    raw = copy.deepcopy(config.model_dump(by_alias=True))
    raw["agents"]["ToolRetrieverAgent"]["subordinates"] = ["TaskExecutorAgent"]
    with pytest.raises(Exception, match="cycle"):
        SystemConfig.model_validate(raw)


def test_duplicate_a2a_port_rejected(config):
    raw = copy.deepcopy(config.model_dump(by_alias=True))
    raw["agents"]["CoderAgent"]["a2a"]["port"] = raw["agents"]["ResearchAgent"]["a2a"]["port"]
    with pytest.raises(Exception, match="port"):
        SystemConfig.model_validate(raw)


# ── built system invariants ──────────────────────────────────────────────────

def test_all_agents_built_under_their_names(config, system):
    for name in config.agents:
        assert system.agent(name).name == name


def test_orchestrator_roster_matches_prompt_and_tools(config, system):
    orchestrator = system.agent("OrchestratorAgent")
    enabled = [a.name for a in config.enabled_subordinates("OrchestratorAgent")]
    attached = [t.agent.name for t in orchestrator.tools if hasattr(t, "agent")]
    assert attached == enabled
    for name in enabled:
        assert name in orchestrator.instruction, f"{name} wired but not in the prompt"
    for sub in config.agent("OrchestratorAgent").subordinates:
        if not config.agent(sub).is_enabled():
            assert sub not in orchestrator.instruction, f"{sub} disabled but still in the prompt"


def test_prompts_advertise_exactly_the_attached_function_tools(config, system):
    """Every attached function tool is named in the prompt and vice versa."""
    for name, cfg in config.agents.items():
        if cfg.cls != "llm" or not cfg.prompt:
            continue
        agent = system.agent(name)
        instruction = agent.instruction
        for tool in agent.tools:
            tool_name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
            if tool_name is None or hasattr(tool, "get_tools") or hasattr(tool, "agent"):
                continue  # toolsets resolve at runtime; AgentTools live in <<AGENTS>>
            assert tool_name in instruction, f"{name}: tool {tool_name} not in prompt"


def test_no_unfilled_placeholders(config, system):
    for name, cfg in config.agents.items():
        instruction = getattr(system.agent(name), "instruction", "") or ""
        assert "<<" not in instruction, f"{name}: unfilled placeholder in prompt"


def test_pipeline_state_injections_are_optional(config, system):
    """ADK {state_key} injections that depend on an upstream agent having called
    a tool must use the optional `{key?}` form, or a degenerate run (empty web
    search, no retrieval) crashes the agent with a KeyError mid-turn."""
    state_keys = ("accumulated_tools", "filtered_tools", "accumulated_web_mcps")
    for name in config.agents:
        instruction = getattr(system.agent(name), "instruction", "") or ""
        for key in state_keys:
            assert "{" + key + "}" not in instruction, (
                f"{name}: bare ADK injection {{{key}}} — must be optional {{{key}?}}"
            )


def test_critic_prompt_embeds_current_roster(config):
    ctx = PromptContext(config=config.agent("OrchestratorAgent"), system=config)
    critic_prompt = REGISTRY.prompt("pre_action_critic")(ctx)
    for sub in config.enabled_subordinates("OrchestratorAgent"):
        assert sub.name in critic_prompt


def test_planner_roster_uses_real_agent_names(config, system):
    instruction = system.agent("PlannerAgent").instruction
    for sub in config.enabled_subordinates("OrchestratorAgent"):
        if sub.name != "PlannerAgent":
            assert sub.name in instruction
    # The old prose aliases must be gone.
    assert "Experiment Agent" not in instruction
    assert "Hypothesis Agent" not in instruction


def test_orchestrator_tool_discovery_gate(config, system):
    """Structural invariant: the retrieval tool is documented iff attached, and
    when attached the retrieve_tools gate is positioned BEFORE the routing roster
    so the orchestrator checks for ready-made tools before delegating. (The exact
    wording of the gate is content the prompt author may tune.)"""
    cfg = config.agent("OrchestratorAgent")
    instruction = system.agent("OrchestratorAgent").instruction
    has_retrieval = "retrieval" in cfg.tools
    assert ("retrieve_tools" in instruction) == has_retrieval
    if has_retrieval:
        # The gate must come before the routing roster (so it's read first).
        assert instruction.index("retrieve_tools") < instruction.index(
            "Delegate by the NATURE"
        )


def test_executor_sufficiency_and_discovery_guardrails(config, system):
    """Two routing guardrails are present when the relevant agents are wired:
    the ExperimentAgent must be able to abstain (NO_MATCHING_TOOL) when retrieved
    tools don't implement the task, and the orchestrator must do tool DISCOVERY
    itself rather than delegating "does a tool exist" to the Executor."""
    exp = system.agent("ExperimentAgent").instruction
    assert "NO_MATCHING_TOOL" in exp
    assert "Recommend CoderAgent" in exp

    cfg = config.agent("OrchestratorAgent")
    if "retrieval" in cfg.tools and "TaskExecutorAgent" in cfg.subordinates:
        orch = system.agent("OrchestratorAgent").instruction
        assert "retrieve_tools" in orch
        assert 'delegate "check if a tool exists"' in orch.lower() or \
               'Do NOT delegate "check if a tool exists"' in orch


def test_dataset_collector_is_a_coder_subordinate_sharing_the_sandbox(config, system):
    """The DatasetCollectorAgent is wired under CoderAgent, named in the coder's
    prompt, and uses the coder toolset — so it works in the same per-session
    sandbox workspace (the data it downloads lands where the coder builds on it)."""
    coder = config.agent("CoderAgent")
    assert "DatasetCollectorAgent" in coder.subordinates
    collector = config.agent("DatasetCollectorAgent")
    # Both share the coder toolset -> same workspace-state anchor -> same sandbox.
    assert "coder" in collector.tools and "coder" in coder.tools
    # The coder prompt advertises the subordinate (rendered from config).
    coder_instruction = system.agent("CoderAgent").instruction
    assert "DatasetCollectorAgent" in coder_instruction
    assert "SAME sandbox" in coder_instruction
    # Reached only through the coder — not a standalone A2A service.
    assert collector.a2a is None
    # Built and attached as an AgentTool on the coder.
    attached = [t.agent.name for t in system.agent("CoderAgent").tools if hasattr(t, "agent")]
    assert attached == ["DatasetCollectorAgent"]


def test_orchestrator_prompt_documents_only_wired_critics(config, system):
    cfg = config.agent("OrchestratorAgent")
    instruction = system.agent("OrchestratorAgent").instruction
    assert ("Pre-action critic" in instruction) == (
        "pre_action_critique" in cfg.callbacks.after_model
    )
    assert ("Post-action critic" in instruction) == (
        "post_action_critique" in cfg.callbacks.after_tool
    )
