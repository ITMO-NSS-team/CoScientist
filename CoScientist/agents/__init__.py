"""LLM Agents module — agents are assembled from CoScientist/agents/system.yaml.

The YAML is the single source of truth for the system layout (agents, tools,
callbacks, prompts, HITL, A2A exposure). This module builds the in-process
system once and re-exports the agent instances under their historical names so
existing imports keep working.
"""
import copy

from CoScientist.assembly import build_system
from CoScientist.assembly.schema import load_config
from CoScientist.logging import multi_agent_tracer
from CoScientist.agents.llm_repair import install_json_repair
from opik.integrations.adk import track_adk_agent_recursive

# Guard the LiteLlm tool-call JSON boundary process-wide BEFORE any runner executes:
# a malformed tool-call payload (qwen truncation / missing comma) must not kill the run.
# Idempotent; installed once at first import of the agents package (CLI + web both hit this).
install_json_repair()

_system = build_system()

orchestrator_agent = _system.root
root_agent = orchestrator_agent
planner_agent = _system.agent("PlannerAgent")
hypotheses_agent = _system.agent("HypothesesAgent")
research_agent = _system.agent("ResearchAgent")
task_execution_agent = _system.agent("TaskExecutorAgent")
medical_agent = _system.agent("MedicalAgent")
coder_agent = _system.agent("CoderAgent")
tool_agent = _system.agent("ToolPreparerAgent")
tool_retriever_agent = _system.agent("ToolRetrieverAgent")
tool_reranker_agent = _system.agent("ToolReranker")
tool_websearcher_agent = _system.agent("ToolWebSearcherAgent")
fedot_agent = _system.agent("ExperimentAgent")

track_adk_agent_recursive(orchestrator_agent, multi_agent_tracer)


def build_for_mode():
    """Build an AgentSystem configured for the current start mode from settings.

    Reads ``settings.web.start_mode``:
      * ``"init"`` (default) — InitAgent is root (sequential: PlannerAgent →
        OrchestratorAgent).
      * ``"orchestrator"`` — OrchestratorAgent is root, with PlannerAgent
        added to its subordinates so it can be invoked on demand.

    Other runtime-tunable parameters (e.g. ``max_searches``) are read from
    ``settings.web`` by individual components at build time.

    Returns:
        An :class:`~CoScientist.assembly.assembler.AgentSystem`.
    """
    from CoScientist.config import get_settings
    start_mode = get_settings().web.start_mode

    if start_mode == "init":
        return build_system()

    if start_mode != "orchestrator":
        raise ValueError(f"Unknown start_mode {start_mode!r}; expected 'init' or 'orchestrator'")

    # Load a fresh config and patch it for orchestrator-as-root mode.
    raw_config = load_config()
    patched = copy.deepcopy(raw_config)

    # Make OrchestratorAgent the root.
    patched.agents["OrchestratorAgent"].root = True
    patched.agents["InitAgent"].root = False

    # Add PlannerAgent to OrchestratorAgent's subordinates (if not already).
    orch_subs = patched.agents["OrchestratorAgent"].subordinates
    if "PlannerAgent" not in orch_subs:
        orch_subs.insert(0, "PlannerAgent")

    # Re-validate the patched config and build.
    system = build_system(config=patched)
    track_adk_agent_recursive(system.root, multi_agent_tracer)
    return system


__all__ = [
    "orchestrator_agent",
    "root_agent",
    "planner_agent",
    "fedot_agent",
    "research_agent",
    "hypotheses_agent",
    "medical_agent",
    "coder_agent",
    "tool_retriever_agent",
    "tool_reranker_agent",
    "tool_websearcher_agent",
    "task_execution_agent",
    "tool_agent",
    "build_for_mode",
]
