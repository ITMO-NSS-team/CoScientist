"""OrchestratorAgent — top-level coordinator (in-process ADK mode).

The roster is driven by the agent catalog (:mod:`CoScientist.agents.catalog`) —
the single source of truth for which agents are enabled, their prompt
descriptions, and their order. We map each catalog name to its LlmAgent
instance and attach the enabled ones as tools.

For A2A mode (each sub-agent as a remote HTTP service) see
:mod:`CoScientist.a2a.orchestrator`.
"""
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.agent_tool import AgentTool

from CoScientist.agents import catalog
from CoScientist.agents.definitions.coder_agent import coder_agent
from CoScientist.agents.common import agent_tools, make_llm
# post_action_critique is intentionally not wired (after_tool_callback disabled below).
from CoScientist.agents.callbacks import pre_action_critique
from CoScientist.agents.definitions.hypotheses_agent import hypotheses_agent
from CoScientist.agents.callbacks import before_model_modifier as med_before_model
from CoScientist.agents.definitions.medical_agent import medical_agent
from CoScientist.agents.definitions.planner_agent import planner_agent
from CoScientist.agents.prompts import build_orchestrator_instruction
from CoScientist.agents.definitions.research_agent import research_agent
from CoScientist.agents.definitions.task_execution_agent import task_execution_agent
from CoScientist.logging import multi_agent_tracer
from opik.integrations.adk import track_adk_agent_recursive

# Catalog name -> the LlmAgent instance it refers to.
_AGENT_INSTANCES = {
    "PlannerAgent": planner_agent,
    "HypothesesAgent": hypotheses_agent,
    "ResearchAgent": research_agent,
    "TaskExecutorAgent": task_execution_agent,
    "MedicalAgent": medical_agent,
    "CoderAgent": coder_agent,
}


def _resolve_agent(name: str):
    inst = _AGENT_INSTANCES.get(name)
    if inst is None:
        raise ValueError(
            f"Catalog agent {name!r} has no instance in _AGENT_INSTANCES "
            "(orchestrator_agent.py). Add it there or fix the catalog name."
        )
    return inst


_orchestrator_subagents = [
    AgentTool(agent=_resolve_agent(spec.name)) for spec in catalog.enabled_agents()
]

orchestrator_agent = LlmAgent(
    name="OrchestratorAgent",
    model=make_llm(),
    instruction=build_orchestrator_instruction(),
    description="Main Orchestrator Agent",
    before_model_callback=med_before_model,
    after_model_callback=pre_action_critique,
    # after_tool_callback=post_action_critique,
    tools=agent_tools(_orchestrator_subagents, hitl=False),
)

track_adk_agent_recursive(orchestrator_agent, multi_agent_tracer)

__all__ = ["orchestrator_agent"]
