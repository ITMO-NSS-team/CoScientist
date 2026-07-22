"""LLM Agents module — agents are assembled from CoScientist/agents/system.yaml.

The YAML is the single source of truth for the system layout (agents, tools,
callbacks, prompts, HITL, A2A exposure). This module builds the in-process
system once and re-exports the agent instances under their historical names so
existing imports keep working.
"""
from CoScientist.assembly import build_system
from CoScientist.logging import multi_agent_tracer
from CoScientist.agents.llm_repair import install_json_repair

# Guard the LiteLlm tool-call JSON boundary process-wide BEFORE any runner executes:
# a malformed tool-call payload (qwen truncation / missing comma) must not kill the run.
# Idempotent; installed once at first import of the agents package (CLI + web both hit this).
install_json_repair()

_system = build_system()

# The full assembled system — for callers that need to iterate every agent of
# the ACTIVE profile (e.g. the web app wiring HITL handlers) instead of relying
# on the historical fixed names below.
agent_system = _system

# root == the delegation-tree orchestrator (OrchestratorAgent). Lifecycle flow
# (pre/post stages) lives in system.yaml `pipeline` and is driven by the manager
# — it is deliberately NOT the root, so this stays the real LLM orchestrator.
#
# Historical named exports. Alternative profiles ($COSCIENTIST_CONFIG, e.g.
# "microfluidics") declare only a subset of the agents — the missing ones
# resolve to None so importing this module keeps working for every profile.
orchestrator_agent = _system.root
root_agent = orchestrator_agent

# Agents that run as pipeline stages (pre/post) around the orchestrator.
pipeline_pre_agents = [_system.agent(n) for n in _system.config.pipeline.pre]
pipeline_post_agents = [_system.agent(n) for n in _system.config.pipeline.post]
planner_agent = _system.agents.get("PlannerAgent")
hypotheses_agent = _system.agents.get("HypothesesAgent")
research_agent = _system.agents.get("ResearchAgent")
task_execution_agent = _system.agents.get("TaskExecutorAgent")
medical_agent = _system.agents.get("MedicalAgent")
coder_agent = _system.agents.get("CoderAgent")
tool_agent = _system.agents.get("ToolPreparerAgent")
tool_retriever_agent = _system.agents.get("ToolRetrieverAgent")
tool_reranker_agent = _system.agents.get("ToolReranker")
tool_websearcher_agent = _system.agents.get("ToolWebSearcherAgent")
fedot_agent = _system.agents.get("ExperimentAgent")
result_aggregator_agent = _system.agents.get("ResultAggregatorAgent")
tz_agent = _system.agents.get("TZAgent")

# Attach the Opik tracer only when tracing is enabled (see OPIK__ENABLED).
if multi_agent_tracer is not None:
    from opik.integrations.adk import track_adk_agent_recursive

    track_adk_agent_recursive(orchestrator_agent, multi_agent_tracer)
    for _stage in pipeline_pre_agents + pipeline_post_agents:
        track_adk_agent_recursive(_stage, multi_agent_tracer)

__all__ = [
    "agent_system",
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
    "result_aggregator_agent",
    "pipeline_pre_agents",
    "pipeline_post_agents",
    "tz_agent",
]
