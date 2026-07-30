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

# `orchestrator_agent` == the delegation-tree orchestrator (the real LLM). It is
# the config `root` (root: true in system.yaml) and stays the named export every
# caller (prompts, A2A) expects.
#
# Historical named exports. Alternative profiles ($COSCIENTIST_CONFIG, e.g.
# "microfluidics") declare only a subset of the agents — the missing ones
# resolve to None so importing this module keeps working for every profile.
orchestrator_agent = _system.root
root_agent = orchestrator_agent

# Agents that run as pipeline stages (pre/post) around the orchestrator.
pipeline_pre_agents = [_system.agent(n) for n in _system.config.pipeline.pre]
pipeline_post_agents = [_system.agent(n) for n in _system.config.pipeline.post]

# The RUN root: the whole lifecycle (pre → orchestrator → post/aggregator) is one
# ADK SequentialAgent, driven by a single Runner.run_async so it is ONE invocation
# = ONE trace, with the Result Aggregator as the terminal child (it reads the graph
# the orchestrator populated and writes the report). When no pipeline stages are
# declared, the orchestrator IS the run root (no needless wrapper).
if pipeline_pre_agents or pipeline_post_agents:
    from google.adk.agents.sequential_agent import SequentialAgent

    run_root = SequentialAgent(
        name="ResearchPipeline",
        description="Full research lifecycle: orchestrator run then report synthesis.",
        sub_agents=[*pipeline_pre_agents, orchestrator_agent, *pipeline_post_agents],
    )
else:
    run_root = orchestrator_agent

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
# Tracking the run root covers the whole SequentialAgent (orchestrator + pipeline
# stages) so the entire lifecycle lands in ONE trace.
if multi_agent_tracer is not None:
    from opik.integrations.adk import track_adk_agent_recursive

    track_adk_agent_recursive(run_root, multi_agent_tracer)

__all__ = [
    "agent_system",
    "orchestrator_agent",
    "root_agent",
    "run_root",
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
