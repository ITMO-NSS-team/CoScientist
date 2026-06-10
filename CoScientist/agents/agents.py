from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.agent_tool import AgentTool
from google.adk.planners import PlanReActPlanner

import litellm

from CoScientist.config import get_settings

from CoScientist.agents.prompts import hypotheses_instruction, research_instruction, fedot_instruction, build_orchestrator_instruction, tool_retriever_instruction, planner_instruction, tool_reranker_instruction, tool_websearcher_instruction, tool_scoring_instruction, medical_instruction, coder_instruction
from CoScientist.agents import catalog
from CoScientist.agents.callbacks import before_tool_reranker_model, after_tool_reranker_agent, after_fullset_reranker_agent, print_research_agent_tool_call
from CoScientist.agents.critic_agent import (
    pre_action_critique,
    post_action_critique,
)
from CoScientist.agents.custom_agents import WebToolsDeployerAgent
from CoScientist.agents.med_callbacks import before_model_modifier as med_before_model, med_agent_before_model
from CoScientist.agents.research_callbacks import papers_agent_before_model

from CoScientist.tools import fedot_toolset_instance, websearch_toolset_instance, retrieval_toolset_instance, search_mcp_servers, med_toolset_instance, coder_toolset_instance, paper_analysis_toolset_instance, papers_search_toolset_instance
from CoScientist.storage import RetrievalFinalResult, ToolRanking, MCPRanking


from CoScientist.hitl import HITLToolset
from CoScientist.hitl.session_agent import SessionAgent
from CoScientist.hitl.handler import AbstractHITLHandler, ConsoleHITLHandler
from CoScientist.hitl.callbacks import make_hitl_after_callback, make_hitl_before_callback
from CoScientist.hitl.models import HITLAction
from CoScientist.hitl.tool import get_hitl_tools

from CoScientist.logging import multi_agent_tracer

from opik.integrations.adk import track_adk_agent_recursive

from typing import Any


settings = get_settings()

MODEL = settings.llm.main_model
litellm.api_key = settings.llm.openai_api_key
# Silence litellm's "Provider List: https://docs.litellm.ai/docs/providers" spam.
# It fires when litellm can't map a model prefix (e.g. "qwen/...") to a known
# provider during cost/token bookkeeping — harmless, but it floods the console.
litellm.suppress_debug_info = True
hitl_enabled = settings.hitl.enabled
hitl_handler=ConsoleHITLHandler() if hitl_enabled else None

# The CoderAgent runs on a dedicated (stronger) model — its multi-step tool-use
# benefits from more capability. Falls back to the main model when unset.
#
# Routing mirrors the other agents exactly: the provider prefix in the model
# string (e.g. "openrouter/qwen/...") selects the provider/base-URL, and the
# global `litellm.api_key` (set above) carries the key. We deliberately do NOT
# pass `api_base` here — doing so makes litellm strip the provider prefix, fail
# to re-infer the provider, and spam "Provider List: ..." warnings.
CODER_MODEL = settings.llm.coder_model or settings.llm.main_model

def _build_coder_llm() -> LiteLlm:
    return LiteLlm(model=CODER_MODEL)

def _agent_tools(base_tools: Any, hitl_tools: bool = False) -> list:
    """Helper to add HITL tools directly if hitl_tools=True and global hitl is enabled."""
    if isinstance(base_tools, list):
        tools = list(base_tools)
    else:
        tools = [base_tools] if base_tools else []
        
    if hitl_enabled and hitl_tools:
        tools.extend(get_hitl_tools())
    return tools

hypotheses_agent = LlmAgent(
    name="HypothesesAgent",
    model=LiteLlm(model=MODEL),
    instruction=hypotheses_instruction,
    description="Agent to generate scientific hypotheses and ideas for given task",
    output_key="hypotheses",
    tools=_agent_tools([], hitl_tools=False),
    #before_agent_callback=make_hitl_before_callback(hitl_handler) if hitl_enabled else None,
    #after_agent_callback=make_hitl_after_callback(hitl_handler, HITLAction.APPROVE) if hitl_enabled else None,
)

research_agent = LlmAgent(
    name="ResearchAgent",
    model=LiteLlm(model=MODEL),
    instruction=research_instruction,
    description="Agent to answer questions and knowledge mining using Literature and Web Search.",
    output_key="search_results",
    # Drop any optional MCP toolsets that aren't configured (None) so the agent
    # still builds when their URLs are unset.
    tools=_agent_tools([t for t in [websearch_toolset_instance,
                                    paper_analysis_toolset_instance,
                                    papers_search_toolset_instance] if t is not None],
                       hitl_tools=True),
    before_model_callback=papers_agent_before_model,
    after_tool_callback=print_research_agent_tool_call,
    #before_agent_callback=make_hitl_before_callback(hitl_handler) if hitl_enabled else None,
    #after_agent_callback=make_hitl_after_callback(hitl_handler, HITLAction.APPROVE) if hitl_enabled else None,
)

#----------------- TOOLS SEARCHING AGENTS --------------------------

tool_retriever_agent = LlmAgent(
    name='ToolRetrieverAgent',
    model=LiteLlm(model=MODEL),
    instruction=tool_retriever_instruction,
    description="Agent to retrieve relevant MCP servers from RAG database of MCP tools for given task.",
    tools=retrieval_toolset_instance,
    output_key="retrieved_tools"
)

tool_reranker_agent = LlmAgent(
    name='ToolReranker',
    model=LiteLlm(model=MODEL),
    instruction=tool_reranker_instruction,
    description="Agent to rerank retrieved MCP servers from RAG database of MCP tools for given task.",
    output_schema=ToolRanking,
    before_model_callback=before_tool_reranker_model,
    after_agent_callback=after_tool_reranker_agent,
    output_key="reranked_tools"
)

local_tools_extractor = SequentialAgent(
    name="LocalToolsExtractorAgent",
    sub_agents=[tool_retriever_agent, tool_reranker_agent],
    description="Agent to extract relevant ready-to-use tools from local storage",
)

tool_websearcher_agent = LlmAgent(
    name='ToolWebSearcherAgent',
    model=LiteLlm(model=MODEL),
    instruction=tool_websearcher_instruction,
    description="Agent to web-search relevant MCP servers from public web storages.",
    # output_schema=RetrievalFinalResult,
    tools=[search_mcp_servers],
    output_key="retrieved_web_tools"
)

tool_searcher = ParallelAgent(
     name="ParallelToolSearcherAgent",
     sub_agents=[local_tools_extractor, tool_websearcher_agent],
     description="Runs multiple search agents in parallel to gather relevant MCP servers."
 )

tool_fullset_reranker_agent = LlmAgent(
    name='FullSetToolReranker',
    model=LiteLlm(model=MODEL),
    instruction=tool_scoring_instruction,
    description="Agent to score found web MCP servers given already available local MCP servers for given task.",
    output_schema=MCPRanking,
    after_agent_callback=after_fullset_reranker_agent,
    output_key="reranked_web_servers"
)

web_tools_deployer = WebToolsDeployerAgent(
        name="WebToolsDeployerAgent",
        description="Agent to deploy found web mcp servers"
    )

tool_agent = SequentialAgent(name='ToolPreparerAgent',
        sub_agents=[tool_searcher, tool_fullset_reranker_agent, web_tools_deployer],
        description="Agent to find and prepare relevant mcp servers for current task",
    )
#------------------------------------------------------------------
#----------------- EXPERIMENT AGENTS ------------------------------

fedot_agent = LlmAgent(
    name="ExperimentAgent",
    model=LiteLlm(model=MODEL),
    instruction=fedot_instruction,
    description="Agent to invoke MAS for solving given task. Uses MCP tools",
    output_key="fedot_results",
    tools=_agent_tools(fedot_toolset_instance, hitl_tools=False),
    #before_agent_callback=make_hitl_before_callback(hitl_handler) if hitl_enabled else None,
    #after_agent_callback=make_hitl_after_callback(hitl_handler, HITLAction.APPROVE) if hitl_enabled else None,
)



task_execution_agent = SequentialAgent(
    name="TaskExecutorAgent",
    sub_agents=[tool_agent, fedot_agent],
    description="Agent to complete experiments and run calculations. Use it for any computation and idea validation. It can use a lot of MCP tools",
)

medical_agent = LlmAgent(
    name="MedicalAgent",
    model=LiteLlm(model=MODEL),
    instruction=medical_instruction,
    description="Agent for medical and clinical questions: PubMed literature search, PICO extraction, study taxonomy, and DICOM image analysis.",
    output_key="medical_results",
    tools=med_toolset_instance,
    before_model_callback=med_agent_before_model,
)

coder_agent = LlmAgent(
    name="CoderAgent",
    model=_build_coder_llm(),
    instruction=coder_instruction,
    description=(
        "General-purpose coder / sandbox agent. Writes and runs code, executes "
        "shell and git commands (clone/commit/push), manages files, installs "
        "dependencies, collects and processes data, and runs long jobs in an "
        "isolated workspace. Use it whenever a task requires doing software/data "
        "engineering rather than calling a ready-made service."
    ),
    output_key="coder_results",
    tools=_agent_tools(coder_toolset_instance, hitl_tools=False),
)

#------------------------------------------------------------------

planner = PlanReActPlanner()

planner_agent = SessionAgent(
    name="PlannerAgent",
    model=LiteLlm(model=MODEL),
    instruction=planner_instruction,
    description="Generates a roadmap for solving the task",
    output_key="planner_roadmap",
    plan_file_path="roadmap.txt",
    planner=planner,
    hitl_handler=hitl_handler,
    #before_agent_callback=make_hitl_before_callback(hitl_handler) if hitl_enabled else None,
    #after_agent_callback=make_hitl_after_callback(hitl_handler, HITLAction.APPROVE) if hitl_enabled else None,
)    

# Orchestrator sub-agents are driven by the agent catalog (single source of truth
# for which agents are enabled, their prompt descriptions, and their order). Map
# each catalog name to its LlmAgent instance and attach the enabled ones.
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
            "(agents.py). Add it there or fix the catalog name."
        )
    return inst

_orchestrator_subagents = [
    AgentTool(agent=_resolve_agent(spec.name))
    for spec in catalog.enabled_agents()
]

orchestrator_agent = LlmAgent(
    name="OrchestratorAgent",
    model=LiteLlm(model=MODEL),
    #planner=planner,
    instruction=build_orchestrator_instruction(),
    description="Main Orchestrator Agent",
    before_model_callback=med_before_model,
    after_model_callback=pre_action_critique,
    # after_tool_callback=post_action_critique,
    tools=_agent_tools(_orchestrator_subagents, hitl_tools=False),
)

track_adk_agent_recursive(orchestrator_agent, multi_agent_tracer)

