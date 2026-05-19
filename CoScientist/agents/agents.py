
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools import FunctionTool
from google.genai import types
from google.adk.planners import BasePlanner, BuiltInPlanner, PlanReActPlanner

import litellm

from CoScientist.config import get_settings

from CoScientist.agents.prompts import hypotheses_instruction, research_instruction, fedot_instruction, orchestrator_instruction, tool_retriever_instruction, planner_instruction, tool_reranker_instruction, tool_websearcher_instruction, tool_scoring_instruction
from CoScientist.agents.callbacks import before_tool_reranker_model, after_tool_reranker_agent, after_fullset_reranker_agent
from CoScientist.agents.critic_agent import (
    pre_action_critique,
    post_action_critique,
)
from CoScientist.agents.custom_agents import WebToolsDeployerAgent

from CoScientist.tools import fedot_toolset_instance, websearch_toolset_instance, retrieval_toolset_instance, search_mcp_servers
from CoScientist.storage import RetrievalFinalResult, ToolRanking, MCPRanking


from CoScientist.hitl import HITLToolset
from CoScientist.hitl.session_agent import SessionAgent
from CoScientist.hitl.handler import AbstractHITLHandler, ConsoleHITLHandler
from CoScientist.hitl.callbacks import make_hitl_after_callback, make_hitl_before_callback
from CoScientist.hitl.models import HITLAction
from CoScientist.hitl.tool import get_hitl_tools

from CoScientist.logging import multi_agent_tracer

from opik.integrations.adk import track_adk_agent_recursive

from typing import Dict, Any, Optional


settings = get_settings()

MODEL = settings.llm.main_model
litellm.api_key = settings.llm.openai_api_key
hitl_enabled = settings.hitl.enabled
hitl_handler=ConsoleHITLHandler() if hitl_enabled else None

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
    tools=_agent_tools([], hitl_tools=True),
    #before_agent_callback=make_hitl_before_callback(hitl_handler) if hitl_enabled else None,
    #after_agent_callback=make_hitl_after_callback(hitl_handler, HITLAction.APPROVE) if hitl_enabled else None,
)

research_agent = LlmAgent(
    name="ResearchAgent",
    model=LiteLlm(model=MODEL),
    instruction=research_instruction,
    description="Agent to answer questions and knowledge mining using Literature and Web Search.",
    output_key="search_results",
    tools=_agent_tools([websearch_toolset_instance], hitl_tools=True),
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
    tools=_agent_tools(fedot_toolset_instance, hitl_tools=True),
    #before_agent_callback=make_hitl_before_callback(hitl_handler) if hitl_enabled else None,
    #after_agent_callback=make_hitl_after_callback(hitl_handler, HITLAction.APPROVE) if hitl_enabled else None,
)



task_execution_agent = SequentialAgent(
    name="TaskExecutorAgent",
    sub_agents=[tool_agent, fedot_agent],
    description="Agent to complete experiments and run calculations. Use it for any computation and idea validation. It can use a lot of MCP tools",
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

orchestrator_agent = LlmAgent(
    name="OrchestratorAgent",
    model=LiteLlm(model=MODEL),
    #planner=planner,
    instruction=orchestrator_instruction,
    description="Main Orchestrator Agent",
    after_model_callback=pre_action_critique,
    after_tool_callback=post_action_critique,
    tools=_agent_tools([
        AgentTool(agent=planner_agent),
        AgentTool(agent=hypotheses_agent), 
        AgentTool(agent=research_agent), 
        AgentTool(agent=task_execution_agent)
    ], hitl_tools=True),
)

track_adk_agent_recursive(orchestrator_agent, multi_agent_tracer)

