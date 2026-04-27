
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools import FunctionTool
from google.genai import types
from google.adk.planners import BasePlanner, BuiltInPlanner, PlanReActPlanner

import litellm

from CoScientist.config import get_settings
from CoScientist.agents.prompts import hypotheses_instruction, research_instruction, fedot_instruction, orchestrator_instruction, tool_retriever_instruction, planner_instruction
from CoScientist.hitl.session_agent import SessionAgent
from CoScientist.tools import fedot_toolset_instance, websearch_toolset_instance, retrieval_toolset_instance
from CoScientist.storage import RetrievalFinalResult
from CoScientist.hitl import HITLToolset
from CoScientist.hitl.handler import AbstractHITLHandler, ConsoleHITLHandler
from CoScientist.hitl.callbacks import make_hitl_after_callback, make_hitl_before_callback
from CoScientist.hitl.models import HITLAction
from CoScientist.logging import multi_agent_tracer
from CoScientist.hitl.tool import get_hitl_tools

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

tool_retriever_agent = LlmAgent(
    name='ToolRetrieverAgent',
    model=LiteLlm(model=MODEL),
    instruction=tool_retriever_instruction,
    description="Agent to retrieve relevant MCP servers from RAG database of MCP tools for given task.",
    output_schema=RetrievalFinalResult,
    tools=_agent_tools(retrieval_toolset_instance, hitl_tools=True),
    output_key="retrieved_tools",
    #before_agent_callback=make_hitl_before_callback(hitl_handler) if hitl_enabled else None,
    #after_agent_callback=make_hitl_after_callback(hitl_handler, HITLAction.APPROVE) if hitl_enabled else None,
)

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
    sub_agents=[tool_retriever_agent, fedot_agent],
    description="Agent to complete experiments and run calculations. Use it for any computation and idea validation. It can use a lot of MCP tools",
)

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
    tools=_agent_tools([
        AgentTool(agent=planner_agent),
        AgentTool(agent=hypotheses_agent), 
        AgentTool(agent=research_agent), 
        AgentTool(agent=task_execution_agent)
    ], hitl_tools=True),
)

track_adk_agent_recursive(orchestrator_agent, multi_agent_tracer)

