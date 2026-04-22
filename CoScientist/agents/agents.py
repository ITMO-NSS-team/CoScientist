
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
from CoScientist.hitl.looping_agent import SessionAgent
from CoScientist.tools import fedot_toolset_instance, websearch_toolset_instance, retrieval_toolset_instance
from CoScientist.storage import RetrievalFinalResult
from CoScientist.hitl import HITLToolset
from CoScientist.hitl.handler import AbstractHITLHandler, ConsoleHITLHandler
from CoScientist.hitl.callbacks import make_hitl_after_callback, make_hitl_before_callback
from CoScientist.hitl.models import HITLAction
from CoScientist.logging import multi_agent_tracer


from opik.integrations.adk import track_adk_agent_recursive


from typing import Dict, Any, Optional
import uuid
import os
import asyncio
import json


settings = get_settings()

MODEL = settings.llm.main_model
litellm.api_key = settings.llm.openai_api_key



async def create_agents(hitl_handler: Optional[AbstractHITLHandler] = None):
    """Create all agents with optional HITL support.

    Args:
        hitl_handler: HITL handler instance. If None and HITL is enabled
                      in settings, ConsoleHITLHandler is used.

    Returns:
        Dictionary with all agent instances.
    """
    hitl_enabled = settings.hitl.enabled
    hitl_agents = set(settings.hitl.agents_requiring_approval)

    # Create HITL toolset if enabled
    hitl_tools = []
    handler = None
    if hitl_enabled:
        handler = hitl_handler or ConsoleHITLHandler()
        hitl_toolset = HITLToolset(handler=handler)
        hitl_tools = await hitl_toolset.get_tools(None)

    def _agent_tools(agent_name: str, base_tools: list = None) -> list:
        """Append HITL tools to base tools if agent is in agents_requiring_approval."""
        base = list(base_tools) if base_tools else []

        # HITL tools
        if hitl_enabled and agent_name in hitl_agents:
            base.extend(hitl_tools)
        return base

    research_agent = LlmAgent(
        name="ResearchAgent",
        model=LiteLlm(model=MODEL),
        instruction=research_instruction,
        description="Agent to answer questions and knowledge mining using Literature and Web Search.",
        output_key="search_results",
        tools=_agent_tools("ResearchAgent", [websearch_toolset_instance]),
        #before_agent_callback=make_hitl_before_callback(handler) if (hitl_enabled and "ResearchAgent" in hitl_agents) else None,
        #after_agent_callback=make_hitl_after_callback(handler, HITLAction.APPROVE) if (hitl_enabled and "ResearchAgent" in hitl_agents) else None,
    )

    hypotheses_base_tools = [
        *retrieval_toolset_instance,
        AgentTool(agent=research_agent),
    ]
    hypotheses_agent = LlmAgent(
        name="HypothesesAgent",
        model=LiteLlm(model=MODEL),
        instruction=hypotheses_instruction,
        description="Agent to generate actionable scientific hypotheses grounded in available tools and literature",
        output_key="hypotheses",
        tools=_agent_tools("HypothesesAgent", hypotheses_base_tools),
        #before_agent_callback=make_hitl_before_callback(handler) if (hitl_enabled and "HypothesesAgent" in hitl_agents) else None,
        #after_agent_callback=make_hitl_after_callback(handler, HITLAction.APPROVE) if (hitl_enabled and "HypothesesAgent" in hitl_agents) else None,
    )

    tool_retriever_agent = LlmAgent(
        name='ToolRetrieverAgent',
        model=LiteLlm(model=MODEL),
        instruction=tool_retriever_instruction,
        description="Agent to retrieve relevant MCP servers from RAG database of MCP tools for given task.",
        output_schema=RetrievalFinalResult,
        tools=_agent_tools("ToolRetrieverAgent", retrieval_toolset_instance),
        output_key="retrieved_tools",
        before_agent_callback=make_hitl_before_callback(handler) if (hitl_enabled and "ToolRetrieverAgent" in hitl_agents) else None,
        #after_agent_callback=make_hitl_after_callback(handler, HITLAction.APPROVE) if (hitl_enabled and "ToolRetrieverAgent" in hitl_agents) else None,
    )

    fedot_agent = LlmAgent(
        name="ExperimentAgent",
        model=LiteLlm(model=MODEL),
        instruction=fedot_instruction,
        description="Agent to invoke MAS for solving given task. Uses MCP tools",
        output_key="fedot_results",
        tools=_agent_tools("ExperimentAgent", fedot_toolset_instance),
        before_agent_callback=make_hitl_before_callback(handler) if (hitl_enabled and "ExperimentAgent" in hitl_agents) else None,
        #after_agent_callback=make_hitl_after_callback(handler, HITLAction.APPROVE) if (hitl_enabled and "ExperimentAgent" in hitl_agents) else None,
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
        #before_agent_callback=make_hitl_before_callback(handler) if (hitl_enabled and "PlannerAgent" in hitl_agents) else None,
        #after_agent_callback=make_hitl_after_callback(handler, HITLAction.APPROVE) if (hitl_enabled and "PlannerAgent" in hitl_agents) else None,

        # only for SessionAgent
        hitl_handler=handler if (hitl_enabled and "PlannerAgent" in hitl_agents) else None,
    )    

    orchestrator_agent = LlmAgent(
        name="OrchestratorAgent",
        model=LiteLlm(model=MODEL),
        instruction=orchestrator_instruction,
        description="Main Orchestrator Agent",
        output_key="orchestrator_answer",
        tools=_agent_tools("OrchestratorAgent", [AgentTool(agent=planner_agent),AgentTool(agent=hypotheses_agent), AgentTool(agent=research_agent), AgentTool(agent=task_execution_agent)]),
        #before_agent_callback=make_hitl_before_callback(handler) if (hitl_enabled and "OrchestratorAgent" in hitl_agents) else None,
        #after_agent_callback=make_hitl_after_callback(handler, HITLAction.APPROVE) if (hitl_enabled and "OrchestratorAgent" in hitl_agents) else None,
    )

    track_adk_agent_recursive(orchestrator_agent, multi_agent_tracer)

    return {
        "hypotheses_agent": hypotheses_agent,
        "research_agent": research_agent,
        "tool_retriever_agent": tool_retriever_agent,
        "fedot_agent": fedot_agent,
        "task_execution_agent": task_execution_agent,
        "planner_agent": planner_agent,
        "orchestrator_agent": orchestrator_agent,
    }


# Default agents (without HITL handler — uses ConsoleHITLHandler if HITL enabled)
_default_agents = asyncio.run(create_agents())

hypotheses_agent = _default_agents["hypotheses_agent"]
research_agent = _default_agents["research_agent"]
tool_retriever_agent = _default_agents["tool_retriever_agent"]
fedot_agent = _default_agents["fedot_agent"]
task_execution_agent = _default_agents["task_execution_agent"]
planner_agent = _default_agents["planner_agent"]
orchestrator_agent = _default_agents["orchestrator_agent"]
