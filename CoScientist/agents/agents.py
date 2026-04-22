
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.lite_llm import LiteLlm 
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools import FunctionTool
from google.genai import types
from google.adk.planners import BasePlanner, BuiltInPlanner, PlanReActPlanner

import litellm

from CoScientist.config import get_settings
from CoScientist.agents.prompts import hypotheses_instruction, research_instruction, fedot_instruction, orchestrator_instruction, tool_retriever_instruction, tool_reranker_instruction
from CoScientist.tools import fedot_toolset_instance, websearch_toolset_instance, retrieval_toolset_instance
from CoScientist.storage import RetrievalFinalResult, ToolRanking
from CoScientist.logging import multi_agent_tracer
from CoScientist.agents.callbacks import before_tool_reranker_model, after_tool_reranker_agent

from opik.integrations.adk import track_adk_agent_recursive


from typing import Dict, Any
import uuid
import os
import asyncio
import json


settings = get_settings()

MODEL = settings.llm.main_model
litellm.api_key = settings.llm.openai_api_key

hypotheses_agent = LlmAgent(
    name="HypothesesAgent",
    model=LiteLlm(model=MODEL),
    instruction=hypotheses_instruction,
    description="Agent to generate scientific hypotheses and ideas for given task",
    output_key="hypotheses"
)

research_agent = LlmAgent(
    name="ResearchAgent",
    model=LiteLlm(model=MODEL),
    instruction=research_instruction,
    description="Agent to answer questions and knowledge mining using Literature and Web Search.",
    output_key="search_results",
    tools=[websearch_toolset_instance]
)


thinking_config = types.ThinkingConfig(
    include_thoughts=False,   # Ask the model to include its thoughts in the response
    thinking_budget=256      # Limit the 'thinking' to 256 tokens (adjust as needed)
)

planner = BuiltInPlanner(
    thinking_config=thinking_config
)

tool_retriever_agent = LlmAgent(
    name='ToolRetriever',
    model=LiteLlm(model=MODEL),
    instruction=tool_retriever_instruction,
    description="Agent to retrieve relevant MCP servers from RAG database of MCP tools for given task.",
    # planner=planner,
    # output_schema=RetrievalFinalResult,
    tools=retrieval_toolset_instance,
    output_key="retrieved_tools"
)

tool_reranker_agent = LlmAgent(
    name='ToolReranker',
    model=LiteLlm(model=MODEL),
    instruction=tool_reranker_instruction,
    description="Agent to rerank retrieved MCP servers from RAG database of MCP tools for given task.",
    # planner=planner,
    output_schema=ToolRanking,
    before_model_callback=before_tool_reranker_model,
    after_agent_callback=after_tool_reranker_agent,
    output_key="reranked_tools"
)

fedot_agent = LlmAgent(
    name="ExperimentAgent",
    model=LiteLlm(model=MODEL),
    instruction=fedot_instruction,
    description="Agent to invoke MAS for solving given task. Uses MCP tools",
    output_key="fedot_results",
    tools=fedot_toolset_instance
)

task_execution_agent = SequentialAgent(
    name="TaskExecutorAgent",
    sub_agents=[tool_retriever_agent, tool_reranker_agent, fedot_agent],
    description="Agent to complete experiments and run calculations. Use it for any computation and idea validation. It can use a lot of MCP tools",
)

orchestrator_agent = LlmAgent(
    name="OrchestratorAgent",
    model=LiteLlm(model=MODEL),
    instruction=orchestrator_instruction,
    description="Main Orchestrator Agent",
    tools=[AgentTool(agent=hypotheses_agent), AgentTool(agent=research_agent), AgentTool(agent=task_execution_agent)],
)

track_adk_agent_recursive(orchestrator_agent, multi_agent_tracer)



