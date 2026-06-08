"""Backward-compatible re-export of all agents.

Each agent is now defined in its own module; import from there for clarity.
This file exists so existing code using
  `from CoScientist.agents.agents import <agent>`
continues to work without changes.
"""
from CoScientist.agents.hypotheses_agent import hypotheses_agent
from CoScientist.agents.medical_agent import medical_agent
from CoScientist.agents.orchestrator_agent import orchestrator_agent
from CoScientist.agents.planner_agent import planner_agent
from CoScientist.agents.research_agent import research_agent
from CoScientist.agents.task_execution_agent import (
    fedot_agent,
    task_execution_agent,
    tool_agent,
    tool_reranker_agent,
    tool_retriever_agent,
    tool_websearcher_agent,
)
from CoScientist.logging import multi_agent_tracer
from opik.integrations.adk import track_adk_agent_recursive

track_adk_agent_recursive(orchestrator_agent, multi_agent_tracer)

__all__ = [
    "hypotheses_agent",
    "medical_agent",
    "orchestrator_agent",
    "planner_agent",
    "research_agent",
    "fedot_agent",
    "task_execution_agent",
    "tool_agent",
    "tool_reranker_agent",
    "tool_retriever_agent",
    "tool_websearcher_agent",
]
