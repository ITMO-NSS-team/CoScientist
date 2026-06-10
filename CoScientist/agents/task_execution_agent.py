"""Re-export shim for TaskExecutorAgent and its tool-pipeline sub-agents.

The agents are defined in :mod:`CoScientist.agents.agents` (single source of
truth, driven by the agent catalog). This module keeps the per-agent import
path stable for the A2A servers.
"""
from CoScientist.agents.agents import (
    fedot_agent,
    task_execution_agent,
    tool_agent,
    tool_reranker_agent,
    tool_retriever_agent,
    tool_websearcher_agent,
)

__all__ = [
    "task_execution_agent",
    "fedot_agent",
    "tool_agent",
    "tool_reranker_agent",
    "tool_retriever_agent",
    "tool_websearcher_agent",
]
