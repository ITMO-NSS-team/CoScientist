"""Agent definitions (the LlmAgent/SequentialAgent instances)."""
from CoScientist.agents.definitions.coder_agent import coder_agent
from CoScientist.agents.definitions.hypotheses_agent import hypotheses_agent
from CoScientist.agents.definitions.medical_agent import medical_agent
from CoScientist.agents.definitions.orchestrator_agent import orchestrator_agent
from CoScientist.agents.definitions.planner_agent import planner_agent
from CoScientist.agents.definitions.research_agent import research_agent
from CoScientist.agents.definitions.task_execution_agent import (
    fedot_agent,
    task_execution_agent,
    tool_agent,
    tool_reranker_agent,
    tool_retriever_agent,
    tool_websearcher_agent,
)

__all__ = [
    "coder_agent",
    "hypotheses_agent",
    "medical_agent",
    "orchestrator_agent",
    "planner_agent",
    "research_agent",
    "task_execution_agent",
    "fedot_agent",
    "tool_agent",
    "tool_reranker_agent",
    "tool_retriever_agent",
    "tool_websearcher_agent",
]
