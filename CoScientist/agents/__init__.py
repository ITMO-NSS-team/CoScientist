"""LLM Agents module — agents are assembled from CoScientist/agents/system.yaml.

The YAML is the single source of truth for the system layout (agents, tools,
callbacks, prompts, HITL, A2A exposure). This module builds the in-process
system once and re-exports the agent instances under their historical names so
existing imports keep working.
"""
import os
from typing import Any

_AGENT_NAMES = {
    "orchestrator_agent": "root",
    "root_agent": "root",
    "planner_agent": "PlannerAgent",
    "hypotheses_agent": "HypothesesAgent",
    "research_agent": "ResearchAgent",
    "task_execution_agent": "TaskExecutorAgent",
    "medical_agent": "MedicalAgent",
    "coder_agent": "CoderAgent",
    "tool_agent": "ToolPreparerAgent",
    "tool_retriever_agent": "ToolRetrieverAgent",
    "tool_reranker_agent": "ToolReranker",
    "tool_websearcher_agent": "ToolWebSearcherAgent",
    "fedot_agent": "ExperimentAgent",
}

__all__ = list(_AGENT_NAMES)

_system = None


def _get_system():
    global _system
    if _system is None:
        from CoScientist.assembly import build_system

        _system = build_system()
        if not os.getenv("A2A_DISABLE_OPIK"):
            from CoScientist.logging import multi_agent_tracer
            from opik.integrations.adk import track_adk_agent_recursive

            track_adk_agent_recursive(_system.root, multi_agent_tracer)
    return _system


def __getattr__(name: str) -> Any:
    if name not in _AGENT_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    system = _get_system()
    target = _AGENT_NAMES[name]
    value = system.root if target == "root" else system.agent(target)
    globals()[name] = value
    return value
