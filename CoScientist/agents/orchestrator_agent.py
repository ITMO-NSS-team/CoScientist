"""Re-export shim for the in-process OrchestratorAgent.

The agent is defined in :mod:`CoScientist.agents.agents` (single source of
truth, driven by the agent catalog). For the A2A orchestrator (sub-agents as
remote services) see :mod:`CoScientist.a2a.orchestrator`.
"""
from CoScientist.agents.agents import orchestrator_agent

__all__ = ["orchestrator_agent"]
