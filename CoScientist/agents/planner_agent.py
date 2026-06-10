"""Re-export shim for PlannerAgent.

The agent is defined in :mod:`CoScientist.agents.agents` (single source of
truth, driven by the agent catalog). This module keeps the per-agent import
path stable for the A2A servers.
"""
from CoScientist.agents.agents import planner_agent

__all__ = ["planner_agent"]
