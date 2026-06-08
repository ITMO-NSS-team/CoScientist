"""Entry point for `adk web` / `adk api_server`.

Set A2A_MODE=1 to use the A2A orchestrator (sub-agents must be running).
Default (A2A_MODE unset) uses the in-process ADK orchestrator.
"""
import os

if os.getenv("A2A_MODE"):
    from CoScientist.a2a.orchestrator import orchestrator_a2a_agent as root_agent
else:
    from CoScientist.agents import orchestrator_agent as root_agent