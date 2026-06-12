"""Entry point for `adk web` / `adk api_server`.

Set A2A_MODE=1 to use the A2A orchestrator (sub-agents must be running).
Default (A2A_MODE unset) uses the in-process ADK orchestrator.

Exported as an ADK ``App`` so the event-logger plugin rides along: every
agent's thoughts, tool calls and tool results are printed to the console,
same as the A2A servers do. Disable with LOG_AGENT_EVENTS=0.
"""
import os

from google.adk.apps import App

from CoScientist.logging.event_logger import EventLoggerPlugin

if os.getenv("A2A_MODE"):
    from CoScientist.a2a.orchestrator import orchestrator_a2a_agent as root_agent
else:
    from CoScientist.agents import orchestrator_agent as root_agent

# adk web keys sessions by the agents-dir entry name, so App.name must match
# the directory name ("CoScientist").
app = App(name="CoScientist", root_agent=root_agent, plugins=[EventLoggerPlugin()])
