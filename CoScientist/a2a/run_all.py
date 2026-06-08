"""Launch all CoScientist A2A agent servers in a single process.

Usage (from /app):
    python -m CoScientist.a2a.run_all

Each sub-agent runs on its own port; the orchestrator runs on port 8000.
Port defaults can be overridden via environment variables (see config.py).
"""
from __future__ import annotations

import asyncio
import logging

import uvicorn
from a2a.types import AgentCard, AgentCapabilities, AgentSkill

from CoScientist.a2a.config import AGENT_PORTS, AGENT_URLS
from CoScientist.a2a.orchestrator import orchestrator_a2a_agent
from CoScientist.a2a.server import make_a2a_app
from CoScientist.a2a.servers.hypotheses import app as hypotheses_app
from CoScientist.a2a.servers.medical import app as medical_app
from CoScientist.a2a.servers.planner import app as planner_app
from CoScientist.a2a.servers.research import app as research_app
from CoScientist.a2a.servers.task_execution import app as task_execution_app

logger = logging.getLogger(__name__)

_SERVERS: list[tuple[str, object, int]] = [
    ("PlannerAgent",      planner_app,        AGENT_PORTS["planner"]),
    ("HypothesesAgent",   hypotheses_app,     AGENT_PORTS["hypotheses"]),
    ("ResearchAgent",     research_app,       AGENT_PORTS["research"]),
    ("TaskExecutorAgent", task_execution_app, AGENT_PORTS["task_execution"]),
    ("MedicalAgent",      medical_app,        AGENT_PORTS["medical"]),
]


def _build_orchestrator_app():
    card = AgentCard(
        name="OrchestratorAgent",
        description="Main CoScientist Orchestrator — coordinates all sub-agents via A2A",
        url=AGENT_URLS["orchestrator"],
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=True),
        defaultInputModes=["text/plain"],
        defaultOutputModes=["text/plain"],
        skills=[
            AgentSkill(
                id="orchestrate",
                name="Orchestrate",
                description="Coordinate scientific research agents to solve complex tasks",
                tags=["orchestration", "science", "multi-agent"],
            )
        ],
    )
    return make_a2a_app(orchestrator_a2a_agent, card, "orchestrator")


async def _serve(app, port: int, label: str) -> None:
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)
    logger.info("Starting %s on port %d", label, port)
    await server.serve()


async def main() -> None:
    orchestrator_app = _build_orchestrator_app()
    tasks = [
        _serve(app, port, label)
        for label, app, port in _SERVERS
    ]
    tasks.append(_serve(orchestrator_app, AGENT_PORTS["orchestrator"], "OrchestratorAgent"))

    print("Starting CoScientist A2A agents:")
    for label, _, port in _SERVERS:
        print(f"  {label:<22} → http://localhost:{port}/")
    print(f"  {'OrchestratorAgent':<22} → http://localhost:{AGENT_PORTS['orchestrator']}/")
    print()

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
