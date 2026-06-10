"""Launch all CoScientist A2A agent servers in a single process.

Usage (from /app):
    python -m CoScientist.a2a.run_all

Each sub-agent runs on its own port; the orchestrator runs on port 8000.
Port defaults can be overridden via environment variables (see config.py).
"""
from __future__ import annotations

import asyncio
import logging
import signal

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


def _make_server(app, port: int) -> uvicorn.Server:
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)
    # We install a single shared signal handler below, so disable uvicorn's
    # per-server handlers (six servers each grabbing SIGINT shut down messily).
    server.config.install_signal_handlers = False
    return server


async def main() -> None:
    orchestrator_app = _build_orchestrator_app()

    specs = [
        (label, app, port) for label, app, port in _SERVERS
    ] + [("OrchestratorAgent", orchestrator_app, AGENT_PORTS["orchestrator"])]

    servers = [_make_server(app, port) for _, app, port in specs]

    # One shared shutdown path: signal all servers to exit gracefully.
    loop = asyncio.get_running_loop()

    def _request_shutdown() -> None:
        logger.info("Shutdown requested; stopping all A2A servers...")
        for server in servers:
            server.should_exit = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_shutdown)
        except NotImplementedError:  # e.g. on Windows
            signal.signal(sig, lambda *_: _request_shutdown())

    print("Starting CoScientist A2A agents:")
    for label, _, port in specs:
        print(f"  {label:<22} → http://localhost:{port}/")
    print()

    await asyncio.gather(*(server.serve() for server in servers))


if __name__ == "__main__":
    asyncio.run(main())
