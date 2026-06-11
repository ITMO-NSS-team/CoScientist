"""Reusable factory for wrapping any ADK agent as an A2A FastAPI application."""
import os
from typing import Any, Optional

# ADK marks its A2A executor as experimental; suppress unless the caller opted in.
if not os.getenv("ADK_A2A_EXPERIMENTAL_WARNINGS"):
    os.environ.setdefault("ADK_SUPPRESS_A2A_EXPERIMENTAL_FEATURE_WARNINGS", "1")

from a2a.server.apps.jsonrpc.fastapi_app import A2AFastAPIApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard
from fastapi import FastAPI
from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor
from google.adk.agents.base_agent import BaseAgent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService


# Substrings that mark a settings key as sensitive; matched case-insensitively.
_SECRET_HINTS = ("key", "password", "secret", "token", "login", "credential")
_REDACTED = "***redacted***"


def _redact(value: Any) -> Any:
    """Recursively mask values whose key looks like a secret.

    Used so we never ship API keys / passwords to the Opik dashboard as trace
    metadata. Matches by key name, so it also covers nested third-party settings.
    """
    if isinstance(value, dict):
        return {
            k: (_REDACTED if any(h in k.lower() for h in _SECRET_HINTS) else _redact(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


def _attach_opik_tracer(agent: BaseAgent, app_name: str) -> None:
    """Attach an Opik tracer recursively so all events/callbacks land in the
    Opik dashboard (same project as the in-process orchestrator).

    Disabled by setting A2A_DISABLE_OPIK=1 (useful for offline local debugging).
    """
    if os.getenv("A2A_DISABLE_OPIK"):
        return
    try:
        from opik.integrations.adk import OpikTracer, track_adk_agent_recursive

        from CoScientist.config import get_settings

        settings = get_settings()
        tracer = OpikTracer(
            name=f"a2a-{app_name}",
            metadata=_redact(settings.model_dump()),
            project_name="adk-coscientist",
        )
        track_adk_agent_recursive(agent, tracer)
    except Exception as exc:  # never let tracing break the server
        import logging

        logging.getLogger(__name__).warning(
            "Opik tracing disabled for %s: %s", app_name, exc
        )


def make_a2a_app(
    agent: BaseAgent,
    agent_card: AgentCard,
    app_name: str,
    *,
    session_service: Optional[InMemorySessionService] = None,
) -> FastAPI:
    """Wrap an ADK agent as an A2A-compatible FastAPI application.

    Args:
        agent: The ADK agent to expose.
        agent_card: A2A AgentCard describing this agent.
        app_name: Unique name for the ADK Runner app.
        session_service: Optional session service (creates InMemory one if omitted).

    Returns:
        A FastAPI application implementing the A2A JSON-RPC protocol.
    """
    _attach_opik_tracer(agent, app_name)
    from CoScientist.a2a.event_logger import EventLoggerPlugin
    from CoScientist.graph.emitter import GraphEmitterPlugin

    runner = Runner(
        agent=agent,
        app_name=app_name,
        session_service=session_service or InMemorySessionService(),
        artifact_service=InMemoryArtifactService(),
        plugins=[EventLoggerPlugin(), GraphEmitterPlugin()],
    )
    executor = A2aAgentExecutor(runner=runner)
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )
    return A2AFastAPIApplication(agent_card=agent_card, http_handler=handler).build()
