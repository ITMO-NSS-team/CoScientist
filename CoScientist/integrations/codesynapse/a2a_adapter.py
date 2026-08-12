"""A2A 0.3 adapter for persistent façade task records."""

from __future__ import annotations

from typing import Any

from a2a.server.agent_execution.agent_executor import AgentExecutor
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks.task_store import TaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Artifact,
    DataPart,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
)

from CoScientist.integrations.codesynapse.auth import sha256_json, sha256_text
from CoScientist.integrations.codesynapse.facade import CodesynapseFacade, StartRequest
from CoScientist.integrations.codesynapse.models import A2ATaskRecord, ArtifactPart, RunState


def _a2a_state(state: RunState) -> TaskState:
    if state == RunState.COMPLETED:
        return TaskState.completed
    if state == RunState.CANCELLED:
        return TaskState.canceled
    if state in {RunState.FAILED, RunState.INTERRUPTED}:
        return TaskState.failed
    return TaskState.working


def _artifact_parts(part: ArtifactPart) -> list[Any]:
    if part.text is not None:
        return [TextPart(text=part.text)]
    if part.data is not None:
        return [DataPart(data=part.data)]
    return [DataPart(data={"artifact_id": part.artifact_id, "checksum_sha256": part.checksum_sha256})]


def task_from_record(record: A2ATaskRecord, *, context_id: str) -> Task:
    """Produce a current A2A task view from durable façade state."""

    artifacts = []
    if record.artifacts is not None:
        for part in (
            record.artifacts.final_report,
            record.artifacts.structured_result,
            record.artifacts.artifacts_manifest,
            record.artifacts.error,
        ):
            if part is not None:
                artifacts.append(
                    Artifact(
                        artifact_id=f"{record.a2a_task_id}:{part.name}",
                        name=part.name,
                        description=part.mime_type,
                        parts=_artifact_parts(part),
                    )
                )
    return Task(
        id=record.a2a_task_id,
        context_id=context_id,
        status=TaskStatus(state=_a2a_state(record.state)),
        artifacts=artifacts or None,
        metadata={
            "external_run_id": record.external_run_id,
            "coscientist_run_id": record.coscientist_run_id,
        },
    )


class FacadeTaskStore(TaskStore):
    """A2A task store view backed by the façade's durable task repository."""

    def __init__(self, facade: CodesynapseFacade, verifier=None) -> None:
        self._facade = facade
        self._verifier = verifier

    async def save(self, task: Task, context=None) -> None:  # A2A persists through the façade instead.
        return None

    async def get(self, task_id: str, context=None) -> Task | None:
        record = await self._facade.get_task(task_id)
        return task_from_record(record, context_id=record.coscientist_run_id) if record else None

    async def delete(self, task_id: str, context=None) -> None:
        return None


class FacadeAgentExecutor(AgentExecutor):
    """Starts a façade task and immediately publishes its A2A ``working`` view."""

    def __init__(self, facade: CodesynapseFacade, verifier=None) -> None:
        self._facade = facade
        self._verifier = verifier

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        metadata = context.metadata
        claims = None
        if self._verifier is not None:
            encoded_token = metadata.get("integration_jwt")
            if not encoded_token:
                raise ValueError("Codesynapse metadata requires integration_jwt")
            claims = await self._verifier.verify(str(encoded_token))
        for name in ("external_run_id", "tenant_id", "project_id"):
            if not metadata.get(name):
                raise ValueError(f"Codesynapse metadata requires {name}")
        if claims is not None and any(
            getattr(claims, name) != str(metadata[name])
            for name in ("external_run_id", "tenant_id", "project_id")
        ):
            raise ValueError("JWT scope does not match A2A metadata")
        research_request = context.get_user_input()
        integration_context = metadata.get("context") or {}
        if not isinstance(integration_context, dict):
            raise ValueError("Codesynapse metadata context must be a JSON object")
        if claims is not None:
            trace_callback_url = str(metadata.get("trace_callback_url") or "")
            trace_capability_token = str(metadata.get("trace_capability_token") or "")
            control_token_hash = str(metadata.get("control_token_hash") or "")
            signed_values_match = (
                claims.research_request_sha256 == sha256_text(research_request)
                and claims.context_sha256 == sha256_json(integration_context)
                and claims.trace_callback_url == trace_callback_url
                and claims.trace_capability_token_hash == sha256_text(trace_capability_token)
                and claims.control_token_hash == control_token_hash
            )
            if not signed_values_match:
                raise ValueError("JWT does not bind the supplied Codesynapse context")
        record = await self._facade.start(
            StartRequest(
                external_run_id=str(metadata["external_run_id"]),
                tenant_id=str(metadata["tenant_id"]),
                project_id=str(metadata["project_id"]),
                research_request=research_request,
                context=integration_context,
                control_token_hash=metadata.get("control_token_hash"),
                a2a_task_id=context.task_id,
                trace_callback_url=metadata.get("trace_callback_url"),
                trace_capability_token=metadata.get("trace_capability_token"),
            )
        )
        await event_queue.enqueue_event(task_from_record(record, context_id=context.context_id or record.coscientist_run_id))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        if context.task_id is None:
            raise ValueError("task id is required for cancellation")
        await self._facade.cancel(context.task_id)
        record = await self._facade.get_task(context.task_id)
        if record is not None:
            await event_queue.enqueue_event(task_from_record(record, context_id=context.context_id or record.coscientist_run_id))


def make_agent_card(url: str) -> AgentCard:
    """Return the fixed, versioned card registered in a Codesynapse tenant."""

    return AgentCard(
        name="coscientist",
        description="Long-running scientific research pipeline with Codesynapse HITL and task polling.",
        url=url,
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/markdown", "application/json"],
        skills=[
            AgentSkill(
                id="research",
                name="research",
                description="Run the complete CoScientist research pipeline.",
                tags=["research", "long-running", "hitl"],
            )
        ],
    )
