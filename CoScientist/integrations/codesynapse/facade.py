"""Long-running, idempotent façade over the full CoScientist pipeline."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from CoScientist.integrations.codesynapse.executor import PipelineExecutor
from CoScientist.integrations.codesynapse.hitl import CodesynapseHITLHandler, HITLRequestTimeout
from CoScientist.integrations.codesynapse.models import (
    A2ATaskRecord,
    ArtifactPart,
    IntegrationRun,
    RunState,
    TerminalArtifacts,
)
from CoScientist.integrations.codesynapse.state import transition
from CoScientist.integrations.codesynapse.store import DuplicateIdentityError, IntegrationStore
from CoScientist.integrations.codesynapse.trace import TraceRecorder


class StartRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    external_run_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    research_request: str = Field(min_length=1)
    context: dict[str, Any] = Field(default_factory=dict)
    coscientist_run_id: str | None = None
    control_token_hash: str | None = None
    trace_callback_url: str | None = Field(default=None, exclude=True)
    trace_capability_token: str | None = Field(default=None, exclude=True)
    trace_recorder: object | None = Field(default=None, exclude=True)
    a2a_task_id: str | None = None


class CodesynapseFacade:
    """Creates one durable task per external identity and executes it asynchronously."""

    def __init__(
        self,
        *,
        store: IntegrationStore,
        executor: PipelineExecutor,
        delivery_factory: Callable[[StartRequest], object] | None = None,
        lease_ttl_seconds: float = 3600.0,
    ) -> None:
        self._store = store
        self._executor = executor
        self._delivery_factory = delivery_factory
        self._jobs: dict[str, asyncio.Task[None]] = {}
        self._handlers: dict[str, CodesynapseHITLHandler] = {}
        self._task_ids_by_run: dict[str, str] = {}
        self._cancelling_tasks: set[str] = set()
        self._start_lock = asyncio.Lock()
        self._lease_owner_id = str(uuid4())
        self._lease_ttl_seconds = lease_ttl_seconds

    def set_delivery_factory(self, delivery_factory: Callable[[StartRequest], object]) -> None:
        """Attach the transport adapter without coupling façade state to HTTP."""

        self._delivery_factory = delivery_factory

    async def start(self, request: StartRequest, *, run_in_background: bool = True) -> A2ATaskRecord:
        """Idempotently create an A2A task, returning a working task immediately."""

        async with self._start_lock:
            existing = await self._store.get_run(request.external_run_id)
            if existing is not None:
                if existing.a2a_task_id is None:
                    raise RuntimeError("existing integration run has no A2A task id")
                task = await self._store.get_task(existing.a2a_task_id)
                if task is None:
                    raise RuntimeError("existing integration run has no persisted task")
                return task

            coscientist_run_id = request.coscientist_run_id or str(uuid4())
            a2a_task_id = request.a2a_task_id or str(uuid4())
            run = IntegrationRun(
                external_run_id=request.external_run_id,
                tenant_id=request.tenant_id,
                project_id=request.project_id,
                state=RunState.STARTING,
                a2a_task_id=a2a_task_id,
                coscientist_run_id=coscientist_run_id,
                control_token_hash=request.control_token_hash,
            )
            task = A2ATaskRecord(
                a2a_task_id=a2a_task_id,
                external_run_id=request.external_run_id,
                coscientist_run_id=coscientist_run_id,
                state=RunState.RUNNING,
            )
            # Write the dependent task first: a process crash can then leave an
            # auditable orphan, but never a durable run that cannot answer
            # canonical A2A tasks/get. The unique indexes decide the winner.
            try:
                await self._store.save_task(task)
            except DuplicateIdentityError:
                existing_task = await self._store.get_task_by_external_run(request.external_run_id)
                if existing_task is None:
                    raise
                task = existing_task
                run = run.model_copy(update={
                    "a2a_task_id": task.a2a_task_id,
                    "coscientist_run_id": task.coscientist_run_id,
                })
            try:
                await self._store.create_run(run)
            except DuplicateIdentityError:
                # The process-local lock cannot coordinate two façade replicas.
                # MongoDB's unique external_run_id index is the cross-process
                # idempotency boundary; return the task created by the winner.
                existing = await self._store.get_run(request.external_run_id)
                if existing is None or existing.a2a_task_id is None:
                    raise
                task = await self._store.get_task(existing.a2a_task_id)
                if task is None:
                    raise RuntimeError("existing integration run has no persisted task")
                return task
            a2a_task_id = task.a2a_task_id
            coscientist_run_id = task.coscientist_run_id
            self._task_ids_by_run[coscientist_run_id] = a2a_task_id

        if run_in_background:
            self._jobs[a2a_task_id] = asyncio.create_task(self._execute(request, a2a_task_id, coscientist_run_id))
            return task
        await self._execute(request, a2a_task_id, coscientist_run_id)
        completed = await self._store.get_task(a2a_task_id)
        if completed is None:
            raise RuntimeError("task disappeared during execution")
        return completed

    async def get_task(self, a2a_task_id: str) -> A2ATaskRecord | None:
        return await self._store.get_task(a2a_task_id)

    async def cancel(self, a2a_task_id: str) -> bool:
        """Idempotently cancel a live task and publish a terminal cancelled view."""

        task = await self._store.get_task(a2a_task_id)
        if task is None:
            return False
        if task.state in {RunState.COMPLETED, RunState.FAILED, RunState.CANCELLED, RunState.INTERRUPTED}:
            return task.state == RunState.CANCELLED
        run = await self._store.get_run(task.external_run_id)
        if run is not None and run.state in {RunState.RUNNING, RunState.WAITING_FOR_HUMAN}:
            run.state = transition(run.state, RunState.CANCELLING)
            run.updated_at = datetime.now(timezone.utc)
            await self._store.save_run(run)
        handler = self._handlers.get(task.coscientist_run_id)
        if handler is not None:
            await handler.cancel_pending()
        self._cancelling_tasks.add(a2a_task_id)
        job = self._jobs.pop(a2a_task_id, None)
        if job is not None:
            job.cancel()
            await asyncio.gather(job, return_exceptions=True)
        persisted_task = await self._store.get_task(a2a_task_id)
        if persisted_task is not None and persisted_task.state in {
            RunState.COMPLETED,
            RunState.FAILED,
            RunState.CANCELLED,
            RunState.INTERRUPTED,
        }:
            return persisted_task.state == RunState.CANCELLED
        if run is not None:
            run.state = RunState.CANCELLED
            run.terminal_reason = "cancelled"
            run.updated_at = datetime.now(timezone.utc)
            await self._store.save_run(run)
        await self._store.save_task(
            self._error_task(
                task.a2a_task_id,
                task.external_run_id,
                task.coscientist_run_id,
                RunState.CANCELLED,
                "cancelled",
            )
        )
        return True

    async def resolve_hitl(self, coscientist_run_id: str, request_id: str, response) -> bool:
        handler = self._handlers.get(coscientist_run_id)
        return await handler.resolve(request_id, response) if handler is not None else False

    async def cancel_by_run(self, coscientist_run_id: str) -> bool:
        task_id = self._task_ids_by_run.get(coscientist_run_id)
        return await self.cancel(task_id) if task_id else False

    async def _execute(self, request: StartRequest, a2a_task_id: str, coscientist_run_id: str) -> None:
        if not await self._store.acquire_run_lease(
            request.external_run_id, self._lease_owner_id, self._lease_ttl_seconds
        ):
            return
        run = await self._store.get_run(request.external_run_id)
        if run is None:
            await self._store.release_run_lease(request.external_run_id, self._lease_owner_id)
            return
        run.state = transition(run.state, RunState.RUNNING)
        run.updated_at = datetime.now(timezone.utc)
        await self._store.save_run(run)
        dispatcher = self._delivery_factory(request) if self._delivery_factory is not None else None

        async def flush_trace_event(event) -> None:
            if dispatcher is not None:
                await dispatcher.flush_run(coscientist_run_id)

        recorder = TraceRecorder(
            self._store,
            run_id=coscientist_run_id,
            tenant_id=request.tenant_id,
            project_id=request.project_id,
            on_event=flush_trace_event if dispatcher is not None else None,
        )
        async def emit_hitl_event(event_type: str, **fields: Any) -> None:
            current = await self._store.get_run(request.external_run_id)
            if current is not None:
                if event_type == "hitl.requested" and current.state == RunState.RUNNING:
                    current.state = transition(current.state, RunState.WAITING_FOR_HUMAN)
                    await self._store.save_run(current)
                elif event_type in {"hitl.resolved", "hitl.expired"} and current.state == RunState.WAITING_FOR_HUMAN:
                    current.state = transition(current.state, RunState.RUNNING)
                    await self._store.save_run(current)
            await recorder.emit(event_type, **fields)

        handler = CodesynapseHITLHandler(run_id=coscientist_run_id, emit=emit_hitl_event)
        self._handlers[coscientist_run_id] = handler
        await recorder.emit("run.started", data={"external_run_id": request.external_run_id})
        try:
            execution_request = request.model_copy(update={"coscientist_run_id": coscientist_run_id, "trace_recorder": recorder})
            report = await self._executor.execute(execution_request, handler)
            run.state = transition(run.state, RunState.COMPLETED)
            task = A2ATaskRecord(
                a2a_task_id=a2a_task_id,
                external_run_id=request.external_run_id,
                coscientist_run_id=coscientist_run_id,
                state=RunState.COMPLETED,
                artifacts=TerminalArtifacts(
                    state=RunState.COMPLETED,
                    final_report=ArtifactPart(name="final_report", mime_type="text/markdown", text=report),
                ),
            )
            await recorder.emit("run.completed")
        except asyncio.CancelledError:
            cancelled = a2a_task_id in self._cancelling_tasks
            state = RunState.CANCELLED if cancelled else RunState.INTERRUPTED
            error_code = "cancelled" if cancelled else "interrupted"
            run.state = state
            run.terminal_reason = error_code
            task = self._error_task(a2a_task_id, request.external_run_id, coscientist_run_id, state, error_code)
            await recorder.emit("run.cancelled" if cancelled else "run.failed", data={"error_code": error_code})
        except HITLRequestTimeout as exc:
            run.state = RunState.FAILED
            task = self._error_task(a2a_task_id, request.external_run_id, coscientist_run_id, RunState.FAILED, "hitl_timeout", str(exc))
            await recorder.emit("run.failed", data={"error_code": "hitl_timeout", "message": str(exc)})
        except Exception as exc:  # terminal failures are surfaced as structured error artifacts
            run.state = RunState.FAILED
            task = self._error_task(a2a_task_id, request.external_run_id, coscientist_run_id, RunState.FAILED, "execution_failed", str(exc))
            await recorder.emit("run.failed", data={"error_code": "execution_failed", "message": str(exc)})
        try:
            run.updated_at = datetime.now(timezone.utc)
            await self._store.save_run(run)
            await self._store.save_task(task)
        finally:
            await self._store.release_run_lease(request.external_run_id, self._lease_owner_id)
            self._jobs.pop(a2a_task_id, None)
            self._handlers.pop(coscientist_run_id, None)
            self._cancelling_tasks.discard(a2a_task_id)

    @staticmethod
    def _error_task(
        a2a_task_id: str,
        external_run_id: str,
        coscientist_run_id: str,
        state: RunState,
        error_code: str,
        message: str | None = None,
    ) -> A2ATaskRecord:
        return A2ATaskRecord(
            a2a_task_id=a2a_task_id,
            external_run_id=external_run_id,
            coscientist_run_id=coscientist_run_id,
            state=state,
            artifacts=TerminalArtifacts(
                state=state,
                error=ArtifactPart(
                    name="error",
                    mime_type="application/json",
                    data={"error_code": error_code, "message": message or error_code, "retryable": False},
                ),
            ),
        )

    async def interrupt_non_terminal_tasks(self) -> int:
        """MVP restart rule: pending in-memory execution cannot be resumed."""

        tasks = await self._store.non_terminal_tasks()
        jobs: list[asyncio.Task[None]] = []
        for task in tasks:
            job = self._jobs.pop(task.a2a_task_id, None)
            if job is not None:
                job.cancel()
                jobs.append(job)
        if jobs:
            await asyncio.gather(*jobs, return_exceptions=True)
        for task in tasks:
            run = await self._store.get_run(task.external_run_id)
            if run is not None:
                run.state = RunState.INTERRUPTED
                run.terminal_reason = "interrupted"
                run.updated_at = datetime.now(timezone.utc)
                await self._store.save_run(run)
            await self._store.save_task(
                self._error_task(
                    task.a2a_task_id,
                    task.external_run_id,
                    task.coscientist_run_id,
                    RunState.INTERRUPTED,
                    "interrupted",
                )
            )
        return len(tasks)
