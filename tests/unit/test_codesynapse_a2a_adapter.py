import asyncio
from types import SimpleNamespace

import pytest

from CoScientist.integrations.codesynapse.a2a_adapter import FacadeAgentExecutor, task_from_record
from CoScientist.integrations.codesynapse.auth import claims_from_payload, sha256_json, sha256_text
from CoScientist.integrations.codesynapse.facade import CodesynapseFacade
from CoScientist.integrations.codesynapse.models import A2ATaskRecord, ArtifactPart, RunState, TerminalArtifacts
from CoScientist.integrations.codesynapse.store import InMemoryIntegrationStore


def test_completed_record_becomes_a2a_completed_task_with_named_artifact():
    task = task_from_record(
        A2ATaskRecord(
            a2a_task_id="task-1", external_run_id="external-1", coscientist_run_id="run-1",
            state=RunState.COMPLETED,
            artifacts=TerminalArtifacts(
                state=RunState.COMPLETED,
                final_report=ArtifactPart(name="final_report", mime_type="text/markdown", text="# Report"),
            ),
        ),
        context_id="context-1",
    )

    assert task.status.state == "completed"
    assert task.artifacts[0].name == "final_report"
    assert task.artifacts[0].parts[0].root.text == "# Report"


def test_interrupted_record_becomes_a2a_failed_task_with_structured_error():
    task = task_from_record(
        A2ATaskRecord(
            a2a_task_id="task-1", external_run_id="external-1", coscientist_run_id="run-1",
            state=RunState.INTERRUPTED,
            artifacts=TerminalArtifacts(
                state=RunState.INTERRUPTED,
                error=ArtifactPart(name="error", mime_type="application/json", data={"error_code": "interrupted"}),
            ),
        ),
        context_id="context-1",
    )

    assert task.status.state == "failed"
    assert task.artifacts[0].parts[0].root.data["error_code"] == "interrupted"


def test_signed_a2a_context_must_bind_all_mutable_run_inputs():
    async def scenario():
        request_text = "Find a hypothesis"
        integration_context = {"artifact": "s3://bucket/input"}
        trace_token = "trace-capability"
        control_hash = sha256_text("control-capability")
        claims = claims_from_payload({
            "iss": "codesynapse", "aud": "coscientist", "tenant_id": "tenant-1",
            "project_id": "project-1", "external_run_id": "external-1",
            "research_request_sha256": sha256_text(request_text),
            "context_sha256": sha256_json(integration_context),
            "trace_callback_url": "http://codesynapse/internal/trace",
            "trace_capability_token_hash": sha256_text(trace_token),
            "control_token_hash": control_hash,
        })

        class Verifier:
            async def verify(self, token):
                return claims

        class Executor:
            async def execute(self, request, hitl_handler):
                return "report"

        events = []

        class Queue:
            async def enqueue_event(self, event):
                events.append(event)

        request_context = SimpleNamespace(
            metadata={
                "integration_jwt": "signed", "external_run_id": "external-1", "tenant_id": "tenant-1",
                "project_id": "project-1", "context": integration_context,
                "trace_callback_url": "http://codesynapse/internal/trace", "trace_capability_token": trace_token,
                "control_token_hash": control_hash,
            },
            task_id="task-1", context_id="context-1", get_user_input=lambda: request_text,
        )
        adapter = FacadeAgentExecutor(CodesynapseFacade(store=InMemoryIntegrationStore(), executor=Executor()), verifier=Verifier())
        await adapter.execute(request_context, Queue())
        assert events[0].status.state == "working"

        request_context.metadata["context"] = {"artifact": "s3://attacker/input"}
        with pytest.raises(ValueError, match="does not bind"):
            await adapter.execute(request_context, Queue())

    asyncio.run(scenario())
