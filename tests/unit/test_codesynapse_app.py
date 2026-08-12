from fastapi.testclient import TestClient

from CoScientist.integrations.codesynapse.app import create_app
from CoScientist.integrations.codesynapse.facade import CodesynapseFacade
from CoScientist.integrations.codesynapse.settings import CodesynapseIntegrationSettings
from CoScientist.integrations.codesynapse.store import InMemoryIntegrationStore


class _Executor:
    async def execute(self, request, hitl_handler):
        return "report"


def test_facade_app_exposes_a2a_agent_card_and_health_check():
    store = InMemoryIntegrationStore()
    app = create_app(
        CodesynapseIntegrationSettings(a2a_public_url="http://testserver"),
        facade=CodesynapseFacade(store=store, executor=_Executor()),
        store=store,
    )
    client = TestClient(app)

    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/.well-known/agent-card.json").json()["name"] == "coscientist"


def test_a2a_start_returns_working_task_before_pipeline_completes():
    class BlockingExecutor:
        async def execute(self, request, hitl_handler):
            import asyncio

            await asyncio.sleep(0.05)
            return "report"

    store = InMemoryIntegrationStore()
    app = create_app(
        CodesynapseIntegrationSettings(a2a_public_url="http://testserver"),
        facade=CodesynapseFacade(store=store, executor=BlockingExecutor()),
        store=store,
    )
    client = TestClient(app)
    response = client.post("/", json={
        "jsonrpc": "2.0",
        "id": "request-1",
        "method": "message/send",
        "params": {
            "message": {"messageId": "message-1", "role": "user", "parts": [{"kind": "text", "text": "Find a hypothesis"}]},
            "metadata": {"external_run_id": "external-1", "tenant_id": "tenant-1", "project_id": "project-1"},
            "configuration": {"blocking": False},
        },
    })

    assert response.status_code == 200
    assert response.json()["result"]["status"]["state"] == "working"
