import asyncio

from CoScientist.integrations.codesynapse.delivery import TraceOutboxDispatcher
from CoScientist.integrations.codesynapse.models import TraceEvent
from CoScientist.integrations.codesynapse.store import InMemoryIntegrationStore


def test_outbox_dispatcher_delivers_pending_events_once_in_sequence_order():
    async def scenario():
        store = InMemoryIntegrationStore()
        for sequence in (2, 1):
            await store.append_event(
                TraceEvent(event_id=f"event-{sequence}", run_id="run-1", sequence=sequence, tenant_id="tenant-1", project_id="project-1", type="tool.completed")
            )
        delivered = []

        class Client:
            async def deliver(self, events):
                delivered.append([event.sequence for event in events])
                return True

        dispatcher = TraceOutboxDispatcher(store, Client())
        assert await dispatcher.flush_run("run-1") == 2
        assert await dispatcher.flush_run("run-1") == 0
        assert delivered == [[1, 2]]

    asyncio.run(scenario())
