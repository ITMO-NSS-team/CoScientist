import asyncio

from CoScientist.integrations.codesynapse.delivery import TraceDeliveryClient
from CoScientist.integrations.codesynapse.models import TraceEvent


class _Response:
    status_code = 204


def test_delivery_acknowledges_only_successful_callback():
    async def scenario():
        captured = {}

        async def post(url, *, headers, json):
            captured.update(url=url, headers=headers, body=json)
            return _Response()

        client = TraceDeliveryClient(
            callback_url="http://codesynapse/internal/events",
            capability_token="capability",
            post=post,
        )
        delivered = await client.deliver([
            TraceEvent(
                event_id="event-1", run_id="run-1", sequence=1,
                tenant_id="tenant-1", project_id="project-1", type="run.started",
            )
        ])

        assert delivered
        assert captured["headers"]["Authorization"] == "Bearer capability"
        assert captured["body"]["events"][0]["event_id"] == "event-1"

    asyncio.run(scenario())
