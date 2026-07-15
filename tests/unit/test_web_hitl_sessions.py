import asyncio

from CoScientist.hitl.models import HITLAction, HITLRequest
from CoScientist.web.handler import WebHITLHandler


class _Socket:
    def __init__(self):
        self.messages = []

    async def send_json(self, payload):
        self.messages.append(payload)


def test_hitl_request_and_response_are_scoped_to_session():
    async def scenario():
        handler = WebHITLHandler()
        first_key = ("user_a", "session_a")
        second_key = ("user_b", "session_b")
        first_socket = _Socket()
        second_socket = _Socket()
        await handler.attach_websocket(first_socket, first_key)
        await handler.attach_websocket(second_socket, second_key)

        request_task = asyncio.create_task(handler.handle_request(HITLRequest(
            agent_name="PlannerAgent",
            action_type=HITLAction.APPROVE,
            message="Approve plan",
            context={
                "output": "plan",
                "_session": {"user_id": first_key[0], "session_id": first_key[1]},
            },
        )))
        await asyncio.sleep(0)

        assert len(first_socket.messages) == 1
        assert second_socket.messages == []
        payload = first_socket.messages[0]
        assert "_session" not in payload["context"]

        assert not handler.resolve_request(
            payload["request_id"],
            {"action": "approve", "approved": True},
            second_key,
        )
        assert not request_task.done()
        assert handler.resolve_request(
            payload["request_id"],
            {"action": "approve", "approved": True},
            first_key,
        )
        response = await request_task
        assert response.approved

    asyncio.run(scenario())

