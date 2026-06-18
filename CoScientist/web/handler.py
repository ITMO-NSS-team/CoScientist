import asyncio
import json
import uuid
from typing import Optional

from CoScientist.hitl.handler import AbstractHITLHandler
from CoScientist.hitl.models import HITLRequest, HITLResponse, HITLAction


class WebHITLHandler(AbstractHITLHandler):

    def __init__(self):
        # request_id -> asyncio.Future[HITLResponse]
        self._pending: dict[str, asyncio.Future] = {}
        self._websocket = None
        # event log that the frontend can poll
        self._event_log: list[dict] = []

    def __deepcopy__(self, memo):
        return self

    def set_websocket(self, ws):
        self._websocket = ws

    # Seconds to wait for a HITL response before auto-approving (prevents infinite hang)
    HITL_TIMEOUT_SECONDS: int = 300

    async def handle_request(self, request: HITLRequest) -> HITLResponse:
        request_id = str(uuid.uuid4())

        payload = {
            "type": "hitl_request",
            "request_id": request_id,
            "agent_name": request.agent_name,
            "action_type": request.action_type.value,
            "message": request.message,
            "options": request.options,
            "context": request.context,
            "invoked_via": request.invoked_via,
        }

        self._event_log.append(payload)

        if self._websocket:
            await self._websocket.send_json(payload)

        # Wait for response from browser
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending[request_id] = future

        try:
            response_data = await asyncio.wait_for(
                asyncio.shield(future),
                timeout=self.HITL_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            self._pending.pop(request_id, None)
            print(f"[WebHITLHandler] HITL timeout for request {request_id[:8]}, auto-approving.")
            response_data = {"action": "approve", "approved": True}

        action = HITLAction(response_data.get("action", "approve"))
        return HITLResponse(
            action=action,
            approved=response_data.get("approved", False),
            selected_option=response_data.get("selected_option"),
            instructions=response_data.get("instructions"),
            free_input=response_data.get("free_input"),
        )

    def resolve_request(self, request_id: str, response_data: dict):
        """Called when the browser sends a HITL response."""
        future = self._pending.pop(request_id, None)
        if future and not future.done():
            future.set_result(response_data)

    def reset(self):
        """Cancel all pending requests and clear state."""
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.cancel()
        self._pending.clear()
        self._event_log.clear()

    def get_event_log(self) -> list[dict]:
        return list(self._event_log)

    def clear_event_log(self):
        self._event_log.clear()