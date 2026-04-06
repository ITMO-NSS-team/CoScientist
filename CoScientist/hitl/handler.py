"""HITL handlers — abstract interface and implementations."""

import asyncio
from abc import ABC, abstractmethod

from CoScientist.hitl.models import HITLRequest, HITLResponse, HITLAction


class AbstractHITLHandler(ABC):
    """Abstract interface for handling HITL requests.

    Implement this for different UIs: console, web chat, Telegram, etc.
    """

    @abstractmethod
    async def handle_request(self, request: HITLRequest) -> HITLResponse:
        """Process a HITL request and return the human's response."""
        ...


class ConsoleHITLHandler(AbstractHITLHandler):
    """Simple console-based HITL handler (for local development/testing)."""

    async def handle_request(self, request: HITLRequest) -> HITLResponse:
        print(f"\n{'=' * 60}")
        print(f"[HITL] Agent '{request.agent_name}' requests: {request.action_type.value}")
        print(f"Message: {request.message}")

        if request.options:
            print("\nOptions:")
            for i, opt in enumerate(request.options, 1):
                print(f"  {i}. {opt}")

        user_input = await asyncio.to_thread(input, "\nYour response: ")

        if request.action_type == HITLAction.SELECT and request.options:
            try:
                idx = int(user_input.strip()) - 1
                if 0 <= idx < len(request.options):
                    return HITLResponse(
                        action=HITLAction.SELECT,
                        selected_option=request.options[idx],
                        approved=True,
                    )
            except (ValueError, IndexError):
                pass
            return HITLResponse(
                action=HITLAction.SELECT,
                selected_option=user_input.strip(),
                free_input=user_input.strip(),
                approved=True,
            )

        elif request.action_type == HITLAction.APPROVE:
            approved = user_input.strip().lower() in ("y", "yes", "da", "1", "ok")
            return HITLResponse(
                action=HITLAction.APPROVE,
                approved=approved,
                free_input=user_input.strip() if not approved else None,
            )

        elif request.action_type == HITLAction.EDIT:
            return HITLResponse(
                action=HITLAction.EDIT,
                edited_content=user_input.strip(),
                approved=True,
            )

        else:
            return HITLResponse(
                action=request.action_type,
                free_input=user_input.strip(),
                approved=True,
            )


class CallbackHITLHandler(AbstractHITLHandler):
    """Queue-based HITL handler for integration with web UI / chat bots.

    External code (web server, chat bot) reads requests from the queue
    and submits responses back.
    """

    def __init__(self):
        self._request_queue: asyncio.Queue[HITLRequest] = asyncio.Queue()
        self._response_queue: asyncio.Queue[HITLResponse] = asyncio.Queue()

    async def handle_request(self, request: HITLRequest) -> HITLResponse:
        """Put request into queue and wait for external response."""
        await self._request_queue.put(request)
        if request.timeout_seconds:
            response = await asyncio.wait_for(
                self._response_queue.get(),
                timeout=request.timeout_seconds,
            )
        else:
            response = await self._response_queue.get()
        return response

    async def get_pending_request(self) -> HITLRequest:
        """Called by UI/chat bot to get the current HITL request."""
        return await self._request_queue.get()

    async def submit_response(self, response: HITLResponse) -> None:
        """Called by UI/chat bot to submit the human's response."""
        await self._response_queue.put(response)

    def has_pending_request(self) -> bool:
        """Check if there is a pending HITL request (non-blocking)."""
        return not self._request_queue.empty()
