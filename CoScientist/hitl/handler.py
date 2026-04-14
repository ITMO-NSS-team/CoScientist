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
        print(f"[HITL] Agent '{request.agent_name}' requests: {request.action_type.value}. Invoked_via: {request.invoked_via}")
        print(f"Message: {request.message}")

        if request.context and "output" in request.context:
            print(f"\nPROPOSED PLAN/OUTPUT:")
            print(f"{'-' * 30}")
            print(f"{request.context['output']}")
            print(f"{'-' * 30}")

        if request.options:
            print("\nOptions:")
            for i, opt in enumerate(request.options, 1):
                print(f"  {i}. {opt}")

        is_simple_toggle = (request.invoked_via == "callback" and request.action_type == HITLAction.APPROVE)
        
        print("\nAction Menu:")
        if is_simple_toggle:
            print("  1. Approve (Proceed with agent execution)")
            print("  2. Reject (Skip this agent's execution)")
        else:
            print("  1. Approve (Accept and proceed)")
            print("  2. Edit (Provide feedback / request changes to agent)")
            print("  3. Free input (Custom answer, then proceed to the next agent)")
            print("  4. Stop program (Exit completely)")
        
        while True:
            choice = await asyncio.to_thread(input, f"\nSelect action (1-{2 if is_simple_toggle else 4}): ")
            choice = choice.strip()

            if choice == "1":
                return HITLResponse(
                    action=HITLAction.APPROVE,
                    approved=True
                )
            elif choice == "2":
                if is_simple_toggle:
                    return HITLResponse(
                        action=HITLAction.REJECT,
                        approved=False,
                        instructions="Human rejected execution."
                    )
                else:
                    feedback = await asyncio.to_thread(input, "Enter your feedback/changes: ")
                    return HITLResponse(
                        action=HITLAction.EDIT,
                        approved=False,
                        instructions=feedback
                    )
            elif choice == "3" and not is_simple_toggle:
                user_msg = await asyncio.to_thread(input, "Enter your input: ")
                user_msg = user_msg.strip()
                
                if request.action_type == HITLAction.SELECT and request.options:
                    try:
                        idx = int(user_msg) - 1
                        if 0 <= idx < len(request.options):
                            return HITLResponse(
                                action=HITLAction.SELECT,
                                selected_option=request.options[idx],
                                approved=True,
                            )
                    except (ValueError, IndexError):
                        pass
                    
                    # If not a number, treat as free selection or choice
                    return HITLResponse(
                        action=HITLAction.SELECT,
                        selected_option=user_msg,
                        free_input=user_msg,
                        approved=True,
                    )
                else:
                    return HITLResponse(
                        action=HITLAction.PROVIDE_INPUT,
                        free_input=user_msg,
                        approved=True,
                    )
            elif choice == "4" and not is_simple_toggle:
                print("\nStopping program execution based on user request...")
                import sys
                sys.exit(0)
            else:
                print(f"Invalid choice. Please enter a valid option.")



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
