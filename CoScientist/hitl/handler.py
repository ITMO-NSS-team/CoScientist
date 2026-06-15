"""HITL handlers — abstract interface and implementations."""

import asyncio
import sys
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

    def __init__(self) -> None:
        # Per-signature counter to break headless reject->retry loops (F015).
        self._seen: dict[str, int] = {}

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
            print("  3. Stop program (Exit completely)")
        
        # Non-interactive session (no TTY, e.g. headless/background/server run): we
        # cannot prompt a human. Fall back to a safe default instead of blocking or
        # letting input() raise EOFError and kill the entire run. Auto-REJECT (not
        # approve) so outward-facing / hard-to-reverse actions are never executed
        # unattended; flip the default here if unattended auto-approval is desired.
        if not sys.stdin.isatty():
            from CoScientist.config.settings import get_settings
            hitl = get_settings().hitl
            sig = f"{request.agent_name}|{request.action_type.value}|{(request.message or '')[:120]}"
            self._seen[sig] = self._seen.get(sig, 0) + 1
            n = self._seen[sig]
            if hitl.headless_auto_approve:
                print("[HITL] Non-interactive — auto-APPROVING (HITL__HEADLESS_AUTO_APPROVE).")
                return HITLResponse(
                    action=HITLAction.APPROVE, approved=True,
                    instructions="Auto-approved: headless run with auto-approve enabled.",
                )
            if n >= hitl.loop_guard_repeats:
                print(f"[HITL] Loop-guard: identical headless request x{n} — auto-APPROVING to break the loop.")
                return HITLResponse(
                    action=HITLAction.APPROVE, approved=True,
                    instructions=f"Auto-approved by loop-guard after {n} identical headless requests.",
                )
            print("[HITL] Non-interactive session — auto-rejecting (no human to approve).")
            return HITLResponse(
                action=HITLAction.REJECT,
                approved=False,
                instructions="Auto-rejected: non-interactive (headless) session, no human available to approve.",
            )

        while True:
            try:
                choice = await asyncio.to_thread(input, f"\nSelect action (1-{2 if is_simple_toggle else 3}): ")
            except EOFError:
                print("[HITL] stdin closed — auto-rejecting (no human to approve).")
                return HITLResponse(
                    action=HITLAction.REJECT,
                    approved=False,
                    instructions="Auto-rejected: stdin closed, no human available to approve.",
                )
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
                    try:
                        feedback = await asyncio.to_thread(input, "Enter your feedback/changes: ")
                    except EOFError:
                        feedback = "Auto-rejected: stdin closed during feedback prompt."
                    return HITLResponse(
                        action=HITLAction.EDIT,
                        approved=False,
                        instructions=feedback
                    )
            elif choice == "3" and not is_simple_toggle:
                print("\nStopping program execution based on user request...")
                sys.exit(0)
            else:
                print(f"Invalid choice. Please enter a valid option.")