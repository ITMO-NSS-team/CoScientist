"""
CoScientist - Main entry point

Runs the multi-agent scientific discovery pipeline:
- Hypothesis generation
- Research
- Experimentation (FEDOT)
- Orchestration
"""
from dotenv import load_dotenv
load_dotenv()

import asyncio
from typing import Optional

from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

from CoScientist.config import get_settings
from CoScientist.agents.agents import create_agents
from CoScientist.hitl import (
    AbstractHITLHandler,
    CallbackHITLHandler,
    HITLRequest,
    HITLResponse,
)

settings = get_settings()


class CoScientistManager:
    """
    Main manager for CoScientist (ADK-based execution).
    """

    def __init__(
        self,
        app_name: str = "coscientist_app",
        user_id: str = "user_1",
        session_id: str = "session_001",
        hitl_handler: Optional[AbstractHITLHandler] = None,
    ):
        self.app_name = app_name
        self.user_id = user_id
        self.session_id = session_id

        self.session_service: Optional[InMemorySessionService] = None
        self.runner: Optional[Runner] = None
        self._initialized = False

        # HITL setup
        self._hitl_handler = hitl_handler
        self._agents = None

    async def initialize(self):
        """Initialize session + runner."""
        if self._initialized:
            return

        # Create agents with HITL handler
        if self._hitl_handler is not None:
            self._agents = await create_agents(hitl_handler=self._hitl_handler)
        else:
            self._agents = await create_agents()
        agent = self._agents["orchestrator_agent"]

        # Session service
        self.session_service = InMemorySessionService()

        await self.session_service.create_session(
            app_name=self.app_name,
            user_id=self.user_id,
            session_id=self.session_id,
        )

        # Runner
        self.runner = Runner(
            agent=agent,
            app_name=self.app_name,
            session_service=self.session_service,
        )

        self._initialized = True

    async def run(self, query: str, verbose: bool = True) -> str:
        """
        Execute a query through the orchestrator agent.

        Args:
            query: user query
            verbose: whether to print events

        Returns:
            Final agent response
        """
        await self.initialize()

        content = types.Content(
            role="user",
            parts=[types.Part(text=query)]
        )

        final_response = "No response"

        async for event in self.runner.run_async(
            user_id=self.user_id,
            session_id=self.session_id,
            new_message=content,
        ):
            if verbose:
                print(
                    f"[Event] {event.author} | {type(event).__name__} | Final={event.is_final_response()}"
                )

            if event.is_final_response():
                if event.content and event.content.parts:
                    final_response = event.content.parts[0].text
                elif event.actions and event.actions.escalate:
                    final_response = f"Escalation: {event.error_message or 'Unknown error'}"


        return final_response

    # --- HITL convenience methods for external UI integration ---

    async def get_hitl_request(self) -> Optional[HITLRequest]:
        """Get pending HITL request (for web UI / chat bot integration).

        Only works when hitl_handler is CallbackHITLHandler.
        """
        if isinstance(self._hitl_handler, CallbackHITLHandler):
            return await self._hitl_handler.get_pending_request()
        return None

    async def submit_hitl_response(self, response: HITLResponse) -> None:
        """Submit human response to a HITL request (for web UI / chat bot integration).

        Only works when hitl_handler is CallbackHITLHandler.
        """
        if isinstance(self._hitl_handler, CallbackHITLHandler):
            await self._hitl_handler.submit_response(response)

    def has_pending_hitl(self) -> bool:
        """Check if there is a pending HITL request (non-blocking)."""
        if isinstance(self._hitl_handler, CallbackHITLHandler):
            return self._hitl_handler.has_pending_request()
        return False

    async def close(self):
        """Cleanup (placeholder)."""
        # If you switch to persistent sessions later, close here
        pass

# Convenience functions
async def create_manager(
    hitl_handler: Optional[AbstractHITLHandler] = None,
) -> CoScientistManager:
    """Create and initialize a CoScientistManager.

    Args:
        hitl_handler: Optional HITL handler. Pass CallbackHITLHandler for
                      web UI integration, or None for console-based HITL.
    """
    manager = CoScientistManager(hitl_handler=hitl_handler)
    await manager.initialize()
    return manager


# Export public API
__all__ = [
    # Main classes
    "CoScientistManager",
    # Functions
    "create_manager"
]

# CLI entrypoint
if __name__ == "__main__":
    async def main():

        manager = await create_manager()

        print("CoScientist (ADK) initialized\n")

        try:
            while True:
                query = input("Enter query (or 'exit'): ")

                if query.lower() in {"exit", "quit"}:
                    break

                result = await manager.run(query)

                print("\n=== Final Response ===")
                print(result)
                print()

        finally:
            await manager.close()

    asyncio.run(main())
