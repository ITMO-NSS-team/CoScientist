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
import logging

from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

from CoScientist.config import get_settings
from CoScientist.agents import orchestrator_agent
from CoScientist.agents.research_callbacks import cleanup_uploaded_papers
from CoScientist.hitl import (
    AbstractHITLHandler,
    HITLRequest,
    HITLResponse,
)

settings = get_settings()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    
        # Session service
        self.session_service = InMemorySessionService()

        await self.session_service.create_session(
            app_name=self.app_name,
            user_id=self.user_id,
            session_id=self.session_id,
        )

        # Runner
        self.runner = Runner(
            agent=orchestrator_agent,
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
                    final_response = event.content.parts[0].text or ""
                elif event.actions and event.actions.escalate:
                    final_response = f"Escalation: {event.error_message or 'Unknown error'}"


        return final_response

    async def close(self):
        """Cleanup session-related resources and uploaded paper artifacts."""
        try:
            await asyncio.to_thread(cleanup_uploaded_papers, self.user_id, self.session_id)
        except Exception as exc:
            logger.error(f"Warning: failed to cleanup uploaded papers for session {self.session_id}: {exc}")

# Convenience functions
async def create_manager() -> CoScientistManager:
    """Create and initialize a CoScientistManager."""
    manager = CoScientistManager()
    await manager.initialize()
    return manager


# Export public API
__all__ = [
    # Main classes
    "CoScientistManager",
    # Models
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
