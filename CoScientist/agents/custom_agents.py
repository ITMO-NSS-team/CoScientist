"""Custom (non-LLM) agent classes referenced from system.yaml via custom:<name>."""
from typing import AsyncGenerator, List, Dict, Any
from typing_extensions import override

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event

class WebToolsDeployerAgent(BaseAgent):
    """
    Custom agent for deploying found web mcp servers.
    """

    # --- Field Declarations for Pydantic ---
    # model_config allows setting Pydantic configurations if needed, e.g., arbitrary_types_allowed
    model_config = {"arbitrary_types_allowed": True}


    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """
        Implements the custom orchestration logic for the story workflow.
        Uses the instance attributes assigned by Pydantic.
        """

        current_state = ctx.session.state

        filtered_mcps: List[Dict[str, Any]] = current_state.get('filtered_mcps', [])

        #TODO: Implement full deploying strategy after side tools are ready
        deployed_mcps = []
        ctx.session.state['deployed_mcps'] = deployed_mcps
        yield Event(author=self.name, invocation_id=ctx.invocation_id)

        # if not filtered_mcps:
        #     return
