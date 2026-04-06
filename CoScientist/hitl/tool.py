"""HITL Toolset — tools that agents call to request human input."""

from typing import Any, Dict, List, Optional

from google.adk.tools import BaseTool, FunctionTool
from google.adk.tools.base_toolset import BaseToolset
from google.adk.agents.readonly_context import ReadonlyContext

from CoScientist.hitl.models import HITLRequest, HITLAction
from CoScientist.hitl.handler import AbstractHITLHandler


class HITLToolset(BaseToolset):
    """Toolset providing HITL tools to agents.

    Agents call these tools when they need human confirmation,
    selection, or input before proceeding.
    """

    def __init__(self, handler: AbstractHITLHandler, prefix: str = "hitl_"):
        self._handler = handler
        self.tool_name_prefix = prefix

    async def get_tools(
        self, readonly_context: Optional[ReadonlyContext] = None
    ) -> List[BaseTool]:
        return [
            FunctionTool(self.request_approval),
            FunctionTool(self.request_selection),
            FunctionTool(self.request_input),
        ]

    async def close(self) -> None:
        pass

    async def request_approval(
        self,
        agent_name: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Request human approval for an action.

        Use this tool when you need the user to confirm or reject
        a proposed action before proceeding.

        Args:
            agent_name: Name of the agent requesting approval.
            message: Description of what needs approval.
            context: Additional context for the human.

        Returns:
            Dictionary with 'approved' (bool) and optional 'feedback' (str).
        """
        request = HITLRequest(
            agent_name=agent_name,
            action_type=HITLAction.APPROVE,
            message=message,
            context=context or {},
        )
        response = await self._handler.handle_request(request)
        return {
            "approved": response.approved,
            "feedback": response.free_input,
        }

    async def request_selection(
        self,
        agent_name: str,
        message: str,
        options: List[str],
    ) -> Dict[str, Any]:
        """Ask the human to select from a list of options.

        Use this tool when you have generated multiple proposals
        (e.g. hypotheses, plans) and need the user to choose the best one.

        Args:
            agent_name: Name of the agent requesting selection.
            message: Explanation of what to select and why.
            options: List of options for the human to choose from.

        Returns:
            Dictionary with 'selected' (str) and 'approved' (bool).
        """
        request = HITLRequest(
            agent_name=agent_name,
            action_type=HITLAction.SELECT,
            message=message,
            options=options,
        )
        response = await self._handler.handle_request(request)
        return {
            "selected": response.selected_option,
            "approved": response.approved,
        }

    async def request_input(
        self,
        agent_name: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Request free-form input from the human.

        Use this tool when you need additional information,
        clarification, or guidance from the user.

        Args:
            agent_name: Name of the agent requesting input.
            message: What information is needed and why.
            context: Additional context for the human.

        Returns:
            Dictionary with 'input' (str) and 'approved' (bool).
        """
        request = HITLRequest(
            agent_name=agent_name,
            action_type=HITLAction.PROVIDE_INPUT,
            message=message,
            context=context or {},
        )
        response = await self._handler.handle_request(request)
        return {
            "input": response.free_input,
            "approved": response.approved,
        }
