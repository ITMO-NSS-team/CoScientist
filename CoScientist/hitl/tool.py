"""HITL Toolset — tools that agents call to request human input."""

import os
from typing import Any, Dict, List, Optional

from google.adk.tools import BaseTool, FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.base_toolset import BaseToolset
from google.adk.agents.readonly_context import ReadonlyContext

from CoScientist.config import get_settings
from CoScientist.hitl.models import HITLRequest, HITLAction
from CoScientist.hitl.handler import AbstractHITLHandler, ConsoleHITLHandler
from CoScientist.graph.session_scope import session_key

settings = get_settings()

def a2a_mode() -> bool:
    """True when this process serves agents over A2A (set by a2a/serve.py).

    Over A2A there is no console/websocket to the human, so the blocking tools
    below would hang the server; the A2A-native long-running variants are used
    instead (see CoScientist/hitl/a2a_tools.py).
    """
    return os.getenv("COSCIENTIST_A2A_MODE", "") not in ("", "0", "false", "False")


def get_hitl_tools() -> list:
    if a2a_mode():
        from CoScientist.hitl.a2a_tools import get_a2a_hitl_tools
        return get_a2a_hitl_tools()
    return [
        FunctionTool(hitl_toolset.request_approval),
        FunctionTool(hitl_toolset.request_selection)
    ]

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
            FunctionTool(self.request_selection)
        ]

    async def close(self) -> None:
        pass

    async def request_approval(
        self,
        agent_name: str,
        message: str,
        tool_context: ToolContext,
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
        request_context = dict(context or {})
        user_id, session_id = session_key(tool_context)
        request_context["_session"] = {
            "user_id": user_id,
            "session_id": session_id,
        }
        request = HITLRequest(
            agent_name=agent_name,
            action_type=HITLAction.APPROVE,
            message=f"Agent '{agent_name}' requests approval for the following action: {message}",
            context=request_context,
            invoked_via="tool"
        )
        response = await self._handler.handle_request(request)
        return {
            "approved": response.approved,
            "feedback": response.instructions or response.free_input or "No feedback provided.",
        }

    async def request_selection(
        self,
        agent_name: str,
        message: str,
        options: List[str],
        tool_context: ToolContext,
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
        user_id, session_id = session_key(tool_context)
        request = HITLRequest(
            agent_name=agent_name,
            action_type=HITLAction.SELECT,
            message=message,
            options=options,
            context={
                "_session": {
                    "user_id": user_id,
                    "session_id": session_id,
                }
            },
            invoked_via="tool"
        )
        response = await self._handler.handle_request(request)
        return {
            "selected": response.selected_option,
            "approved": response.approved,
            "feedback": response.instructions or response.free_input or "No feedback provided.",
        }

hitl_toolset = HITLToolset(handler=ConsoleHITLHandler())
