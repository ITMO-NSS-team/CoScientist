"""HITL tools for agents served over A2A.

Over A2A there is no console and no websocket to the human: the agent runs as an
HTTP service and the *caller* is the one who can answer. Blocking inside the tool
(what ConsoleHITLHandler does) would hang the server, which is why HITL did not
work in A2A mode.

A2A has a native mechanism for exactly this, and ADK implements it: a call to a
**long-running** tool is emitted to the client and puts the A2A task into
``input-required`` (see google/adk/a2a/converters/long_running_functions.py).
The run then yields instead of blocking; the client answers by sending back a
FunctionResponse for that call id, and the run resumes with the human's answer.

So these tools return IMMEDIATELY with a "pending" payload and are wrapped in
``LongRunningFunctionTool``. Their names/signatures match the in-process tools
(CoScientist/hitl/tool.py) so prompts, docs and the unknown-tool guard are
unchanged — only the delivery mechanism differs.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.tools.tool_context import ToolContext

from CoScientist.graph.session_scope import session_key


def _pending(agent_name: str, question: str, **extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "status": "pending_human_input",
        "agent_name": agent_name,
        "question": question,
        "message": (
            "Sent to the human via A2A; this task is now 'input-required'. Do NOT "
            "call this tool again and do not assume an answer — the human's reply "
            "arrives as the response to this call, then continue from it."
        ),
    }
    payload.update(extra)
    return payload


async def request_approval(
    agent_name: str,
    message: str,
    tool_context: ToolContext = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Ask the human to approve an action (yes / no, or a free-text answer).

    Use before expensive, outward-facing or hard-to-reverse actions, and whenever
    a decision is genuinely the human's. The human may reply yes/no OR with
    free-text instructions ("other") — follow whatever they send.

    Args:
        agent_name: Name of the agent requesting approval.
        message: What needs approval, in plain language.
        context: Optional extra context for the human.

    Returns:
        The human's answer: {'approved': bool, 'feedback': str}.
    """
    user_id, session_id = session_key(tool_context) if tool_context is not None else (None, None)
    return _pending(
        agent_name,
        message,
        expected_response={"approved": "bool", "feedback": "str (optional free-text answer)"},
        context=dict(context or {}),
        session={"user_id": user_id, "session_id": session_id},
    )


async def request_selection(
    agent_name: str,
    message: str,
    options: List[str],
    tool_context: ToolContext = None,
) -> Dict[str, Any]:
    """Ask the human to choose among options you generated (or answer their own way).

    Use when several alternatives (hypotheses, plans, thresholds) need a human
    decision. The human may pick one of `options` OR reply with their own answer
    ("other") — honour whichever they send.

    Args:
        agent_name: Name of the agent requesting the choice.
        message: What to choose and why it matters.
        options: 2-4 concrete options.

    Returns:
        The human's answer: {'selected': str, 'approved': bool, 'feedback': str}.
    """
    user_id, session_id = session_key(tool_context) if tool_context is not None else (None, None)
    return _pending(
        agent_name,
        message,
        options=list(options or []),
        expected_response={"selected": "one of options (or the human's own answer)",
                           "approved": "bool", "feedback": "str (optional)"},
        session={"user_id": user_id, "session_id": session_id},
    )


def get_a2a_hitl_tools() -> list:
    """The HITL tools to attach when serving an agent over A2A."""
    return [LongRunningFunctionTool(request_approval),
            LongRunningFunctionTool(request_selection)]
