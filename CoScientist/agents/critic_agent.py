"""
Critic agents for the orchestrator.

Two callbacks are wired onto the OrchestratorAgent:

  * `pre_action_critique`   -> after_model_callback
        Runs after the orchestrator LLM has decided on its next action(s)
        but BEFORE those actions execute. Inspects the chosen function
        calls (which sub-agent, with what args) in light of the task and
        history, and votes:
          - APPROVE -> let the calls execute as-is
          - REVISE  -> mutate the args in place and let the calls execute
          - REJECT  -> replace the response with a text message that
                       describes why the plan was rejected; the
                       orchestrator will then re-decide on its next turn

  * `post_action_critique`  -> after_tool_callback
        Runs after a sub-agent (tool) returns. Evaluates the result and
        annotates it with a `_critic` directive when the result is
        insufficient or wrong, leaving the original payload intact.

Both critics are themselves LLM calls returning strict JSON.
"""

from __future__ import annotations

from opik import track

import json
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, List, Optional

import litellm
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from CoScientist.config import get_settings
from CoScientist.agents.prompts import (
    pre_action_critic_instruction,
    post_action_critic_instruction,
)


_settings = get_settings()
_CRITIC_MODEL = _settings.llm.main_model


# ---------------------------------------------------------------------------
# Verdict enums
# ---------------------------------------------------------------------------
class PreVerdict(str, Enum):
    APPROVE = "approve"
    REVISE = "revise"
    REJECT = "reject"


class PostVerdict(str, Enum):
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"
    WRONG = "wrong"


# ---------------------------------------------------------------------------
# Trajectory parsing (from session history on the callback context)
# ---------------------------------------------------------------------------
def _session_contents(callback_context: CallbackContext) -> List[types.Content]:
    """
    Best-effort extraction of the session's message history from the
    callback context. Falls back to an empty list if the internal layout
    differs across ADK versions.
    """
    inv = getattr(callback_context, "_invocation_context", None) or getattr(
        callback_context, "invocation_context", None
    )
    if inv is None:
        return []
    session = getattr(inv, "session", None)
    if session is None:
        return []
    events = getattr(session, "events", None) or []
    contents: List[types.Content] = []
    for ev in events:
        c = getattr(ev, "content", None)
        if c is not None and getattr(c, "parts", None):
            contents.append(c)
    return contents


def _extract_completed_trajectory(
    contents: List[types.Content],
) -> List[Dict[str, Any]]:
    """
    Pair every function_call with its matching function_response. Calls
    without a response yet are skipped — they belong to the in-flight
    decision the pre-critic is currently evaluating.
    """
    responses_by_id: Dict[str, Any] = {}
    for c in contents:
        if c.role != "user" or not c.parts:
            continue
        for p in c.parts:
            fr = getattr(p, "function_response", None)
            if fr is not None:
                fr_id = getattr(fr, "id", "")
                # Skip responses without an id — otherwise they all collapse to
                # the "" key and overwrite each other, mis-pairing calls.
                if fr_id:
                    responses_by_id[fr_id] = getattr(fr, "response", None)

    trajectory: List[Dict[str, Any]] = []
    pending_thought: Optional[str] = None

    for c in contents:
        if c.role != "model" or not c.parts:
            continue
        for p in c.parts:
            if getattr(p, "text", None) and getattr(p, "thought", False):
                pending_thought = p.text
                continue
            fc = getattr(p, "function_call", None)
            if fc is None:
                continue
            call_id = getattr(fc, "id", "") or ""
            if call_id not in responses_by_id:
                pending_thought = None
                continue
            trajectory.append(
                {
                    "thought": pending_thought,
                    "tool": getattr(fc, "name", ""),
                    "args": dict(getattr(fc, "args", {}) or {}),
                    "response": responses_by_id[call_id],
                }
            )
            pending_thought = None
    return trajectory


def _extract_pending_calls(llm_response: LlmResponse) -> List[Dict[str, Any]]:
    """
    Pull (thought?, function_call) pairs out of the orchestrator's freshly
    produced LlmResponse. These are the calls about to execute.
    """
    if llm_response is None or llm_response.content is None:
        return []
    parts = llm_response.content.parts or []
    pending_thought: Optional[str] = None
    calls: List[Dict[str, Any]] = []
    for p in parts:
        if getattr(p, "text", None) and getattr(p, "thought", False):
            pending_thought = p.text
            continue
        fc = getattr(p, "function_call", None)
        if fc is None:
            continue
        calls.append(
            {
                "thought": pending_thought,
                "tool": getattr(fc, "name", ""),
                "args": dict(getattr(fc, "args", {}) or {}),
                "_part": p,  # kept so REVISE can mutate args in place
            }
        )
        pending_thought = None
    return calls


# ---------------------------------------------------------------------------
# Formatting / truncation
# ---------------------------------------------------------------------------
def _truncate(value: Any, limit: int = 1500) -> str:
    if value is None:
        return ""
    s = value if isinstance(value, str) else json.dumps(value, default=str)
    if len(s) <= limit:
        return s
    return s[:limit] + f"...[truncated {len(s) - limit} chars]"


def _format_trajectory(trajectory: List[Dict[str, Any]]) -> str:
    if not trajectory:
        return "(no completed prior steps)"
    lines: List[str] = []
    for i, step in enumerate(trajectory, 1):
        lines.append(f"--- Completed step {i} ---")
        if step.get("thought"):
            lines.append(f"Reasoning: {_truncate(step['thought'], 500)}")
        lines.append(f"Tool called: {step['tool']}")
        lines.append(f"Args: {_truncate(step['args'], 400)}")
        lines.append(f"Result: {_truncate(step['response'], 1000)}")
    return "\n".join(lines)


def _format_pending_calls(calls: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for i, call in enumerate(calls, 1):
        lines.append(f"--- Proposed action {i} ---")
        if call.get("thought"):
            lines.append(f"Reasoning: {_truncate(call['thought'], 500)}")
        lines.append(f"Tool to call: {call['tool']}")
        lines.append(f"Args: {_truncate(call['args'], 600)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM critic invocation
# ---------------------------------------------------------------------------
async def _invoke_critic_llm(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """Returns parsed JSON dict; on any failure returns {} (permissive default).

    Uses the async litellm API so the critic's network call does not block the
    orchestrator's event loop (the callbacks run inside it).
    """
    try:
        resp = await litellm.acompletion(
            model=_CRITIC_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        raw = resp["choices"][0]["message"]["content"]
        return json.loads(raw)
    except Exception as e:
        print(f"[Critic] LLM call failed ({e!r}); defaulting to permissive verdict.")
        return {}


# ---------------------------------------------------------------------------
# Pre-action critic  (after_model_callback)
# ---------------------------------------------------------------------------
def _apply_revisions(
    pending: List[Dict[str, Any]], revised_calls: List[Dict[str, Any]]
) -> None:
    """
    Mutate the LlmResponse's function_call parts in place using the critic's
    revised args. Match by index — the critic is told to return one entry
    per proposed action in the same order.
    """
    for i, call in enumerate(pending):
        if i >= len(revised_calls):
            break
        rev = revised_calls[i]
        new_args = rev.get("args")
        if not isinstance(new_args, dict):
            continue
        part = call["_part"]
        fc = getattr(part, "function_call", None)
        if fc is None:
            continue
        try:
            fc.args = new_args
        except Exception:  
            try:
                object.__setattr__(fc, "args", new_args)
            except Exception:  
                pass

def _first_text(content) -> str:
    """First text part of a Content, or '' if none (e.g. an image-only message)."""
    if content is None or not getattr(content, "parts", None):
        return ""
    for p in content.parts:
        if getattr(p, "text", None):
            return p.text
    return ""


@track(name="pre_action_critique")
async def pre_action_critique(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    """after_model_callback for the OrchestratorAgent."""
    pending = _extract_pending_calls(llm_response)

    # No tool calls in this response -> nothing to critique.
    if not pending:
        return None

    contents = _session_contents(callback_context)
    # user_content may be absent or start with a non-text part (image/DICOM upload).
    user_task = _first_text(getattr(callback_context, "user_content", None))
    trajectory = _extract_completed_trajectory(contents)

    user_prompt = (
        f"ORIGINAL TASK:\n{user_task}\n\n"
        f"COMPLETED TRAJECTORY:\n{_format_trajectory(trajectory)}\n\n"
        f"PROPOSED NEXT ACTION(S) (not yet executed):\n{_format_pending_calls(pending)}\n\n"
        "Decide whether to approve, revise, or reject these proposed actions. "
        "Respond as strict JSON."
    )

    print(f"pre action critic invoked with such prompt: {user_prompt}")

    payload = await _invoke_critic_llm(pre_action_critic_instruction, user_prompt)
    verdict_raw = (payload.get("verdict") or "approve").lower().strip()
    feedback = (payload.get("feedback") or "").strip()
    revised_calls = payload.get("revised_calls") or []

    # Audit trail
    state = callback_context.state
    history = state.get("critic_pre_history", [])
    history.append(
        {
            "verdict": verdict_raw,
            "feedback": feedback,
            "proposed": [{"tool": c["tool"], "args": c["args"]} for c in pending],
        }
    )
    state["critic_pre_history"] = history

    print(f"pre action critic returned: {payload}")
    if verdict_raw == PreVerdict.APPROVE.value:
        print(f"pre action critic returned None")
        return None

    if verdict_raw == PreVerdict.REVISE.value:
        if revised_calls:
            _apply_revisions(pending, revised_calls)
        if feedback and llm_response.content and llm_response.content.parts:
            llm_response.content.parts.insert(
                0,
                types.Part(text=f"[CRITIC REVISION]: {feedback}", thought=True),
            )
        print(f"pre action critic returned revision: {feedback}")
        return None

    if verdict_raw == PreVerdict.REJECT.value:
        msg = (
            "I am rejecting my own proposed action(s). "
            f"Reason: {feedback or 'the plan does not advance the task'}. "
            "I will reconsider which agent to call and with what arguments, "
            "given the original task and the completed trajectory so far."
        )
        print(f"pre action critic rejected with msg {msg}")
        return LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text=msg)])
        )

    return None


# ---------------------------------------------------------------------------
# Post-action critic  (after_tool_callback)
# ---------------------------------------------------------------------------
@track(name="post_action_critique")
async def post_action_critique(
    tool: BaseTool,
    args: Dict[str, Any],
    tool_context: ToolContext,
    tool_response: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """after_tool_callback for the OrchestratorAgent."""
    user_prompt = (
        f"TOOL CALLED: {tool.name}\n"
        f"ARGS: {_truncate(args, 800)}\n"
        f"RESULT: {_truncate(tool_response, 3000)}\n\n"
        "Evaluate whether this result is sufficient to advance the task, "
        "needs refinement, or is wrong. Respond as strict JSON."
    )

    print(f'Post action critic invoked with {user_prompt}')
    payload = await _invoke_critic_llm(post_action_critic_instruction, user_prompt)
    verdict_raw = (payload.get("verdict") or "sufficient").lower().strip()
    feedback = (payload.get("feedback") or "").strip()
    print(f'Post action critic returned with {payload}')
    state = tool_context.state
    history = state.get("critic_post_history", [])
    history.append(
        {"tool": tool.name, "verdict": verdict_raw, "feedback": feedback}
    )
    state["critic_post_history"] = history

    if verdict_raw == PostVerdict.SUFFICIENT.value:
        return None

    annotated = (
        deepcopy(tool_response)
        if isinstance(tool_response, dict)
        else {"result": tool_response}
    )

    if verdict_raw == PostVerdict.INSUFFICIENT.value:
        annotated["_critic"] = {
            "verdict": "insufficient",
            "directive": "REFINE",
            "feedback": feedback
            or "Result is incomplete; refine the query or call a different agent.",
        }
        print(f'post action critic returned insufficient {annotated}')
        return annotated

    if verdict_raw == PostVerdict.WRONG.value:
        annotated["_critic"] = {
            "verdict": "wrong",
            "directive": "REPLAN",
            "feedback": feedback
            or "Result does not address the task; re-plan from scratch.",
        }
        print(f'post action critic returned wrong {annotated}')
        return annotated

    return None