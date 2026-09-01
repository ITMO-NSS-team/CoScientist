"""
Critic callbacks for the orchestrator.

Two callback FACTORIES are wired onto the OrchestratorAgent via system.yaml:

  * `make_pre_action_critique(instruction)`   -> after_model_callback
        Runs after the orchestrator LLM has decided on its next action(s)
        but BEFORE those actions execute. Inspects the chosen function
        calls (which sub-agent, with what args) in light of the task and
        history, and votes:
          - APPROVE -> let the calls execute as-is
          - REVISE  -> mutate the args in place and let the calls execute
          - REJECT  -> replace the response with a text message that
                       describes why the plan was rejected; the
                       orchestrator will then re-decide on its next turn

  * `make_post_action_critique(instruction)`  -> after_tool_callback
        Runs after a sub-agent (tool) returns. Evaluates the result and
        annotates it with a `_critic` directive when the result is
        insufficient or wrong, leaving the original payload intact.

A third factory is wired onto the PLANNER instead, independently of the two
above (system.yaml -> ``PlannerAgent.critic``):

  * `make_plan_critique(instruction)`         -> SessionAgent.plan_critic
        Reviews the roadmap the planner just registered, BEFORE it is accepted
        (and before the human sees it). Returns feedback text to send back for
        one rewrite, or None to accept the plan. The revision budget lives in
        the SessionAgent (`critic_max_rounds`, default 1), not here.

All three critics are themselves LLM calls returning strict JSON. They are
factories (not module-level callbacks) because their system prompts embed the
CURRENT roster — the assembler renders the prompt from the same config that
wires the agents and passes it in.
"""

from __future__ import annotations
from google.adk.agents import callback_context

from opik import track

import asyncio
import json
import os
import time
from copy import deepcopy
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import litellm
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from CoScientist.agents.callbacks.json_output import _extract_json
from CoScientist.config import get_settings


_settings = get_settings()
_CRITIC_MODEL = _settings.llm.main_model

# A critic call is a bare litellm call: it bypasses `RetryingLiteLlm` and with it
# every protection agent calls get — the wall-clock deadline (`llm_timeout:` in
# system.yaml), the retry, the proxy pre-flight. It is awaited inline in the
# orchestrator's event loop (and the plan critic inside SessionAgent's review
# loop), so a call that does not come back stops the whole run.
#
# The deadline is a backstop, not a duplicate of litellm's `timeout=`: that one
# is httpx's, i.e. a PER-READ budget, so it only bounds a peer that goes silent.
# A peer that keeps the socket busy — or anything that stalls outside the read
# loop — is bounded by nothing else (cf. `deadline_s` in agents/common.py, and
# the ToolReranker note in system.yaml).
# 90s, not 60: measured directly, a verdict on this stack costs ~15s, and the
# budget has to absorb a slow route on top of that. One long attempt beats two
# short ones — same wall time, but a single connection that is merely slow gets
# to finish instead of being cut twice.
_CRITIC_TIMEOUT_S = float(os.getenv("CRITIC__TIMEOUT", "90"))
# Kept under the wall-clock deadline on purpose. Given the same number both
# clocks race and the deadline wins, reporting a bare `TimeoutError()` that says
# nothing about where it hung; a lower one lets litellm report the fault it can
# actually name ("Timeout passed=45.0, time taken=45.05").
_CRITIC_HTTP_TIMEOUT_S = _CRITIC_TIMEOUT_S * 0.75
# One retry only: the critic runs on every orchestrator turn, so its worst case
# has to stay a bounded fraction of a turn. The deadline itself is NOT retried
# (see below) — a second full budget on a route that just proved it cannot
# answer in time only doubles the stall.
_CRITIC_MAX_ATTEMPTS = int(os.getenv("CRITIC__MAX_ATTEMPTS", "2"))
# The cap covers REASONING PLUS the answer, which is why it is nowhere near the
# size of a verdict. The main model reasons before it answers — measured at
# 164-186 reasoning tokens against a 242-251 token completion on a short plan,
# and reasoning grows with the plan under review. A cap that binds does not
# yield a shorter verdict, it yields an EMPTY one: the budget is spent before
# the JSON starts, `content` comes back "" and the critic silently approves.
# So this is a guard against a runaway generation, set well above the envelope,
# not a budget meant to bite. `finish_reason` is logged when it does.
_CRITIC_MAX_TOKENS = int(os.getenv("CRITIC__MAX_TOKENS", "2000"))


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


class PlanVerdict(str, Enum):
    APPROVE = "approve"
    REVISE = "revise"


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


def _usage_note(resp: Any) -> str:
    """`completion=N (reasoning=M) of cap` — what the answer actually spent."""
    usage = getattr(resp, "usage", None)
    if usage is None:
        return f"no usage reported, cap={_CRITIC_MAX_TOKENS}"
    details = getattr(usage, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", None)
    return (
        f"completion_tokens={getattr(usage, 'completion_tokens', '?')}"
        + (f" (reasoning={reasoning})" if reasoning else "")
        + f" of cap={_CRITIC_MAX_TOKENS}"
    )


# ---------------------------------------------------------------------------
# LLM critic invocation
# ---------------------------------------------------------------------------
async def _invoke_critic_llm(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """Returns the critic's parsed JSON verdict; on any failure approves.

    Uses the async litellm API so the critic's network call does not block the
    orchestrator's event loop (the callbacks run inside it), under a hard
    wall-clock deadline so a stalled provider cannot park the run forever
    (see `_CRITIC_TIMEOUT_S`). A timed-out or otherwise failed critic approves:
    a verdict nobody could produce must not hold up the work.
    """
    # Shares the agent tree's notion of a retryable upstream fault, and its
    # proxy pre-flight. Imported lazily to keep this module out of
    # `agents.common`'s import graph.
    from CoScientist.agents.common import RetryingLiteLlm, _is_transient

    for attempt in range(1, _CRITIC_MAX_ATTEMPTS + 1):
        started = time.perf_counter()
        try:
            # Same pre-flight every agent call gets: a proxy whose VPN is down
            # accepts the connection and then answers nothing, so without this
            # the critic buys the full deadline twice over to learn that. Fails
            # in ~5s with the reason named. No-op when the proxy is off.
            await RetryingLiteLlm._verify_proxy_reachable()
            # `timeout=` is only httpx's per-read budget; `asyncio.timeout` is
            # what actually caps a provider that stalls mid-response.
            async with asyncio.timeout(_CRITIC_TIMEOUT_S):
                resp = await litellm.acompletion(
                    model=_CRITIC_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    # No `response_format`: OpenRouter hands it to the provider
                    # as a constrained decode, which this repo has already seen
                    # hang with no timeout (ToolReranker carries no
                    # `output_schema` for that reason — see system.yaml). Every
                    # critic prompt already mandates bare JSON, and
                    # `_extract_json` copes with fences or prose around it.
                    temperature=0.0,
                    max_tokens=_CRITIC_MAX_TOKENS,
                    timeout=_CRITIC_HTTP_TIMEOUT_S,
                    num_retries=0,
                )

            # The critic bypasses the agent tree, so no model callback prices it —
            # but it runs on every orchestrator turn and is not free.
            from CoScientist.logging.metrics import record_completion
            record_completion(resp, model=_CRITIC_MODEL, agent="Critic")

            choice = resp["choices"][0]
            raw = choice["message"]["content"]
            payload = _extract_json(raw or "")
            if isinstance(payload, dict):
                return payload
            # An empty or unparseable answer needs `finish_reason` and the token
            # counts to be actionable: "length" with the content empty means the
            # cap was spent before the JSON started (a model that reasons out
            # loud first), which is a different fix from a model that answered
            # in prose.
            print(
                f"[Critic] unparseable verdict ({_truncate(raw, 300)!r}); "
                f"finish_reason={choice.get('finish_reason')!r}, "
                f"{_usage_note(resp)}; defaulting to permissive verdict."
            )
            return {"verdict": "approve"}
        except Exception as e:  # noqa: BLE001 — a critic never takes the run down
            # `_is_transient` counts a TimeoutError as retryable, which is right
            # for an agent (its deadline covers time-to-FIRST-response, so a
            # retry costs little). Here the deadline covers the whole call: once
            # it blows, a second attempt buys the same wait again for a route
            # that just failed to answer within it. Give up and approve instead.
            deadline_blown = isinstance(e, TimeoutError)
            retryable = (
                attempt < _CRITIC_MAX_ATTEMPTS
                and not deadline_blown
                and _is_transient(e)
            )
            # Elapsed time and route are what tell a stalled provider apart from
            # a stalled transport after the fact: the deadline reports a bare
            # `TimeoutError()`, which on its own names neither.
            route = (
                f"via proxy {_settings.services.proxy_url}"
                if _settings.web.use_proxy
                else "direct"
            )
            print(
                f"[Critic] LLM call failed after {time.perf_counter() - started:.1f}s "
                f"({route}, attempt {attempt}/{_CRITIC_MAX_ATTEMPTS}, {e!r}); "
                + ("retrying." if retryable else "defaulting to permissive verdict.")
            )
            if not retryable:
                return {"verdict": "approve"}
            await asyncio.sleep(min(1.5 ** attempt, 8.0))

    return {"verdict": "approve"}


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

    The critic may only override the VALUES of args the orchestrator already
    chose; it cannot drop existing keys or introduce new ones. Every sub-agent
    is an ``AgentTool`` that requires a ``request`` key, so a naive replace
    would let the critic strip it (KeyError at call time) or inject keys the
    tool does not accept. We therefore keep the original args and apply only
    overrides for keys that already exist.
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
        original = dict(getattr(fc, "args", {}) or {})
        overrides = {k: v for k, v in new_args.items() if k in original}
        ignored = [k for k in new_args if k not in original]
        if ignored:
            print(
                f"[Critic] ignoring revision keys not in original args for "
                f"{call.get('tool')}: {ignored}"
            )
        safe_args = {**original, **overrides}
        if safe_args == original:
            continue
        try:
            fc.args = safe_args
        except Exception:
            try:
                object.__setattr__(fc, "args", safe_args)
            except Exception:
                pass

def _format_retrieved_tools(state: Any) -> str:
    """Render the tools the orchestrator's retrieve_tools surfaced, so the critic
    can judge whether a chosen agent actually MATCHES an available tool.

    Reads `accumulated_tools` (what the orchestrator's own retrieve_tools calls
    accumulated this turn). Returns '' when nothing was retrieved.
    """
    try:
        tools = state.get("accumulated_tools") or []
    except Exception:  # noqa: BLE001 — state shape varies across ADK versions
        return ""
    if not tools:
        return ""
    lines: List[str] = []
    for t in tools[:20]:
        if not isinstance(t, dict):
            continue
        name = t.get("tool") or t.get("name") or t.get("tool_name") or "?"
        desc = t.get("description") or t.get("desc") or ""
        lines.append(f"- {name}: {_truncate(desc, 160)}")
    return "\n".join(lines)


def _first_text(content) -> str:
    """First text part of a Content, or '' if none (e.g. an image-only message)."""
    if content is None or not getattr(content, "parts", None):
        return ""
    for p in content.parts:
        if getattr(p, "text", None):
            return p.text
    return ""


def make_pre_action_critique(instruction: str) -> Callable:
    """Build the orchestrator's after_model_callback with the given critic prompt."""

    @track(name="pre_action_critique")
    async def pre_action_critique(
        callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        """after_model_callback for the OrchestratorAgent."""
        pending = _extract_pending_calls(llm_response)

        # No tool calls in this response -> nothing to critique.
        if not pending:
            return None

        #######
        # Auto-approve task management tools to save LLM calls and prevent false rejections.
        MANAGEMENT_TOOLS = {"update_task_status", "request_approval"}
        if all(call.get("tool") in MANAGEMENT_TOOLS for call in pending):
            print(f"pre_action_critique auto-approved management tools: {[c.get('tool') for c in pending]}")
            return None

        contents = _session_contents(callback_context)
        # user_content may be absent or start with a non-text part (image/DICOM upload).
        user_task = _first_text(getattr(callback_context, "user_content", None))
        trajectory = _extract_completed_trajectory(contents)
        #####

        contents = _session_contents(callback_context)
        # user_content may be absent or start with a non-text part (image/DICOM upload).
        user_task = _first_text(getattr(callback_context, "user_content", None))
        trajectory = _extract_completed_trajectory(contents)

        # Tools the orchestrator's retrieve_tools surfaced — lets the critic judge
        # whether a chosen compute agent actually matches an available tool.
        retrieved = _format_retrieved_tools(callback_context.state)
        retrieved_block = (
            f"RETRIEVED TOOLS (from retrieve_tools this turn):\n{retrieved}\n\n"
            if retrieved else ""
        )

        user_prompt = (
            f"ORIGINAL TASK:\n{user_task}\n\n"
            f"COMPLETED TRAJECTORY:\n{_format_trajectory(trajectory)}\n\n"
            f"{retrieved_block}"
            f"PROPOSED NEXT ACTION(S) (not yet executed):\n{_format_pending_calls(pending)}\n\n"
            "Decide whether to approve, revise, or reject these proposed actions. "
            "Respond as strict JSON."
        )

        print(f"pre action critic invoked with such prompt: {user_prompt}")

        payload = await _invoke_critic_llm(instruction, user_prompt)
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

    return pre_action_critique


# ---------------------------------------------------------------------------
# Post-action critic  (after_tool_callback)
# ---------------------------------------------------------------------------
def make_post_action_critique(instruction: str) -> Callable:
    """Build the orchestrator's after_tool_callback with the given critic prompt."""

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
        payload = await _invoke_critic_llm(instruction, user_prompt)
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

    return post_action_critique


# ---------------------------------------------------------------------------
# Plan critic  (SessionAgent.plan_critic — the PLANNER's own critic)
# ---------------------------------------------------------------------------
def make_plan_critique(instruction: str) -> Callable:
    """Build the planner's plan critic with the given critic prompt.

    Not an ADK callback: an after_model/after_agent callback can only rewrite
    or replace an output, and a plan critique is worthless unless the PLANNER
    itself redoes the roadmap. The returned coroutine is handed to the
    SessionAgent, which owns the generate → review → revise loop and caps it
    (`critic_max_rounds`, default 1 — the critic gets one say).

    Contract: ``await plan_critique(task, plan) -> feedback | None``. A None
    (approve, empty feedback, or a failed LLM call) accepts the plan as-is.
    """

    @track(name="plan_critique")
    async def plan_critique(task: str, plan: str) -> Optional[str]:
        user_prompt = (
            f"ORIGINAL TASK:\n{task or '(not available)'}\n\n"
            f"PROPOSED PLAN (as registered, in execution order):\n{_truncate(plan, 6000)}\n\n"
            "Decide whether to approve this plan or send it back for its one "
            "revision. Respond as strict JSON."
        )

        print(f"plan critic invoked with such prompt: {user_prompt}")

        payload = await _invoke_critic_llm(instruction, user_prompt)
        verdict_raw = (payload.get("verdict") or "approve").lower().strip()
        feedback = (payload.get("feedback") or "").strip()

        print(f"plan critic returned: {payload}")

        # Anything but an explicit, substantiated "revise" accepts the plan:
        # a revision round the critic cannot justify only costs a rewrite.
        if verdict_raw != PlanVerdict.REVISE.value or not feedback:
            return None
        return feedback

    return plan_critique