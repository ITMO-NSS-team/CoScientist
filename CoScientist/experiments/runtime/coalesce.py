"""Orchestrator helpers for ExperimentModuleAgent call shaping.

Two structural (never keyword-based) safety nets live here:

1. ``coalesce_experiment_module_calls`` — if the orchestrator accidentally fans
   out several ExperimentModuleAgent calls in one turn, merge them into a single
   self-contained brief so the module builds ONE ExperimentPlan.

2. ``enforce_experiment_module_first`` — give the Experiment Module the FIRST
   shot at any ask before literature research *or* a top-level McpBuilder hop,
   decided by EXECUTION STATE (has the module run this session?). After the
   module started, orch Research/McpBuilder is only the NO_MATCHING_TOOL
   fallback — including after phase=completed. The module then decides
   research-vs-compute from its own inventory. A first-shot McpBuilder call
   is rewritten to the module (same as Research). Each rewrite flags
   ``GATE_ROUTED_STATE_KEY`` so the early feasibility gate may apply.
"""

from __future__ import annotations

import logging
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.genai import types

from .shared import GATE_ROUTED_STATE_KEY

logger = logging.getLogger(__name__)

_EM_NAME = "ExperimentModuleAgent"
_RESEARCH_NAME = "ResearchAgent"
_MCP_BUILDER_NAME = "McpBuilderAgent"
# Set by research_init, or by ContextInit from the user's original_request.
_ROOT_GOAL_STATE_KEY = "orchestrator_root_goal"
_FRAME_STATE_KEY = "research_frame"


def _text_parts(content: object) -> str:
    parts = getattr(content, "parts", None) if content is not None else None
    return "\n".join(
        t.strip() for p in (parts or [])
        if (t := getattr(p, "text", "") or "").strip()
    ).strip()


def _canonical_ask(callback_context: CallbackContext, fallback: str) -> str:
    """User's original ask — never a reworded Research/McpBuilder brief.

    Order: research_init goal, ContextInit frame.original_request, user_content, brief.
    """
    state = getattr(callback_context, "state", None)
    getter = getattr(state, "get", None) if state is not None else None
    if callable(getter):
        root = getter(_ROOT_GOAL_STATE_KEY)
        if isinstance(root, str) and root.strip():
            return root.strip()
        raw = getter(_FRAME_STATE_KEY)
        if isinstance(raw, dict):
            text = raw.get("original_request")
            if isinstance(text, str) and text.strip():
                return text.strip()
        text = getattr(raw, "original_request", None) if raw is not None else None
        if isinstance(text, str) and text.strip():
            return text.strip()
    user = _text_parts(getattr(callback_context, "user_content", None))
    return user or fallback


def _experiment_module_attempted(state: object) -> bool:
    """True once the Experiment Module has started for this session.

    ``experiment_source_request`` is persisted the moment the module's
    ToolPreparer runs; ``experiment_runtime``/``experiment_context`` appear once
    planning begins. Any of them means the module already had its shot (and, if
    it bailed, emitted NO_MATCHING_TOOL), so the gate must not re-route Research.
    """
    getter = getattr(state, "get", None)
    if not callable(getter):
        return False
    for key in ("experiment_source_request", "experiment_runtime", "experiment_context"):
        if getter(key):
            return True
    return False


def enforce_experiment_module_first(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> Optional[LlmResponse]:
    """after_model: rewrite a first-shot ResearchAgent call into the module.

    Deterministic gate keyed on execution state, not on the request text:
      * same-turn Research/McpBuilder + EM → drop the parallel hop (Dual EM);
      * after the module started, orch Research only if NO_MATCHING_TOOL;
      * first-shot Research/McpBuilder (no EM yet) → rewrite to the module.
    The module ``request`` is the user's original ask (research_init goal,
    ContextInit ``research_frame.original_request``, or this turn's
    user_content) — never a reworded Research/McpBuilder brief, which would
    steer the module into a literature-only plan. The module's inventory (not
    a keyword rule) then decides whether the ask is computable; if it is not,
    it returns NO_MATCHING_TOOL and the orchestrator falls back to ResearchAgent
    on the next turn.
    """
    content = getattr(llm_response, "content", None)
    parts = list(getattr(content, "parts", None) or [])
    if not parts:
        return None
    state = callback_context.state

    research_idxs: list[int] = []
    mcp_idxs: list[int] = []
    has_module_call = False
    for i, part in enumerate(parts):
        name = getattr(getattr(part, "function_call", None), "name", None)
        if name == _EM_NAME:
            has_module_call = True
        elif name == _RESEARCH_NAME:
            research_idxs.append(i)
        elif name == _MCP_BUILDER_NAME:
            mcp_idxs.append(i)

    def _drop_parallel(idxs: list[int], *, reason: str) -> None:
        if not idxs:
            return
        drop = set(idxs)
        kept = [p for i, p in enumerate(parts) if i not in drop]
        if not kept:
            kept = [types.Part(text=(
                "Experiment Module already owns this session; literature and "
                "infra hops go through EM routes, not a parallel orchestrator call."
            ))]
        content.parts = kept
        logger.warning(
            "[%s] dropped parallel %s call(s) (%s)",
            getattr(callback_context, "agent_name", None) or "orchestrator",
            ",".join(sorted({
                getattr(getattr(parts[i], "function_call", None), "name", "") or "?"
                for i in idxs
            })),
            reason,
        )

    # Same-turn Research/McpBuilder + EM is Dual EM: science/compute stays in EM.
    if has_module_call:
        _drop_parallel(research_idxs + mcp_idxs, reason="same_turn_em")
        return None

    from CoScientist.experiments.runtime.guards import NO_MATCHING_TOOL_STATE_KEY

    getter = getattr(state, "get", None)
    no_match = bool(callable(getter) and getter(NO_MATCHING_TOOL_STATE_KEY))
    # After the module started, orch Research/McpBuilder only on NO_MATCHING_TOOL.
    # phase=completed is not a literature lane — science/compute stays in EM.
    if _experiment_module_attempted(state) and not no_match:
        _drop_parallel(research_idxs + mcp_idxs, reason="em_in_progress")
        return None
    if _experiment_module_attempted(state):
        return None

    def _brief_of(fc: object) -> str:
        args = dict(getattr(fc, "args", None) or {})
        brief = args.get("request") or args.get("query") or args.get("input") or args.get("message")
        return brief.strip() if isinstance(brief, str) else ""

    rewrite_idxs = list(research_idxs)
    rewrite_idxs.extend(mcp_idxs)
    if not rewrite_idxs:
        return None

    for i in rewrite_idxs:
        fc = getattr(parts[i], "function_call", None)
        if fc is None:
            continue
        fc.name = _EM_NAME
        # The module reads ``request``. Prefer the canonical top-level goal so a
        # reworded literature / infra delegation brief cannot bias the module;
        # fall back to whatever brief the orchestrator wrote otherwise.
        args = dict(getattr(fc, "args", None) or {})
        brief = _brief_of(fc)
        chosen = _canonical_ask(callback_context, brief)
        if chosen:
            args["request"] = chosen
        fc.args = args
    if hasattr(state, "__setitem__"):
        # Mark this ask as structurally rerouted (not an explicit orchestrator
        # pick) so assess_experiment_inventory_feasibility may early-exit it
        # with NO_MATCHING_TOOL if the module's inventory has no compute signal.
        state[GATE_ROUTED_STATE_KEY] = True
    logger.warning(
        "[%s] routed first-shot Research/McpBuilder call(s) → ExperimentModuleAgent "
        "(module decides compute-vs-research from inventory)",
        getattr(callback_context, "agent_name", None) or "orchestrator",
    )
    return None  # in-place mutation is enough


def coalesce_experiment_module_calls(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> Optional[LlmResponse]:
    """If the orchestrator fans out N ExperimentModuleAgent calls, keep one.

    Merges every ``request`` into a single self-contained brief so the module
    builds one ExperimentPlan instead of N interleaved runtimes.
    """
    content = getattr(llm_response, "content", None)
    parts = list(getattr(content, "parts", None) or [])
    if not parts:
        return None

    em_idxs: list[int] = []
    requests: list[str] = []
    for i, part in enumerate(parts):
        fc = getattr(part, "function_call", None)
        if fc is None or getattr(fc, "name", None) != _EM_NAME:
            continue
        em_idxs.append(i)
        args = dict(getattr(fc, "args", None) or {})
        req = args.get("request")
        if isinstance(req, str) and req.strip():
            requests.append(req.strip())

    if len(em_idxs) <= 1:
        return None

    merged = (
        "Complete the following computational experiment as ONE stage. "
        "Build a single ExperimentPlan covering all items below in order "
        "(with depends_on / artifact handoff as needed):\n\n"
        + "\n\n".join(f"{n}. {r}" for n, r in enumerate(requests, 1))
    )
    keep_i = em_idxs[0]
    keep_fc = getattr(parts[keep_i], "function_call", None)
    if keep_fc is not None:
        keep_fc.args = dict(getattr(keep_fc, "args", None) or {})
        keep_fc.args["request"] = merged

    drop = set(em_idxs[1:])
    content.parts = [p for i, p in enumerate(parts) if i not in drop]
    agent = getattr(callback_context, "agent_name", None) or "orchestrator"
    logger.warning(
        "[%s] coalesced %d ExperimentModuleAgent calls into 1",
        agent,
        len(em_idxs),
    )
    return None  # in-place mutation is enough


def suppress_experiment_module_after_completed(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> Optional[LlmResponse]:
    """after_model: do not re-enter the module after result HITL accepted the stage."""
    state = getattr(callback_context, "state", None)
    getter = getattr(state, "get", None) if state is not None else None
    runtime = getter("experiment_runtime") if callable(getter) else None
    if not isinstance(runtime, dict) or runtime.get("phase") != "completed":
        return None
    content = getattr(llm_response, "content", None)
    parts = list(getattr(content, "parts", None) or [])
    if not parts:
        return None
    em_idxs = [
        i for i, part in enumerate(parts)
        if getattr(getattr(part, "function_call", None), "name", None) == _EM_NAME
    ]
    if not em_idxs:
        return None
    kept = [p for i, p in enumerate(parts) if i not in set(em_idxs)]
    if not kept:
        summary = getter("experiment_summary") if callable(getter) else None
        if not isinstance(summary, str) or not summary.strip():
            summary = (
                "Experiment stage already completed for this session; "
                "not starting a second plan."
            )
        kept = [types.Part(text=summary)]
    content.parts = kept
    logger.warning(
        "[%s] suppressed ExperimentModuleAgent call(s) after phase=completed",
        getattr(callback_context, "agent_name", None) or "orchestrator",
    )
    return None


__all__ = [
    "coalesce_experiment_module_calls",
    "enforce_experiment_module_first",
    "suppress_experiment_module_after_completed",
]
