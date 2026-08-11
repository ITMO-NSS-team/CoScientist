"""Orchestrator helpers for ExperimentModuleAgent call shaping.

Two structural (never keyword-based) safety nets live here:

1. ``coalesce_experiment_module_calls`` — if the orchestrator accidentally fans
   out several ExperimentModuleAgent calls in one turn, merge them into a single
   self-contained brief so the module builds ONE ExperimentPlan.

2. ``enforce_experiment_module_first`` — give the Experiment Module the FIRST
   shot at any ask before literature research, decided by EXECUTION STATE (has
   the module run this session?), never by matching words in the request. The
   module then decides research-vs-compute from its own inventory: on a
   NO_MATCHING_TOOL verdict the module has already run, so this gate no longer
   fires and the orchestrator's ResearchAgent fallback proceeds normally.
"""

from __future__ import annotations

import logging
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse

logger = logging.getLogger(__name__)

_EM_NAME = "ExperimentModuleAgent"
_RESEARCH_NAME = "ResearchAgent"
# Set by research_init: the orchestrator's canonical top-level goal.
_ROOT_GOAL_STATE_KEY = "orchestrator_root_goal"


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
      * fires only when the turn calls ResearchAgent and NOT the module, and
      * the module has never run this session.
    The module ``request`` prefers the orchestrator's canonical top-level goal
    (``research_init(question=...)``) over the ResearchAgent delegation brief:
    the latter is often a reworded "find literature on X" that would steer the
    module into a literature plan even when the real ask is compute. The
    module's inventory (not a keyword rule) then decides whether the ask is
    computable; if it is not, it returns NO_MATCHING_TOOL and the orchestrator
    falls back to ResearchAgent on the next turn.
    """
    content = getattr(llm_response, "content", None)
    parts = list(getattr(content, "parts", None) or [])
    if not parts:
        return None
    state = callback_context.state
    if _experiment_module_attempted(state):
        return None

    root_goal = state.get(_ROOT_GOAL_STATE_KEY) if hasattr(state, "get") else None
    root_goal = root_goal.strip() if isinstance(root_goal, str) else ""

    research_idxs: list[int] = []
    has_module_call = False
    for i, part in enumerate(parts):
        name = getattr(getattr(part, "function_call", None), "name", None)
        if name == _EM_NAME:
            has_module_call = True
        elif name == _RESEARCH_NAME:
            research_idxs.append(i)
    if has_module_call or not research_idxs:
        return None

    for i in research_idxs:
        fc = getattr(parts[i], "function_call", None)
        if fc is None:
            continue
        fc.name = _EM_NAME
        # The module reads ``request``. Prefer the canonical top-level goal so a
        # reworded literature delegation brief cannot bias the module's plan;
        # fall back to whatever brief the orchestrator wrote otherwise.
        args = dict(getattr(fc, "args", None) or {})
        brief = args.get("request") or args.get("query") or args.get("input") or args.get("message")
        brief = brief.strip() if isinstance(brief, str) else ""
        chosen = root_goal or brief
        if chosen:
            args["request"] = chosen
        fc.args = args
    logger.warning(
        "[%s] routed first-shot ResearchAgent call(s) → ExperimentModuleAgent "
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


__all__ = ["coalesce_experiment_module_calls", "enforce_experiment_module_first"]
