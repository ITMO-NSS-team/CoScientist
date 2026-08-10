"""Orchestrator helpers for ExperimentModuleAgent call shaping."""

from __future__ import annotations

import logging
import re
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.genai import types

logger = logging.getLogger(__name__)

_EM_NAME = "ExperimentModuleAgent"
_RESEARCH_NAME = "ResearchAgent"
_COMPUTE_ASK_RE = re.compile(
    r"(?i)\b("
    r"generate|suggest|design|create|discover|propose|compute|predict|dock|"
    r"molecule|molecules|inhibitor|analogs?|candidates?|smiles|screening|"
    r"сгенерир|предложи|разработ|молекул|ингибитор"
    r")\b"
)


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


def _user_ask(callback_context: CallbackContext) -> str:
    state = getattr(callback_context, "state", None) or {}
    for key in ("experiment_source_request", "user_query", "query"):
        val = state.get(key) if hasattr(state, "get") else None
        if isinstance(val, str) and val.strip():
            return val.strip()
    user_content = getattr(callback_context, "user_content", None)
    parts = getattr(user_content, "parts", None) or []
    chunks: list[str] = []
    for part in parts:
        text = getattr(part, "text", None)
        if isinstance(text, str) and text.strip():
            chunks.append(text.strip())
    return "\n".join(chunks).strip()


def redirect_research_to_experiment_module(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> Optional[LlmResponse]:
    """Rewrite ResearchAgent fan-out into one ExperimentModuleAgent on compute asks.

    Prevents literature-only paths for Generate/Suggest/… molecule tasks when EM
    is available (experiments profile). No-op if EM is already called this turn.
    """
    content = getattr(llm_response, "content", None)
    parts = list(getattr(content, "parts", None) or [])
    if not parts:
        return None

    research_idxs: list[int] = []
    research_reqs: list[str] = []
    has_em = False
    for i, part in enumerate(parts):
        fc = getattr(part, "function_call", None)
        name = getattr(fc, "name", None) if fc is not None else None
        if name == _EM_NAME:
            has_em = True
        elif name == _RESEARCH_NAME:
            research_idxs.append(i)
            args = dict(getattr(fc, "args", None) or {})
            req = args.get("request")
            if isinstance(req, str) and req.strip():
                research_reqs.append(req.strip())

    if has_em or not research_idxs:
        return None

    ask = _user_ask(callback_context)
    blob = " ".join([ask, *research_reqs])
    if not _COMPUTE_ASK_RE.search(blob):
        return None

    brief_bits = research_reqs or ([ask] if ask else [])
    if not brief_bits:
        return None
    if len(brief_bits) == 1:
        brief = brief_bits[0]
    else:
        brief = (
            "Complete the following computational experiment as ONE stage. "
            "Build a single ExperimentPlan covering all items below:\n\n"
            + "\n\n".join(f"{n}. {r}" for n, r in enumerate(brief_bits, 1))
        )

    drop = set(research_idxs)
    new_parts = [p for i, p in enumerate(parts) if i not in drop]
    new_parts.append(
        types.Part.from_function_call(name=_EM_NAME, args={"request": brief})
    )
    content.parts = new_parts
    agent = getattr(callback_context, "agent_name", None) or "orchestrator"
    logger.warning(
        "[%s] redirected %d ResearchAgent call(s) → ExperimentModuleAgent",
        agent,
        len(research_idxs),
    )
    return None


__all__ = [
    "coalesce_experiment_module_calls",
    "redirect_research_to_experiment_module",
]
