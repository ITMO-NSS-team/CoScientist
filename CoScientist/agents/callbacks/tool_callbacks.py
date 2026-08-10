import os
import re

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from typing import Any, Callable, Dict, Iterable, List, Optional

import logging
logger = logging.getLogger(__name__)

# ── Executor tool-match thresholds (the Coder↔Executor redirect mechanism) ───
# A retrieved tool counts as a real match only at/above _KEEP. When NOTHING
# clears _KEEP we look at the single best score:
#   * best >= _ABSTAIN  -> marginal salvage: take top-2 and proceed (cautious).
#   * best <  _ABSTAIN  -> ABSTAIN: leave the tool set empty and flag a no-match,
#                          so ExperimentAgent redirects to CoderAgent instead of
#                          "solving" the task with an unrelated tool (e.g. running
#                          a GAN trainer for a "train a transformer" task).
_TOOL_KEEP_SCORE = float(os.getenv("EXECUTOR_TOOL_KEEP_SCORE", "0.3"))
_TOOL_ABSTAIN_SCORE = float(os.getenv("EXECUTOR_TOOL_ABSTAIN_SCORE", "0.2"))

# State key carrying the executor's tool-match verdict for the redirect guard.
TOOL_MATCH_STATE_KEY = "executor_tool_match"

# Set after a successful fedot_tool capture so Fedot/Coder cannot re-enter.
FEDOT_DELIVERABLE_READY_KEY = "fedot_deliverable_ready"
FEDOT_DELIVERABLE_READY_TOKEN = "FEDOT_DELIVERABLE_READY"

def before_tool_reranker_model(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> None:
    """Skips ToolRetriever context"""

    new_contents = []

    for content in llm_request.contents:
        # A content may have empty parts or a non-text first part (function
        # call/response) — guard before reading .text.
        first_text = content.parts[0].text if content.parts else None
        if first_text == 'For context:':
            continue
        new_contents.append(content)

    llm_request.contents = new_contents
    return


# Set when ToolReranker scores were applied from after_model (skip after_agent).
_TOOL_RERANK_APPLIED_KEY = "_tool_rerank_applied"


def _score_items_from_reranked_state(raw: Any) -> List[Dict[str, Any]]:
    """Normalize ``reranked_tools`` state (dict / model / list) to score dicts."""
    if raw is None:
        return []
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump()
    if isinstance(raw, dict):
        tools = raw.get("tools") or []
    elif isinstance(raw, list):
        tools = raw
    else:
        return []
    out: List[Dict[str, Any]] = []
    for t in tools:
        if hasattr(t, "model_dump"):
            t = t.model_dump()
        if not isinstance(t, dict):
            continue
        try:
            out.append({"index": int(t["index"]), "score": float(t["score"])})
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _llm_response_text(llm_response: LlmResponse, *, include_thoughts: bool) -> str:
    content = getattr(llm_response, "content", None)
    parts = getattr(content, "parts", None) if content is not None else None
    if not parts:
        return ""
    chunks: List[str] = []
    for p in parts:
        text = getattr(p, "text", None)
        if not text:
            continue
        if not include_thoughts and getattr(p, "thought", False):
            continue
        chunks.append(text)
    return "".join(chunks)


def _score_items_from_llm_response(llm_response: LlmResponse) -> Optional[List[Dict[str, Any]]]:
    """Parse ToolRanking scores from the model response (not from output_key state).

    Prefer non-thought text (post-sanitize path); fall back to thoughts — GLM often
    parks the JSON ranking in a thought part while the logger shows the plain text empty.
    """
    from CoScientist.agents.callbacks.json_output import _extract_json, _normalize_ranking_payload

    for include_thoughts in (False, True):
        text = _llm_response_text(llm_response, include_thoughts=include_thoughts)
        if not text.strip():
            continue
        extracted = _extract_json(text)
        if extracted is None:
            continue
        items = _score_items_from_reranked_state(_normalize_ranking_payload(extracted))
        if items:
            return items
    return None


def apply_tool_rerank_scores(state: Any, score_items: List[Dict[str, Any]]) -> None:
    """Filter ``accumulated_tools`` by rerank scores; set match verdict + filtered_tools."""
    rerank_map: Dict[int, float] = {int(t["index"]): float(t["score"]) for t in score_items}
    acc_tools: List[Dict[str, Any]] = list(state.get("accumulated_tools") or [])

    filtered_tools: List[Dict[str, Any]] = [
        tool for tool in acc_tools
        if rerank_map.get(tool.get('tool_index', -1), 0) >= _TOOL_KEEP_SCORE
    ]

    best_score = max(rerank_map.values(), default=0.0)
    matched = bool(filtered_tools)

    if not filtered_tools and best_score >= _TOOL_ABSTAIN_SCORE:
        # Marginal salvage: nothing cleared _KEEP but the best is not hopeless —
        # take top-2 and proceed cautiously (preserves the old behaviour here).
        top_ids = {
            idx for idx, _ in sorted(
                rerank_map.items(), key=lambda x: x[1], reverse=True
            )[:2]
        }
        filtered_tools = [t for t in acc_tools if t.get('tool_index', -1) in top_ids]
        matched = bool(filtered_tools)
    # else (best < _ABSTAIN): ABSTAIN — leave filtered_tools empty so the
    # redirect guard on ExperimentAgent sends the task to CoderAgent instead of
    # running an unrelated tool.

    # Record the verdict for the redirect guard / the orchestrator's critic /
    # the FEDOT hard-stop (a False "matched" here means a DIFFERENT capability
    # is being asked for, so fedot_artifact_handoff.should_hard_stop_fedot must
    # let this step through even if a prior deliverable is already captured).
    state[TOOL_MATCH_STATE_KEY] = {
        "matched": matched,
        "best_score": round(best_score, 3),
        "kept": len(filtered_tools),
    }
    state['filtered_tools'] = filtered_tools
    state['accumulated_tools'] = []
    state['retrieval_queries'] = []
    state[_TOOL_RERANK_APPLIED_KEY] = True


def after_tool_reranker_model(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    """after_model: apply ToolReranker scores from the response body.

    ``output_key`` is often invisible in ``after_agent`` (ADK state-delta timing),
    which produced false ``best_score=0.0`` / empty ``filtered_tools``. Reading the
    ranking JSON here (after ``sanitize_json_output``) avoids that race.
    """
    if any(
        getattr(p, "function_call", None)
        for p in (getattr(getattr(llm_response, "content", None), "parts", None) or [])
    ):
        return None
    items = _score_items_from_llm_response(llm_response)
    if not items:
        logger.warning(
            "[%s] after_model tool rerank: no parseable scores in response",
            _agent_name(callback_context),
        )
        return None
    apply_tool_rerank_scores(callback_context.state, items)
    return None


def after_tool_reranker_agent(
    callback_context: CallbackContext
) -> None:
    """Fallback: apply scores from ``output_key`` state if after_model did not."""

    current_state = callback_context.state
    if current_state.get(_TOOL_RERANK_APPLIED_KEY):
        return None

    score_items = _score_items_from_reranked_state(current_state.get("reranked_tools"))
    if not score_items:
        # Still record an empty verdict so redirect_when_no_tools sees best_score=0
        # rather than a stale prior-turn match.
        apply_tool_rerank_scores(current_state, [])
        return None

    apply_tool_rerank_scores(current_state, score_items)
    return None


def after_fullset_reranker_agent(
    callback_context: CallbackContext
) -> None:
    """Adds ToolReranker output to state"""

    current_state = callback_context.state
    reranked_mcps: List[Dict[str, Any]] = (current_state.get('reranked_web_servers') or {}).get('mcp_scores', [])

    # Binary deploy score (0/1) per MCP index — truthiness selects deploy.
    rerank_map: Dict[int, bool] = {t['index']: t['score'] for t in reranked_mcps}
    acc_mcps: List[Dict[str, Any]] = current_state.get('accumulated_web_mcps', [])

    filtered_mcps: List[Dict[str, Any]] = [
        mcp for mcp in acc_mcps
        if rerank_map.get(mcp.get('index', -1), False)
    ]

    callback_context.state['filtered_mcps'] = filtered_mcps
    callback_context.state['accumulated_web_mcps'] = []
    callback_context.state['retrieval_queries_mcp'] = []
    return

def before_get_task(callback_context: CallbackContext):
    """Ensure the current session has a task list before the agent runs.

    Task data already lives in ADK session state.  In particular, do not reload
    it from process-global storage here: that used to resurrect stale plans and
    mix concurrent users.
    """
    if callback_context.state.get("active_tasks") is None:
        callback_context.state["active_tasks"] = []
    return None


def inject_graph_root(callback_context: CallbackContext):
    """Give the agent the session graph root and relevant global memory.

    state['graph_root'] (rendered via the {graph_root?} placeholder) gets:
      1. the system root — every agent + its capabilities + this session's trace;
      2. relevant facts accumulated by all completed local research sessions,
         retrieved for the current query so agents build on prior findings.
    Best-effort — the graph must never break a run.
    """
    parts = []
    query = ""
    try:
        from CoScientist.graph.memory import get_knowledge_graph
        knowledge_graph = get_knowledge_graph(callback_context)
        parts.append(knowledge_graph.root_summary())
        goals = [h for h in knowledge_graph.history(limit=50) if h.get("kind") == "goal"]
        query = goals[-1]["label"] if goals else ""
    except Exception:  # noqa: BLE001
        pass
    try:
        from CoScientist.graph.memory_store import get_knowledge_memory
        knowledge_memory = get_knowledge_memory(callback_context)
        mem = knowledge_memory.relevant_summary(query)
        if mem:
            parts.append(mem)
    except Exception:  # noqa: BLE001
        pass
    callback_context.state['graph_root'] = "\n\n".join(p for p in parts if p)
    return None

# Recognisable token the orchestrator prompt / post-critic key off to re-route.
NO_MATCHING_TOOL_TOKEN = "NO_MATCHING_TOOL"


def redirect_when_no_tools(
    callback_context: CallbackContext,
) -> Optional[types.Content]:
    """before_agent_callback for ExperimentAgent: abstain → redirect to CoderAgent.

    By the time ExperimentAgent runs, the tool-prep pipeline has set
    ``executor_tool_match``. If no retrieved tool matched the task (and no web
    MCP was deployed), running FEDOT would just pick the nearest-but-wrong tool
    (the "train a GAN for a transformer task" failure). Instead we short-circuit
    the agent and return a structured redirect so the orchestrator sends the
    step to CoderAgent.
    """
    state = callback_context.state
    verdict = state.get(TOOL_MATCH_STATE_KEY) or {}
    has_local = bool(state.get("filtered_tools"))
    has_web = bool(state.get("filtered_mcps"))

    # Only abstain on an explicit no-match verdict with nothing usable.
    if verdict.get("matched") or has_local or has_web:
        return None

    best = verdict.get("best_score", 0.0)
    message = (
        f"{NO_MATCHING_TOOL_TOKEN}: No ready-made MCP tool matches this task "
        f"(best tool relevance was {best}, below the bar). This looks like custom "
        "engineering — a specific architecture, a named repository/example code, "
        "or writing and running code — which no existing tool covers. Do NOT "
        "treat a tool that shares only the verb (e.g. 'train a GAN' for a 'train a "
        "transformer' request) as a match. Recommend re-routing this step to "
        "CoderAgent."
    )
    logger.info("[ExperimentAgent] abstaining (no matching tool, best=%s) → CoderAgent", best)
    state["fedot_results"] = message
    return types.Content(role="model", parts=[types.Part(text=message)])


def _artifact_urls(artifacts: Any) -> List[str]:
    urls: List[str] = []
    for art in artifacts or []:
        if isinstance(art, dict):
            for key in ("results_presigned_url", "url", "presigned_url"):
                val = art.get(key)
                if val:
                    urls.append(str(val))
                    break
        elif isinstance(art, str) and art.strip():
            urls.append(art.strip())
    return urls


def refuse_when_fedot_deliverable(
    callback_context: CallbackContext,
) -> Optional[types.Content]:
    """before_agent: hard-stop Fedot/Coder once the ask's deliverable is done.

    Soft prompt STOP alone does not prevent ADK re-entry after a successful
    ``fedot_tool``. Uses ``should_hard_stop_fedot``, which is deliberately
    conservative: it does NOT fire when the current step's tool-match verdict
    abstained or names a new tool, i.e. whenever the orchestrator genuinely
    still needs another agent call (a gen→dock handoff, or a distinct step
    routed to CoderAgent) this returns None and lets the agent run normally.
    """
    from CoScientist.tools.fedot_artifact_handoff import should_hard_stop_fedot

    state = callback_context.state
    if not should_hard_stop_fedot(state):
        return None
    urls = _artifact_urls(state.get("fedot_artifacts"))
    body = "\n".join(urls) if urls else "(see session state fedot_artifacts)"
    message = (
        f"{FEDOT_DELIVERABLE_READY_TOKEN}: S3/artifacts already captured. "
        "Do NOT call fedot_tool, CoderAgent, or retrieve again. "
        "Hand these URLs to the orchestrator for Final Response:\n"
        f"{body}"
    )
    logger.info(
        "[%s] refusing re-entry — fedot deliverable already ready (%s url(s))",
        _agent_name(callback_context),
        len(urls),
    )
    state["fedot_results"] = message
    return types.Content(role="model", parts=[types.Part(text=message)])


def make_unknown_tool_guard(valid_names: Iterable[str]) -> Callable:
    """Build an after_model_callback that intercepts hallucinated tool calls.

    When the LLM emits a function call whose name is NOT a real tool of the
    agent, ADK raises and kills the whole run before any tool/agent callback can
    react (e.g. CoderAgent calling `find` directly instead of
    `execute_bash("find ...")`). This guard catches that in the model response
    and replaces it with a corrective message, so the agent re-plans on its next
    turn instead of crashing the orchestration.
    """
    valid = set(valid_names)

    def guard(
        callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        content = getattr(llm_response, "content", None)
        parts = getattr(content, "parts", None) if content is not None else None
        if not parts:
            return None
        unknown = []
        for p in parts:
            fc = getattr(p, "function_call", None)
            name = getattr(fc, "name", None) if fc is not None else None
            if name and name not in valid:
                unknown.append(name)
        if not unknown:
            return None
        bad = ", ".join(sorted(set(unknown)))
        allowed = ", ".join(sorted(valid))
        logger.warning("[%s] hallucinated tool call(s): %s", _agent_name(callback_context), bad)
        msg = (
            f"The tool(s) `{bad}` do not exist — they are not in your tool list. "
            f"Your only tools are: {allowed}. Shell programs (find, grep, ls, cat, "
            "wc, git, sed, awk, …) are NOT tools — run them INSIDE execute_bash, "
            "e.g. execute_bash(command=\"find . -name '*.py' | wc -l\"). "
            "Re-issue your request calling ONLY a tool from the list above."
        )
        return LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text=msg)])
        )

    return guard


def _agent_name(callback_context: CallbackContext) -> str:
    return getattr(callback_context, "agent_name", None) or "agent"


def print_research_agent_tool_call(
    tool: BaseTool,
    args: Dict[str, Any],
    tool_context: ToolContext,
    tool_response: Any,
) -> None:
    """Print tool calls and persist downloaded S3 keys to session state."""
    try:
        logger.info(f"\n[ResearchAgent tool called] {tool.name}")
        logger.info(f"[ResearchAgent tool args] {args}")
    except Exception as e:
        logger.error(f"Error in print_research_agent_tool_call: {e}")

    if tool.name != "download_papers_from_search":
        return

    try:
        papers = (tool_response or {}).get("metadata", {}).get("papers", [])
        new_keys = [p["s3_key"] for p in papers if p.get("s3_key")]
        if not new_keys:
            return
        existing: List[str] = tool_context.state.get("downloaded_paper_s3_keys", [])
        merged_keys: List[str] = existing + [k for k in new_keys if k not in existing]
        tool_context.state["downloaded_paper_s3_keys"] = merged_keys
        logger.info(
            "Registered %d downloaded paper S3 key(s) in session state.",
            len(merged_keys),
        )
    except Exception as e:
        logger.error("Failed to persist downloaded paper S3 keys: %s", e)

def capture_mcp_artifacts(
    tool: BaseTool,
    args: Dict[str, Any],
    tool_context: ToolContext,
    tool_response: Any,
) -> None:
    """after_tool: stash figure/table artifact URLs a tool returned into
    ``state['mcp_artifacts']`` so the graph-first Result Aggregator's
    ``format_results`` downloads them into the report folder.

    Many MCP tools (e.g. the tox-antitargets suite) render a plot server-side and
    return a presigned URL to it (commonly ``metadata.figure.artifact``). That link
    only lives in the tool result; with the aggregator running ``include_contents:
    none`` it never reaches the report unless captured here — at the AGENT's own
    tool boundary, which fires for sub-agent (AgentTool) MCP calls where an
    App-level plugin does not.
    """
    try:
        from CoScientist.reporting.collect import find_artifact_urls
        urls = find_artifact_urls(tool_response)
    except Exception:  # noqa: BLE001 — capture must never break a tool call
        return
    if not urls:
        return
    try:
        existing = list(tool_context.state.get("mcp_artifacts") or [])
        seen = {a.get("url") for a in existing if isinstance(a, dict)}
        name = getattr(tool, "name", None)
        for u in urls:
            if u in seen:
                continue
            seen.add(u)
            existing.append({"url": u, "tool": name})
        tool_context.state["mcp_artifacts"] = existing
        logger.info("capture_mcp_artifacts: %s → +%d artifact URL(s) (%d total)",
                    name, len(urls), len(existing))
    except Exception as e:  # noqa: BLE001
        logger.error("capture_mcp_artifacts failed: %s", e)


class SearchLimiter:

    _STATE_KEY = "_search_limiter_count"

    def __init__(self, max_searches: int = 5):
        self.max_searches = max_searches

    def limit_searches(self, tool, args: dict, tool_context: ToolContext) -> Optional[dict]:
        # Match "search" as a whole name token, NOT as a substring: otherwise
        # "re-search" tools (research_commit, research_context_slice, …) are
        # wrongly counted as searches and blocked once the cap is hit, which
        # stops agents recording anything in the research graph.
        tokens = re.split(r"[^a-z]+", tool.name.lower())
        if "search" not in tokens:
            return None

        count = tool_context.state.get(self._STATE_KEY, 0)
        count += 1
        tool_context.state[self._STATE_KEY] = count

        if count > self.max_searches:
            return {
                "result": (
                    f"Search limit reached ({self.max_searches} searches allowed). "
                    "You MUST now synthesize your answer from the results you already have. "
                    "Do NOT attempt any more searches."
                )
            }
        return None
