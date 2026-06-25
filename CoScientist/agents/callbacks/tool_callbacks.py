from CoScientist.tools.task_tracker import task_tracker_instance
import os

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


def after_tool_reranker_agent(
    callback_context: CallbackContext
) -> None:
    """Adds ToolReranker output to state"""

    current_state = callback_context.state
    reranked_tools: Dict[str, float] = (current_state.get('reranked_tools') or {}).get('tools', [])

    rerank_map: Dict[int, float] = {t['index']: t['score'] for t in reranked_tools}
    acc_tools: List[Dict[str, Any]] = current_state.get('accumulated_tools', [])

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

    # Record the verdict for the redirect guard / the orchestrator's critic.
    callback_context.state[TOOL_MATCH_STATE_KEY] = {
        "matched": matched,
        "best_score": round(best_score, 3),
        "kept": len(filtered_tools),
    }
    callback_context.state['filtered_tools'] = filtered_tools
    callback_context.state['accumulated_tools'] = []
    callback_context.state['retrieval_queries'] = []
    return


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
    """Get task before agent is called"""  
    active_tasks = task_tracker_instance.get_active_tasks(readonly_context=callback_context)  
    callback_context.state['active_tasks'] = active_tasks
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

class SearchLimiter:

    _STATE_KEY = "_search_limiter_count"

    def __init__(self, max_searches: int = 5):
        self.max_searches = max_searches

    def limit_searches(self, tool, args: dict, tool_context: ToolContext) -> Optional[dict]:
        if "search" not in tool.name.lower():
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