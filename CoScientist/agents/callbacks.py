from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.tool_context import ToolContext
from google.adk.models import LlmRequest
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from typing import List, Dict, Any, Optional
from google.adk.models.llm_response import LlmResponse  
from google.genai import types  

from typing import Optional, List, Dict, Any
import json
import ast

import logging
logger = logging.getLogger(__name__)
from CoScientist.storage.models import ToolRanking, MCPRanking
import re

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


def after_tool_reranker_model_callback(
    callback_context: CallbackContext,
    llm_response: LlmResponse
) -> None:
    """Adds ToolReranker output to state"""

    current_state = callback_context.state
    reranked_tools: Dict[str, float] = (current_state.get('reranked_tools') or {}).get('tools', [])

    rerank_map: Dict[int, float] = {t['index']: t['score'] for t in reranked_tools}
    acc_tools: List[Dict[str, Any]] = current_state.get('accumulated_tools', [])

    filtered_tools: List[Dict[str, Any]] = [
        tool for tool in acc_tools
        if rerank_map.get(tool.get('tool_index', -1), 0) >= 0.3
    ]

    if not filtered_tools:
        # fallback: take top-2 by rerank score
        top_indices = sorted(
            rerank_map.items(),
            key=lambda x: x[1],
            reverse=True
        )[:2]

        top_ids = {idx for idx, _ in top_indices}

        filtered_tools = [
            tool for tool in acc_tools
            if tool.get('tool_index', -1) in top_ids
        ]

    callback_context.state['filtered_tools'] = filtered_tools
    callback_context.state['accumulated_tools'] = []
    callback_context.state['retrieval_queries'] = []
    return


def after_fullset_reranker_model_callback(
    callback_context: CallbackContext,
    llm_response: LlmResponse
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


def print_research_agent_tool_call(
    tool: BaseTool,
    args: Dict[str, Any],
    tool_context: ToolContext,
    tool_response: Any,
) -> None:
    """Print the tool name and args when ResearchAgent invokes a tool."""
    try:
        logger.info(f"\n[ResearchAgent tool called] {tool.name}")
        logger.info(f"[ResearchAgent tool args] {args}")
    except Exception as e:
        logger.error(f"Error in print_research_agent_tool_call: {e}")

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

def normalize_json_response(  
    callback_context: CallbackContext,  
    llm_response: LlmResponse,  
) -> None:  
    """Normalize single-quoted Python dict to valid JSON (in-place)."""  
    if not (llm_response.content and llm_response.content.parts):  
        return None  
  
    for i, part in enumerate(llm_response.content.parts):  
        if not part.text:  
            continue  
        text = part.text.strip()  
        try:  
            json.loads(text)  
        except json.JSONDecodeError:  
            try:  
                obj = ast.literal_eval(text)  
                llm_response.content.parts[i].text = json.dumps(obj)
            except Exception:
                pass
  
    return None  