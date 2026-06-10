from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from typing import List, Dict, Any

import logging
logger = logging.getLogger(__name__)

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
        if rerank_map.get(tool['tool_index'], 0) >= 0.3
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
            if tool['tool_index'] in top_ids
        ]

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
        if rerank_map.get(mcp['index'], False)
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