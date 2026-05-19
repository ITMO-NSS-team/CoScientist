from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest

from typing import Optional, List, Dict, Any

from CoScientist.storage.models import ToolRanking

def before_tool_reranker_model(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> None:
    """Skips ToolRetriever context"""

    new_contents = []

    for content in llm_request.contents:
        if content.parts[0].text == 'For context:':
            continue
        else:
            new_contents.append(content)
    
    llm_request.contents = new_contents
    return


def after_tool_reranker_agent(
    callback_context: CallbackContext
) -> None:
    """Adds ToolReranker output to state"""

    current_state = callback_context.state
    reranked_tools: Dict[str, float] = current_state['reranked_tools']['tools']

    rerank_map: Dict[int, float] = {t['index']: t['score'] for t in reranked_tools}
    acc_tools: List[Dict[str, Any]] = current_state['accumulated_tools']

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
    reranked_mcps: Dict[str, bool] = current_state['reranked_web_servers']['mcp_scores']

    rerank_map: Dict[int, float] = {t['index']: t['score'] for t in reranked_mcps}
    acc_mcps: List[Dict[str, Any]] = current_state['accumulated_web_mcps']

    filtered_mcps: List[Dict[str, Any]] = [
        mcp for mcp in acc_mcps
        if rerank_map.get(mcp['index'], False)
    ]

    callback_context.state['filtered_mcps'] = filtered_mcps
    callback_context.state['accumulated_web_mcps'] = []
    callback_context.state['retrieval_queries_mcp'] = []
    return