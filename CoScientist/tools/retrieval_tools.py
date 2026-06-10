"""Tools for fedotmas inference"""

import asyncio
from typing import List, Optional, Dict, Any

from google.adk.tools import BaseTool, ToolContext
from google.adk.tools.base_toolset import BaseToolset
from google.adk.agents.readonly_context import ReadonlyContext

from CoScientist.storage import RetrievalToolResult

from rag_tools import create_manager, MCPServer
from rag_tools.storage import PostgresClient
from rag_tools.config.settings import get_settings
from rag_tools.retrieval import APIEmbedder, APIReranker, BM25Reranker, HybridReranker
from rag_tools.storage.models import RetrievalResult

settings = get_settings()


class RetrievalToolSet(BaseToolset):
    """Toolset for rag tool usage"""
    def __init__(self, prefix: str = "rag_"):
        super().__init__()
        self.tool_name_prefix = prefix
        self.wrapper = None

    def get_tools(
        self,
        readonly_context: Optional[ReadonlyContext]
    ) -> List[BaseTool]:

        tools = [self.retrieve_tools, self.get_server_info]

        return tools
        
    async def close(self) -> None:
        await asyncio.sleep(0)  # Placeholder for async cleanup if needed


    async def retrieve_tools(self, query: str,
                                    tool_context: ToolContext = None
                                    ) -> Dict[str, Any]:
        """
        Tool for retrieving MCP tools from DB using RAG. 
        
        Args:
            query: query to use for tools lookup in database using RAG.
        
        Returns:
            List ot the most relevant tools in db which can be used to solve the task .
        """
        embedder = APIEmbedder(settings.api_embedding)
        api_reranker = APIReranker(settings.api_reranker)
        bm2_reranker = BM25Reranker(settings.bm_reranker)
        reranker = HybridReranker([api_reranker, bm2_reranker], settings.hybrid_reranker)
        manager = await create_manager(settings, embedder, reranker)

        try:
            retrieved_tools: List[RetrievalResult] = await manager.retrieve_tools(
                query=query,
                top_k=settings.rag.default_top_k,
                rerank=True,
                rerank_top_k=settings.rag.rerank_top_k,
                min_score=settings.rag.min_relevance_score)

            results = [
                RetrievalToolResult(
                    tool=r.name,
                    server_id=r.server_id,
                    description=r.description,
                    score=r.rerank_score,
                )
                for r in retrieved_tools
            ]
        finally:
            # Always release the manager's DB/HTTP connections, even on error.
            await manager.close()

        if tool_context is None:
            return {
                "status": "success",
                "result": [r.model_dump() for r in results],
                "accumulated_count": len(results),
                "message": f"Retrieved {len(results)} tools (no session accumulation).",
            }

        # ACCUMULATE into state
        accumulated = tool_context.state.get('accumulated_tools', [])
        existing_tools = {t['tool'] for t in accumulated}
        last_idx = len(accumulated) + 1

        for tool_result in results:
            if tool_result.tool not in existing_tools:
                accumulated.append({
                    'tool': tool_result.tool,
                    'server_id': tool_result.server_id,
                    'description': tool_result.description,
                    'score': tool_result.score,
                    'tool_index': last_idx,
                    'retrieval_query': query,  # Track which query found this
                })
                last_idx += 1
        
        tool_context.state['accumulated_tools'] = accumulated
        tool_context.state['retrieval_queries'] = tool_context.state.get('retrieval_queries', []) + [query]

        return {
            "status": "success",
            "result": [r.model_dump() for r in results],
            "accumulated_count": len(accumulated),
            "message": f"Retrieved {len(results)} tools. Total accumulated: {len(accumulated)}."
        }

    async def get_server_info(self, server_id: str) -> Dict[str, Any]:
        """
        Returns MCP server metadata. 
        
        Args:
            server_id: server id to look up for.
        
        Returns:
            Server metadata.
        """

        postgres = PostgresClient(settings.postgres)
        await postgres.initialize()

        server: MCPServer = await postgres.get_server(server_id)

        await postgres.close()

        return {
            "status": "success",
            "result": server,
        }



    
async def list_available_tools(query: str) -> Dict[str, Any]:
    """Search the MCP tool registry for ready-to-use tools relevant to a task.

    Call this EARLY to learn which capabilities ALREADY EXIST as MCP tools, then
    analyze the returned list to decide how to proceed: if the tools you need are
    present, run the experiment via TaskExecutorAgent (it executes these tools
    through FEDOT.MAS); if the needed capability is missing, gather more context
    first (research / hypotheses) or write code (CoderAgent).

    Args:
        query: a short description of the capability/task you need tools for, e.g.
            "generate candidate molecules and predict their activity against a protein".

    Returns:
        dict: {"status": "success", "count": <int>,
               "tools": [{"name", "server_id", "description", "score"}]}  sorted by relevance.
    """
    embedder = APIEmbedder(settings.api_embedding)
    reranker = HybridReranker(
        [APIReranker(settings.api_reranker), BM25Reranker(settings.bm_reranker)],
        settings.hybrid_reranker,
    )
    manager = await create_manager(settings, embedder, reranker)
    try:
        retrieved: List[RetrievalResult] = await manager.retrieve_tools(
            query=query,
            top_k=settings.rag.default_top_k,
            rerank=True,
            rerank_top_k=settings.rag.rerank_top_k,
            min_score=settings.rag.min_relevance_score,
        )
        tools = [
            {
                "name": r.name,
                "server_id": r.server_id,
                "description": (r.description or "").strip()[:200],
                "score": round(float(r.rerank_score or 0.0), 3),
            }
            for r in retrieved
        ]
    finally:
        await manager.close()

    return {"status": "success", "count": len(tools), "tools": tools}


retrieval_toolset = RetrievalToolSet()
retrieval_toolset_instance = retrieval_toolset.get_tools(None)

