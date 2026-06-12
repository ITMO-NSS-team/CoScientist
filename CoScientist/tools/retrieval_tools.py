"""Tools for fedotmas inference"""

import asyncio
import logging
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

logger = logging.getLogger(__name__)


async def _create_rag_manager():
    """Build a rag_tools manager (embedder + hybrid reranker + backing stores).

    Raises on backing-store connection failure (Postgres/Qdrant). Callers wrap
    this so a tool-registry outage degrades to "no tools" instead of killing the
    whole agent run — literature/knowledge queries don't need the registry.
    """
    embedder = APIEmbedder(settings.api_embedding)
    reranker = HybridReranker(
        [APIReranker(settings.api_reranker), BM25Reranker(settings.bm_reranker)],
        settings.hybrid_reranker,
    )
    return await create_manager(settings, embedder, reranker)


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
        try:
            manager = await _create_rag_manager()
        except Exception as exc:
            logger.warning(
                "retrieve_tools: tool registry unavailable (%s); returning empty result", exc
            )
            return {
                "status": "unavailable",
                "result": [],
                "accumulated_count": len(tool_context.state.get('accumulated_tools', [])) if tool_context else 0,
                "message": f"Tool registry unavailable ({type(exc).__name__}); proceed without RAG tool retrieval.",
            }

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
        except Exception as exc:
            logger.warning("retrieve_tools: retrieval failed (%s); returning empty result", exc)
            return {
                "status": "unavailable",
                "result": [],
                "accumulated_count": len(tool_context.state.get('accumulated_tools', [])) if tool_context else 0,
                "message": f"Tool retrieval failed ({type(exc).__name__}).",
            }
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
        try:
            await postgres.initialize()
            server: MCPServer = await postgres.get_server(server_id)
        except Exception as exc:
            # Same graceful-degradation contract as retrieve_tools (F009.A2): a
            # registry/DB outage must not crash the agent run.
            logger.warning(
                "get_server_info: server registry unavailable (%s); returning unavailable", exc
            )
            return {
                "status": "unavailable",
                "result": None,
                "message": f"Server registry unavailable ({type(exc).__name__}).",
            }
        finally:
            try:
                await postgres.close()
            except Exception:
                pass

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
    try:
        manager = await _create_rag_manager()
    except Exception as exc:
        logger.warning(
            "list_available_tools: tool registry unavailable (%s); returning empty list", exc
        )
        return {
            "status": "unavailable",
            "count": 0,
            "tools": [],
            "message": f"Tool registry unavailable ({type(exc).__name__}); proceed without RAG tool retrieval.",
        }

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
                # FULL description — never truncate. A tool's description carries the
                # info the orchestrator needs to ground a request (e.g. the concrete
                # cases / datasets / models a parameterized tool supports). Cutting it
                # hid those and led to invented case names / "false success" (F015c/F015h).
                "description": (r.description or "").strip(),
                "score": round(float(r.rerank_score or 0.0), 3),
            }
            for r in retrieved
        ]
    except Exception as exc:
        logger.warning("list_available_tools: retrieval failed (%s); returning empty list", exc)
        return {
            "status": "unavailable",
            "count": 0,
            "tools": [],
            "message": f"Tool retrieval failed ({type(exc).__name__}).",
        }
    finally:
        await manager.close()

    return {"status": "success", "count": len(tools), "tools": tools}


async def list_server_tools(server_id: str) -> Dict[str, Any]:
    """List ALL tools of one MCP server, with FULL (untruncated) descriptions.

    Use this AFTER `list_available_tools` surfaces a relevant server: RAG search
    returns only the top-k matching tools and a 500-char description snippet, whereas
    this returns the server's COMPLETE toolset and each tool's full description and
    input schema — including the concrete cases / datasets / models a parameterized
    tool supports (e.g. which disease cases a generator has). Use it to ground a
    request against the tool's real options instead of assuming a value.

    Args:
        server_id: the `server_id` from a `list_available_tools` result.

    Returns:
        {"status": "success", "server_id", "count",
         "tools": [{"name", "description", "input_schema", "tags"}]}
    """
    postgres = PostgresClient(settings.postgres)
    try:
        await postgres.initialize()
        tools = await postgres.get_tools_by_server(server_id)
    except Exception as exc:
        logger.warning(
            "list_server_tools: server registry unavailable (%s); returning unavailable", exc
        )
        return {
            "status": "unavailable",
            "server_id": server_id,
            "count": 0,
            "tools": [],
            "message": f"Server registry unavailable ({type(exc).__name__}).",
        }
    finally:
        try:
            await postgres.close()
        except Exception:
            pass

    return {
        "status": "success",
        "server_id": server_id,
        "count": len(tools),
        "tools": [
            {
                "name": t.name,
                "description": (t.description or "").strip(),  # FULL — never truncate
                "input_schema": t.input_schema,
                "tags": t.tags,
            }
            for t in tools
        ],
    }


retrieval_toolset = RetrievalToolSet()
retrieval_toolset_instance = retrieval_toolset.get_tools(None)

