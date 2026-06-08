"""Tools for websearch"""
from typing import List, Optional, Dict, Any
import asyncio
import inspect

from CoScientist.tools.utils import tool, toolset
from CoScientist.config import get_settings

from google.adk.tools import FunctionTool, BaseTool
from google.adk.tools.base_toolset import BaseToolset
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams


settings = get_settings()
PAPER_ANALYSIS_URL = settings.mcp.paper_analysis_url
PAPERS_SEARCH_URL = settings.mcp.papers_search_url

websearch_toolset_instance = McpToolset(
    connection_params=StreamableHTTPConnectionParams(
        url=f"https://mcp.tavily.com/mcp/?tavilyApiKey={settings.services.tavily_api_key}"
    ),
)

paper_analysis_toolset_instance = McpToolset(
    connection_params=StreamableHTTPConnectionParams(
        url=PAPER_ANALYSIS_URL
    ),
)

papers_search_toolset_instance = McpToolset(
    connection_params=StreamableHTTPConnectionParams(
        url=PAPERS_SEARCH_URL
    ),
)


# class WebSearchToolset(BaseToolset):
#     """Toolset for websearch usage"""
#     def __init__(self, prefix: str = "web_"):
#         self.tool_name_prefix = prefix

#     async def get_tools(
#         self,
#         readonly_context: Optional[ReadonlyContext]
#     ) -> List[BaseTool]:

#         tools = []
#         return tools
        
#     async def close(self) -> None:
#         await asyncio.sleep(0)  # Placeholder for async cleanup if needed

#     @toolset
#     async def tavily_toolset(self) -> McpToolset:

#         return McpToolset(
#                 connection_params=StreamableHTTPConnectionParams(
#                     url=f"https://mcp.tavily.com/mcp/?tavilyApiKey={settings.services.tavily_api_key}"
#                 ),
#             )
    
# websearch_toolset_instance = WebSearchToolset()
# websearch_toolset_instance = asyncio.run(websearch_toolset_instance.get_tools(None))