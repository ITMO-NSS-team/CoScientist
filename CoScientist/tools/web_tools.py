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

websearch_toolset_instance = McpToolset(
    connection_params=StreamableHTTPConnectionParams(
        url=f"https://mcp.tavily.com/mcp/?tavilyApiKey={settings.services.tavily_api_key}",
        # Default ADK timeout is 5s — too short for Tavily MCP's list_tools /
        # search cold-start latency ("Failed to get tools from MCP server:
        # TimeoutError"). 90s is ample for the cold start, while bounding how
        # long a genuinely stuck MCP call can hang the whole request (300s
        # caused 5-minute hangs in batch runs).
        timeout=90.0,
        sse_read_timeout=120.0,
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