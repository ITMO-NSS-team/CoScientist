from CoScientist.tools_web_search.engine import MCPSearchTool
from CoScientist.tools_web_search.models import MCPSearchResult

from google.adk.tools import ToolContext


_tool = MCPSearchTool()  # share across calls


async def search_mcp_servers(query: str,
                            tool_context: ToolContext = None) -> str:
    """
    Search public MCP server registries for servers matching the query.

    Use this tool to discover MCP servers for APIs, integrations, tools,
    databases, browser automation, research workflows, and other capabilities.

    Args:
        query: Natural language query describing the desired MCP server
            functionality or integration.
            Examples:
            - "github"
            - "youtube transcription"
            - "postgres database"
            - "browser automation"

    Returns:
        Compact LLM-friendly text containing up to 15 matching MCP servers,
        including descriptions, metadata, registry pages, and repository links.
    """
    async with _tool:
        result: MCPSearchResult = await _tool.search(query)

    tool_context.state['found_web_mcp_tools'] = result
    return result.to_agent_text(limit=15)