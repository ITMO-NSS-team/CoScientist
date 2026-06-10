"""Tools for websearch / literature research (MCP toolsets)."""
from typing import Optional

from CoScientist.config import get_settings

from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams


settings = get_settings()
PAPER_ANALYSIS_URL = settings.mcp.paper_analysis_url
PAPERS_SEARCH_URL = settings.mcp.papers_search_url


def _http_mcp_toolset(url: Optional[str]) -> Optional[McpToolset]:
    """Build an HTTP MCP toolset, or None when the URL is not configured.

    Returning None (instead of crashing at import on a missing URL) lets the app
    start without these optional services; the ResearchAgent simply runs without
    the corresponding toolset. Set the URLs in .env to enable them.
    """
    if not url:
        return None
    return McpToolset(connection_params=StreamableHTTPConnectionParams(url=url))


# Tavily websearch MCP — DISABLED. The lab VPN breaks the external TLS/SSE stream
# to mcp.tavily.com: the streamable-HTTP session negotiates but `list_tools` never
# returns, so ADK raises ConnectionError and kills the entire ResearchAgent run
# (every benchmark request failed at step 1 because of this). Re-enable once
# Tavily traffic bypasses the VPN (route mcp.tavily.com direct). When None, the
# ResearchAgent relies on the literature MCP tools (paper_analysis / papers_search).
websearch_toolset_instance: Optional[McpToolset] = None
# websearch_toolset_instance = McpToolset(
#     connection_params=StreamableHTTPConnectionParams(
#         url=f"https://mcp.tavily.com/mcp/?tavilyApiKey={settings.services.tavily_api_key}"
#     ),
# )

# Optional paper-analysis / paper-search MCP servers — only built when configured
# (MCP__PAPER_ANALYSIS_URL / MCP__PAPERS_SEARCH_URL in .env).
paper_analysis_toolset_instance = _http_mcp_toolset(PAPER_ANALYSIS_URL)
papers_search_toolset_instance = _http_mcp_toolset(PAPERS_SEARCH_URL)
