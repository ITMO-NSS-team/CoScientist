"""Tools for fedotmas inference"""

import asyncio
from typing import List, Optional, Dict, Any

from google.adk.tools import BaseTool, ToolContext
from google.adk.tools.base_toolset import BaseToolset
from google.adk.agents.readonly_context import ReadonlyContext

from fedotmas import MAS, HttpMCPServer
from fedotmas.plugins import LoggingPlugin

from CoScientist.tools.fedot_artifact_plugin import ArtifactCapturePlugin
from rag_tools import MCPServer
from rag_tools.storage import PostgresClient
from rag_tools.config.settings import get_settings

settings = get_settings()

class FedotMASToolset(BaseToolset):
    """Toolset for fedotmas usage"""
    def __init__(self, prefix: str = "fedot_"):
        super().__init__()
        self.tool_name_prefix = prefix

    def get_tools(
        self,
        readonly_context: Optional[ReadonlyContext]
    ) -> List[BaseTool]:

        tools = [self.fedot_tool]
        return tools
        
    async def close(self) -> None:
        await asyncio.sleep(0)  # Placeholder for async cleanup if needed

    async def fedot_tool(self, task_description: str,  tool_context: ToolContext = None) -> Dict[str, Any]:
        """
        Tool for generating and executing multi-agent pipelines via FEDOT.MAS. Use it for experiments completion and calculations
        
        Args:
            task_description: Clear description of the task, including goals,
                            inputs, constraints, and expected outputs.
        
        Returns:
            Result of the executed MAS pipeline.
        """
        state = tool_context.state if tool_context is not None else {}

        postgres = PostgresClient(settings.postgres)
        await postgres.initialize()
        try:
            filtered_tools = state.get('filtered_tools', [])
            server_ids = set([t['server_id'] for t in filtered_tools])
            servers: List[MCPServer] = [await postgres.get_server(server_id) for server_id in server_ids]
        finally:
            # Always release the DB connection, even if a lookup raised.
            await postgres.close()

        servers = [server for server in servers if (server is not None and server.protocol == 'http')]
        servers_payload = {server.name: HttpMCPServer(url=server.url, description=server.description)
                           for server in servers}

        # Deployed web MCPs are stored in state as dicts (see WebToolsDeployerAgent).
        web_servers = state.get('deployed_mcps', [])
        web_servers_payload = {
            s['name']: HttpMCPServer(url=s['url'], description=s.get('description', ''))
            for s in web_servers
        }
        servers_payload.update(web_servers_payload)

        # An after_tool_callback plugin captures S3 artifact links
        # (results_presigned_url) at the tool-call boundary, BEFORE the FEDOT.MAS
        # sub-agents paraphrase them away or hallucinate molecules.
        # NB: passing plugins= REPLACES the MAS default (just LoggingPlugin in this
        # fedotmas version), so re-include it alongside the capture plugin.
        cap = ArtifactCapturePlugin()
        result = None
        status, err = "success", None
        try:
            mas = MAS(
                mcp_servers=servers_payload,
                plugins=[LoggingPlugin(), cap],
            )
            result = await mas.run(task_description)
        except Exception as e:
            status, err = "error", f"FEDOT.MAS run failed: {e}"

        # Fallback: scan the returned MAS state for presigned URLs the plugin may
        # have missed (only when a result actually came back).
        if result is not None:
            import re as _re
            import json as _json
            try:
                _txt = _json.dumps(result, default=str, ensure_ascii=False)
            except Exception:
                _txt = str(result)
            _known = {a.get("url") for a in cap.captured}
            for _u in dict.fromkeys(_re.findall(r"https?://[^\s\"'<>)\\]+X-Amz-[^\s\"'<>)\\]+", _txt)):
                if _u not in _known:
                    cap.captured.append({"url": _u, "tool": "fedot_state_scan"})

        # Surface the REAL artifacts in the return value AND shared session state, so
        # the link survives the sub-agents' paraphrasing.
        if cap.captured and tool_context is not None:
            tool_context.state["fedot_artifacts"] = cap.captured

        ret = {"status": status, "artifacts": cap.captured}
        if result is not None:
            ret["result"] = result
        if err:
            ret["error"] = err
        return ret

    
fedot_toolset = FedotMASToolset()
fedot_toolset_instance = fedot_toolset.get_tools(None)