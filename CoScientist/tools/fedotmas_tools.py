"""Tools for fedotmas inference"""

import asyncio
from typing import List, Optional, Dict, Any

from google.adk.tools import BaseTool, ToolContext
from google.adk.tools.base_toolset import BaseToolset
from google.adk.agents.readonly_context import ReadonlyContext

from fedotmas import MAS, HttpMCPServer

from CoScientist.tools.local_mcp_registry import local_server

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

        filtered_tools = state.get('filtered_tools', [])
        server_ids = {tool['server_id'] for tool in filtered_tools}
        servers = [local_server(server_id) for server_id in server_ids]

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

        try:
            mas = MAS(mcp_servers=servers_payload)
            result = await mas.run(task_description)
        except Exception as e:
            return {"status": "error", "error": f"FEDOT.MAS run failed: {e}"}

        return {
            "status": "success",
            "result": result,
        }

    
fedot_toolset = FedotMASToolset()
fedot_toolset_instance = fedot_toolset.get_tools(None)
