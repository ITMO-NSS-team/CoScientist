"""Tools for fedotmas inference"""

import asyncio
import json
import os
import re
from typing import List, Optional, Dict, Any

from google.adk.tools import BaseTool, ToolContext
from google.adk.tools.base_toolset import BaseToolset
from google.adk.agents.readonly_context import ReadonlyContext

from fedotmas import HttpMCPServer
from fedotmas.plugins import LangfusePlugin, LoggingPlugin, WebSearchLimitPlugin

from CoScientist.tools.fedot_artifact_plugin import ArtifactCapturePlugin, merge_artifacts
from CoScientist.tools.fedot_artifact_handoff import (
    bind_upstream_inputs_to_task,
    materialize_tables_from_artifacts,
    record_fedot_producer_tools,
    should_hard_stop_fedot,
    tables_from_state,
)
from CoScientist.tools.fedot_mas_patch import PatchedMAS, ensure_fedot_openai_proxy_compat
from rag_tools import MCPServer
from rag_tools.storage import PostgresClient
from rag_tools.config.settings import get_settings

settings = get_settings()

# Duplicated from tool_callbacks.FEDOT_DELIVERABLE_READY_KEY (avoid import cycle).
_FEDOT_DELIVERABLE_READY_KEY = "fedot_deliverable_ready"

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
        # Hard-stop after deliverable — unless a new consumer tool was retrieved
        # (gen→dock handoff), see fedot_artifact_handoff.should_hard_stop_fedot.
        # Kill-switch: COSCIENTIST_FEDOT_HARD_STOP=0.
        if should_hard_stop_fedot(state):
            arts = list(state.get("fedot_artifacts") or [])
            return {
                "status": "success",
                "artifacts": arts,
                "already_delivered": True,
                "message": (
                    "FEDOT deliverable already captured; refusing a second run. "
                    "Use existing artifacts / URLs for the Final Response."
                ),
            }
        servers = []
        postgres = PostgresClient(settings.postgres)
        try:
            await postgres.initialize()
            try:
                server_ids = set([t['server_id'] for t in filtered_tools])
                servers = [await postgres.get_server(server_id) for server_id in server_ids]
            finally:
                # Always release the DB connection, even if a lookup raised.
                await postgres.close()
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "fedot_tool: database lookup failed (%s); proceeding with empty servers list", exc
            )

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
        if not servers_payload:
            return {
                "status": "error",
                "artifacts": [],
                "error": (
                    "No runnable HTTP MCP server was selected. Retrieve the requested "
                    "tool again instead of starting FEDOT.MAS or escalating to Coder."
                ),
            }

        # Schema-driven handoff: if prior MCP CSVs are in state and the current
        # filtered tools' input_schema names match CSV headers, bind those values
        # into the task so the next step cannot invent placeholders.
        tables = tables_from_state(state)
        if tables and tool_context is not None and not state.get("fedot_artifact_tables"):
            tool_context.state["fedot_artifact_tables"] = tables
        task_description = bind_upstream_inputs_to_task(task_description, tables, filtered_tools)

        # F010.A3/A4: an after_tool_callback plugin captures S3 artifact links
        # (results_presigned_url) at the tool-call boundary, BEFORE FEDOT.MAS sub-agents
        # paraphrase them away / hallucinate molecules.
        # NB: passing plugins= REPLACES MAS defaults, so re-include them.
        cap = ArtifactCapturePlugin()
        # Cap runaway MAS/MCP hangs (e.g. an MCP server stuck retrying a dead
        # connection). Artifacts already captured before the timeout are still
        # returned below — a link produced before a failure/timeout is never lost.
        fedot_timeout_s = float(os.getenv("COSCIENTIST_FEDOT_TIMEOUT_S", "600"))
        web_search_limit = int(os.getenv("COSCIENTIST_FEDOT_WEB_SEARCH_LIMIT", "4"))
        result = None
        status, err = "success", None
        try:
            # openai/<id> for MASConfig validation; PatchedMAS strips it again
            # on the wire to the OpenAI-compatible proxy.
            ensure_fedot_openai_proxy_compat()
            mas = PatchedMAS(
                mcp_servers=servers_payload,
                plugins=[
                    LoggingPlugin(),
                    WebSearchLimitPlugin(max_calls_per_agent=web_search_limit),
                    LangfusePlugin(trace_name="coscientist:fedot"),
                    cap,
                ],
            )
            result = await mas.run(task_description, timeout=fedot_timeout_s)
        except (asyncio.TimeoutError, TimeoutError):
            status, err = "timeout", f"FEDOT.MAS exceeded {fedot_timeout_s}s"
        except Exception as e:
            status, err = "error", f"FEDOT.MAS run failed: {e}"

        # Fallback (F010.A4): scan the final MAS state for presigned URLs the
        # plugin's after_tool hook may have missed (only when a result actually
        # came back). Cheap safety net on top of ArtifactCapturePlugin, not a
        # replacement for it.
        if result is not None:
            try:
                _txt = json.dumps(result, default=str, ensure_ascii=False)
            except Exception:
                _txt = str(result)
            known_urls = {a.get("url") for a in cap.captured}
            scanned = [
                {"url": u, "tool": "fedot_state_scan"}
                for u in dict.fromkeys(re.findall(r"https?://[^\s\"'<>)\\]+X-Amz-[^\s\"'<>)\\]+", _txt))
                if u not in known_urls
            ]
            if scanned:
                cap.captured = merge_artifacts(cap.captured, scanned)

        # Surface the REAL artifacts in the return value AND shared session state — so the
        # link survives even when FEDOT.MAS timed out AFTER generation (F015 Mode B fix).
        # Merge with artifacts from any prior fedot_tool call in this session.
        if cap.captured and tool_context is not None:
            previous = merge_artifacts(list(tool_context.state.get("fedot_artifacts") or []), cap.captured)
            # Materialize CSV tables onto artifacts (durable) + parallel state key.
            # Never let a bad/non-CSV presigned URL fail the whole tool after MAS success.
            try:
                tables = await asyncio.to_thread(
                    materialize_tables_from_artifacts,
                    previous,
                    list(tool_context.state.get("fedot_artifact_tables") or []),
                )
            except Exception:
                tables = list(tool_context.state.get("fedot_artifact_tables") or [])
            tool_context.state["fedot_artifacts"] = previous
            tool_context.state["fedot_artifact_tables"] = tables
            # Even on timeout: if we already have S3 links, treat as delivered.
            tool_context.state[_FEDOT_DELIVERABLE_READY_KEY] = True
            record_fedot_producer_tools(tool_context.state, filtered_tools)

        ret = {"status": status, "artifacts": cap.captured}
        if result is not None:
            ret["result"] = result
        if err:
            ret["error"] = err
        # Success without a captured S3 artifact is a soft signal for retry —
        # never invent molecule payloads here.
        if status == "success" and not cap.captured:
            ret["empty_artifacts"] = True
        return ret

    
fedot_toolset = FedotMASToolset()
fedot_toolset_instance = fedot_toolset.get_tools(None)
