"""AlembicAgent toolset — turn a scientific GitHub repo into a running MCP server.

The single capability the alembic pipeline provides (clone → build env →
generate + validate a FastMCP server → ship it as a Docker image) is exposed
here as ADK tools so the CoScientist orchestrator can delegate "wrap repo X as an
MCP server" to the :class:`AlembicAgent`.

The heavy lifting is delegated to :mod:`CoScientist.alembic.builder`, which only
ever shells out to the Docker CLI — the build runs inside an ephemeral container
(the security boundary), never in this process. Because a build can take many
minutes, the blocking Docker work runs in a worker thread so the A2A event loop
stays responsive.

The local alembic package is imported lazily (and as ``CoScientist.alembic.*``)
to avoid the import-time cost and to keep the unrelated PyPI ``alembic`` package
out of the way.
"""
from __future__ import annotations

import asyncio
import shutil
from typing import Any, Dict, List, Optional

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools import BaseTool, ToolContext
from google.adk.tools.base_toolset import BaseToolset

# Session-state key under which served containers are tracked, so a later
# stop_mcp_server call (a separate A2A turn) can find them.
_SERVERS_STATE_KEY = "alembic_servers"

# Keep tool responses readable: reports can be long.
_REPORT_TRUNC = 4000


def _trunc(text: str, n: int = _REPORT_TRUNC) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[:n] + "\n…(truncated)"


def _extract_tool_names(server_report: str) -> List[str]:
    """Best-effort list of the tool names the generated server exposes.

    The coder's ``server.md`` carries a ``samples:`` block describing how to
    invoke each ``@mcp.tool()``. The exact layout is not guaranteed, so this is
    tolerant: try YAML, then fall back to a light scan. Never raises.
    """
    names: List[str] = []
    if not server_report:
        return names
    try:
        import re

        import yaml

        # Pull any fenced ```yaml ... ``` blocks plus the raw text as candidates.
        candidates = re.findall(r"```(?:ya?ml)?\s*(.*?)```", server_report, re.DOTALL)
        candidates.append(server_report)
        for blob in candidates:
            try:
                data = yaml.safe_load(blob)
            except Exception:
                continue
            samples = data.get("samples") if isinstance(data, dict) else None
            if isinstance(samples, dict):
                names.extend(str(k) for k in samples)
            elif isinstance(samples, list):
                for item in samples:
                    if isinstance(item, dict):
                        tool = item.get("tool") or item.get("name")
                        if tool:
                            names.append(str(tool))
            if names:
                break
        if not names:
            # Last resort: explicit `tool: <name>` lines anywhere in the report.
            names = re.findall(r"(?im)^\s*-?\s*tool:\s*([A-Za-z_]\w*)", server_report)
    except Exception:
        return []
    # De-duplicate, preserve order.
    return list(dict.fromkeys(names))


class AlembicToolset(BaseToolset):
    """Tools that build and serve an MCP server from a GitHub repository."""

    def __init__(self, prefix: str = "alembic_"):
        super().__init__()
        self.tool_name_prefix = prefix

    def get_tools(
        self, readonly_context: Optional[ReadonlyContext] = None
    ) -> List[BaseTool]:
        return [self.build_mcp_server, self.stop_mcp_server]

    async def close(self) -> None:
        await asyncio.sleep(0)  # nothing persistent to release

    @staticmethod
    def _docker_available() -> bool:
        return shutil.which("docker") is not None

    async def build_mcp_server(
        self,
        repo_url: str,
        serve: bool = True,
        tool_context: ToolContext = None,
    ) -> Dict[str, Any]:
        """Turn a scientific GitHub repository into a deployable MCP server.

        Runs the full alembic pipeline inside Docker: clone the repo, build its
        environment, generate a FastMCP server (with pytest tests) exposing the
        repo's main functionality as tools, validate every tool end-to-end, and
        commit the result to an image. When ``serve`` is true, also launch the
        image so the MCP server is reachable over HTTP.

        This is expensive and long-running (often several minutes to tens of
        minutes for heavy ML repos). Call it once per repository.

        Args:
            repo_url: GitHub URL of the repository to wrap (e.g.
                ``https://github.com/Roestlab/massformer``).
            serve: If true (default), start the built image and return the live
                MCP server URL. If false, only build and commit the image.

        Returns:
            On success: ``{"status": "success", "image", "tools", and (when
            served) "url"/"container", plus "validation_summary"}``.
            On failure: ``{"status": "error", "error": <message>}``.
        """
        if not self._docker_available():
            return {
                "status": "error",
                "error": "Docker CLI not found on PATH. The alembic build runs "
                         "inside Docker; this agent needs access to a Docker daemon.",
            }

        from CoScientist.alembic.builder import AlembicBuildError, build_and_serve

        try:
            result = await asyncio.to_thread(
                build_and_serve, repo_url, serve=serve
            )
        except AlembicBuildError as exc:
            return {"status": "error", "error": str(exc)}
        except Exception as exc:  # docker missing daemon, etc.
            return {"status": "error", "error": f"alembic build failed: {exc}"}

        build = result.build
        tools = _extract_tool_names(build.server_report)
        out: Dict[str, Any] = {
            "status": "success",
            "repo": build.repo,
            "image": build.image,
            "tools": tools,
            "validation_summary": _trunc(build.validation_report),
        }
        if result.serve is not None:
            out["url"] = result.serve.url
            out["container"] = result.serve.container
            out["port"] = result.serve.port
            # Track the live container so stop_mcp_server can find it later.
            if tool_context is not None:
                servers = list(tool_context.state.get(_SERVERS_STATE_KEY, []))
                servers.append(
                    {"container": result.serve.container,
                     "url": result.serve.url,
                     "repo": build.repo}
                )
                tool_context.state[_SERVERS_STATE_KEY] = servers
        return out

    async def stop_mcp_server(
        self,
        container: str,
        tool_context: ToolContext = None,
    ) -> Dict[str, Any]:
        """Stop and remove a running MCP-server container started by build_mcp_server.

        Args:
            container: The container name returned by build_mcp_server (the
                ``container`` field).

        Returns:
            ``{"status": "stopped", "container": <name>}`` (best-effort; stopping
            an unknown container is a no-op).
        """
        if not self._docker_available():
            return {"status": "error", "error": "Docker CLI not found on PATH."}

        from CoScientist.alembic.builder import stop_container

        await asyncio.to_thread(stop_container, container)
        if tool_context is not None:
            servers = [
                s for s in tool_context.state.get(_SERVERS_STATE_KEY, [])
                if s.get("container") != container
            ]
            tool_context.state[_SERVERS_STATE_KEY] = servers
        return {"status": "stopped", "container": container}


alembic_toolset = AlembicToolset()
# A list of bound tool callables (mirrors fedot_toolset_instance) so the assembly
# layer can attach them directly and validate prompt/tool consistency by name.
alembic_toolset_instance = alembic_toolset.get_tools(None)
