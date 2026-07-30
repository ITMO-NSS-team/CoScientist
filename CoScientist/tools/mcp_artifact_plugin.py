"""Generic capture of artifact links returned by MCP tools.

Many MCP tools (e.g. the tox-antitargets suite) render a figure/table server-side
and return a *presigned URL* to it inside the tool result — commonly under a
``metadata.figure.artifact`` key or a ``*_presigned_url`` key. That link only
lives in the tool result envelope: with the Result Aggregator running graph-first
(``include_contents: none``) it never reaches the report unless it is captured.

This App-level plugin fires at every tool-call boundary and stashes each artifact
URL it finds into ``state["mcp_artifacts"]`` (a ``*_artifacts`` key the report
collector already downloads). It must run BEFORE the tool-result truncation plugin
so it still sees the full, untruncated URL.

Distinct from :mod:`CoScientist.tools.fedot_artifact_plugin`, which captures S3
CSV links *inside* a nested FEDOT.MAS run; this one is generic and global.
"""
from __future__ import annotations

import logging
from typing import Any

from google.adk.plugins import BasePlugin

from CoScientist.reporting.collect import find_artifact_urls

logger = logging.getLogger(__name__)

_STATE_KEY = "mcp_artifacts"


class McpArtifactCapturePlugin(BasePlugin):
    """Capture artifact (figure/table) URLs from tool results into session state."""

    def __init__(self, name: str = "mcp_artifact_capture") -> None:
        super().__init__(name)

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result):  # noqa: ANN001
        try:
            urls = find_artifact_urls(result)
        except Exception:  # noqa: BLE001 — capture must never break a tool call
            return None
        if not urls:
            return None
        try:
            existing = list(tool_context.state.get(_STATE_KEY) or [])
            seen = {a.get("url") for a in existing if isinstance(a, dict)}
            tool_name = getattr(tool, "name", None)
            for url in urls:
                if url in seen:
                    continue
                seen.add(url)
                existing.append({"url": url, "tool": tool_name})
            tool_context.state[_STATE_KEY] = existing
            logger.info(
                "mcp_artifact_capture: %s → %d artifact URL(s) (%d total)",
                tool_name, len(urls), len(existing),
            )
        except Exception:  # noqa: BLE001
            pass
        return None  # never mutate the tool result itself
