"""In-process ``format_results`` tool for the Result Aggregator agent.

Collects the run's figures and data tables into the per-run report folder and
returns ready-to-embed markdown blocks. Runs in-process (not via the
result-aggregator MCP server) so it can read ADK session state and the sandbox
workspace directly; it needs no running container.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext

from CoScientist.config.report import ReportConfig
from CoScientist.reporting.collect import collect_artifacts

logger = logging.getLogger(__name__)


def _session_id(tool_context: ToolContext) -> str:
    inv = (getattr(tool_context, "_invocation_context", None)
           or getattr(tool_context, "invocation_context", None))
    session = getattr(inv, "session", None) if inv is not None else None
    sid = getattr(session, "id", None) if session else None
    return sid or "default_session"


def _graph_nodes(tool_context: ToolContext) -> list:
    """Research-graph nodes (with full, untruncated attrs) for the current
    session, so artifact URLs an agent committed to Evidence nodes can be
    downloaded. Best-effort — never break report generation if the graph is
    disabled/empty."""
    try:
        from CoScientist.graph.research.agent_tools import get_research_graph
        graph = get_research_graph(tool_context)
        return graph.full().get("nodes", []) or []
    except Exception as exc:  # noqa: BLE001
        logger.info("format_results: no research graph nodes (%s)", exc)
        return []


def _state_to_dict(state: Any) -> Dict[str, Any]:
    """ADK ``tool_context.state`` is a ``State`` wrapper (merged session + delta),
    not a plain dict — ``dict(state)`` mis-reads it as a sequence. Convert safely."""
    if state is None:
        return {}
    if isinstance(state, dict):
        return dict(state)
    to_dict = getattr(state, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:
            pass
    try:
        return {k: state[k] for k in state.keys()}
    except Exception:
        return dict(getattr(state, "_value", {}) or {})


async def format_results(tool_context: ToolContext) -> Dict[str, Any]:
    """Collect every figure and data table this run produced into the report
    folder and return markdown blocks (image embeds + tables) to embed verbatim
    in the final report. Call this FIRST, before writing the report."""
    state = _state_to_dict(getattr(tool_context, "state", {}))
    session_id = _session_id(tool_context)
    cfg = ReportConfig.from_mapping(state.get("report_config"))

    result = collect_artifacts(
        session_id=session_id,
        state=state,
        reports_root=cfg.reports_root,
        graph_nodes=_graph_nodes(tool_context),
    )
    logger.info(
        "format_results: session=%s figures=%d tables=%d",
        session_id, len(result["figures"]), len(result["tables"]),
    )
    return {
        "status": "success",
        "report_dir": result["report_dir"],
        "figures_count": len(result["figures"]),
        "tables_count": len(result["tables"]),
        "formatted_markdown": result["blocks_markdown"],
    }


# Registered under the "result_formatter" tool key (see assembly/bindings.py).
result_formatter_tool = FunctionTool(format_results)

# Back-compat alias: assembly bindings import ``result_formatter_toolset_instance``.
result_formatter_toolset_instance = result_formatter_tool


class ResultFormatterToolset:  # kept for import compatibility; no longer used
    """Deprecated: the formatter is now an in-process FunctionTool."""


__all__ = [
    "format_results",
    "result_formatter_tool",
    "result_formatter_toolset_instance",
    "ResultFormatterToolset",
]
