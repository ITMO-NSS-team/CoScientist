"""Toolset module."""
from importlib import import_module
from typing import Any

# Must run before any MCP toolset is used: fail-fast backport for truncated SSE
# frames from remote MCP servers (see mcp_patches docstring).
import CoScientist.tools.mcp_patches  # noqa: F401

_EXPORTS = {
    "FedotMASToolset": ("CoScientist.tools.fedotmas_tools", "FedotMASToolset"),
    "fedot_toolset_instance": ("CoScientist.tools.fedotmas_tools", "fedot_toolset_instance"),
    "websearch_toolset_instance": ("CoScientist.tools.research_tools", "websearch_toolset_instance"),
    "paper_analysis_toolset_instance": ("CoScientist.tools.research_tools", "paper_analysis_toolset_instance"),
    "papers_search_toolset_instance": ("CoScientist.tools.research_tools", "papers_search_toolset_instance"),
    "RetrievalToolSet": ("CoScientist.tools.retrieval_tools", "RetrievalToolSet"),
    "retrieval_toolset_instance": ("CoScientist.tools.retrieval_tools", "retrieval_toolset_instance"),
    "search_mcp_servers": ("CoScientist.tools.servers_web_search", "search_mcp_servers"),
    "med_toolset_instance": ("CoScientist.tools.med_tools", "med_toolset_instance"),
    "CoderToolset": ("CoScientist.tools.coder_tools", "CoderToolset"),
    "coder_toolset_instance": ("CoScientist.tools.coder_tools", "coder_toolset_instance"),
    "TaskTrackerToolset": ("CoScientist.tools.task_tracker", "TaskTrackerToolset"),
    "task_tracker_instance": ("CoScientist.tools.task_tracker", "task_tracker_instance"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
