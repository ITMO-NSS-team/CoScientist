"""Unit tests for the ToolReranker accumulated_tools guard (Bug D, no network).

The ToolReranker instruction interpolates {accumulated_tools}; if the upstream
ToolRetriever didn't populate it, ADK raises "Context variable not found" and
kills the run. before_tool_reranker_agent seeds it to [] so the prompt renders.
"""
from types import SimpleNamespace

from dotenv import load_dotenv

load_dotenv()

from CoScientist.agents.callbacks import before_tool_reranker_agent


def test_seeds_accumulated_tools_when_absent():
    ctx = SimpleNamespace(state={})
    before_tool_reranker_agent(ctx)
    assert ctx.state["accumulated_tools"] == []


def test_does_not_overwrite_existing():
    existing = [{"tool": "search_papers"}]
    ctx = SimpleNamespace(state={"accumulated_tools": existing})
    before_tool_reranker_agent(ctx)
    assert ctx.state["accumulated_tools"] == existing
