"""Unit tests for ResilientAgentTool (no network / VPN required).

ResilientAgentTool wraps a sub-agent so that a blank result from the stock
AgentTool (which can happen when a thinking model ends a long tool loop on a
thought-only / tool event) falls back to the agent's ``output_key`` state,
where ADK has already saved the real answer. These tests pin that behaviour.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from dotenv import load_dotenv

load_dotenv()

from google.adk.tools.agent_tool import AgentTool
from CoScientist.agents.agents import ResilientAgentTool


def _make_tool(output_key="search_results"):
    # Bypass AgentTool.__init__ (needs a full LlmAgent); we only exercise run_async.
    tool = ResilientAgentTool.__new__(ResilientAgentTool)
    tool.agent = SimpleNamespace(name="ResearchAgent", output_key=output_key)
    return tool


def _run(tool, state):
    ctx = SimpleNamespace(state=state)
    return asyncio.run(tool.run_async(args={"request": "q"}, tool_context=ctx))


def test_falls_back_to_output_key_when_result_blank():
    tool = _make_tool()
    with patch.object(AgentTool, "run_async", new=AsyncMock(return_value="")):
        assert _run(tool, {"search_results": "REAL ANSWER"}) == "REAL ANSWER"


def test_whitespace_only_result_triggers_fallback():
    tool = _make_tool()
    with patch.object(AgentTool, "run_async", new=AsyncMock(return_value="  \n ")):
        assert _run(tool, {"search_results": "REAL ANSWER"}) == "REAL ANSWER"


def test_nonblank_result_passes_through_unchanged():
    tool = _make_tool()
    with patch.object(AgentTool, "run_async", new=AsyncMock(return_value="direct answer")):
        # fallback present but must NOT be used when the direct result is good
        assert _run(tool, {"search_results": "fallback"}) == "direct answer"


def test_blank_result_and_empty_state_returns_blank():
    tool = _make_tool()
    with patch.object(AgentTool, "run_async", new=AsyncMock(return_value="")):
        assert _run(tool, {}) == ""


def test_no_output_key_returns_blank():
    tool = _make_tool(output_key=None)
    with patch.object(AgentTool, "run_async", new=AsyncMock(return_value="")):
        assert _run(tool, {"search_results": "ignored"}) == ""
