from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from CoScientist.graph.session_scope import (
    GRAPH_SCOPE_SESSION_KEY,
    GRAPH_SCOPE_USER_KEY,
)
from CoScientist.logging import tool_activity


@pytest.fixture
def sink(monkeypatch):
    received: list[tuple[tuple[str, str], dict]] = []

    async def collect(key, payload):
        received.append((key, payload))

    monkeypatch.setattr(tool_activity, "_sink", collect)
    return received


def agent_tool_context(agent_name: str = "ResearchAgent", call_id: str = "fc_1"):
    """A sub-agent context as AgentTool builds it: child session, parent state."""
    return SimpleNamespace(
        agent_name=agent_name,
        function_call_id=call_id,
        state={
            GRAPH_SCOPE_USER_KEY: "user_1",
            GRAPH_SCOPE_SESSION_KEY: "session_1",
        },
        session=SimpleNamespace(user_id="child", id="child_session_xyz"),
    )


def test_subagent_tool_use_is_reported_under_the_parent_session(sink):
    plugin = tool_activity.ToolActivityPlugin()
    tool = SimpleNamespace(name="tavily_search")
    context = agent_tool_context()

    async def scenario():
        assert await plugin.before_tool_callback(
            tool=tool, tool_args={"query": "ibuprofen"}, tool_context=context
        ) is None
        assert await plugin.after_tool_callback(
            tool=tool, tool_args={}, tool_context=context, result={"answer": "206.29"}
        ) is None

    asyncio.run(scenario())

    # The transient AgentTool child session must not leak: both records belong
    # to the public web session that owns the run.
    assert [key for key, _ in sink] == [("user_1", "session_1")] * 2
    call, result = (payload for _, payload in sink)
    assert call["phase"] == "call"
    assert call["author"] == "ResearchAgent"
    assert call["tool"] == "tavily_search"
    assert call["args"] == {"query": "ibuprofen"}
    assert result["phase"] == "result"
    assert result["result"] == {"answer": "206.29"}
    # Both records carry the call id, so a consumer pairs them exactly instead
    # of matching on tool name (which breaks when a tool is called twice).
    assert call["call_id"] == result["call_id"] == "fc_1"


def test_call_id_is_optional(sink):
    plugin = tool_activity.ToolActivityPlugin()
    context = agent_tool_context()
    del context.function_call_id

    asyncio.run(plugin.before_tool_callback(
        tool=SimpleNamespace(name="tavily_search"), tool_args={}, tool_context=context,
    ))

    (_, payload), = sink
    assert payload["call_id"] is None


def test_tool_error_is_reported(sink):
    plugin = tool_activity.ToolActivityPlugin()

    asyncio.run(plugin.on_tool_error_callback(
        tool=SimpleNamespace(name="tavily_search"),
        tool_args={},
        tool_context=agent_tool_context(),
        error=RuntimeError("rate limited"),
    ))

    (_, payload), = sink
    assert payload["phase"] == "error"
    assert payload["error"] == "rate limited"


def test_oversized_payload_degrades_to_truncated_text(sink):
    plugin = tool_activity.ToolActivityPlugin()

    asyncio.run(plugin.after_tool_callback(
        tool=SimpleNamespace(name="read_file"),
        tool_args={},
        tool_context=agent_tool_context("CoderAgent"),
        result={"blob": "x" * 50_000},
    ))

    (_, payload), = sink
    assert isinstance(payload["result"], str)
    assert len(payload["result"]) <= tool_activity._PREVIEW_LIMIT + 2
    # The full (untruncated) value rides alongside the preview so the web app
    # can stash it for on-demand fetch — it must actually contain everything.
    assert payload["result_truncated"] is True
    # Well under the (much larger) full-value ceiling, so structure survives.
    assert payload["result_full"] == {"blob": "x" * 50_000}


def test_small_payload_carries_no_full_copy(sink):
    """No point doubling memory for a value that was never truncated."""
    plugin = tool_activity.ToolActivityPlugin()

    asyncio.run(plugin.after_tool_callback(
        tool=SimpleNamespace(name="tavily_search"),
        tool_args={},
        tool_context=agent_tool_context(),
        result={"answer": "206.29"},
    ))

    (_, payload), = sink
    assert payload["result_truncated"] is False
    assert "result_full" not in payload


def test_a_failing_sink_cannot_break_the_run(monkeypatch):
    async def broken(key, payload):
        raise RuntimeError("sink down")

    monkeypatch.setattr(tool_activity, "_sink", broken)
    plugin = tool_activity.ToolActivityPlugin()

    assert asyncio.run(plugin.before_tool_callback(
        tool=SimpleNamespace(name="tavily_search"),
        tool_args={},
        tool_context=agent_tool_context(),
    )) is None


def test_plugin_is_inert_without_a_sink(monkeypatch):
    monkeypatch.setattr(tool_activity, "_sink", None)
    plugin = tool_activity.ToolActivityPlugin()

    assert asyncio.run(plugin.before_tool_callback(
        tool=SimpleNamespace(name="tavily_search"),
        tool_args={},
        tool_context=agent_tool_context(),
    )) is None


def test_a_call_record_carries_the_tool_s_own_description(sink):
    """Consumers classify an unknown MCP tool by what it says it does."""
    plugin = tool_activity.ToolActivityPlugin()
    tool = SimpleNamespace(
        name="qsr_lookup_v2",
        description="  Search PubMed\n  for clinical trials matching a query.  ",
    )
    context = agent_tool_context()

    asyncio.run(plugin.before_tool_callback(
        tool=tool, tool_args={"q": "aspirin"}, tool_context=context
    ))

    payload = sink[0][1]
    assert payload["description"] == (
        "Search PubMed for clinical trials matching a query."
    )


def test_a_description_is_capped_and_optional(sink):
    plugin = tool_activity.ToolActivityPlugin()
    context = agent_tool_context()

    asyncio.run(plugin.before_tool_callback(
        tool=SimpleNamespace(name="verbose", description="x " * 500),
        tool_args={}, tool_context=context,
    ))
    asyncio.run(plugin.before_tool_callback(
        tool=SimpleNamespace(name="bare"), tool_args={}, tool_context=context,
    ))

    verbose, bare = (payload for _, payload in sink)
    assert len(verbose["description"]) == tool_activity._DESCRIPTION_LIMIT
    # Nothing to say is not the same as an empty string on the wire.
    assert "description" not in bare


def test_a_result_record_does_not_repeat_the_description(sink):
    """It never changes mid-call, and every frame goes to every open tab."""
    plugin = tool_activity.ToolActivityPlugin()
    tool = SimpleNamespace(name="tavily_search", description="Search the web.")

    asyncio.run(plugin.after_tool_callback(
        tool=tool, tool_args={}, tool_context=agent_tool_context(), result={"ok": 1},
    ))

    assert "description" not in sink[0][1]
