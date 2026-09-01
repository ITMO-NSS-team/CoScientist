"""ADK plugin that streams every tool call/result to an observer sink.

Why a plugin and not the event stream: subordinate agents are attached as
``AgentTool`` (see ``assembly/assembler.py``), and an AgentTool delegation runs
its agent in a *nested* Runner with its own child ADK session. Only the
delegation's own function_call/function_response surfaces in the parent
``run_async`` stream — everything the sub-agent does inside (e.g.
``ResearchAgent`` calling ``tavily_search``) never reaches a consumer of the
top-level events. Tool callbacks, on the other hand, fire in nested runners
too, which is exactly why the console trace in ``event_logger`` sees them.

The web UI registers a sink here to show the full picture live. With no sink
registered the plugin is inert, so CLI/A2A runs pay nothing for it.

The sink is invoked as ``await sink(session_key, payload)`` and must never be
able to break a run: every dispatch is guarded.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Awaitable, Callable, Optional

from google.adk.plugins.base_plugin import BasePlugin

from CoScientist.graph.session_scope import SessionKey, session_key

logger = logging.getLogger("CoScientist.logging.tool_activity")

ToolActivitySink = Callable[[SessionKey, dict], Awaitable[None]]

# The live broadcast only ever carries a compact preview of each value — a raw
# result can be megabytes (search dumps, file contents), and every connected
# tab would otherwise pay for that on every call. The full value is rendered
# too (see `_FULL_LIMIT`) but only handed to the web app for on-demand
# storage/retrieval (the ToolsViewer's "Show full result"), never broadcast.
_PREVIEW_LIMIT = 1500
# Ceiling on the *full* value kept for on-demand fetch — generous enough to
# cover virtually any real tool output, but still bounded so one pathological
# call (a multi-MB dump) can't be pulled whole into a session's memory.
_FULL_LIMIT = 2_000_000
# A tool's own one-liner, carried on the `call` record so an observer can say
# what a tool *does* when its name means nothing. MCP servers are built at
# runtime here (see ``McpBuilderAgent``), so their tool names are whatever the
# source repository happened to call them — the description is the only stable
# thing about a tool nobody has ever seen before.
_DESCRIPTION_LIMIT = 200

_sink: Optional[ToolActivitySink] = None


def set_tool_activity_sink(sink: Optional[ToolActivitySink]) -> None:
    """Register (or clear, with ``None``) the observer for tool activity."""
    global _sink
    _sink = sink


def _render(value: Any, limit: int) -> tuple[Any, bool]:
    """Return ``(rendered, truncated)`` — a JSON-safe rendering of ``value``
    capped at ``limit`` characters.

    Structure is preserved while it stays under the cap — observers render a
    dict of arguments far more readably than its escaped JSON text. Anything
    larger degrades to a truncated string.
    """
    if value is None or isinstance(value, (bool, int, float)):
        return value, False
    try:
        text = value if isinstance(value, str) else json.dumps(
            value, ensure_ascii=False, default=str,
        )
    except (TypeError, ValueError):
        text = str(value)
    truncated = len(text) > limit
    if truncated:
        text = text[:limit] + " …"
    if isinstance(value, str) or truncated:
        return text, truncated
    try:
        return json.loads(text), False
    except (TypeError, ValueError):
        return text, False


def _preview_and_full(value: Any) -> tuple[Any, Any, bool]:
    """Return ``(preview, full, truncated)`` for one args/result/error value.

    ``truncated`` reflects only the small preview cap — ``full`` is what the
    ToolsViewer fetches on demand when the user asks to see everything, so a
    caller should skip storing/sending it at all when ``truncated`` is False
    (the preview already *is* the complete value).
    """
    preview, truncated = _render(value, _PREVIEW_LIMIT)
    if not truncated:
        return preview, preview, False
    full, _ = _render(value, _FULL_LIMIT)
    return preview, full, True


def _short_description(tool: Any) -> Optional[str]:
    """The tool's own description, on one line and capped.

    Only the first sentence-ish is useful to a consumer classifying the call,
    and a full MCP description can run to several paragraphs.
    """
    text = getattr(tool, "description", None)
    if not isinstance(text, str):
        return None
    text = " ".join(text.split())
    if not text:
        return None
    return text[:_DESCRIPTION_LIMIT]


def _agent_name(tool_context: Any) -> str:
    return getattr(tool_context, "agent_name", None) or "system"


def _call_id(tool_context: Any) -> Optional[str]:
    """The id ADK assigns to this function call.

    Lets an observer pair a result with its own call instead of guessing by
    tool name — which is wrong as soon as an agent runs the same tool twice.
    """
    call_id = getattr(tool_context, "function_call_id", None)
    return str(call_id) if call_id else None


class ToolActivityPlugin(BasePlugin):
    """Report every tool call, result, and error to the registered sink."""

    def __init__(self, name: str = "tool_activity") -> None:
        super().__init__(name=name)

    async def _dispatch(self, tool_context: Any, payload: dict) -> None:
        sink = _sink
        if sink is None:
            return
        try:
            key = session_key(tool_context)
        except Exception:  # noqa: BLE001 - context shapes vary across ADK paths
            return
        payload.setdefault("timestamp", datetime.now().isoformat())
        try:
            await sink(key, payload)
        except Exception as exc:  # noqa: BLE001 - an observer must not fail a run
            logger.warning("Tool activity sink failed: %s", exc)

    async def before_tool_callback(self, *, tool, tool_args, tool_context) -> None:
        preview, full, truncated = _preview_and_full(tool_args)
        payload = {
            "phase": "call",
            "author": _agent_name(tool_context),
            "tool": getattr(tool, "name", "?"),
            "call_id": _call_id(tool_context),
            "args": preview,
            "args_truncated": truncated,
        }
        description = _short_description(tool)
        if description:
            payload["description"] = description
        if truncated:
            payload["args_full"] = full
        await self._dispatch(tool_context, payload)
        return None  # never override the tool's own execution

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result) -> None:
        preview, full, truncated = _preview_and_full(result)
        payload = {
            "phase": "result",
            "author": _agent_name(tool_context),
            "tool": getattr(tool, "name", "?"),
            "call_id": _call_id(tool_context),
            "result": preview,
            "result_truncated": truncated,
        }
        if truncated:
            payload["result_full"] = full
        await self._dispatch(tool_context, payload)
        return None

    async def on_tool_error_callback(self, *, tool, tool_args, tool_context, error) -> None:
        # ADK re-raises a tool error when no plugin supplies a replacement
        # response, so ``after_tool_callback`` never fires for it: this is the
        # only closing record such a call will ever get.
        preview, full, truncated = _preview_and_full(str(error))
        payload = {
            "phase": "error",
            "author": _agent_name(tool_context),
            "tool": getattr(tool, "name", "?"),
            "call_id": _call_id(tool_context),
            "error": preview,
            "error_truncated": truncated,
        }
        if truncated:
            payload["error_full"] = full
        await self._dispatch(tool_context, payload)
        return None
