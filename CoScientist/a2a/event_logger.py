"""ADK plugin that logs every agent's inner activity to the server's stdout.

Attached to each A2A server's Runner, so when an agent delegates to a sub-agent
the sub-agent's OWN reasoning and tool use show up on its server's console
(aggregated by run_all). This surfaces what each agent does internally —
thoughts, tool calls, and tool results — not just the final result that comes
back over A2A.

Disable with A2A_LOG_EVENTS=0.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

from google.adk.plugins.base_plugin import BasePlugin

# ANSI colors (no-op if the terminal ignores them)
_DIM = "\033[2m"
_BOLD = "\033[1m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_MAGENTA = "\033[35m"
_RESET = "\033[0m"


def _enabled() -> bool:
    return os.getenv("A2A_LOG_EVENTS", "1") not in ("0", "false", "False")


def _short(value: Any, limit: int = 500) -> str:
    s = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return s if len(s) <= limit else s[:limit] + " …"


def _agent(ctx: Any) -> str:
    return getattr(ctx, "agent_name", None) or "?"


def _emit(line: str) -> None:
    print(line, flush=True)


class EventLoggerPlugin(BasePlugin):
    """Prints agent thoughts, tool calls, and tool results as they happen."""

    def __init__(self, name: str = "event_logger") -> None:
        super().__init__(name=name)

    async def after_model_callback(self, *, callback_context, llm_response) -> Optional[Any]:
        if not _enabled() or llm_response is None or llm_response.content is None:
            return None
        agent = _agent(callback_context)
        for part in (llm_response.content.parts or []):
            text = getattr(part, "text", None)
            fc = getattr(part, "function_call", None)
            if text and getattr(part, "thought", False):
                _emit(f"{_DIM}{_MAGENTA}[{agent}] 💭 {_short(text, 700)}{_RESET}")
            elif fc is not None:
                args = _short(dict(getattr(fc, "args", {}) or {}), 400)
                _emit(f"{_YELLOW}[{agent}] 🔧 {_BOLD}{getattr(fc, 'name', '?')}{_RESET}{_YELLOW}({args}){_RESET}")
            elif text:
                _emit(f"{_GREEN}[{agent}] 🗎 {_short(text, 700)}{_RESET}")
        return None

    async def before_tool_callback(self, *, tool, tool_args, tool_context) -> Optional[dict]:
        if _enabled():
            _emit(f"{_CYAN}[{_agent(tool_context)}] ▶ tool {_BOLD}{tool.name}{_RESET}{_CYAN} {_short(tool_args, 400)}{_RESET}")
        return None

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result) -> Optional[dict]:
        if _enabled():
            _emit(f"{_DIM}[{_agent(tool_context)}] ◀ {tool.name} → {_short(result, 500)}{_RESET}")
        return None

    async def on_tool_error_callback(self, *, tool, tool_args, tool_context, error) -> Optional[dict]:
        if _enabled():
            _emit(f"{_BOLD}[{_agent(tool_context)}] ✖ {tool.name} error: {_short(str(error), 300)}{_RESET}")
        return None
