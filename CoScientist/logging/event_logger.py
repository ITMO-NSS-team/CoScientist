"""ADK plugin that logs every agent's inner activity to stdout.

Attached in two places:
- each A2A server's Runner (``a2a/server.py``) — a sub-agent's reasoning and
  tool use show up on its own server's console (aggregated by run_all);
- the ``App`` exported from ``agent.py`` — so plain ``adk web`` /
  ``adk api_server`` runs get the same console trace without A2A.

This surfaces what each agent does internally — thoughts, tool calls, and tool
results — not just the final answer.

The same trace is also appended to a log file (ANSI stripped) — handy when the
terminal scrollback can't hold a long run. Path: AGENT_LOG_FILE (default
``/app/agent_events.log``); set it to "" to keep only the console output.

Disable everything with LOG_AGENT_EVENTS=0 (the older A2A_LOG_EVENTS=0 still
works).
"""
from __future__ import annotations

import json
import os
import re
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

# File copy of the console trace (ANSI stripped). Best-effort: an unwritable
# path disables the file sink rather than breaking the run. Under run_all every
# A2A server appends to the same file — O_APPEND keeps each capped line intact,
# so they aggregate just like the shared console.
_LOG_FILE = os.getenv("AGENT_LOG_FILE", "/app/agent_events.log")
_ANSI_RE = re.compile(r"\033\[[0-9;]*m")
_log_fh = None        # open handle, lazily created
_log_disabled = False  # set once if opening failed, so we stop retrying


def _get_log_fh():
    global _log_fh, _log_disabled
    if _log_disabled or not _LOG_FILE:
        return None
    if _log_fh is None:
        try:
            _log_fh = open(_LOG_FILE, "a", buffering=1)  # line-buffered
        except OSError:
            _log_disabled = True
            return None
    return _log_fh


def _enabled() -> bool:
    value = os.getenv("LOG_AGENT_EVENTS") or os.getenv("A2A_LOG_EVENTS") or "1"
    return value not in ("0", "false", "False")


def _short(value: Any, limit: int = 500) -> str:
    s = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return s if len(s) <= limit else s[:limit] + " …"


def _agent(ctx: Any) -> str:
    return getattr(ctx, "agent_name", None) or "?"


def _emit(line: str) -> None:
    print(line, flush=True)
    fh = _get_log_fh()
    if fh is not None:
        try:
            fh.write(_ANSI_RE.sub("", line) + "\n")  # line-buffered, flushes on \n
        except OSError:
            pass

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
