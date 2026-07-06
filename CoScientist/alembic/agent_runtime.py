"""Runtime support for driving a single ADK agent turn robustly.

Bundles the cross-cutting concerns that ``main.py``'s pipeline orchestration
shouldn't have to know about:
  - the loguru terminal sink
  - ADK/LiteLLM compatibility patches applied at import time (F19, F22)
  - the guarded single-agent-turn runner (``_run_agent_once`` / ``run_agent``)
    that ``main.py``'s per-stage loop is built on top of
"""
import contextvars
import json
import logging
import sys
from pathlib import Path

from loguru import logger
from google.adk.runners import Runner
from google.genai import types

# ── Loguru: terminal sink ──────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="DEBUG",
    colorize=True,
)
# ──────────────────────────────────────────────────────────────────────────────

# ── Patch ADK tool lookup to return an error response instead of crashing ─────
# When the LLM hallucinates a tool name, ADK raises ValueError and kills the
# entire agent run.  We replace _get_tool with a version that returns a stub
# tool whose run_async echoes the error back to the LLM so it can self-correct.
import google.adk.flows.llm_flows.functions as _adk_fns

_original_get_tool = _adk_fns._get_tool


class _UnknownToolStub:
    """Returned in place of a missing tool; feeds an error back to the LLM."""
    def __init__(self, called_name: str, available: list):
        self.name = called_name
        self.description = f"Unknown tool stub for '{called_name}'."
        # Mirror BaseTool's full public attribute surface (name, description,
        # is_long_running, custom_metadata) since ADK introspects duck-typed
        # tool objects — a partial stub just trades one AttributeError for
        # another the next time a different code path is hit.
        self.is_long_running = False
        self.custom_metadata = None
        self._msg = (
            f"Tool '{called_name}' does not exist. "
            f"You MUST use one of these exact names: "
            f"{', '.join(sorted(available))}. "
            "Retry the call with the correct tool name."
        )

    async def run_async(self, *, args=None, tool_context=None, **_):
        return {"error": self._msg}


def _safe_get_tool(function_call, tools_dict):
    try:
        return _original_get_tool(function_call, tools_dict)
    except (ValueError, KeyError):
        return _UnknownToolStub(function_call.name, list(tools_dict.keys()))


_adk_fns._get_tool = _safe_get_tool
# ──────────────────────────────────────────────────────────────────────────────

# ── F22: detect silent LiteLLM/provider faults ─────────────────────────────────
# OpenRouter (or the upstream backend it proxies to) can surface an internal
# fault as a non-standard finish_reason (observed: "error") instead of a clean
# HTTP error. LiteLLM's own map_finish_reason() has no entry for it, so it
# logs a warning and silently defaults to "stop" — no exception is raised, so
# the agent sees what looks like a normal (if confused/stub) completed turn.
# We can't intercept the raw value inside litellm itself without patching a
# private helper across ~9 different modules that each did `from ... import
# map_finish_reason` (a fragile, version-fragile monkeypatch) — instead we
# hook litellm's own diagnostic logger, which is a stable, public integration
# point, and flag the current asyncio context so the calling stage loop can
# retry the turn immediately instead of burning a guard-retry slot on a bogus
# "final" response.
_transient_provider_fault: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "transient_provider_fault", default=False
)


class _LiteLLMFaultDetector(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        # Use getMessage() rather than record.args/record.msg directly:
        # litellm's own redaction filter interpolates args into record.msg
        # and sets record.args = None *before* other handlers see the
        # record (it runs as an earlier handler on the same logger), so a
        # raw record.args check would silently never match.
        try:
            message = record.getMessage()
        except Exception:
            return
        if "Unmapped finish_reason 'error'" in message:
            _transient_provider_fault.set(True)


logging.getLogger("LiteLLM").addHandler(_LiteLLMFaultDetector())
MAX_TRANSIENT_FAULT_RETRIES = 2  # separate, small budget — not part of MAX_GUARD_RETRIES
# ──────────────────────────────────────────────────────────────────────────────

APP_NAME = "alembic_app"
USER_ID  = "user_1"

TRUNC = 2000  # max chars shown for tool args / responses inline


def _trunc(text: str, n: int = TRUNC) -> str:
    text = str(text).replace("\n", " ")
    return text if len(text) <= n else text[:n] + "…"


def _log_event(agent_name: str, event) -> None:
    """Log a human-readable line for every ADK event."""
    if not event.content or not event.content.parts:
        return

    for part in event.content.parts:
        if part.text:
            snippet = _trunc(part.text.strip())
            if event.is_final_response():
                logger.info(f"[{agent_name}] FINAL: {snippet}")
            else:
                logger.debug(f"[{agent_name}] text:  {snippet}")

        elif hasattr(part, "function_call") and part.function_call:
            fc = part.function_call
            args_str = _trunc(str(fc.args))
            logger.debug(f"[{agent_name}] CALL  {fc.name}({args_str})")

        elif hasattr(part, "function_response") and part.function_response:
            fr = part.function_response
            resp_str = _trunc(str(fr.response))
            logger.debug(f"[{agent_name}] RESP  {fr.name} → {resp_str}")


MAX_TOOL_REPEATS  = 3    # abort if same tool+args combo is called this many times
MAX_STEPS         = 120  # hard ceiling on total events per agent (was 60; complex
                         # debugger fixes routinely need >60 calls)
MAX_GUARD_RETRIES = 3    # re-invoke agent at most this many times when guard fires


async def _run_agent_once(
    agent,
    runner: "Runner",
    session_id: str,
    message: str,
    required_report: str | None,
) -> tuple[str, bool, int, int, bool]:
    """Run one invocation.

    Returns (final_text, wrote_report, steps, total_tokens, transient_fault).
    ``transient_fault`` is True iff LiteLLM logged an unmapped provider
    finish_reason (see F22) during this invocation — the caller should
    retry rather than trust ``final`` as a genuine completed turn.
    """
    _transient_provider_fault.set(False)
    content      = types.Content(role="user", parts=[types.Part(text=message)])
    final        = "Agent did not produce a final response."
    wrote_report = False
    step         = 0
    total_tokens = 0
    last_call    = None
    tool_repeats = 0

    try:
        async for event in runner.run_async(
            user_id=USER_ID, session_id=session_id, new_message=content
        ):
            step += 1
            _log_event(agent.name, event)

            # accumulate token usage when the model reports it
            usage = getattr(event, "usage_metadata", None)
            if usage:
                total_tokens += getattr(usage, "total_token_count", 0) or 0

            if event.content:
                for part in event.content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        fc       = part.function_call
                        call_key = (fc.name, str(fc.args))
                        tool_repeats = tool_repeats + 1 if call_key == last_call else 1
                        last_call    = call_key
                        if tool_repeats >= MAX_TOOL_REPEATS:
                            logger.warning(
                                f"[{agent.name}] ABORT: {fc.name}({_trunc(str(fc.args))}) "
                                f"called {tool_repeats}x with identical args — breaking loop."
                            )
                            return (final, wrote_report, step, total_tokens,
                                    _transient_provider_fault.get())

                    fr = getattr(part, "function_response", None)
                    if fr and fr.name == "write_report" and required_report:
                        report_path = str((fr.response or {}).get("report_path", ""))
                        if required_report in report_path:
                            wrote_report = True

            if step >= MAX_STEPS:
                logger.warning(f"[{agent.name}] ABORT: reached {MAX_STEPS} steps — breaking.")
                return (final, wrote_report, step, total_tokens,
                        _transient_provider_fault.get())

            if event.is_final_response():
                if event.content and event.content.parts:
                    final = event.content.parts[0].text or final
                elif event.actions and event.actions.escalate:
                    final = f"Agent escalated: {event.error_message or 'No message.'}"
                break

    except json.JSONDecodeError as e:
        logger.warning(f"[{agent.name}] invalid JSON in tool call (char {e.pos}): {e.msg} — skipping event.")

    except Exception:
        logger.exception(f"[{agent.name}] ERROR in event loop:")

    return final, wrote_report, step, total_tokens, _transient_provider_fault.get()


async def run_agent(
    agent,
    session_service,
    session_id: str,
    message: str,
    required_report: str | None = None,
    venv_guard_path: str | None = None,
) -> tuple[str, int, int]:
    """Run an agent, retrying if guards (write_report / venv) are not satisfied.

    Returns (final_text, total_steps, total_tokens).
    """
    runner       = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)
    final        = "Agent did not produce a final response."
    total_steps  = 0
    total_tokens = 0

    current_message = message
    for attempt in range(MAX_GUARD_RETRIES + 1):
        # F22: retry immediately on a detected transient provider fault,
        # using the SAME message (the agent did nothing wrong) and a
        # separate small budget — this must not consume a guard-retry slot,
        # which exists for a different purpose (nudging a missed tool call).
        fault_retries = 0
        while True:
            final, wrote_report, steps, tokens, transient_fault = await _run_agent_once(
                agent, runner, session_id, current_message, required_report
            )
            total_steps  += steps
            total_tokens += tokens
            if not transient_fault or fault_retries >= MAX_TRANSIENT_FAULT_RETRIES:
                break
            fault_retries += 1
            logger.warning(
                f"[{agent.name}] transient provider fault (unmapped LLM "
                f"finish_reason) — retrying turn {fault_retries}/"
                f"{MAX_TRANSIENT_FAULT_RETRIES}, not counted against the "
                f"guard-retry budget."
            )

        nudges = []
        if required_report and not wrote_report:
            nudges.append(
                f"You have not called write_report with report_name='{required_report}' yet. "
                f"You MUST call write_report now to save your findings."
            )
        if venv_guard_path and not Path(venv_guard_path).exists():
            nudges.append(
                f"The virtual environment was not created. "
                f"Expected Python binary at: {venv_guard_path}. "
                f"You MUST set up the environment before finishing."
            )

        if not nudges:
            break

        if attempt >= MAX_GUARD_RETRIES:
            logger.warning(f"[guard] Max retries ({MAX_GUARD_RETRIES}) reached — giving up.")
            break

        current_message = "IMPORTANT: " + " ".join(nudges)
        logger.warning(f"[guard] Retry {attempt + 1}/{MAX_GUARD_RETRIES}: {current_message[:120]}")

    return final, total_steps, total_tokens
