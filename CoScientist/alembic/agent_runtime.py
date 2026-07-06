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

from alembic.tools.fs import enable_read_dedup

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

# ── F12: failure taxonomy ───────────────────────────────────────────────────
# Coarse, best-effort classification of a tool-failure's error/traceback text
# into a fixed set of buckets a benchmark run can aggregate across repos.
# Ordering doesn't matter: classify_error() picks whichever needle occurs
# *last* in the text, since generated server.py wraps the real exception in
# `raise RuntimeError(f"... {e.stderr}") from e` (see tools/scripts/
# invoke_tool.py callers) — the wrapper's own class name always appears
# before the wrapped traceback's real exception, so "last match wins"
# recovers the actual root cause instead of the wrapper.
_ERROR_TAXONOMY: list[tuple[str, str]] = [
    ("ModuleNotFoundError",  "ModuleNotFound"),
    ("ImportError",          "Import"),
    ("FileNotFoundError",    "FileNotFound"),
    ("AttributeError",       "AttributeError"),
    ("KeyError",             "KeyError"),
    ("IndexError",           "IndexError"),
    ("NameError",            "NameError"),
    ("TypeError",            "TypeError"),
    ("ValueError",           "ValueError"),
    ("UnicodeDecodeError",   "Encoding"),
    ("IndentationError",     "Syntax"),
    ("SyntaxError",          "Syntax"),
    ("TimeoutExpired",       "Timeout"),
    ("timed out",            "Timeout"),
    ("No matching distribution", "Environment"),
    ("Could not find a version", "Environment"),
    ("error: subprocess-exited-with-error", "Environment"),
    ("Failed building wheel", "Environment"),
    ("CalledProcessError",   "Runtime"),
    ("non-zero exit status", "Runtime"),
]


def classify_error(text: str) -> str:
    """Map a raw error/traceback string to a coarse failure-taxonomy bucket."""
    if not text:
        return "Unknown"
    best_label, best_pos = None, -1
    for needle, label in _ERROR_TAXONOMY:
        pos = text.rfind(needle)
        if pos > best_pos:
            best_pos, best_label = pos, label
    return best_label or "Other"


# Tool names whose function_response carries a structured ok/passed verdict
# we can classify on failure.
_VALIDATION_TOOLS = {"invoke_mcp_tool", "validate_syntax", "run_tests"}


def _tool_outcome(name: str, response) -> tuple[bool | None, str]:
    """Best-effort (ok?, error_text) for a validation-tool's response dict."""
    if not isinstance(response, dict):
        return None, ""
    if name == "invoke_mcp_tool":
        return response.get("ok"), f"{response.get('error','')}\n{response.get('traceback','')}"
    return response.get("passed"), str(response.get("error") or response.get("output") or "")


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


MAX_TOOL_REPEATS  = 3    # abort if same tool+args combo is called this many times in a row
MAX_TOOL_CYCLE    = 3    # abort if the same tool+args recurs this many times TOTAL in one
                         # invocation, even non-consecutively — catches set-cycling loops
                         # (re-reading files A,B,C,A,B,C…) that slip past the consecutive
                         # check above. Observed in the rerun9 BioSPPy explorer, which
                         # re-read the same 10 modules 5-7x each up to the MAX_STEPS ceiling.
MAX_STEPS         = 120  # hard ceiling on total events per agent (was 60; complex
                         # debugger fixes routinely need >60 calls)
MAX_GUARD_RETRIES = 3    # re-invoke agent at most this many times when guard fires


async def _run_agent_once(
    agent,
    runner: "Runner",
    session_id: str,
    message: str,
    required_report: str | None,
) -> tuple[str, bool, int, int, bool, dict, dict, str | None]:
    """Run one invocation.

    Returns (final_text, wrote_report, steps, total_tokens, transient_fault,
    tool_calls, failures_by_class, abort_reason).
    ``transient_fault`` is True iff LiteLLM logged an unmapped provider
    finish_reason (see F22) during this invocation — the caller should
    retry rather than trust ``final`` as a genuine completed turn.
    ``tool_calls`` maps tool name -> call count this invocation (F12).
    ``failures_by_class`` maps taxonomy label -> count this invocation (F12).
    ``abort_reason`` is "tool_repeat" / "tool_cycle" / "max_steps" / None (F12).
    """
    _transient_provider_fault.set(False)
    # Audit N1 follow-up: de-dup repeated reads only for the agent observed to
    # loop (the Explorer, cycling read_file over the same modules); every other
    # agent reads normally so it still gets full content (see tools/fs.py).
    enable_read_dedup(agent.name == "explorer")
    content           = types.Content(role="user", parts=[types.Part(text=message)])
    final             = "Agent did not produce a final response."
    wrote_report      = False
    step              = 0
    total_tokens      = 0
    last_call         = None
    tool_repeats      = 0
    tool_calls: dict[str, int]        = {}
    call_key_counts: dict[tuple, int] = {}
    failures_by_class: dict[str, int] = {}

    def _fault():
        return _transient_provider_fault.get()

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
                        fc = part.function_call
                        tool_calls[fc.name] = tool_calls.get(fc.name, 0) + 1
                        call_key = (fc.name, str(fc.args))
                        tool_repeats = tool_repeats + 1 if call_key == last_call else 1
                        last_call    = call_key
                        if tool_repeats >= MAX_TOOL_REPEATS:
                            logger.warning(
                                f"[{agent.name}] ABORT: {fc.name}({_trunc(str(fc.args))}) "
                                f"called {tool_repeats}x with identical args — breaking loop."
                            )
                            return (final, wrote_report, step, total_tokens, _fault(),
                                    tool_calls, failures_by_class, "tool_repeat")
                        call_key_counts[call_key] = call_key_counts.get(call_key, 0) + 1
                        if call_key_counts[call_key] >= MAX_TOOL_CYCLE:
                            logger.warning(
                                f"[{agent.name}] ABORT: {fc.name}({_trunc(str(fc.args))}) "
                                f"called {call_key_counts[call_key]}x total (non-consecutive "
                                f"cycle) — breaking loop."
                            )
                            return (final, wrote_report, step, total_tokens, _fault(),
                                    tool_calls, failures_by_class, "tool_cycle")

                    fr = getattr(part, "function_response", None)
                    if fr and fr.name == "write_report" and required_report:
                        report_path = str((fr.response or {}).get("report_path", ""))
                        if required_report in report_path:
                            wrote_report = True
                    elif fr and fr.name in _VALIDATION_TOOLS:
                        ok, err_text = _tool_outcome(fr.name, fr.response)
                        if ok is False:
                            label = classify_error(err_text)
                            failures_by_class[label] = failures_by_class.get(label, 0) + 1
                    elif fr and fr.name == "debugger":
                        # F16/F17: the debugger's AgentTool wrapper swallows its
                        # own timeout/retry-exhaustion into a plain text result
                        # rather than raising — surface it as a taxonomy bucket
                        # instead of silently losing the signal.
                        result_text = str((fr.response or {}).get("result", ""))
                        if "timed out after" in result_text or "failed twice" in result_text:
                            failures_by_class["DebuggerTimeout"] = (
                                failures_by_class.get("DebuggerTimeout", 0) + 1
                            )

            if step >= MAX_STEPS:
                logger.warning(f"[{agent.name}] ABORT: reached {MAX_STEPS} steps — breaking.")
                return (final, wrote_report, step, total_tokens, _fault(),
                        tool_calls, failures_by_class, "max_steps")

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

    return final, wrote_report, step, total_tokens, _fault(), tool_calls, failures_by_class, None


async def run_agent(
    agent,
    session_service,
    session_id: str,
    message: str,
    required_report: str | None = None,
    venv_guard_path: str | None = None,
) -> tuple[str, int, int, dict]:
    """Run an agent, retrying if guards (write_report / venv) are not satisfied.

    Returns (final_text, total_steps, total_tokens, stage_metrics). ``stage_metrics``
    (F12) is: {"tool_calls": {name: count}, "failures_by_class": {label: count},
    "guard_retries": int, "transient_fault_retries": int, "abort_reason": str | None}.
    ``abort_reason`` is "tool_repeat"/"tool_cycle"/"max_steps" from the *last* attempt (an
    earlier attempt's abort is cleared once a later retry finishes cleanly),
    "guard_exhausted" if the guard-retry budget ran out while nudges were
    still outstanding, or None if the stage genuinely finished clean.
    """
    runner       = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)
    final        = "Agent did not produce a final response."
    total_steps  = 0
    total_tokens = 0
    tool_calls: dict[str, int]        = {}
    failures_by_class: dict[str, int] = {}
    guard_retries            = 0
    transient_fault_retries  = 0
    abort_reason: str | None = None

    current_message = message
    for attempt in range(MAX_GUARD_RETRIES + 1):
        # F22: retry immediately on a detected transient provider fault,
        # using the SAME message (the agent did nothing wrong) and a
        # separate small budget — this must not consume a guard-retry slot,
        # which exists for a different purpose (nudging a missed tool call).
        fault_retries = 0
        while True:
            (final, wrote_report, steps, tokens, transient_fault,
             call_counts, fail_counts, this_abort) = await _run_agent_once(
                agent, runner, session_id, current_message, required_report
            )
            total_steps  += steps
            total_tokens += tokens
            for k, v in call_counts.items():
                tool_calls[k] = tool_calls.get(k, 0) + v
            for k, v in fail_counts.items():
                failures_by_class[k] = failures_by_class.get(k, 0) + v
            # Reflects only the most recent attempt: a later guard-retry
            # that finishes cleanly must clear an earlier attempt's abort,
            # not accumulate it — otherwise a stage that genuinely
            # succeeded after one retry would still misreport e.g.
            # "tool_repeat" from its first, superseded attempt.
            abort_reason = this_abort
            if not transient_fault or fault_retries >= MAX_TRANSIENT_FAULT_RETRIES:
                break
            fault_retries += 1
            transient_fault_retries += 1
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
            # More specific than whatever the last attempt's abort_reason
            # was (which may be None, e.g. it reached a normal final
            # response but simply never called write_report/set up the
            # venv) — the guard budget itself is what's actually exhausted.
            abort_reason = "guard_exhausted"
            break

        guard_retries += 1
        current_message = "IMPORTANT: " + " ".join(nudges)
        logger.warning(f"[guard] Retry {attempt + 1}/{MAX_GUARD_RETRIES}: {current_message[:120]}")

    stage_metrics = {
        "tool_calls":               tool_calls,
        "failures_by_class":        failures_by_class,
        "guard_retries":            guard_retries,
        "transient_fault_retries":  transient_fault_retries,
        "abort_reason":             abort_reason,
    }
    return final, total_steps, total_tokens, stage_metrics
