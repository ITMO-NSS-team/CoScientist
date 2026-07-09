#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import asyncio
import json
import shutil
import textwrap
import time

from loguru import logger
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

from alembic.agents import explorer_agent, environment_agent, coder_agent, validator_agent
from alembic.tools import WORKDIR, get_repo_name

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

APP_NAME = "alembic_app"
USER_ID  = "user_1"

TRUNC = 2000  # max chars shown for tool args / responses inline


def _trunc(text: str, n: int = TRUNC) -> str:
    text = str(text).replace("\n", " ")
    return text if len(text) <= n else text[:n] + "…"


def _safe(value):
    """Deeply JSON-serializable copy (objects -> str). For on_event payloads."""
    try:
        return json.loads(json.dumps(value, default=str, ensure_ascii=False))
    except (TypeError, ValueError):
        return str(value)


async def _emit(on_event, msg: dict) -> None:
    """Push a UI event to an optional async callback; never crash the run."""
    if on_event is None:
        return
    try:
        await on_event(msg)
    except Exception:
        logger.exception("[on_event] callback raised — ignoring.")


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


# Coarse failure-taxonomy buckets for validation-tool failures, aggregated
# into metrics.json's "failures_by_class" so a benchmark run can report an
# error-distribution table without re-reading every repo's raw logs.
_ERROR_TAXONOMY: list[tuple[str, str]] = [
    ("ModuleNotFoundError",  "ModuleNotFound"),
    ("ImportError",          "ImportError"),
    ("AttributeError",       "AttributeError"),
    ("FileNotFoundError",    "FileNotFound"),
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


MAX_TOOL_REPEATS  = 3    # abort if same tool+args combo is called this many times
MAX_STEPS         = 120  # hard ceiling on total events per agent (was 60; complex
                         # debugger fixes routinely need >60 calls)
MAX_GUARD_RETRIES = 3    # re-invoke agent at most this many times when guard fires

# Per-stage wall-clock budgets (seconds). Caps a hung stage (heavy pip install,
# stuck network) instead of letting a single repo eat 15+ hours of the bench.
STAGE_TIMEOUT = {
    "explorer":    900,    # 15 min — mostly reading + writing the report
    "environment": 2400,   # 40 min — biggest cost: venv + heavy ML deps
    "coder":       1500,   # 25 min — generates server.py + helpers + tests
    "validator":   1800,   # 30 min — syntax + pytest + per-tool invocations
}


async def _run_agent_once(
    agent,
    runner: "Runner",
    session_id: str,
    message: str,
    required_report: str | None,
    on_event=None,
) -> tuple[str, bool, int, int, dict, dict, str | None]:
    """Run one invocation.

    Returns (final_text, wrote_report, steps, total_tokens, tool_calls,
    failures_by_class, abort_reason). ``tool_calls`` maps tool name -> call
    count this invocation. ``failures_by_class`` maps taxonomy label ->
    count this invocation. ``abort_reason`` is "tool_repeat" / "max_steps" /
    None.
    """
    content      = types.Content(role="user", parts=[types.Part(text=message)])
    final        = "Agent did not produce a final response."
    wrote_report = False
    step         = 0
    total_tokens = 0
    last_call    = None
    tool_repeats = 0
    tool_calls: dict[str, int]        = {}
    failures_by_class: dict[str, int] = {}

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
                    if part.text:
                        await _emit(on_event, {
                            "type":  "text",
                            "stage": agent.name,
                            "text":  part.text.strip(),
                            "final": event.is_final_response(),
                        })

                    if hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        await _emit(on_event, {
                            "type":  "tool_call",
                            "stage": agent.name,
                            "name":  fc.name,
                            "args":  _safe(dict(fc.args) if fc.args else {}),
                        })
                        tool_calls[fc.name] = tool_calls.get(fc.name, 0) + 1
                        call_key = (fc.name, str(fc.args))
                        tool_repeats = tool_repeats + 1 if call_key == last_call else 1
                        last_call    = call_key
                        if tool_repeats >= MAX_TOOL_REPEATS:
                            logger.warning(
                                f"[{agent.name}] ABORT: {fc.name}({_trunc(str(fc.args))}) "
                                f"called {tool_repeats}x with identical args — breaking loop."
                            )
                            return (final, wrote_report, step, total_tokens,
                                    tool_calls, failures_by_class, "tool_repeat")

                    fr = getattr(part, "function_response", None)
                    if fr:
                        await _emit(on_event, {
                            "type":     "tool_result",
                            "stage":    agent.name,
                            "name":     fr.name,
                            "response": _safe(fr.response),
                        })
                    if fr and fr.name == "write_report" and required_report:
                        report_path = str((fr.response or {}).get("report_path", ""))
                        if required_report in report_path:
                            wrote_report = True
                    elif fr and fr.name in _VALIDATION_TOOLS:
                        ok, err_text = _tool_outcome(fr.name, fr.response)
                        if ok is False:
                            label = classify_error(err_text)
                            failures_by_class[label] = failures_by_class.get(label, 0) + 1

            if step >= MAX_STEPS:
                logger.warning(f"[{agent.name}] ABORT: reached {MAX_STEPS} steps — breaking.")
                return (final, wrote_report, step, total_tokens,
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

    return final, wrote_report, step, total_tokens, tool_calls, failures_by_class, None


async def run_agent(
    agent,
    session_service,
    session_id: str,
    message: str,
    required_report: str | None = None,
    venv_guard_path: str | None = None,
    on_event=None,
) -> tuple[str, int, int, dict]:
    """Run an agent, retrying if guards (write_report / venv) are not satisfied.

    Returns (final_text, total_steps, total_tokens, stage_metrics). ``stage_metrics``
    is {"tool_calls": {name: count}, "failures_by_class": {label: count},
    "guard_retries": int, "abort_reason": str | None}. ``abort_reason`` reflects the
    last attempt ("tool_repeat"/"max_steps"), "guard_exhausted" if the guard-retry
    budget ran out while nudges were still outstanding, or None if the stage
    finished clean.
    """
    runner       = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)
    final        = "Agent did not produce a final response."
    total_steps  = 0
    total_tokens = 0
    tool_calls: dict[str, int]        = {}
    failures_by_class: dict[str, int] = {}
    guard_retries            = 0
    abort_reason: str | None = None

    current_message = message
    for attempt in range(MAX_GUARD_RETRIES + 1):
        (final, wrote_report, steps, tokens,
         call_counts, fail_counts, abort_reason) = await _run_agent_once(
            agent, runner, session_id, current_message, required_report, on_event
        )
        total_steps  += steps
        total_tokens += tokens
        for k, v in call_counts.items():
            tool_calls[k] = tool_calls.get(k, 0) + v
        for k, v in fail_counts.items():
            failures_by_class[k] = failures_by_class.get(k, 0) + v

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
            abort_reason = "guard_exhausted"
            break

        guard_retries += 1
        current_message = "IMPORTANT: " + " ".join(nudges)
        logger.warning(f"[guard] Retry {attempt + 1}/{MAX_GUARD_RETRIES}: {current_message[:120]}")

    stage_metrics = {
        "tool_calls":        tool_calls,
        "failures_by_class": failures_by_class,
        "guard_retries":     guard_retries,
        "abort_reason":      abort_reason,
    }
    return final, total_steps, total_tokens, stage_metrics


def _banner(stage: int, label: str) -> None:
    sep = "=" * 60
    logger.info(f"\n{sep}\n  STAGE {stage} — {label}\n{sep}")


def _clean_workdir(name: str) -> None:
    """Remove the entire work directory for this repo before a fresh run."""
    repo_dir = WORKDIR / name
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
        logger.debug(f"[clean] removed {repo_dir}")


STAGES = ("explorer", "environment", "coder", "validator")


async def run_pipeline(repo_url: str, resume_from: str | None = None,
                       stop_after: str | None = None, on_event=None):
    name = get_repo_name(repo_url)
    session_service = InMemorySessionService()
    await _emit(on_event, {"type": "pipeline", "status": "start",
                           "repo": name, "repo_url": repo_url})

    if stop_after is not None and stop_after not in STAGES:
        logger.error(f"Unknown --until stage '{stop_after}'. Valid: {', '.join(STAGES)}")
        return

    if resume_from is None:
        _clean_workdir(name)
    else:
        if resume_from not in STAGES:
            logger.error(f"Unknown stage '{resume_from}'. Valid: {', '.join(STAGES)}")
            return
        logger.info(f"[Resume] starting from stage: {resume_from}  (workdir preserved)")

    if (resume_from is not None and stop_after is not None
            and STAGES.index(stop_after) < STAGES.index(resume_from)):
        logger.error(
            f"--until '{stop_after}' is before --resume '{resume_from}' — nothing to run."
        )
        return
    if stop_after is not None:
        logger.info(f"[Until] will stop after completing stage: {stop_after}")

    base = WORKDIR / name
    venv_python = str((base / "output" / ".venv" / "bin" / "python").resolve())

    # ── per-run file sink ──────────────────────────────────────────────────
    log_file = base / "pipeline.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    _file_sink_id = logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG",
        encoding="utf-8",
    )
    logger.info(f"[Run] log → {log_file}")
    # ──────────────────────────────────────────────────────────────────────

    for sid in (f"{name}_explorer", f"{name}_environment",
                f"{name}_coder", f"{name}_validator"):
        await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=sid
        )

    # Structured per-run metrics + failure taxonomy, written to
    # reports/metrics.json in the `finally` block below so run_benchmark.py
    # can aggregate pass-rate-by-stage and error-distribution across a bench.
    pipeline_metrics: dict = {
        "actions_per_stage":       {},
        "tokens_per_stage":        {},
        "durations_per_stage":     {},
        "tool_calls_per_stage":    {},
        "guard_retries_per_stage": {},
        "abort_reason_per_stage":  {},
        "failures_by_class":       {},
        "total_actions":     0,
        "total_tokens":      0,
    }

    def _should_run(stage: str) -> bool:
        idx = STAGES.index(stage)
        if resume_from is not None and idx < STAGES.index(resume_from):
            return False
        if stop_after is not None and idx > STAGES.index(stop_after):
            return False
        return True

    async def _run_stage(stage: str, agent, sid_suffix: str, message: str,
                         **kwargs) -> str:
        """Wrap run_agent in a wall-clock timeout. Returns final text or
        an empty string when the stage timed out (pipeline continues)."""
        started = time.monotonic()
        await _emit(on_event, {"type": "stage", "stage": stage, "status": "running"})
        try:
            final, steps, tokens, stage_metrics = await asyncio.wait_for(
                run_agent(agent, session_service, f"{name}_{sid_suffix}",
                          message, on_event=on_event, **kwargs),
                timeout=STAGE_TIMEOUT[stage],
            )
        except asyncio.TimeoutError:
            logger.error(
                f"[{stage}] STAGE TIMEOUT after {STAGE_TIMEOUT[stage]}s — "
                f"aborting stage, pipeline continues to next stage."
            )
            pipeline_metrics["durations_per_stage"][stage]    = round(time.monotonic() - started, 1)
            pipeline_metrics["abort_reason_per_stage"][stage] = "stage_timeout"
            await _emit(on_event, {"type": "stage", "stage": stage, "status": "timeout"})
            return ""
        pipeline_metrics["durations_per_stage"][stage] = round(time.monotonic() - started, 1)
        pipeline_metrics["actions_per_stage"][stage] = steps
        pipeline_metrics["tokens_per_stage"][stage]  = tokens
        pipeline_metrics["total_actions"]           += steps
        pipeline_metrics["total_tokens"]            += tokens
        pipeline_metrics["tool_calls_per_stage"][stage]    = stage_metrics["tool_calls"]
        pipeline_metrics["guard_retries_per_stage"][stage] = stage_metrics["guard_retries"]
        if stage_metrics["abort_reason"]:
            pipeline_metrics["abort_reason_per_stage"][stage] = stage_metrics["abort_reason"]
        for label, count in stage_metrics["failures_by_class"].items():
            pipeline_metrics["failures_by_class"][label] = (
                pipeline_metrics["failures_by_class"].get(label, 0) + count
            )
        await _emit(on_event, {"type": "stage", "stage": stage, "status": "done",
                               "steps": steps, "tokens": tokens})
        return final

    def _coder_artefacts_present() -> tuple[bool, list[str]]:
        required = [
            base / "output" / "server.py",
            base / "output" / "tests" / "test_server.py",
            base / "reports" / "server.md",
        ]
        missing = [str(p.relative_to(base)) for p in required if not p.exists()]
        return (not missing), missing

    try:
        # ── Stage 1: Explorer ──────────────────────────────────────────────
        if _should_run("explorer"):
            _banner(1, f"Explorer  ({repo_url})")
            await _run_stage("explorer", explorer_agent, "explorer", repo_url,
                             required_report="exploration")
            logger.info(f"[Explorer done] report → {base}/reports/exploration.md")

        # ── Stage 2: Environment ───────────────────────────────────────────
        if _should_run("environment"):
            _banner(2, f"Environment ({repo_url})")
            await _run_stage(
                "environment", environment_agent, "environment", repo_url,
                required_report="environment", venv_guard_path=venv_python,
            )
            logger.info(f"[Environment done] report → {base}/reports/environment.md")

        # ── Stage 3: Coder ─────────────────────────────────────────────────
        if _should_run("coder"):
            _banner(3, f"Coder  ({repo_url})")
            await _run_stage(
                "coder", coder_agent, "coder", repo_url,
                required_report="server",
            )
            logger.info(f"[Coder done] server → {base}/output/server.py")
            logger.info(f"             tests  → {base}/output/tests/test_server.py")
            logger.info(f"             report → {base}/reports/server.md")

        # ── Stage 4: Validator (calls Debugger internally on failures) ─────
        if _should_run("validator"):
            ok, missing = _coder_artefacts_present()
            if not ok:
                logger.error(
                    f"[validator] required artefacts missing: {missing} — "
                    f"skipping validator stage (nothing to validate)."
                )
            else:
                _banner(4, f"Validator  ({repo_url})")
                validator_response = await _run_stage(
                    "validator", validator_agent, "validator", repo_url,
                    required_report="validation",
                )
                logger.info(f"[Validator done] report → {base}/reports/validation.md")

                sep = "=" * 60
                logger.success(
                    f"\n{sep}\n  Pipeline complete: {name}\n"
                    f"  Reports : {base}/reports/\n"
                    f"  Output  : {base}/output/\n"
                    f"  Log     : {log_file}\n{sep}\n\n"
                    f"--- Validator summary ---\n\n"
                    + textwrap.indent((validator_response or "").strip(), "  ")
                )

    finally:
        import sys as _sys
        import traceback as _tb
        exc_type, exc_val, exc_tb = _sys.exc_info()

        reports_dir = base / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        reports_dir.joinpath("metrics.json").write_text(
            json.dumps(pipeline_metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        if exc_val is not None:
            error_payload = {
                "exception": type(exc_val).__name__,
                "message":   str(exc_val),
                "traceback": "".join(_tb.format_exception(exc_type, exc_val, exc_tb)),
            }
            reports_dir.joinpath("error.json").write_text(
                json.dumps(error_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.error(f"[pipeline] error saved → {reports_dir}/error.json")
            await _emit(on_event, {"type": "pipeline", "status": "error",
                                   "repo": name, "message": str(exc_val)})
        else:
            await _emit(on_event, {"type": "pipeline", "status": "complete",
                                   "repo": name, "metrics": pipeline_metrics})

        logger.remove(_file_sink_id)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error(f"Usage: ./main.py <repo_url> [--resume <stage>] [--until <stage>]")
        logger.error(f"       stages: {', '.join(STAGES)}")
        logger.error(f"Example: ./main.py https://github.com/Roestlab/massformer")
        logger.error(f"Example: ./main.py https://github.com/Roestlab/massformer --resume validator")
        logger.error(f"Example: ./main.py https://github.com/Roestlab/massformer --until explorer")
        sys.exit(1)

    repo_url    = sys.argv[1]
    resume_from = None
    if "--resume" in sys.argv:
        idx = sys.argv.index("--resume")
        if idx + 1 >= len(sys.argv):
            logger.error("--resume requires a stage name")
            sys.exit(1)
        resume_from = sys.argv[idx + 1]

    stop_after = None
    if "--until" in sys.argv:
        idx = sys.argv.index("--until")
        if idx + 1 >= len(sys.argv):
            logger.error("--until requires a stage name")
            sys.exit(1)
        stop_after = sys.argv[idx + 1]

    try:
        asyncio.run(run_pipeline(repo_url, resume_from=resume_from, stop_after=stop_after))
    except Exception:
        logger.exception("Pipeline error:")
        raise
