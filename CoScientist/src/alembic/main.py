#!/usr/bin/env python3
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import asyncio

_api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
if not _api_key:
    sys.exit(
        "[alembic] ERROR: neither OPENROUTER_API_KEY nor OPENAI_API_KEY is set. "
        "Pass it via --env-file or -e in your docker run command."
    )
import json
import shutil
import textwrap

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
) -> tuple[str, bool]:
    """Run one invocation. Returns (final_text, wrote_report)."""
    content      = types.Content(role="user", parts=[types.Part(text=message)])
    final        = "Agent did not produce a final response."
    wrote_report = False
    step         = 0
    last_call    = None
    tool_repeats = 0

    try:
        async for event in runner.run_async(
            user_id=USER_ID, session_id=session_id, new_message=content
        ):
            step += 1
            _log_event(agent.name, event)

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
                            return final, wrote_report

                    fr = getattr(part, "function_response", None)
                    if fr and fr.name == "write_report" and required_report:
                        report_path = str((fr.response or {}).get("report_path", ""))
                        if required_report in report_path:
                            wrote_report = True

            if step >= MAX_STEPS:
                logger.warning(f"[{agent.name}] ABORT: reached {MAX_STEPS} steps — breaking.")
                return final, wrote_report

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

    return final, wrote_report


async def run_agent(
    agent,
    session_service,
    session_id: str,
    message: str,
    required_report: str | None = None,
    venv_guard_path: str | None = None,
) -> str:
    """Run an agent, retrying if guards (write_report / venv) are not satisfied."""
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)
    final  = "Agent did not produce a final response."

    current_message = message
    for attempt in range(MAX_GUARD_RETRIES + 1):
        final, wrote_report = await _run_agent_once(
            agent, runner, session_id, current_message, required_report
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

    return final


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


async def run_pipeline(repo_url: str, resume_from: str | None = None):
    name = get_repo_name(repo_url)
    session_service = InMemorySessionService()

    if resume_from is None:
        _clean_workdir(name)
    else:
        if resume_from not in STAGES:
            logger.error(f"Unknown stage '{resume_from}'. Valid: {', '.join(STAGES)}")
            return
        logger.info(f"[Resume] starting from stage: {resume_from}  (workdir preserved)")

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

    def _should_run(stage: str) -> bool:
        if resume_from is None:
            return True
        return STAGES.index(stage) >= STAGES.index(resume_from)

    async def _run_stage(stage: str, agent, sid_suffix: str, message: str,
                         **kwargs) -> str:
        """Wrap run_agent in a wall-clock timeout. Returns final text or
        an empty string when the stage timed out (pipeline continues)."""
        try:
            return await asyncio.wait_for(
                run_agent(agent, session_service, f"{name}_{sid_suffix}",
                          message, **kwargs),
                timeout=STAGE_TIMEOUT[stage],
            )
        except asyncio.TimeoutError:
            logger.error(
                f"[{stage}] STAGE TIMEOUT after {STAGE_TIMEOUT[stage]}s — "
                f"aborting stage, pipeline continues to next stage."
            )
            return ""

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
        logger.remove(_file_sink_id)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error(f"Usage: ./main.py <repo_url> [--resume <stage>]")
        logger.error(f"       stages: {', '.join(STAGES)}")
        logger.error(f"Example: ./main.py https://github.com/Roestlab/massformer")
        logger.error(f"Example: ./main.py https://github.com/Roestlab/massformer --resume validator")
        sys.exit(1)

    repo_url    = sys.argv[1]
    resume_from = None
    if "--resume" in sys.argv:
        idx = sys.argv.index("--resume")
        if idx + 1 >= len(sys.argv):
            logger.error("--resume requires a stage name")
            sys.exit(1)
        resume_from = sys.argv[idx + 1]

    try:
        asyncio.run(run_pipeline(repo_url, resume_from=resume_from))
    except Exception:
        logger.exception("Pipeline error:")
        raise
