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

from alembic.agents import (
    explorer_agent, environment_agent, coder_agent, validator_agent,
    reporter_agent, set_current_repo_url,
)
from alembic.tools import WORKDIR, get_repo_name
from alembic.tools.fs import parse_samples_block
from alembic.tools.invoke import set_skip_tools
from alembic.agent_runtime import APP_NAME, USER_ID, run_agent

# Per-stage wall-clock budgets (seconds). Caps a hung stage (heavy pip install,
# stuck network) instead of letting a single repo eat 15+ hours of the bench.
STAGE_TIMEOUT = {
    "explorer":    900,    # 15 min — mostly reading + writing the report
    "environment": 2400,   # 40 min — biggest cost: venv + heavy ML deps
    "coder":       1500,   # 25 min — generates server.py + helpers + tests
    "validator":   1800,   # 30 min — syntax + pytest + per-tool invocations
}
INVOKE_TIMEOUT              = 120   # invoke_mcp_tool() — F25/F37: resource-heavy calls
                                     # are SKIPPED past this point, not FAILED
DEBUGGER_CALL_TIMEOUT       = 600   # F16: bounds one debugger round-trip

# F32: reserve the last 15% of each stage's budget as a grace window. Once
# elapsed time crosses this fraction, run_agent breaks the agent's current
# attempt early and spends one urgent, deadline-free turn forcing write_report
# — so a stage that's about to blow its hard STAGE_TIMEOUT still saves
# whatever partial findings it has, instead of the wait_for below cancelling
# everything with no report written at all (observed repeatedly: a stage
# grinding through many genuinely-productive-but-slow steps never trips the
# tool_repeat/tool_cycle/max_steps guards, so nothing broke the loop early
# until this).
REPORT_GRACE_FRACTION = 0.85

# F35: hard cap on the last-resort reporter's own call, so the guarantee
# ("the validator stage always ends with a validation.md, one way or
# another") can't itself hang forever. The reporter's toolset is bounded to
# well under a minute of real tool time (validate_syntax/run_tests), so this
# is generous defense-in-depth, not an expected-to-fire limit.
REPORTER_TIMEOUT = 300


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
                       stop_after: str | None = None):
    name = get_repo_name(repo_url)
    set_current_repo_url(repo_url)  # F15: debugger AgentTool stamps this on every call
    session_service = InMemorySessionService()

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
                f"{name}_coder", f"{name}_validator", f"{name}_reporter"):
        await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=sid
        )

    # F12: structured per-run metrics + failure taxonomy, written to
    # reports/metrics.json in the `finally` block below so run_benchmark.py
    # can aggregate pass-rate-by-stage and error-distribution across a bench.
    pipeline_metrics: dict = {
        "actions_per_stage":                {},
        "tokens_per_stage":                 {},
        "durations_per_stage":              {},
        "tool_calls_per_stage":             {},
        "guard_retries_per_stage":          {},
        "transient_fault_retries_per_stage": {},
        "abort_reason_per_stage":           {},
        "failures_by_class":                {},
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
        soft_deadline = started + STAGE_TIMEOUT[stage] * REPORT_GRACE_FRACTION
        try:
            final, steps, tokens, stage_metrics = await asyncio.wait_for(
                run_agent(agent, session_service, f"{name}_{sid_suffix}",
                          message, deadline=soft_deadline, **kwargs),
                timeout=STAGE_TIMEOUT[stage],
            )
        except asyncio.TimeoutError:
            logger.error(
                f"[{stage}] STAGE TIMEOUT after {STAGE_TIMEOUT[stage]}s — "
                f"aborting stage, pipeline continues to next stage."
            )
            pipeline_metrics["durations_per_stage"][stage]   = round(time.monotonic() - started, 1)
            pipeline_metrics["abort_reason_per_stage"][stage] = "stage_timeout"
            return ""
        pipeline_metrics["durations_per_stage"][stage] = round(time.monotonic() - started, 1)
        pipeline_metrics["actions_per_stage"][stage] = steps
        pipeline_metrics["tokens_per_stage"][stage]  = tokens
        pipeline_metrics["total_actions"]           += steps
        pipeline_metrics["total_tokens"]            += tokens
        pipeline_metrics["tool_calls_per_stage"][stage]              = stage_metrics["tool_calls"]
        pipeline_metrics["guard_retries_per_stage"][stage]           = stage_metrics["guard_retries"]
        pipeline_metrics["transient_fault_retries_per_stage"][stage] = stage_metrics["transient_fault_retries"]
        if stage_metrics["abort_reason"]:
            pipeline_metrics["abort_reason_per_stage"][stage] = stage_metrics["abort_reason"]
        for label, count in stage_metrics["failures_by_class"].items():
            pipeline_metrics["failures_by_class"][label] = (
                pipeline_metrics["failures_by_class"].get(label, 0) + count
            )
        return final

    def _coder_artefacts_present() -> tuple[bool, list[str]]:
        required = [
            base / "output" / "server.py",
            base / "output" / "tests" / "test_server.py",
            base / "reports" / "server.md",
        ]
        missing = [str(p.relative_to(base)) for p in required if not p.exists()]
        return (not missing), missing

    def _build_validator_message() -> str:
        """F25: compute the SKIP/invoke split from server.md's samples: block
        in code, register it with invoke_mcp_tool's code-enforced gate
        (set_skip_tools), and hand the validator the exact list up front —
        instead of trusting it to correctly re-derive the same split itself
        from free-text YAML on every run (the AgML failure this closes: a
        tool the Coder itself marked SKIP got invoked anyway with its
        expensive default params, burning the stage budget).
        """
        samples = parse_samples_block(repo_url)
        if not samples:
            set_skip_tools([])
            return repo_url  # unparseable/missing — validator falls back to Step 1's own read
        skip_tools = sorted(
            name for name, v in samples.items()
            if isinstance(v, str) and v.strip().upper() == "SKIP"
        )
        invoke_tools = sorted(name for name in samples if name not in skip_tools)
        set_skip_tools(skip_tools)
        return (
            f"{repo_url}\n\n"
            f"Computed from server.md's samples: block — trust this, it is "
            f"authoritative, not a suggestion to re-derive yourself:\n"
            f"  Invoke in Step 4 ({len(invoke_tools)}): "
            f"{', '.join(invoke_tools) if invoke_tools else '(none)'}\n"
            f"  SKIP — do NOT invoke ({len(skip_tools)}): "
            f"{', '.join(skip_tools) if skip_tools else '(none)'}\n"
            f"This is also enforced in code: calling invoke_mcp_tool on a "
            f"SKIP-listed tool name returns {{\"skipped\": true, \"reason\": "
            f"...}} instead of running it — treat that response as SKIPPED, "
            f"never as a failure to hand to the debugger."
        )

    async def _ensure_validation_report(progress: dict) -> None:
        """F35 last-resort guarantee: if the validator stage ended — via
        hard STAGE_TIMEOUT or by exhausting its guard-retry budget — without
        ever writing validation.md, invoke the separate reporter_agent to
        write SOMETHING rather than leaving zero signal. Uses a FRESH
        session (not the validator's own, which may have been cancelled
        mid-tool-call — replaying that risks the same "Missing tool results
        for tool_call_id(s)" session corruption seen elsewhere this session)
        and a toolset that structurally cannot repeat the validator's own
        failure mode (no debugger, no invoke_mcp_tool — see agents.py).
        ``progress`` is the same dict passed as run_agent's ``progress=``
        for the validator call: mutated in place as events streamed, so it
        still holds a compact summary of how far the run got even if the
        validator's own call was cancelled mid-flight.
        """
        validation_path = base / "reports" / "validation.md"
        if validation_path.exists():
            return
        logger.warning(
            f"[validator] no validation.md after the stage ended — "
            f"invoking the fallback reporter to guarantee one gets written."
        )
        summary_bits = []
        if progress.get("tool_calls"):
            summary_bits.append(f"Tool calls made before running out of time: {progress['tool_calls']}")
        if progress.get("last_debugger_request"):
            summary_bits.append(f"Last debugger request in flight: {progress['last_debugger_request']}")
        if progress.get("last_failure"):
            summary_bits.append(f"Last known validation failure observed: {progress['last_failure']}")
        summary = "\n".join(summary_bits) or "No further detail is available."
        reporter_message = (
            f"{repo_url}\n\nThe validator stage ended without writing a "
            f"validation report (it ran out of its {STAGE_TIMEOUT['validator']}s "
            f"budget). You do not have access to that debugging session — "
            f"here is what's known about it:\n{summary}\n\n"
            f"Perform your own fresh, independent check now and write the "
            f"validation report immediately, per your instructions."
        )
        try:
            await asyncio.wait_for(
                run_agent(reporter_agent, session_service, f"{name}_reporter",
                          reporter_message, required_report="validation"),
                timeout=REPORTER_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.error(
                f"[reporter] fallback reporter ALSO timed out after "
                f"{REPORTER_TIMEOUT}s — giving up, no validation.md written."
            )

    try:
        # ── Stage 1: Explorer ──────────────────────────────────────────────
        if _should_run("explorer"):
            _banner(1, f"Explorer  ({repo_url})")
            explorer_final = await _run_stage("explorer", explorer_agent, "explorer", repo_url,
                             required_report="exploration")
            if explorer_final:
                logger.info(f"[Explorer done] report → {base}/reports/exploration.md")

        # ── Stage 2: Environment ───────────────────────────────────────────
        if _should_run("environment"):
            _banner(2, f"Environment ({repo_url})")
            environment_final = await _run_stage(
                "environment", environment_agent, "environment", repo_url,
                required_report="environment", venv_guard_path=venv_python,
            )
            if environment_final:
                logger.info(f"[Environment done] report → {base}/reports/environment.md")

        # ── Stage 3: Coder ─────────────────────────────────────────────────
        if _should_run("coder"):
            _banner(3, f"Coder  ({repo_url})")
            coder_final = await _run_stage(
                "coder", coder_agent, "coder", repo_url,
                required_report="server",
            )
            if coder_final:
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
                validator_progress: dict = {}
                await _run_stage(
                    "validator", validator_agent, "validator", _build_validator_message(),
                    required_report="validation", progress=validator_progress,
                )
                # F35: fires only if validation.md is still missing — covers
                # both the hard-timeout path (_run_stage returned "") and
                # the guard-exhausted path (run_agent returned normally but
                # never actually called write_report). Either way, the file
                # itself (checked below) is the only source of truth now.
                await _ensure_validation_report(validator_progress)

                sep = "=" * 60
                validation_path = base / "reports" / "validation.md"
                # F35: check the file directly — it's the one source of
                # truth regardless of whether the validator itself wrote it
                # or the fallback reporter had to step in.
                if validation_path.exists():
                    logger.info(f"[Validator done] report → {validation_path}")
                    logger.success(
                        f"\n{sep}\n  Pipeline complete: {name}\n"
                        f"  Reports : {base}/reports/\n"
                        f"  Output  : {base}/output/\n"
                        f"  Log     : {log_file}\n{sep}\n\n"
                        f"--- Validator summary ---\n\n"
                        + textwrap.indent(
                            validation_path.read_text(encoding="utf-8").strip(), "  ",
                        )
                    )
                else:
                    logger.error(
                        f"\n{sep}\n  Pipeline incomplete: {name}\n"
                        f"  Validator stage ended — no {validation_path} was "
                        f"written (the fallback reporter also failed to "
                        f"produce one; see logs above).\n"
                        f"  Reports so far : {base}/reports/\n"
                        f"  Output  : {base}/output/\n"
                        f"  Log     : {log_file}\n{sep}\n"
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

        logger.remove(_file_sink_id)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error(f"Usage: ./main.py <repo_url> [--resume <stage>] [--until <stage>]")
        logger.error(f"       stages: {', '.join(STAGES)}")
        logger.error(f"       --resume <stage>: start from <stage> (workdir preserved)")
        logger.error(f"       --until  <stage>: stop after completing <stage>")
        logger.error(f"Example: ./main.py https://github.com/Roestlab/massformer")
        logger.error(f"Example: ./main.py https://github.com/Roestlab/massformer --resume validator")
        logger.error(f"Example: ./main.py https://github.com/Roestlab/massformer --until explorer")
        sys.exit(1)

    repo_url = sys.argv[1]

    def _stage_arg(flag: str) -> str | None:
        if flag not in sys.argv:
            return None
        idx = sys.argv.index(flag)
        if idx + 1 >= len(sys.argv):
            logger.error(f"{flag} requires a stage name")
            sys.exit(1)
        return sys.argv[idx + 1]

    resume_from = _stage_arg("--resume")
    stop_after  = _stage_arg("--until")

    try:
        asyncio.run(run_pipeline(repo_url, resume_from=resume_from, stop_after=stop_after))
    except Exception:
        logger.exception("Pipeline error:")
        raise
