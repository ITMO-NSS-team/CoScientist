#!/usr/bin/env python3
"""Alembic pipeline orchestration.

Five phases, four stage names the harness knows
(explorer/environment/coder/validator). Two of the phases are deterministic
code gates, not LLM turns — that is the core of the remaster: the LLM proposes,
code disposes.

    clone → EXPLORER ─▶ [plan gate] ─▶ ENVIRONMENT ─▶ CODER ─▶ [static gate] ─▶ VALIDATOR(code loop)
    (LLM)              (AST verify +                (LLM)     (AST import      (deterministic: syntax /
                        layout, code)                          check, code)     tests / invoke + debugger;
                                                                                code renders validation.md)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import asyncio
import json
import shutil
import time

from loguru import logger
from google.adk.sessions import InMemorySessionService

from alembic import config
from alembic.agents import explorer_agent, environment_agent, coder_agent, debugger_agent
from alembic.agent_runtime import classify_error, run_agent
from alembic.contract import (
    EnvSpec, Plan, ToolSpec, Validation, ToolVerdict, SampleSpec,
    parse_json_block, parse_samples, save_plan, write_validation,
)
from alembic.tools import WORKDIR, get_repo_name, invoke_mcp_tool, run_tests, validate_syntax
from alembic.tools.analysis import decide_layout, symbol_table, verify_target
from alembic.tools.invoke import set_skip_tools
from alembic.tools.paths import output_dir, repo_path, reports_dir

STAGES = config.STAGES


# ══════════════════════════════════════════════════════════════════════════════
# Target-task (TM-Bench) support
# ══════════════════════════════════════════════════════════════════════════════
def _load_target_task(cli_value: str | None) -> dict | None:
    """A target-task spec is JSON/YAML text or a path to such a file. Returns
    a dict {description, arguments, returns, example} or None (native mode)."""
    raw = cli_value or config.TARGET_TASK
    if not raw:
        return None
    p = Path(raw)
    text = p.read_text(encoding="utf-8") if p.exists() else raw
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
            v = yaml.safe_load(text)
            return v if isinstance(v, dict) else None
        except Exception:
            logger.warning("[target-task] could not parse spec — ignoring, running native mode.")
            return None


def _target_task_prompt(task: dict | None) -> str:
    if not task:
        return ""
    return (
        "\n\nTARGET TASK (you MUST expose one tool that implements exactly this):\n"
        f"- description: {task.get('description', '')}\n"
        f"- arguments: {json.dumps(task.get('arguments', {}))}\n"
        f"- returns: {json.dumps(task.get('returns', {}))}\n"
        f"- example invocation: {json.dumps(task.get('example', {}))}\n"
        "Ensure a tool matches this capability, signature, and return keys."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline
# ══════════════════════════════════════════════════════════════════════════════
async def run_pipeline(repo_url: str, resume_from: str | None = None,
                       stop_after: str | None = None, target_task_cli: str | None = None):
    name = get_repo_name(repo_url)
    base = WORKDIR / name
    session_service = InMemorySessionService()
    target_task = _load_target_task(target_task_cli)

    for stg in (resume_from, stop_after):
        if stg is not None and stg not in STAGES:
            logger.error(f"Unknown stage '{stg}'. Valid: {', '.join(STAGES)}")
            return

    if resume_from is None:
        _clean_workdir(name)
    else:
        logger.info(f"[Resume] from stage: {resume_from} (workdir preserved)")

    venv_python = str((base / "output" / ".venv" / "bin" / "python").resolve())
    log_file = base / "pipeline.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    sink_id = logger.add(log_file, format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
                         level="DEBUG", encoding="utf-8")
    logger.info(f"[Run] {name} — log → {log_file}"
                + (f"  (target-task mode)" if target_task else ""))

    for suffix in ("explorer", "environment", "coder"):
        await session_service.create_session(app_name=config.APP_NAME, user_id=config.USER_ID,
                                              session_id=f"{name}_{suffix}")

    metrics = _new_metrics()

    def _should_run(stage: str) -> bool:
        idx = STAGES.index(stage)
        if resume_from is not None and idx < STAGES.index(resume_from):
            return False
        if stop_after is not None and idx > STAGES.index(stop_after):
            return False
        return True

    try:
        # ── 1. Explorer ────────────────────────────────────────────────────
        if _should_run("explorer"):
            _banner(1, f"Explorer ({repo_url})")
            msg = repo_url + _target_task_prompt(target_task)
            final = await _run_llm_stage("explorer", explorer_agent, f"{name}_explorer",
                                         msg, metrics, session_service, required_report="exploration")
            _salvage_report(repo_url, "exploration", final, needs_plan=True)
            if final:
                logger.info(f"[Explorer done] → {base}/reports/exploration.md")
            # Plan gate (deterministic): verify proposed tools + decide layout.
            _plan_gate(repo_url)

        # ── 2. Environment ─────────────────────────────────────────────────
        if _should_run("environment"):
            _banner(2, f"Environment ({repo_url})")
            final = await _run_llm_stage("environment", environment_agent, f"{name}_environment",
                                         _environment_message(repo_url), metrics, session_service,
                                         required_report="environment", venv_guard_path=venv_python)
            _salvage_report(repo_url, "environment", final)
            if final:
                logger.info(f"[Environment done] → {base}/reports/environment.md")

        # ── 3. Coder ───────────────────────────────────────────────────────
        if _should_run("coder"):
            _banner(3, f"Coder ({repo_url})")
            final = await _run_llm_stage("coder", coder_agent, f"{name}_coder",
                                         _coder_message(repo_url, target_task), metrics, session_service,
                                         required_report="server")
            _salvage_report(repo_url, "server", final)
            if final:
                logger.info(f"[Coder done] → {base}/output/server.py")

        # ── 4. Validator (deterministic code loop) ─────────────────────────
        if _should_run("validator"):
            ok, missing = _coder_artefacts_present(base)
            if not ok:
                logger.error(f"[validator] required artefacts missing: {missing} — skipping validation.")
            else:
                _banner(4, f"Validator ({repo_url})")
                await _validate(repo_url, name, session_service, metrics)
                _report_completion(name, base, log_file)

    finally:
        _finalize_metrics(base, metrics)
        logger.remove(sink_id)


# ── LLM stage wrapper ─────────────────────────────────────────────────────────
async def _run_llm_stage(stage, agent, session_id, message, metrics, session_service,
                         required_report=None, venv_guard_path=None) -> str:
    started = time.monotonic()
    soft_deadline = started + config.STAGE_TIMEOUT[stage] * config.REPORT_GRACE_FRACTION
    try:
        final, steps, tokens, sm = await asyncio.wait_for(
            run_agent(agent, session_service, session_id, message,
                      required_report=required_report, venv_guard_path=venv_guard_path,
                      deadline=soft_deadline),
            timeout=config.STAGE_TIMEOUT[stage])
    except asyncio.TimeoutError:
        logger.error(f"[{stage}] STAGE TIMEOUT after {config.STAGE_TIMEOUT[stage]}s — continuing.")
        metrics["durations_per_stage"][stage] = round(time.monotonic() - started, 1)
        metrics["abort_reason_per_stage"][stage] = "stage_timeout"
        return ""
    _record_stage(metrics, stage, round(time.monotonic() - started, 1), steps, tokens, sm)
    return final


def _salvage_report(repo_url: str, report_name: str, final_text: str,
                    needs_plan: bool = False) -> None:
    """Guarantee a stage's report exists even when the agent narrated its answer
    instead of calling write_report (a common, intermittent LLM slip that would
    otherwise silently zero the stage). LLM proposes; code disposes: persist the
    agent's final response text when the report is missing/empty — or, for the
    plan-bearing exploration report, when the file holds no parseable JSON plan
    but the final response does."""
    path = reports_dir(repo_url) / f"{report_name}.md"
    existing = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    final_text = final_text or ""
    if final_text.strip() in ("", "Agent did not produce a final response."):
        return  # nothing worth salvaging (timeout / empty / runtime default)
    if needs_plan and not parse_json_block(existing):
        if parse_json_block(final_text):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(final_text, encoding="utf-8")
            logger.warning(f"[{report_name}] salvaged plan from the agent's final response "
                           f"(write_report was not called)")
            return
    if not existing.strip() and final_text.strip():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(final_text, encoding="utf-8")
        logger.warning(f"[{report_name}] salvaged the agent's final response "
                       f"(write_report was not called)")


# ── Plan gate (deterministic) ─────────────────────────────────────────────────
def _plan_gate(repo_url: str) -> Plan:
    """Verify the Explorer's proposed tools against the real repo AST, extract
    real parameter names, drop clear hallucinations, and decide the venv layout.
    Writes plan.json. Purely deterministic — no LLM."""
    repo_dir = repo_path(repo_url).resolve()
    proposal = parse_json_block((reports_dir(repo_url) / "exploration.md").read_text(encoding="utf-8", errors="replace")) \
        if (reports_dir(repo_url) / "exploration.md").exists() else None
    proposal = proposal or {}

    layout = decide_layout(repo_dir)
    env_raw = proposal.get("env", {}) if isinstance(proposal.get("env"), dict) else {}
    env = EnvSpec(
        layout=layout["layout"], server_python=layout["server_python"], repo_python=layout["repo_python"],
        requirements_files=env_raw.get("requirements_files", []),
        dependencies=env_raw.get("dependencies", []),
        system_libs=env_raw.get("system_libs", []),
        weights=env_raw.get("weights", []),
    )

    table = symbol_table(repo_dir)
    tools: list[ToolSpec] = []
    dropped: list[str] = []
    for t in (proposal.get("tools") or [])[: config.MAX_TOOLS + 4]:
        if not isinstance(t, dict) or not t.get("name") or not t.get("target"):
            continue
        v = verify_target(t["target"], table, repo_dir)
        if not v["ok"]:
            dropped.append(f"{t['name']} ({t['target']}: {v['reason']})")
            continue
        tools.append(ToolSpec(name=t["name"], target=t["target"], purpose=t.get("purpose", ""),
                              params=v["params"], verified=True, note=v["reason"]))
        if len(tools) >= config.MAX_TOOLS:
            break

    plan = Plan(repo_url=repo_url, env=env, tools=tools)
    save_plan(plan)
    logger.info(f"[plan gate] layout={env.layout} ({layout['source']}); "
                f"verified {len(tools)} tool(s), dropped {len(dropped)} hallucinated/unverifiable.")
    if dropped:
        logger.info(f"[plan gate] dropped: {'; '.join(dropped)}")
    return plan


# ── Stage messages built from the plan ────────────────────────────────────────
def _environment_message(repo_url: str) -> str:
    from alembic.contract import load_plan
    plan = load_plan(repo_url)
    if not plan:
        return repo_url
    e = plan.env
    lines = [repo_url, "", "Computed environment layout — trust this, it is authoritative:",
             f"  layout: {e.layout}", f"  server_python: {e.server_python}"]
    if e.repo_python:
        lines.append(f"  repo_python: {e.repo_python}  (build .venv-repo on this)")
    if e.requirements_files:
        lines.append(f"  requirements files: {', '.join(e.requirements_files)}")
    if e.system_libs:
        lines.append(f"  likely system libs: {', '.join(e.system_libs)}")
    if e.weights:
        lines.append(f"  external weights to download: {json.dumps(e.weights)}")
    return "\n".join(lines)


def _coder_message(repo_url: str, target_task: dict | None) -> str:
    from alembic.contract import load_plan
    plan = load_plan(repo_url)
    lines = [repo_url]
    if plan and plan.tools:
        lines += ["", "Verified tools to implement (targets confirmed to exist in the repo; "
                  "params are the REAL signature — build argv accordingly):"]
        for t in plan.tools:
            lines.append(f"  - {t.name}  ->  {t.target}  params={t.params}  # {t.purpose}")
    return "\n".join(lines) + _target_task_prompt(target_task)


# ══════════════════════════════════════════════════════════════════════════════
# Validator — deterministic loop calling the debugger only for the hard part
# ══════════════════════════════════════════════════════════════════════════════
async def _validate(repo_url, name, session_service, metrics):
    started = time.monotonic()
    deadline = started + config.STAGE_TIMEOUT["validator"] * config.REPORT_GRACE_FRACTION
    v = Validation()
    problem_summaries: list[str] = []
    dbg_counter = [0]
    failures: dict[str, int] = {}
    n_actions = 0

    async def debug(msg: str) -> str:
        dbg_counter[0] += 1
        sid = f"{name}_debugger_{dbg_counter[0]}"
        await session_service.create_session(app_name=config.APP_NAME, user_id=config.USER_ID, session_id=sid)
        mem = ("\n\nPrevious fix attempts this run (do NOT repeat them):\n- "
               + "\n- ".join(problem_summaries)) if problem_summaries else ""
        try:
            final, _, _, _ = await asyncio.wait_for(
                run_agent(debugger_agent, session_service, sid, msg + mem),
                timeout=config.DEBUGGER_CALL_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning(f"[validator] debugger call timed out after {config.DEBUGGER_CALL_TIMEOUT}s.")
            return "debugger timed out"
        problem_summaries.append(final[:400])
        return final

    # 1. Syntax & imports (static) — repair-loop bounded.
    for attempt in range(config.MAX_STATIC_GATE_RETRIES + 1):
        r = await validate_syntax(repo_url)
        n_actions += 1
        if r.get("passed"):
            v.syntax_ok = True
            break
        v.syntax_error = f"[{r.get('stage')}] {r.get('error', '')}"
        failures[classify_error(v.syntax_error)] = failures.get(classify_error(v.syntax_error), 0) + 1
        if attempt >= config.MAX_STATIC_GATE_RETRIES or time.monotonic() >= deadline:
            break
        summary = await debug(f"Repository: {repo_url}\n\nvalidate_syntax failed at stage "
                              f"'{r.get('stage')}':\n{r.get('error','')}\n\nFix server.py or the "
                              f"named helper, then confirm it imports.")
        v.debugger_actions.append(f"syntax: {summary[:200]}")

    # 2. pytest (only if syntax passed and there's time).
    if v.syntax_ok and time.monotonic() < deadline:
        r = await run_tests(repo_url)
        n_actions += 1
        v.tests_ran = True
        out = r.get("output", "")
        v.tests_passed, v.tests_failed = _parse_pytest_counts(out)
        if not r.get("passed"):
            v.tests_error = out[-800:]
            failures[classify_error(out)] = failures.get(classify_error(out), 0) + 1

    # 3. Per-tool live invocation + repair.
    samples = parse_samples(repo_url)
    set_skip_tools([s.name for s in samples if s.skip])
    for s in samples:
        if time.monotonic() >= deadline:
            v.tools.append(ToolVerdict(s.name, "SKIPPED", "not reached (stage out of time)"))
            continue
        if s.skip:
            v.tools.append(ToolVerdict(s.name, "SKIPPED", s.skip_reason or "marked SKIP"))
            continue
        verdict, extra = await _validate_one_tool(repo_url, s, debug, v, failures, deadline)
        n_actions += extra
        v.tools.append(verdict)

    write_validation(repo_url, name, v)
    logger.info(f"[Validator done] → {reports_dir(repo_url)}/validation.md  "
                f"(syntax={'ok' if v.syntax_ok else 'FAIL'}, "
                f"tools P/F/S={sum(t.status=='PASSED' for t in v.tools)}/"
                f"{sum(t.status=='FAILED' for t in v.tools)}/{sum(t.status=='SKIPPED' for t in v.tools)})")

    metrics["durations_per_stage"]["validator"] = round(time.monotonic() - started, 1)
    metrics["actions_per_stage"]["validator"] = n_actions
    metrics["total_actions"] += n_actions
    for label, c in failures.items():
        metrics["failures_by_class"][label] = metrics["failures_by_class"].get(label, 0) + c


async def _validate_one_tool(repo_url, s: SampleSpec, debug, v: Validation,
                             failures: dict, deadline: float) -> tuple[ToolVerdict, int]:
    """Invoke one tool; on failure debug + independently re-invoke (F24), bounded.
    Returns (verdict, n_extra_actions)."""
    n = 0
    last_error = None
    for attempt in range(3):
        if time.monotonic() >= deadline:
            return ToolVerdict(s.name, "FAILED" if last_error else "SKIPPED",
                               "stage out of time mid-repair" if last_error else "not reached"), n
        r = await invoke_mcp_tool(repo_url, s.name, s.sample_args or {})
        n += 1
        if r.get("skipped"):
            return ToolVerdict(s.name, "SKIPPED", r.get("reason", "")[:200]), n
        if r.get("ok"):
            note = _semantic_note(s, r.get("result"))
            # held-out invocation (F3) — advisory, does not flip the verdict.
            if s.holdout_args:
                hr = await invoke_mcp_tool(repo_url, s.name, s.holdout_args)
                n += 1
                if not hr.get("ok") and not hr.get("skipped"):
                    note = (note + "; " if note else "") + "held-out input failed (possible overfit)"
            return ToolVerdict(s.name, "PASSED", note), n
        # failure
        err = f"{r.get('error','')}\n{r.get('traceback','')}"
        failures[classify_error(err)] = failures.get(classify_error(err), 0) + 1
        if last_error and err.splitlines()[0:1] == last_error.splitlines()[0:1]:
            return ToolVerdict(s.name, "FAILED", (r.get("error") or "")[:200] + " (repeated)"), n
        last_error = err
        if attempt >= 2:
            break
        summary = await debug(
            f"Repository: {repo_url}\n\ninvoke_mcp_tool('{s.name}', {json.dumps(s.sample_args or {})}) failed:\n"
            f"error: {r.get('error','')}\ntraceback: {r.get('traceback','')}\nstderr: {r.get('stderr','')}\n\n"
            f"Fix the missing dependency or the code bug, then re-run this tool to confirm.")
        v.debugger_actions.append(f"invoke {s.name}: {summary[:200]}")
    return ToolVerdict(s.name, "FAILED", (last_error or "").splitlines()[0][:200] if last_error else "failed"), n


def _semantic_note(s: SampleSpec, result) -> str:
    """F2 (advisory): note declared-return keys that are missing from the result."""
    if not s.returns or not isinstance(result, dict):
        return ""
    missing = [k for k in s.returns if k not in result]
    return f"note: result missing declared keys {missing}" if missing else ""


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
def _banner(n: int, label: str) -> None:
    sep = "=" * 60
    logger.info(f"\n{sep}\n  STAGE {n} — {label}\n{sep}")


def _clean_workdir(name: str) -> None:
    d = WORKDIR / name
    if d.exists():
        shutil.rmtree(d)
        logger.debug(f"[clean] removed {d}")


def _coder_artefacts_present(base: Path) -> tuple[bool, list[str]]:
    required = [base / "output" / "server.py", base / "output" / "tests" / "test_server.py",
                base / "reports" / "server.md"]
    missing = [str(p.relative_to(base)) for p in required if not p.exists()]
    return (not missing), missing


def _parse_pytest_counts(output: str) -> tuple[int | None, int | None]:
    import re
    p = re.search(r"(\d+)\s+passed", output)
    f = re.search(r"(\d+)\s+failed", output)
    return (int(p.group(1)) if p else (0 if output else None),
            int(f.group(1)) if f else 0)


def _new_metrics() -> dict:
    return {"actions_per_stage": {}, "tokens_per_stage": {}, "durations_per_stage": {},
            "tool_calls_per_stage": {}, "guard_retries_per_stage": {},
            "transient_fault_retries_per_stage": {}, "abort_reason_per_stage": {},
            "failures_by_class": {}, "total_actions": 0, "total_tokens": 0}


def _record_stage(metrics, stage, duration, steps, tokens, sm) -> None:
    metrics["durations_per_stage"][stage] = duration
    metrics["actions_per_stage"][stage] = steps
    metrics["tokens_per_stage"][stage] = tokens
    metrics["total_actions"] += steps
    metrics["total_tokens"] += tokens
    metrics["tool_calls_per_stage"][stage] = sm["tool_calls"]
    metrics["guard_retries_per_stage"][stage] = sm["guard_retries"]
    metrics["transient_fault_retries_per_stage"][stage] = sm["transient_fault_retries"]
    if sm["abort_reason"]:
        metrics["abort_reason_per_stage"][stage] = sm["abort_reason"]
    for label, c in sm["failures_by_class"].items():
        metrics["failures_by_class"][label] = metrics["failures_by_class"].get(label, 0) + c


def _finalize_metrics(base: Path, metrics: dict) -> None:
    import sys as _sys, traceback as _tb
    exc_type, exc_val, exc_tb = _sys.exc_info()
    d = base / "reports"
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    if exc_val is not None:
        (d / "error.json").write_text(json.dumps({
            "exception": type(exc_val).__name__, "message": str(exc_val),
            "traceback": "".join(_tb.format_exception(exc_type, exc_val, exc_tb)),
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.error(f"[pipeline] error saved → {d}/error.json")


def _report_completion(name: str, base: Path, log_file: Path) -> None:
    sep = "=" * 60
    vp = base / "reports" / "validation.md"
    if vp.exists():
        logger.success(f"\n{sep}\n  Pipeline complete: {name}\n  Reports: {base}/reports/\n{sep}")
    else:
        logger.error(f"\n{sep}\n  Pipeline incomplete: {name} — no validation.md written.\n{sep}")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("Usage: python -m alembic.main <repo_url> [--resume <stage>] "
                     "[--until <stage>] [--target-task <spec>]")
        logger.error(f"       stages: {', '.join(STAGES)}")
        sys.exit(1)

    def _arg(flag: str) -> str | None:
        if flag not in sys.argv:
            return None
        i = sys.argv.index(flag)
        if i + 1 >= len(sys.argv):
            logger.error(f"{flag} requires a value")
            sys.exit(1)
        return sys.argv[i + 1]

    try:
        asyncio.run(run_pipeline(sys.argv[1], resume_from=_arg("--resume"),
                                 stop_after=_arg("--until"), target_task_cli=_arg("--target-task")))
    except Exception:
        logger.exception("Pipeline error:")
        raise
