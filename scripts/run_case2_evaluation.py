#!/usr/bin/env python3
"""Run Case 2 up to the T0 plan-review checkpoint and persist the run.

The experiment review is deliberately fail-closed: a validated plan is saved,
but it is never approved and therefore never executed.  SQLite stores the
session/plan/trace plus the checkpoint manifest listing.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)


def _read_attr(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _extract_plan(state: dict[str, Any]) -> Any:
    runtime = state.get("experiment_runtime") or {}
    return state.get("experiment_plan") or runtime.get("plan")


def _summary(state: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "experiment_em_request",
        "experiment_context",
        "experiment_hypotheses",
        "hypotheses",
        "experiment_plan",
        "experiment_plan_critique",
        "experiment_runtime",
        "experiment_task_results",
        "experiment_summary",
        "retrieval_queries",
        "retrieval_queries_mcp",
        "retrieved_tools",
        "filtered_tools",
        "deployed_mcps",
        "fedot_results",
        "coder_results",
        "final_report",
    )
    return {key: state.get(key) for key in keys if key in state}


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pipeline_evaluation_runs (
            run_id TEXT PRIMARY KEY,
            case_id TEXT NOT NULL,
            variant TEXT NOT NULL,
            prompt TEXT NOT NULL,
            profile TEXT NOT NULL,
            git_commit TEXT NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT NOT NULL,
            status TEXT NOT NULL,
            error TEXT,
            preflight_json TEXT NOT NULL,
            plan_json TEXT,
            summary_json TEXT NOT NULL,
            state_json TEXT NOT NULL,
            report_markdown TEXT NOT NULL,
            report_dir TEXT,
            manifest_json TEXT,
            trace_text TEXT,
            trace_path TEXT
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_pipeline_eval_case_variant "
        "ON pipeline_evaluation_runs(case_id, variant)"
    )
    columns = {row[1] for row in conn.execute("PRAGMA table_info(pipeline_evaluation_runs)")}
    for name, declaration in (
        ("evaluation_mode", "TEXT"),
        ("checkpoint_dir", "TEXT"),
        ("checkpoints_json", "TEXT"),
    ):
        if name not in columns:
            conn.execute(f"ALTER TABLE pipeline_evaluation_runs ADD COLUMN {name} {declaration}")


async def _run(args: argparse.Namespace) -> int:
    load_dotenv(args.env_file, override=False)
    prompt = args.prompt.read_text(encoding="utf-8").strip()
    run_id = f"case2_{args.variant}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    os.environ["COSCIENTIST_CONFIG"] = args.profile
    # Global HITL stays off so ordinary agents never block on ConsoleHITLHandler.
    # ExperimentReviewSessionAgent installs its own fail-closed handler even when
    # global HITL is disabled; with auto-approve off it validates and pauses at T0.
    os.environ["HITL__ENABLED"] = "false"
    os.environ["COSCIENTIST_EXPERIMENT_HITL_AUTO_APPROVE"] = "0"
    os.environ["CHECKPOINTS__ENABLED"] = "true"
    os.environ["CHECKPOINTS__DIR"] = str(args.checkpoint_dir)
    os.environ["POSTGRES__PORT"] = str(args.postgres_port)
    os.environ["AGENT_LOG_FILE"] = str(args.trace)
    os.environ.setdefault("LOG_AGENT_EVENTS", "1")
    os.environ.setdefault("CONTEXT_INIT__ENABLED", "false")

    # Import only after all profile/configuration environment is fixed.
    sys.path.insert(0, str(Path.cwd()))
    from CoScientist.main import CoScientistManager

    started = datetime.now(timezone.utc)
    manager = CoScientistManager(user_id=f"eval_{run_id}", session_id=run_id)
    result: Any = None
    state: dict[str, Any] = {}
    caught: BaseException | None = None
    try:
        result = await manager.run(prompt, verbose=False)
        session = await manager.session_service.get_session(
            app_name=manager.app_name,
            user_id=manager.user_id,
            session_id=manager.session_id,
        )
        state = dict(getattr(session, "state", None) or {}) if session else {}
        caught = getattr(manager, "_run_error", None)
    except BaseException as exc:
        caught = exc
        try:
            session = await manager.session_service.get_session(
                app_name=manager.app_name,
                user_id=manager.user_id,
                session_id=manager.session_id,
            )
            state = dict(getattr(session, "state", None) or {}) if session else {}
        except Exception:
            state = {}
    finally:
        try:
            await manager.close()
        except Exception:
            pass

    finished = datetime.now(timezone.utc)
    report = str(_read_attr(result, "markdown", result or ""))
    report_dir = _read_attr(result, "report_dir")
    manifest = _read_attr(result, "manifest")
    trace_text = args.trace.read_text(encoding="utf-8", errors="replace") if args.trace.exists() else ""
    runtime = state.get("experiment_runtime") or {}
    phase = runtime.get("phase") if isinstance(runtime, dict) else None
    status = "error" if caught else (str(phase) if phase else "completed")
    checkpoint_run_id = f"{manager.app_name}__{run_id}"
    checkpoints: list[dict[str, Any]] = []
    try:
        from CoScientist.checkpoints import LocalZipStore

        store = LocalZipStore(str(args.checkpoint_dir))
        checkpoints = store.list(checkpoint_run_id)
        for row in checkpoints:
            path = store.bundle_path(row["checkpoint_id"])
            row["bundle_path"] = str(path) if path else None
    except Exception as exc:
        checkpoints = [{"error": f"{type(exc).__name__}: {exc}"}]
    preflight = {
        "result_store": "local_sqlite_fallback",
        "registry_postgres_port": args.postgres_port,
        "registry_servers": 0,
        "registry_tools": 0,
        "qdrant_reachable": True,
        "headless_auto_approve": False,
        "evaluation_mode": "plan_checkpoint_only",
        "executor_allowed": False,
        "checkpoint_run_id": checkpoint_run_id,
    }

    args.db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(args.db) as conn:
        _ensure_schema(conn)
        conn.execute(
            """
            INSERT INTO pipeline_evaluation_runs (
                run_id, case_id, variant, prompt, profile, git_commit,
                started_at, finished_at, status, error, preflight_json,
                plan_json, summary_json, state_json, report_markdown,
                report_dir, manifest_json, trace_text, trace_path,
                evaluation_mode, checkpoint_dir, checkpoints_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id, "2", args.variant, prompt, args.profile, _git_commit(),
                started.isoformat(), finished.isoformat(), status,
                f"{type(caught).__name__}: {caught}" if caught else None,
                _json(preflight), _json(_extract_plan(state)), _json(_summary(state)),
                _json(state), report, str(report_dir) if report_dir else None,
                _json(manifest), trace_text, str(args.trace),
                "plan_checkpoint_only", str(args.checkpoint_dir), _json(checkpoints),
            ),
        )
        conn.commit()

    print(_json({
        "run_id": run_id,
        "variant": args.variant,
        "status": status,
        "error": f"{type(caught).__name__}: {caught}" if caught else None,
        "plan_present": _extract_plan(state) is not None,
        "runtime_phase": phase,
        "checkpoint_count": len(checkpoints),
        "checkpoint_labels": [row.get("label") for row in checkpoints],
        "report_dir": str(report_dir) if report_dir else None,
        "db": str(args.db),
        "trace": str(args.trace),
    }))
    return 1 if caught else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=("question_only", "with_refinements"))
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--profile", default="CoScientist/agents/experiments.yaml")
    parser.add_argument("--postgres-port", type=int, default=5433)
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
