#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run one test prompt; write logs/artifacts into the sibling folder.
"""
from __future__ import annotations

import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from dotenv import dotenv_values

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
MANIFEST = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
TESTS = {row["id"]: row for row in MANIFEST["tests"]}


def load_env(test: dict, out: Path) -> dict[str, str]:
    env = os.environ.copy()
    env_file = REPO / ".env" if (REPO / ".env").is_file() else REPO / "CoScientist" / ".env"
    for k, v in dotenv_values(env_file).items():
        if v is not None and k not in env:
            env[k] = v
    env.pop("SESSION_ID", None)
    env["SESSION_ID"] = f"trial_{test['id']}_{int(time.time())}"
    env["COSCIENTIST_CONFIG"] = str(REPO / "CoScientist" / "agents" / "experiments.yaml")
    env["COSCIENTIST_EXPERIMENT_HITL_AUTO_APPROVE"] = "1"
    env["COSCIENTIST_EXPERIMENT_AUDIT_STDOUT"] = "1"
    env["EXPERIMENTS__ROUTE_FEDOT"] = "true"
    env["EXPERIMENTS__ROUTE_ALEMBIC"] = "true" if test.get("route_alembic") else "false"
    env["HYPOTHESES__MAX_ACTIVE"] = "2"
    env["EXPERIMENTS__MAX_GENERATE_NUM"] = str(test.get("max_generate_num", 5))
    if test.get("max_plan_tasks"):
        env["EXPERIMENTS__MAX_PLAN_TASKS"] = str(test["max_plan_tasks"])
    if test.get("alembic_timeout_s"):
        env["EXPERIMENTS__ALEMBIC_TIMEOUT_S"] = str(test["alembic_timeout_s"])
    env["COSCIENTIST_FEDOT_TIMEOUT_S"] = "1200"
    env["GRAPH_SNAPSHOT_DIR"] = str(out / "graph_runs")
    env["RESEARCH_GRAPH_DIR"] = str(out / "graph_runs")
    artifacts = out / "artifacts"
    reports = out / "reports"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    env["EXPERIMENTS__ARTIFACTS_DIR"] = str(artifacts)
    env["EXPERIMENTS__REPORTS_DIR"] = str(reports)
    env["REPORTS_ROOT"] = str(reports)
    return env


def kill_tree(p: subprocess.Popen) -> None:
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except Exception:
        try:
            p.kill()
        except Exception:
            pass


def extract_plan(log_text: str, out: Path) -> None:
    marker = "EXPERIMENT_DESIGN_MATRIX\n"
    if marker not in log_text:
        return
    body = log_text.split(marker, 1)[1]
    cut = re.search(r"\n(?:EXPERIMENT_|INFO:|\x1b\[)", body)
    text = body[: cut.start()] if cut else body[:8000]
    (out / "plan.md").write_text(text.strip() + "\n", encoding="utf-8")


def main() -> int:
    os.chdir(HERE)
    if len(sys.argv) != 2 or sys.argv[1] not in TESTS:
        names = " | ".join(TESTS)
        raise SystemExit(f"usage: run_test.py {{{names}}}")
    test = TESTS[sys.argv[1]]
    prompt_path = HERE / test["file"]
    out = HERE / test["id"]
    out.mkdir(parents=True, exist_ok=True)
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if "\n" in prompt:
        raise SystemExit(f"{prompt_path.name} must be a single line (CLI input() reads one line)")
    (out / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    env = load_env(test, out)
    timeout = int(test.get("timeout_s", 4800))
    log_path = out / "console.log"
    started = time.time()
    print(f"START id={test['id']} session={env['SESSION_ID']} timeout={timeout}s out={out}", flush=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as fh:
        fh.write(
            f"# id={test['id']}\n# session={env['SESSION_ID']}\n"
            f"# timeout={timeout}\n# source={test.get('source')}\n"
            f"# prompt_chars={len(prompt)}\n# prompt_head={prompt[:160]!r}\n"
            f"{'=' * 72}\n"
        )
        fh.flush()
        p = subprocess.Popen(
            ["uv", "run", "python", "-m", "CoScientist", "cli"],
            cwd=str(REPO),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert p.stdin is not None and p.stdout is not None
        p.stdin.write(prompt + "\nexit\n")
        p.stdin.close()
        line_q: queue.Queue[str | None] = queue.Queue()

        def pump() -> None:
            try:
                for raw in p.stdout:
                    line_q.put(raw)
            finally:
                line_q.put(None)

        threading.Thread(target=pump, daemon=True).start()
        rc = None
        timed_out = False
        while True:
            if time.time() - started > timeout:
                timed_out = True
                note = "\n[runner] TIMEOUT - killing process tree\n"
                fh.write(note)
                fh.flush()
                print(note, end="", flush=True)
                kill_tree(p)
                rc = -9
                break
            try:
                line = line_q.get(timeout=0.5)
            except queue.Empty:
                if p.poll() is not None and line_q.empty():
                    rc = p.returncode
                    break
                continue
            if line is None:
                rc = p.poll() if p.poll() is not None else 0
                break
            fh.write(line)
            fh.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
        elapsed = time.time() - started
        footer = f"\n# DONE elapsed={elapsed:.1f}s rc={rc} timed_out={timed_out}\n"
        fh.write(footer)
        print(footer, end="", flush=True)
    extract_plan(log_path.read_text(encoding="utf-8", errors="replace"), out)
    (out / "run.json").write_text(
        json.dumps(
            {
                "id": test["id"],
                "session": env["SESSION_ID"],
                "elapsed_s": round(elapsed, 1),
                "rc": rc,
                "timed_out": timed_out,
                "source": test.get("source"),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0 if rc == 0 and not timed_out else 1


if __name__ == "__main__":
    raise SystemExit(main())
