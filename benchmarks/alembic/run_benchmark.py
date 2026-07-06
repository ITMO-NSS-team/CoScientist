#!/usr/bin/env python3
"""Benchmark the alembic pipeline against a list of repos in parallel.

The base image (``alembic-base:latest``) is built once up-front, then N
workers run ``start_chain.py --no-serve`` concurrently — emulating a
production setting where the base image is pre-baked on a CI host. Servers
are not launched; the produced ``alembic-tool:<repo>`` images stay so you
can ``docker run`` them later.

Usage:
    # parallel run, default 4 workers — outputs land under
    # benchmarks/alembic/runs/<timestamp>/{summary.md,summary.json,logs/}
    python benchmarks/alembic/run_benchmark.py \\
        --repos https://github.com/Roestlab/massformer \\
                https://github.com/whitead/synspace \\
                https://github.com/CrystalEye42/OpenChemIE

    # from a file (one URL per line, '#' = comment), 8 workers, explicit
    # output paths instead of the timestamped default
    python benchmarks/alembic/run_benchmark.py \\
        --repos-file repos.txt \\
        --parallel 8 \\
        --output benchmarks/alembic/runs/my-run/summary.md \\
        --json-output benchmarks/alembic/runs/my-run/summary.json \\
        --log-dir benchmarks/alembic/runs/my-run/logs
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# benchmarks/alembic/run_benchmark.py → project root is 2 levels up
PROJECT_ROOT    = Path(__file__).resolve().parents[2]
COSCIENTIST_DIR = PROJECT_ROOT / "CoScientist"
RUNS_DIR        = Path(__file__).resolve().parent / "runs"

sys.path.insert(0, str(COSCIENTIST_DIR))

from alembic.common import get_repo_name, ensure_base_image

START_CHAIN = COSCIENTIST_DIR / "alembic" / "start_chain.py"
DOCKERFILE  = PROJECT_ROOT / "docker" / "alembic" / "Dockerfile"

AVAILABILITY_TIMEOUT = 15  # seconds — cheap network check, no clone


def check_repo_available(repo_url: str, timeout: int = AVAILABILITY_TIMEOUT) -> tuple[bool, str]:
    """True if ``repo_url`` is reachable and has at least one ref, via
    ``git ls-remote`` — no clone, so a dead/private/empty repo (e.g. the
    Analyze-stroke case: ``fatal: could not read Username``) is caught in
    seconds instead of burning a full pipeline run before the Explorer's
    own clone fails."""
    try:
        r = subprocess.run(
            ["git", "ls-remote", "--exit-code", "--heads", repo_url],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s"
    if r.returncode == 0:
        return True, ""
    lines = [l.strip() for l in (r.stderr or r.stdout).splitlines() if l.strip()]
    return False, (lines[0] if lines else f"git ls-remote exit {r.returncode}")


def run_one(repo_url: str, extra_args: list[str], log_dir: Path,
            idx: int, total: int) -> dict:
    """Invoke start_chain.py --no-serve for one repo; stream logs to a file."""
    name     = get_repo_name(repo_url)
    log_path = log_dir / f"{name}.log"
    print(f"[bench] ↑ start  {name}  ({idx}/{total})  log: {log_path.name}",
          flush=True)

    started = time.time()
    with log_path.open("w", encoding="utf-8") as logf:
        cmd = [sys.executable, str(START_CHAIN), repo_url,
               "--no-serve", *extra_args]
        logf.write(f"$ {' '.join(cmd)}\n\n")
        logf.flush()
        r = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    elapsed = time.time() - started

    record: dict = {
        "repo":         name,
        "url":          repo_url,
        "elapsed_sec":  round(elapsed, 1),
        "exit_code":    r.returncode,
        "log":          str(log_path),
        "error_tail":   None,
        "validation":   None,
    }
    if r.returncode == 0:
        record["validation"] = extract_validation(name)
    else:
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            record["error_tail"] = "\n".join(lines[-60:])
        except OSError:
            pass

    status = "ok" if r.returncode == 0 else f"exit={r.returncode}"
    print(f"[bench] ↓ done   {name}  ({elapsed:.0f}s, {status})",
          flush=True)
    return record


def extract_validation(repo: str) -> dict:
    """Pull validation.md (and metrics.json/error.json) from the committed image and parse them."""
    image = f"alembic-tool:{repo}"
    base  = f"/work/.alembic/{repo}/reports"

    r = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "cat", image, f"{base}/validation.md"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        result: dict = {"error": "validation.md not readable", "stderr": r.stderr[-300:]}
    else:
        result = parse_validation(r.stdout)

    # F12: metrics.json is written unconditionally in main.py's `finally` block,
    # even when the pipeline never reaches (or times out in) the validator stage
    # — pull it regardless of whether validation.md exists, so a partial/failed
    # run still contributes stage-completion and failure-taxonomy data to the
    # benchmark-level aggregate instead of being silently excluded.
    for fname, key in [("metrics.json", "pipeline_metrics"), ("error.json", "pipeline_error")]:
        c = subprocess.run(
            ["docker", "run", "--rm", "--entrypoint", "cat", image, f"{base}/{fname}"],
            capture_output=True, text=True,
        )
        if c.returncode == 0:
            try:
                result[key] = json.loads(c.stdout)
            except json.JSONDecodeError:
                pass

    return result


def parse_validation(md: str) -> dict:
    """Pull headline statuses + per-tool table from validation.md."""
    sections: dict[str, str] = {}
    cur, buf = None, []
    for line in md.splitlines():
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if m:
            if cur is not None:
                sections[cur] = "\n".join(buf).strip()
            cur, buf = m.group(1).strip(), []
        else:
            buf.append(line)
    if cur is not None:
        sections[cur] = "\n".join(buf).strip()

    def first_line(s: str) -> str:
        return s.splitlines()[0].strip() if s else ""

    tools: list[dict] = []
    for line in sections.get("Tool Invocations", "").splitlines():
        m = re.match(r"^-\s+\*\*(.+?)\*\*\s+—\s+(\w+)(.*)", line.strip())
        if m:
            reason = m.group(3).strip().lstrip("(").rstrip(")").strip()
            entry: dict = {"name": m.group(1), "status": m.group(2)}
            if reason:
                entry["reason"] = reason
            tools.append(entry)

    # parse "PASSED — 16 passed, 0 failed" into integers
    tests_str = first_line(sections.get("Tests", ""))
    tests_passed = tests_failed = None
    tm = re.search(r"(\d+)\s+passed", tests_str)
    if tm:
        tests_passed = int(tm.group(1))
    tf = re.search(r"(\d+)\s+failed", tests_str)
    if tf:
        tests_failed = int(tf.group(1))

    return {
        "syntax":          first_line(sections.get("Syntax & Imports", "")),
        "tests":           tests_str,
        "tests_passed":    tests_passed,
        "tests_failed":    tests_failed,
        "overall":         first_line(sections.get("Overall", "")),
        "tools":           tools,
        "tools_created":   len(tools),
        "tools_invoked_ok":      sum(1 for t in tools if t["status"] == "PASSED"),
        "tools_invoked_failed":  sum(1 for t in tools if t["status"] == "FAILED"),
        "tools_invoked_skipped": sum(1 for t in tools if t["status"] == "SKIPPED"),
    }


PIPELINE_STAGES = ("explorer", "environment", "coder", "validator")


def aggregate_metrics(records: list[dict]) -> dict:
    """F12: roll each repo's metrics.json (main.py's ``pipeline_metrics``,
    pulled in by extract_validation()) into pass-rate-by-stage and an
    error-distribution table across the whole bench run."""
    stage_attempted = {s: 0 for s in PIPELINE_STAGES}
    stage_completed = {s: 0 for s in PIPELINE_STAGES}
    failures_by_class: dict[str, int] = {}
    guard_retries_total           = 0
    transient_fault_retries_total = 0
    repos_with_metrics = 0

    for r in records:
        pm = (r.get("validation") or {}).get("pipeline_metrics")
        if not pm:
            continue
        repos_with_metrics += 1
        for s in PIPELINE_STAGES:
            if s in pm.get("durations_per_stage", {}):
                stage_attempted[s] += 1
                if s in pm.get("actions_per_stage", {}):
                    stage_completed[s] += 1
        for label, count in pm.get("failures_by_class", {}).items():
            failures_by_class[label] = failures_by_class.get(label, 0) + count
        guard_retries_total += sum(pm.get("guard_retries_per_stage", {}).values())
        transient_fault_retries_total += sum(
            pm.get("transient_fault_retries_per_stage", {}).values()
        )

    return {
        "repos_with_metrics":  repos_with_metrics,
        "stage_completion":    {
            s: f"{stage_completed[s]}/{stage_attempted[s]}" for s in PIPELINE_STAGES
        },
        "failures_by_class":   dict(
            sorted(failures_by_class.items(), key=lambda kv: -kv[1])
        ),
        "guard_retries_total":            guard_retries_total,
        "transient_fault_retries_total":  transient_fault_retries_total,
    }


def write_summary(records: list[dict], out: Path) -> None:
    """Rewrite the markdown summary (called after every finished worker)."""
    lines = [
        f"# Alembic benchmark — {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"Repos processed: {len(records)}",
        "",
        "| Repo | Time | Exit | Syntax | Tests | Tools (P/F/S) | Overall |",
        "|---|---:|---:|---|---|---|---|",
    ]
    for r in sorted(records, key=lambda x: x["repo"]):
        v = r.get("validation") or {}
        not_run = r["exit_code"] is None  # skipped by the availability check
        tools = v.get("tools", [])
        passed  = sum(1 for t in tools if t["status"] == "PASSED")
        failed  = sum(1 for t in tools if t["status"] == "FAILED")
        skipped = sum(1 for t in tools if t["status"] == "SKIPPED")
        if not_run and v.get("error"):
            overall = f"N/A — {v['error']}"
        elif v.get("error"):
            overall = f"ERROR — {v['error']}"
        else:
            overall = v.get("overall", "—")
        lines.append(
            f"| {r['repo']} "
            f"| {r['elapsed_sec']:.0f}s "
            f"| {'—' if not_run else r['exit_code']} "
            f"| {v.get('syntax','—')} "
            f"| {v.get('tests','—')} "
            f"| {passed}/{failed}/{skipped} "
            f"| {overall} |"
        )

    lines += ["", "## Per-repo details"]
    for r in sorted(records, key=lambda x: x["repo"]):
        not_run = r["exit_code"] is None
        lines += [
            "",
            f"### {r['repo']}",
            f"- URL: {r['url']}",
            f"- Duration: {r['elapsed_sec']}s",
            f"- Exit code: {'N/A — pipeline not run' if not_run else r['exit_code']}",
            f"- Log: {r.get('log') or '—'}",
        ]
        v = r.get("validation") or {}
        if not v:
            lines.append("- validation.md unavailable (pipeline exited non-zero)")
            continue
        if v.get("error"):
            lines.append(f"- {v['error']}")
            continue
        for t in v.get("tools", []):
            lines.append(f"  - {t['name']}: {t['status']}")

    agg = aggregate_metrics(records)
    if agg["repos_with_metrics"]:
        lines += ["", "## Aggregate metrics (F12)",
                  "", f"Repos with metrics.json: {agg['repos_with_metrics']}/{len(records)}",
                  "", "**Stage completion (completed/attempted):**", ""]
        for s in PIPELINE_STAGES:
            lines.append(f"- {s}: {agg['stage_completion'][s]}")
        lines += ["", "**Failure taxonomy (tool-invocation failures across all repos):**", ""]
        if agg["failures_by_class"]:
            for label, count in agg["failures_by_class"].items():
                lines.append(f"- {label}: {count}")
        else:
            lines.append("- (none)")
        lines += [
            "",
            f"- Guard retries (write_report/venv nudges) total: {agg['guard_retries_total']}",
            f"- Transient provider-fault retries (F22) total: {agg['transient_fault_retries_total']}",
        ]

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run alembic on N repos in parallel, summarise outcomes.",
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--repos", nargs="+",
                     help="Explicit list of repo URLs to benchmark.")
    src.add_argument("--repos-file", type=Path,
                     help="File with one URL per line ('#' starts a comment).")

    ap.add_argument("--parallel", type=int, default=4,
                    help="How many pipelines to run concurrently (default 4).")
    ap.add_argument("--output", type=Path, default=None,
                    help="Markdown summary path (default: "
                         "benchmarks/alembic/runs/<timestamp>/summary.md).")
    ap.add_argument("--log-dir", type=Path, default=None,
                    help="Per-repo log dir (default: "
                         "benchmarks/alembic/runs/<timestamp>/logs).")
    ap.add_argument("--json-output", type=Path, default=None,
                    help="Optional JSON dump of all per-repo records (default: "
                         "benchmarks/alembic/runs/<timestamp>/summary.json).")
    ap.add_argument("--rebuild-base", action="store_true",
                    help="Force rebuild of alembic-base:latest before workers start.")
    ap.add_argument("--platform", default=None,
                    help="Pass-through to docker --platform (build + run).")
    ap.add_argument("--until", default=None,
                    choices=("explorer", "environment", "coder", "validator"),
                    help="Stop each repo's pipeline after completing this stage "
                         "(forwarded to start_chain --until). E.g. --until "
                         "explorer runs only exploration across all repos. Note: "
                         "for stages before 'validator' there is no validation.md, "
                         "so the summary's tool columns read ERROR while the "
                         "per-stage metrics (durations, stage completion) are "
                         "still collected from metrics.json.")
    ap.add_argument("--skip-availability-check", action="store_true",
                    help="Skip the pre-flight 'git ls-remote' reachability "
                         "check and run the pipeline on every repo as-is.")
    return ap.parse_args()


def main() -> None:
    ns = parse_args()

    # Default all three outputs into one shared, timestamped run folder so
    # ad-hoc invocations self-organize under benchmarks/alembic/runs/
    # instead of scattering files at the project root.
    if ns.output is None or ns.log_dir is None or ns.json_output is None:
        run_dir = RUNS_DIR / datetime.now().strftime("%Y-%m-%d_%H%M%S")
        if ns.output is None:
            ns.output = run_dir / "summary.md"
        if ns.log_dir is None:
            ns.log_dir = run_dir / "logs"
        if ns.json_output is None:
            ns.json_output = run_dir / "summary.json"

    if ns.repos:
        repos = ns.repos
    else:
        repos = [
            line.strip()
            for line in ns.repos_file.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    if not repos:
        sys.exit("[bench] no repos to run")

    ns.log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[bench] {len(repos)} repos, {ns.parallel} parallel workers")
    print(f"[bench] logs   → {ns.log_dir}")
    print(f"[bench] summary→ {ns.output}")

    records: list[dict] = []
    lock = threading.Lock()

    if ns.skip_availability_check:
        available = repos
    else:
        print(f"[bench] checking reachability of {len(repos)} repos "
              f"(git ls-remote, {AVAILABILITY_TIMEOUT}s timeout each)...")
        available = []
        with ThreadPoolExecutor(max_workers=ns.parallel) as pool:
            checks = {pool.submit(check_repo_available, url): url for url in repos}
            for fut in as_completed(checks):
                url = checks[fut]
                ok, reason = fut.result()
                name = get_repo_name(url)
                if ok:
                    available.append(url)
                else:
                    print(f"[bench] ✗ skip    {name}  — unreachable: {reason}",
                          flush=True)
                    records.append({
                        "repo":        name,
                        "url":         url,
                        "elapsed_sec": 0,
                        "exit_code":   None,
                        "log":         None,
                        "validation":  {"error": f"repo unreachable: {reason}"},
                    })
        if not available:
            write_summary(records, ns.output)
            sys.exit("[bench] no reachable repos — nothing to run")
        print(f"[bench] {len(available)}/{len(repos)} repos reachable, "
              f"{len(repos) - len(available)} skipped")

    ensure_base_image(DOCKERFILE, PROJECT_ROOT, platform=ns.platform, rebuild=ns.rebuild_base)

    extra: list[str] = []
    if ns.platform:
        extra += ["--platform", ns.platform]
    if ns.until:
        extra += ["--until", ns.until]

    total = len(available)

    def flush_outputs() -> None:
        with lock:
            write_summary(records, ns.output)
            if ns.json_output:
                # F12: wrap the flat per-repo list with the cross-repo
                # aggregate (stage pass-rates + failure taxonomy) so
                # summary.json alone is enough for a paper table, without
                # re-parsing every repo's metrics.json by hand.
                ns.json_output.write_text(
                    json.dumps(
                        {"repos": records, "aggregate": aggregate_metrics(records)},
                        indent=2, ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

    try:
        with ThreadPoolExecutor(max_workers=ns.parallel) as pool:
            futures = {
                pool.submit(run_one, url, extra, ns.log_dir, i + 1, total): url
                for i, url in enumerate(available)
            }
            for fut in as_completed(futures):
                try:
                    rec = fut.result()
                except Exception as e:
                    url = futures[fut]
                    rec = {
                        "repo":        get_repo_name(url),
                        "url":         url,
                        "elapsed_sec": 0,
                        "exit_code":   -1,
                        "log":         None,
                        "validation":  {"error": f"worker raised {type(e).__name__}: {e}"},
                    }
                with lock:
                    records.append(rec)
                flush_outputs()
    except KeyboardInterrupt:
        print("\n[bench] interrupted by user — partial results saved", file=sys.stderr)

    flush_outputs()
    print(f"\n[bench] done. summary → {ns.output}")
    if ns.json_output:
        print(f"[bench]              json → {ns.json_output}")


if __name__ == "__main__":
    main()
