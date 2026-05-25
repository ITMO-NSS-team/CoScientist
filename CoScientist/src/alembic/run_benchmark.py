#!/usr/bin/env python3
"""Benchmark the alembic pipeline against a list of repos in parallel.

The base image (``alembic-base:latest``) is built once up-front, then N
workers run ``start_chain.py --no-serve`` concurrently — emulating a
production setting where the base image is pre-baked on a CI host. Servers
are not launched; the produced ``alembic-tool:<repo>`` images stay so you
can ``docker run`` them later.

Usage:
    # parallel run, default 4 workers
    python CoScientist/src/alembic/run_benchmark.py \\
        --repos https://github.com/Roestlab/massformer \\
                https://github.com/whitead/synspace \\
                https://github.com/CrystalEye42/OpenChemIE

    # from a file (one URL per line, '#' = comment), 8 workers, JSON dump
    python CoScientist/src/alembic/run_benchmark.py \\
        --repos-file repos.txt \\
        --parallel 8 \\
        --json-output bench.json
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

ROOT        = Path(__file__).resolve().parents[3]
START_CHAIN = Path(__file__).resolve().parent / "start_chain.py"
DOCKERFILE  = ROOT / "docker" / "alembic" / "Dockerfile"
BASE_IMAGE  = "alembic-base:latest"


def _repo_name(url: str) -> str:
    return re.sub(r"\.git$", "", url.rstrip("/").split("/")[-1])


def _image_exists(name: str) -> bool:
    r = subprocess.run(
        ["docker", "image", "inspect", name],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return r.returncode == 0


def ensure_base(platform: str | None, rebuild: bool) -> None:
    """Build alembic-base:latest once before fanning out workers."""
    if not rebuild and _image_exists(BASE_IMAGE):
        print(f"[bench] base image {BASE_IMAGE} present — reusing.")
        return
    cmd = ["docker", "build"]
    if platform:
        cmd += ["--platform", platform]
    cmd += ["-t", BASE_IMAGE, "-f", str(DOCKERFILE), str(ROOT)]
    print(f"[bench] building base image: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(r.returncode)


def run_one(repo_url: str, extra_args: list[str], log_dir: Path,
            idx: int, total: int) -> dict:
    """Invoke start_chain.py --no-serve for one repo; stream logs to a file."""
    name     = _repo_name(repo_url)
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
        "validation":   None,
    }
    if r.returncode == 0:
        record["validation"] = extract_validation(name)

    status = "ok" if r.returncode == 0 else f"exit={r.returncode}"
    print(f"[bench] ↓ done   {name}  ({elapsed:.0f}s, {status})",
          flush=True)
    return record


def extract_validation(repo: str) -> dict:
    """Pull validation.md out of the just-committed image and parse it."""
    image = f"alembic-tool:{repo}"
    path  = f"/work/.alembic/{repo}/reports/validation.md"
    r = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "cat", image, path],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return {"error": "validation.md not readable", "stderr": r.stderr[-300:]}
    return parse_validation(r.stdout)


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
        m = re.match(r"^-\s+\*\*(.+?)\*\*\s+—\s+(\w+)", line.strip())
        if m:
            tools.append({"name": m.group(1), "status": m.group(2)})

    return {
        "syntax":  first_line(sections.get("Syntax & Imports", "")),
        "tests":   first_line(sections.get("Tests", "")),
        "overall": first_line(sections.get("Overall", "")),
        "tools":   tools,
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
        tools = v.get("tools", [])
        passed  = sum(1 for t in tools if t["status"] == "PASSED")
        failed  = sum(1 for t in tools if t["status"] == "FAILED")
        skipped = sum(1 for t in tools if t["status"] == "SKIPPED")
        lines.append(
            f"| {r['repo']} "
            f"| {r['elapsed_sec']:.0f}s "
            f"| {r['exit_code']} "
            f"| {v.get('syntax','—')} "
            f"| {v.get('tests','—')} "
            f"| {passed}/{failed}/{skipped} "
            f"| {v.get('overall','—')} |"
        )

    lines += ["", "## Per-repo details"]
    for r in sorted(records, key=lambda x: x["repo"]):
        lines += [
            "",
            f"### {r['repo']}",
            f"- URL: {r['url']}",
            f"- Duration: {r['elapsed_sec']}s",
            f"- Exit code: {r['exit_code']}",
            f"- Log: {r.get('log','—')}",
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
    ap.add_argument("--output", type=Path,
                    default=ROOT / "alembic_bench.md",
                    help="Markdown summary path (default: ./alembic_bench.md).")
    ap.add_argument("--log-dir", type=Path,
                    default=ROOT / "alembic_bench_logs",
                    help="Per-repo log dir (default: ./alembic_bench_logs).")
    ap.add_argument("--json-output", type=Path, default=None,
                    help="Optional JSON dump of all per-repo records.")
    ap.add_argument("--rebuild-base", action="store_true",
                    help="Force rebuild of alembic-base:latest before workers start.")
    ap.add_argument("--platform", default=None,
                    help="Pass-through to docker --platform (build + run).")
    return ap.parse_args()


def main() -> None:
    ns = parse_args()

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

    ensure_base(ns.platform, ns.rebuild_base)

    extra: list[str] = []
    if ns.platform:
        extra += ["--platform", ns.platform]

    records: list[dict] = []
    lock = threading.Lock()
    total = len(repos)

    def flush_outputs() -> None:
        with lock:
            write_summary(records, ns.output)
            if ns.json_output:
                ns.json_output.write_text(
                    json.dumps(records, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

    try:
        with ThreadPoolExecutor(max_workers=ns.parallel) as pool:
            futures = {
                pool.submit(run_one, url, extra, ns.log_dir, i + 1, total): url
                for i, url in enumerate(repos)
            }
            for fut in as_completed(futures):
                try:
                    rec = fut.result()
                except Exception as e:
                    url = futures[fut]
                    rec = {
                        "repo":        _repo_name(url),
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
