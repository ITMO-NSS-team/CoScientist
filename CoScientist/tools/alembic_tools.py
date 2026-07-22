"""ADK function tools wrapping the Alembic pipeline (GitHub repo → MCP server).

Alembic (CoScientist/alembic) builds a validated FastMCP tool server from a
scientific repository inside Docker. A full build takes tens of minutes, so the
tools here are job-based: ``build_mcp_server`` launches ``start_chain.py`` as a
host-side background subprocess and returns a ``job_id``; ``check_mcp_build``
reports progress (current pipeline stage, log tail) and, once the serve
container is up, the resulting MCP endpoint URL.

Jobs are process-wide (like the coder's local job registry), so over A2A —
where every orchestrator delegation is a fresh session — a later delegation can
find and continue an earlier build via ``list_mcp_builds``.
"""
from __future__ import annotations

import re
import secrets
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from google.adk.tools import ToolContext

# /<root>/CoScientist/tools/alembic_tools.py -> /<root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]
START_CHAIN = PROJECT_ROOT / "CoScientist" / "alembic" / "start_chain.py"
# Host-side stdout logs of the build subprocesses (the pipeline's own logs live
# inside the build container; this is the start_chain wrapper output).
LOG_DIR = PROJECT_ROOT / ".alembic" / "a2a_builds"

_LOG_TAIL_LINES = 15
_MAX_JOBS = 200  # cap registry size; evict oldest finished jobs past this

_JOBS: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()

# Patterns over start_chain.py / alembic.main output.
_URL_RE = re.compile(r"url\s*:\s*(http://\S+/mcp)")
_IMAGE_RE = re.compile(r"image\s*:\s*(\S+)")
_CONTAINER_RE = re.compile(r"container\s*:\s*(\S+)")
_STAGE_RE = re.compile(r"STAGE (\d) — (\S+)")


def _repo_name(repo_url: str) -> str:
    """Last path segment of a repo URL, without a trailing ``.git``
    (same rule as alembic.common.get_repo_name, kept local so importing this
    module never touches the alembic package's top-level path setup)."""
    return re.sub(r"\.git$", "", repo_url.rstrip("/").split("/")[-1])


def _evict_finished_jobs() -> None:
    if len(_JOBS) <= _MAX_JOBS:
        return
    for job_id, rec in list(_JOBS.items()):
        if rec["status"] != "running":
            del _JOBS[job_id]
        if len(_JOBS) <= _MAX_JOBS:
            return


def _read_log(rec: Dict[str, Any]) -> str:
    try:
        return Path(rec["log_file"]).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _finalize(rec: Dict[str, Any], returncode: int) -> None:
    """Parse the finished build's log into the job record (under _LOCK)."""
    text = _read_log(rec)
    rec["returncode"] = returncode
    rec["finished_at"] = time.time()
    if returncode == 0:
        url = _URL_RE.search(text)
        image = _IMAGE_RE.search(text)
        container = _CONTAINER_RE.search(text)
        rec["status"] = "done"
        rec["mcp_url"] = url.group(1) if url else None
        rec["image"] = image.group(1) if image else None
        rec["container"] = container.group(1) if container else None
    else:
        rec["status"] = "failed"


def _runner(rec: Dict[str, Any]) -> None:
    log_path = Path(rec["log_file"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                [sys.executable, str(START_CHAIN), rec["repo_url"]],
                stdout=log, stderr=subprocess.STDOUT, cwd=PROJECT_ROOT,
            )
            with _LOCK:
                rec["pid"] = proc.pid
            returncode = proc.wait()
    except OSError as exc:  # docker/python missing, log dir unwritable, ...
        with _LOCK:
            rec["status"] = "failed"
            rec["error"] = f"could not launch the build subprocess: {exc}"
            rec["finished_at"] = time.time()
        return
    with _LOCK:
        _finalize(rec, returncode)


def _snapshot(rec: Dict[str, Any], with_log_tail: bool = True) -> Dict[str, Any]:
    """The tool-facing view of one job record (call under _LOCK)."""
    out = {
        "job_id": rec["job_id"],
        "repo_url": rec["repo_url"],
        "status": rec["status"],
        "elapsed_seconds": round((rec.get("finished_at") or time.time()) - rec["started_at"]),
    }
    text = _read_log(rec) if (with_log_tail or rec["status"] != "running") else ""
    stages = _STAGE_RE.findall(text)
    if stages:
        out["stage"] = f"{stages[-1][0]}/5 {stages[-1][1]}"
    if rec["status"] == "running":
        if with_log_tail:
            out["log_tail"] = "\n".join(text.splitlines()[-_LOG_TAIL_LINES:])
        out["note"] = ("The build is still running (a full build takes tens of "
                       "minutes). Do other work and call "
                       f"check_mcp_build('{rec['job_id']}') again later — do not "
                       "poll in a tight loop.")
    elif rec["status"] == "done":
        out["mcp_url"] = rec.get("mcp_url")
        out["image"] = rec.get("image")
        out["container"] = rec.get("container")
        if not rec.get("mcp_url"):
            out["note"] = ("Build finished but no MCP URL was printed — the image "
                           f"{rec.get('image') or 'alembic-tool:<repo>'} was likely "
                           "built with serving skipped; check the log.")
    else:
        out["error"] = rec.get("error") or "\n".join(text.splitlines()[-_LOG_TAIL_LINES:])
    return out


async def build_mcp_server(
    repo_url: str,
    force_rebuild: bool = False,
    tool_context: Optional[ToolContext] = None,
) -> Dict[str, Any]:
    """Start an Alembic build: turn a GitHub repository into a served MCP tool
    server (clone → env → generated+validated tools → FastMCP server in Docker).

    The build runs in the background. This returns immediately with a job_id;
    track it with check_mcp_build(job_id).

    Args:
        repo_url: GitHub repository URL, e.g. "https://github.com/whitead/synspace".
        force_rebuild: Start a fresh build even if this repo already has a
            finished (or running) build in this process.

    Returns:
        status "running" with the job_id to check later; or the existing job for
        this repo (already running/done) unless force_rebuild is set.
    """
    repo_url = (repo_url or "").strip()
    if not re.match(r"^(https?://|git@)\S+/\S+", repo_url):
        return {"status": "error",
                "error": f"repo_url does not look like a git repository URL: {repo_url!r}"}

    with _LOCK:
        if not force_rebuild:
            # Prefer a live build; else the most recent finished one.
            same = [r for r in _JOBS.values() if r["repo_url"] == repo_url]
            for rec in reversed(same):
                if rec["status"] == "running":
                    snap = _snapshot(rec, with_log_tail=False)
                    snap["note"] = ("A build for this repository is already running — "
                                    f"reusing it. Track it with check_mcp_build('{rec['job_id']}').")
                    return snap
            for rec in reversed(same):
                if rec["status"] == "done":
                    snap = _snapshot(rec)
                    snap["note"] = ("This repository was already built in this process — "
                                    "reusing the result. Pass force_rebuild=true to rebuild.")
                    return snap

        job_id = f"{_repo_name(repo_url)}-{secrets.token_hex(3)}"
        rec: Dict[str, Any] = {
            "job_id": job_id,
            "repo_url": repo_url,
            "status": "running",
            "started_at": time.time(),
            "log_file": str(LOG_DIR / f"{job_id}.log"),
        }
        _evict_finished_jobs()
        _JOBS[job_id] = rec

    threading.Thread(target=_runner, args=(rec,), daemon=True,
                     name=f"alembic-build-{job_id}").start()
    return {
        "status": "running",
        "job_id": job_id,
        "repo_url": repo_url,
        "note": ("Build started (base image → pipeline → docker commit → serve). "
                 "A full build takes tens of minutes: report the job_id back, do "
                 f"other work, and call check_mcp_build('{job_id}') later."),
    }


async def check_mcp_build(
    job_id: str, tool_context: Optional[ToolContext] = None
) -> Dict[str, Any]:
    """Check an Alembic build started by build_mcp_server.

    Args:
        job_id: The id returned by build_mcp_server.

    Returns:
        status "running" with the current pipeline stage and a log tail;
        "done" with the served MCP endpoint (mcp_url), image and container;
        or "failed" with the error tail of the build log.
    """
    with _LOCK:
        rec = _JOBS.get(job_id)
        if rec is None:
            return {"status": "error",
                    "error": f"unknown job_id {job_id!r} — use list_mcp_builds() "
                             "to see the builds known to this process."}
        return _snapshot(rec)


async def list_mcp_builds(tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
    """List every Alembic build known to this process (running and finished).

    Use this to recover a lost job_id or to find an MCP server that was already
    built for a repository in an earlier delegation/session.

    Returns:
        builds: one summary per job (job_id, repo_url, status, stage/mcp_url).
    """
    with _LOCK:
        return {"builds": [_snapshot(rec, with_log_tail=False)
                           for rec in _JOBS.values()]}


ALEMBIC_TOOLS = [build_mcp_server, check_mcp_build, list_mcp_builds]

__all__ = ["ALEMBIC_TOOLS", "build_mcp_server", "check_mcp_build", "list_mcp_builds"]
