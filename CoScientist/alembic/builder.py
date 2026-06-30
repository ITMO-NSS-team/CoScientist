"""Programmatic driver for the alembic Docker build → commit → serve flow.

This is the single source of truth for turning a scientific GitHub repo into a
running FastMCP server image. It mirrors what a human does with
``start_chain.py`` but exposes plain functions that **return structured results
and raise on failure** instead of calling ``sys.exit`` — so it can be driven by
the CLI *and* by the CoScientist AlembicAgent toolset
(``CoScientist/tools/alembic_tools.py``).

Everything runs through ``docker``: the container is the security boundary (the
pipeline agents may run arbitrary shell), so the host process here only ever
shells out to the Docker CLI and never imports the in-container pipeline code.

Imports stay RELATIVE (``from .common import ...``) so this module works whether
the package is reached as the top-level ``alembic`` (script / in-container) or as
``CoScientist.alembic`` (in-process from the A2A server).
"""
from __future__ import annotations

import os
import platform as _platform
import random
import secrets
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dotenv import dotenv_values

from .common import BASE_IMAGE, ensure_base_image, get_repo_name

# /<root>/CoScientist/alembic/builder.py -> /<root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DOCKERFILE = PROJECT_ROOT / "docker" / "alembic" / "Dockerfile"
TOOL_REPO = "alembic-tool"
PORT_RANGE = (20000, 30000)
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"

# Reports the pipeline writes inside the container; we read them out before the
# build container is dropped so callers can see what the server exposes.
_CONTAINER_WORKDIR = os.environ.get("ALEMBIC_WORKDIR", "/work/.alembic")

# Env vars forwarded into both the build and serve containers (the build needs
# the LLM keys; serve needs whatever the generated server reads at runtime).
PASSTHROUGH_ENV = (
    "OPENROUTER_API_KEY", "OPENAI_API_KEY", "TAVILY_API_KEY",
    "GOOGLE_API_KEY", "GEMINI_API_KEY",
    "MODEL", "MCP_URLS", "OR_APP_NAME", "FEDOTMAS_DEFAULT_MODEL",
)


class AlembicBuildError(RuntimeError):
    """A docker step in the build/serve flow exited non-zero."""


@dataclass
class BuildResult:
    """Outcome of building (and committing) the tool image for a repo."""

    repo: str
    image: str
    # Reports the in-container pipeline produced (best-effort; "" if unread).
    validation_report: str = ""
    server_report: str = ""


@dataclass
class ServeResult:
    """A running MCP-server container for a built repo image."""

    repo: str
    image: str
    container: str
    port: int
    url: str


@dataclass
class BuildAndServeResult:
    build: BuildResult
    serve: Optional[ServeResult] = None
    # docker stdio captured from the failing step, when something went wrong.
    logs: List[str] = field(default_factory=list)


def default_platform() -> Optional[str]:
    """``linux/amd64`` on Apple Silicon so old x86-only wheels (dgl 0.9, etc.)
    pull and run via Rosetta; native everywhere else (``None`` = no flag)."""
    if sys.platform == "darwin" and _platform.machine() in ("arm64", "aarch64"):
        return "linux/amd64"
    return None


def _run(cmd: List[str], **kw) -> subprocess.CompletedProcess:
    print(f"[alembic-builder] $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, **kw)


def _random_port() -> int:
    return random.randint(*PORT_RANGE)


def _env_args(env_file: Optional[Path]) -> List[str]:
    args: List[str] = []
    if env_file and Path(env_file).exists():
        for k, v in dotenv_values(env_file).items():
            if v is not None:
                args += ["-e", f"{k}={v}"]
    for var in PASSTHROUGH_ENV:
        if var in os.environ:
            args += ["-e", f"{var}={os.environ[var]}"]
    return args


def ensure_base(*, platform: Optional[str] = None, rebuild: bool = False) -> None:
    """Build ``alembic-base:latest`` once if it is missing (or forced)."""
    ensure_base_image(BASE_DOCKERFILE, PROJECT_ROOT, platform=platform, rebuild=rebuild)


def _read_container_report(container: str, repo: str, name: str) -> str:
    """Best-effort ``cat`` of ``reports/<name>`` inside the build container."""
    path = f"{_CONTAINER_WORKDIR}/{repo}/reports/{name}"
    r = _run(
        ["docker", "exec", container, "sh", "-c", f"cat {path} 2>/dev/null"],
        capture_output=True, text=True,
    )
    return r.stdout if r.returncode == 0 else ""


def build_image(
    repo_url: str,
    *,
    env_file: Optional[Path] = DEFAULT_ENV_FILE,
    platform: Optional[str] = None,
    gpus: Optional[str] = None,
    resume: Optional[str] = None,
) -> BuildResult:
    """Run the alembic pipeline in a container and ``docker commit`` the result.

    Returns a :class:`BuildResult` (incl. the validation/server reports the
    pipeline wrote). Raises :class:`AlembicBuildError` if the pipeline or commit
    fails; the failed build container is kept for inspection.
    """
    repo = get_repo_name(repo_url)
    cname = f"alembic-build-{repo}-{secrets.token_hex(3)}"
    tool_image = f"{TOOL_REPO}:{repo}"

    cmd = ["docker", "run", "--name", cname]
    if platform:
        cmd += ["--platform", platform]
    if gpus:
        cmd += ["--gpus", gpus]
    cmd += _env_args(env_file)
    cmd += [BASE_IMAGE, "build", repo_url]
    if resume:
        cmd += ["--resume", resume]

    r = _run(cmd)
    if r.returncode != 0:
        raise AlembicBuildError(
            f"alembic pipeline failed (exit {r.returncode}). "
            f"Build container kept for inspection: {cname} "
            f"(docker logs {cname})."
        )

    # Capture the inter-agent reports BEFORE we drop the build container.
    validation_report = _read_container_report(cname, repo, "validation.md")
    server_report = _read_container_report(cname, repo, "server.md")

    # ── Pre-commit cleanup — keep secrets out of the saved image ──────
    # 1. Wipe pipeline.log: agent stderr may have echoed API keys.
    _run(
        ["docker", "exec", cname, "sh", "-c",
         "rm -f /work/.alembic/*/pipeline.log"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    # 2. Blank every sensitive env var in the committed image's Config.Env so
    #    `docker inspect <image>` never exposes a key passed at build time. The
    #    serve container still receives real values via its own --env-file.
    keys_to_scrub = set(PASSTHROUGH_ENV)
    if env_file and Path(env_file).exists():
        for line in Path(env_file).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                keys_to_scrub.add(line.split("=", 1)[0].strip())
    change_args: List[str] = []
    for key in sorted(keys_to_scrub):
        change_args += ["--change", f"ENV {key}="]

    print(f"[alembic-builder] committing {cname} -> {tool_image}")
    c = _run(["docker", "commit", *change_args, cname, tool_image])
    if c.returncode != 0:
        raise AlembicBuildError(
            f"docker commit failed (exit {c.returncode}); container kept: {cname}"
        )
    _run(["docker", "rm", cname],
         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return BuildResult(
        repo=repo,
        image=tool_image,
        validation_report=validation_report,
        server_report=server_report,
    )


def serve_image(
    repo_url: str,
    tool_image: str,
    *,
    env_file: Optional[Path] = DEFAULT_ENV_FILE,
    platform: Optional[str] = None,
    gpus: Optional[str] = None,
    host: str = "localhost",
) -> ServeResult:
    """Launch a committed tool image and map its MCP port to a random host port.

    Raises :class:`AlembicBuildError` if the container fails to start.
    """
    repo = get_repo_name(repo_url)
    port = _random_port()
    cname = f"alembic-serve-{repo}-{secrets.token_hex(3)}"

    cmd = ["docker", "run", "-d", "--name", cname, "-p", f"{port}:8000"]
    if platform:
        cmd += ["--platform", platform]
    if gpus:
        cmd += ["--gpus", gpus]
    cmd += _env_args(env_file)
    cmd += [tool_image, "serve", repo_url]

    r = _run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise AlembicBuildError(
            f"failed to start MCP server container (exit {r.returncode}): "
            f"{(r.stderr or '').strip()}"
        )
    return ServeResult(
        repo=repo,
        image=tool_image,
        container=cname,
        port=port,
        url=f"http://{host}:{port}/mcp",
    )


def stop_container(container: str) -> None:
    """Stop and remove a serve container (best-effort)."""
    _run(["docker", "stop", container],
         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _run(["docker", "rm", container],
         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def build_and_serve(
    repo_url: str,
    *,
    serve: bool = True,
    env_file: Optional[Path] = DEFAULT_ENV_FILE,
    platform: Optional[str] = None,
    gpus: Optional[str] = None,
    resume: Optional[str] = None,
    rebuild_base: bool = False,
    host: str = "localhost",
) -> BuildAndServeResult:
    """End-to-end: ensure the base image, build+commit the repo, optionally serve.

    A convenience wrapper used by both the CLI and the AlembicAgent toolset.
    """
    if platform is None:
        platform = default_platform()
    ensure_base(platform=platform, rebuild=rebuild_base)
    build = build_image(
        repo_url, env_file=env_file, platform=platform, gpus=gpus, resume=resume
    )
    served = None
    if serve:
        served = serve_image(
            repo_url, build.image, env_file=env_file, platform=platform,
            gpus=gpus, host=host,
        )
    return BuildAndServeResult(build=build, serve=served)


__all__ = [
    "AlembicBuildError",
    "BuildAndServeResult",
    "BuildResult",
    "ServeResult",
    "build_and_serve",
    "build_image",
    "default_platform",
    "ensure_base",
    "serve_image",
    "stop_container",
]
