#!/usr/bin/env python3
"""Build an isolated MCP tool container from a GitHub repository.

Flow:
  1. Build ``alembic-base:latest`` (Dockerfile under ``docker/alembic/``)
     if it is not already present locally.
  2. Run a *build* container that executes the alembic pipeline against
     ``<repo_url>``. The pipeline clones, sets up a venv, generates
     ``server.py`` and validates it — all inside the container.
  3. On successful exit, ``docker commit`` the container to
     ``alembic-tool:<repo-name>``. The "improved" image now carries the
     cloned repo, its venv and the generated FastMCP server.
  4. Launch the committed image with a random host port mapped to the
     container's ``$MCP_PORT`` so the MCP server is reachable from the host.

The actual Docker steps live in ``alembic.builder`` (shared with the
CoScientist AlembicAgent toolset); this module is the thin CLI over them.

Run from anywhere:
    python CoScientist/alembic/start_chain.py <repo_url>
"""
from __future__ import annotations

import argparse
import sys

# Import the stdlib ``logging`` NOW, while it still resolves to the stdlib —
# putting ``CoScientist/`` on sys.path below makes ``CoScientist/logging``
# shadow it, and ``builder`` -> ``dotenv`` does ``import logging`` transitively.
# Caching the stdlib module in sys.modules first keeps that import correct.
import logging  # noqa: F401
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from alembic.builder import (
    AlembicBuildError,
    DEFAULT_ENV_FILE,
    build_image,
    default_platform,
    ensure_base,
    serve_image,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="start_chain",
        description="Build a fully isolated MCP tool container from a GitHub repo.",
    )
    ap.add_argument("repo_url", help="GitHub repository URL")
    ap.add_argument("--rebuild-base", action="store_true",
                    help="Force rebuild of alembic-base:latest")
    ap.add_argument("--gpus", default=None,
                    help='Docker --gpus value, e.g. "all". Off by default.')
    ap.add_argument("--platform", default=None,
                    help='Docker --platform value, e.g. "linux/amd64". '
                         'Auto-set to linux/amd64 on Apple Silicon so x86-only '
                         'wheels (dgl, old torchvision, etc.) run via Rosetta. '
                         'Pass "native" to force the host architecture.')
    ap.add_argument("--resume", default=None,
                    choices=("explorer", "environment", "coder", "validator"),
                    help="Resume the alembic pipeline from a specific stage")
    ap.add_argument("--no-serve", action="store_true",
                    help="Build and commit only; do not launch the MCP server")
    ap.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE,
                    help=f"Path to .env to inject (default: {DEFAULT_ENV_FILE})")
    return ap.parse_args()


def main() -> None:
    ns = parse_args()
    if ns.platform is None:
        ns.platform = default_platform()
        if ns.platform:
            print(f"[start-chain] Apple Silicon detected — defaulting to "
                  f"--platform {ns.platform} (Rosetta). "
                  f"Pass --platform native to override.")
    elif ns.platform == "native":
        ns.platform = None

    ensure_base(platform=ns.platform, rebuild=ns.rebuild_base)

    try:
        build = build_image(
            ns.repo_url, env_file=ns.env_file, platform=ns.platform,
            gpus=ns.gpus, resume=ns.resume,
        )
    except AlembicBuildError as exc:
        sys.stderr.write(f"\n[start-chain] {exc}\n")
        sys.exit(1)

    if ns.no_serve:
        print(f"[start-chain] built {build.image} (serve skipped).")
        return

    try:
        served = serve_image(
            ns.repo_url, build.image, env_file=ns.env_file,
            platform=ns.platform, gpus=ns.gpus,
        )
    except AlembicBuildError as exc:
        sys.stderr.write(f"\n[start-chain] {exc}\n")
        sys.exit(1)

    print(
        "\n[start-chain] MCP server up.\n"
        f"  image     : {served.image}\n"
        f"  container : {served.container}\n"
        f"  url       : {served.url}\n"
        f"  logs      : docker logs -f {served.container}\n"
        f"  stop      : docker stop {served.container} && docker rm {served.container}\n"
        f"  relaunch  : docker run -d -p <port>:8000 {served.image} serve {ns.repo_url}"
    )


if __name__ == "__main__":
    main()
