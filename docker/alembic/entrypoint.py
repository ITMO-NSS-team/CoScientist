#!/usr/bin/env python3
"""Alembic container entrypoint.

Modes:
  entrypoint.py build <repo_url> [--resume <stage>] [--until <stage>]
      Runs the alembic pipeline; intended to be followed by `docker commit`.
  entrypoint.py serve <repo_url>
      Starts the generated FastMCP server on $MCP_PORT (HTTP transport).
  entrypoint.py shell
      Drops into bash for debugging.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

USAGE = (
    "Usage:\n"
    "  entrypoint.py build <repo_url> [--resume <stage>] [--until <stage>]\n"
    "  entrypoint.py serve <repo_url>\n"
    "  entrypoint.py shell\n"
)


def _usage(code: int = 64) -> None:
    sys.stderr.write(USAGE)
    sys.exit(code)


def _repo_name(repo_url: str) -> str:
    return repo_url.rstrip("/").split("/")[-1].removesuffix(".git")


def _run_build(args: list[str]) -> None:
    if not args:
        _usage()
    os.chdir("/work")
    os.environ["PYTHONPATH"] = "/app:" + os.environ.get("PYTHONPATH", "")
    os.execvp("python", ["python", "-m", "alembic.main", *args])


def _run_serve(args: list[str]) -> None:
    if not args:
        _usage()
    repo_url = args[0]
    workdir  = Path(os.environ.get("ALEMBIC_WORKDIR", "/work/.alembic"))
    output   = workdir / _repo_name(repo_url) / "output"
    server   = output / "server.py"
    venv_py  = output / ".venv" / "bin" / "python"

    if not server.exists():
        sys.stderr.write(f"[entrypoint] No server.py found at {server}\n")
        sys.exit(1)
    if not venv_py.exists() or not os.access(venv_py, os.X_OK):
        sys.stderr.write(f"[entrypoint] No venv python at {venv_py}\n")
        sys.exit(1)

    port = os.environ.get("MCP_PORT", "8000")
    print(f"[entrypoint] Starting MCP server: {server} on 0.0.0.0:{port}",
          flush=True)

    os.chdir(output)
    env = os.environ.copy()
    env["SERVER_PATH"] = str(server)
    os.execve(str(venv_py),
              [str(venv_py), "/usr/local/bin/serve.py"],
              env)


def main() -> None:
    if len(sys.argv) < 2:
        _usage()

    mode, *args = sys.argv[1:]

    if mode == "build":
        _run_build(args)
    elif mode == "serve":
        _run_serve(args)
    elif mode == "shell":
        os.execvp("/bin/bash", ["/bin/bash"])
    elif mode in ("help", "--help", "-h"):
        _usage(0)
    else:
        sys.stderr.write(f"[entrypoint] Unknown mode: {mode}\n")
        _usage()


if __name__ == "__main__":
    main()
