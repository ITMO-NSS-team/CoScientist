"""Shell-execution tools exposed to the agents: bash / bash_env."""
import asyncio
import subprocess
from pathlib import Path

from alembic.tools.paths import MAX_BYTES


def _glob_command(stripped: str) -> dict | None:
    """Handle the custom ``glob <pattern>`` shortcut. Returns None if not glob."""
    first = stripped.split()[0] if stripped else ""
    if first != "glob":
        return None
    parts = stripped.split(None, 1)
    if len(parts) < 2:
        return {"error": "glob requires a pattern argument."}
    pattern = parts[1]
    if pattern.startswith("/"):
        root, pat = Path("/"), pattern.lstrip("/")
    else:
        root, pat = Path("."), pattern
    matched = sorted(str(p) for p in root.glob(pat))
    return {"matches": matched}


def _run_shell(command: str, timeout: int) -> dict:
    """Shared body for ``bash`` / ``bash_env``: glob shortcut + shell run."""
    stripped = command.strip()
    if not stripped:
        return {"error": "empty command"}

    glob_result = _glob_command(stripped)
    if glob_result is not None:
        return glob_result

    try:
        result = subprocess.run(
            stripped, shell=True, capture_output=True, text=True, timeout=timeout,
        )
        output = result.stdout
        if result.returncode != 0 and result.stderr:
            output += "\n[stderr] " + result.stderr
        return {"output": output[:MAX_BYTES]}
    except subprocess.TimeoutExpired:
        return {"error": f"Command timed out after {timeout} seconds."}


async def bash(command: str) -> dict:
    """Run a shell command with a 15 s timeout.

    The pipeline is intended to run inside an ephemeral container, so any
    command line is accepted — the container is the security boundary. The
    custom ``glob <pattern>`` shortcut is still recognised as a convenience.

    Examples:
        bash("ls .alembic/massformer/repos")
        bash("grep -r 'def train' .alembic/massformer/repos -l")
        bash("head -n 30 .alembic/massformer/repos/README.md")
        bash("glob .alembic/massformer/repos/**/*.yaml")
        bash("python -m py_compile .alembic/massformer/output/server.py && echo OK")
    """
    # F23: offload the blocking subprocess.run to a worker thread — ADK calls
    # non-async tools with a plain synchronous call (no run_in_executor), so a
    # sync bash()/bash_env() here would freeze the whole event loop for the
    # duration of the command, silently defeating any asyncio.wait_for-based
    # timeout (per-debugger-call and per-stage alike) wrapping this turn.
    from alembic.main import BASH_TIMEOUT  # deferred: see main.py's timeout block
    return await asyncio.to_thread(_run_shell, command, BASH_TIMEOUT)


async def bash_env(command: str) -> dict:
    """Run a shell command with a 900 s timeout — for slow installs and downloads.

    Same semantics as ``bash``, just a longer timeout so package managers
    (pip / uv / apt-get / conda) have time to download and build, and so a
    pretrained-model-weight download (huggingface-cli / hf_hub_download,
    potentially multi-GB) has room to finish (F6). Inherits this process's
    full environment, so HF_TOKEN (if set) is automatically visible to any
    huggingface_hub/huggingface-cli call run through this tool — never pass
    it explicitly on the command line.

    Examples:
        bash_env("uv venv .alembic/massformer/output/.venv --python 3.11")
        bash_env("uv pip install --python .alembic/massformer/output/.venv/bin/python torch torchvision")
        bash_env("pip install -r .alembic/massformer/repos/requirements.txt")
        bash_env("which python3")
        # System libs (container only, /var/lib/apt/lists is empty):
        bash_env("apt-get update && apt-get install -y --no-install-recommends libpoppler-cpp-dev")
        # Pretrained weights (F6) — HF_TOKEN used automatically, never inline it:
        bash_env("huggingface-cli download MahmoodLab/UNI2-h --local-dir .alembic/UNI/repos/checkpoints")
    """
    from alembic.main import BASH_ENV_TIMEOUT  # deferred: see main.py's timeout block
    return await asyncio.to_thread(_run_shell, command, BASH_ENV_TIMEOUT)
