"""Venv creation and dependency-compatibility checking."""
import asyncio
import json
import shlex
import subprocess
from pathlib import Path

from alembic.config import VENV_COMPAT_TIMEOUT, VENV_SETUP_TIMEOUT
from alembic.tools.paths import COMPAT_CHECK_SCRIPT, output_dir, repo_path
from alembic.tools.shell import record_env_command


def _pip_install(use_uv: bool, python: str, venv_dir: Path, *args: str) -> None:
    """Install into the venv via uv (when available) or the venv's own pip.

    Every subprocess call is bounded — a stalled resolver or a build that
    drops into an interactive prompt must not hang forever.
    """
    cmd = (["uv", "pip", "install", "--python", python, *args]
           if use_uv
           else [str(venv_dir / "bin" / "pip"), "install", *args])
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=VENV_SETUP_TIMEOUT)
    record_env_command(shlex.join(cmd))


async def setup_venv(packages: list[str] | None = None,
                     requirements_file: str | None = None,
                     python_version: str | None = None) -> dict:
    """Create the server .venv and install dependencies.

    Uses `uv` when available, falls back to `python -m venv` + `pip`.
    Always installs `fastmcp`, `pytest`, `mcp` automatically. Editable
    installs are intentionally NOT supported (they almost always fail on
    Cython/C-extension repos) — pass runtime deps via ``packages`` or a
    ``requirements_file`` instead.

    Args:
        packages:          Extra pip-installable package names.
        requirements_file: Path to requirements.txt relative to cloned repo root.
        python_version:    Python version string, e.g. "3.11".

    Examples:
        setup_venv(requirements_file="requirements.txt", python_version="3.11")
        setup_venv(packages=["numpy", "scipy"])
    """
    # Run on a worker thread — see bash()/bash_env() in shell.py for why.
    return await asyncio.to_thread(
        _setup_venv_sync, packages, requirements_file, python_version)


def _setup_venv_sync(packages: list[str] | None,
                     requirements_file: str | None,
                     python_version: str | None) -> dict:
    # LLMs often pass a single package as a bare string, or a comma/space list —
    # coerce so `["fastmcp",...] + packages` never TypeErrors and half-builds the venv.
    if isinstance(packages, str):
        packages = [p for p in packages.replace(",", " ").split() if p]
    out_dir  = output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    venv_dir = out_dir / ".venv"
    python   = str(venv_dir / "bin" / "python")

    use_uv = subprocess.run(["which", "uv"], capture_output=True).returncode == 0

    try:
        if use_uv:
            cmd = ["uv", "venv", str(venv_dir)]
            if python_version:
                cmd += ["--python", python_version]
        else:
            py_bin = f"python{python_version}" if python_version else "python"
            cmd = [py_bin, "-m", "venv", str(venv_dir)]
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=VENV_SETUP_TIMEOUT)
        record_env_command(shlex.join(cmd))
    except subprocess.CalledProcessError as e:
        return {"success": False, "error": f"venv creation failed: {e.stderr.strip()}"}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"venv creation timed out after {VENV_SETUP_TIMEOUT}s"}

    errors = []
    if requirements_file:
        req_path = repo_path() / requirements_file
        if req_path.exists():
            try:
                _pip_install(use_uv, python, venv_dir, "-r", str(req_path))
            except subprocess.CalledProcessError as e:
                errors.append(f"requirements install failed: {e.stderr.strip()}")
            except subprocess.TimeoutExpired:
                errors.append(f"requirements install timed out after {VENV_SETUP_TIMEOUT}s")
        else:
            errors.append(f"requirements file not found: {req_path}")

    install_pkgs = ["fastmcp", "pytest", "mcp"] + (packages or [])
    try:
        _pip_install(use_uv, python, venv_dir, *install_pkgs)
    except subprocess.CalledProcessError as e:
        errors.append(f"package install failed: {e.stderr.strip()}")
    except subprocess.TimeoutExpired:
        errors.append(f"package install timed out after {VENV_SETUP_TIMEOUT}s")

    if errors:
        return {"success": False, "venv": str(venv_dir), "error": "; ".join(errors)}
    return {"success": True, "venv": str(venv_dir), "python": python}


def ensure_pkg(python: str, import_name: str, pip_name: str | None = None) -> str | None:
    """Deterministically make sure ``import_name`` is importable under ``python``,
    installing ``pip_name`` (default = import_name) if not. Called by the env
    gate, not the LLM — a debugger that rebuilds a venv by hand often drops
    pytest (tools venv) or fastmcp (server venv) that setup_venv would add.
    Returns an error string or None."""
    if subprocess.run([python, "-c", f"import {import_name}"], capture_output=True).returncode == 0:
        return None
    cmd = ["uv", "pip", "install", "--python", python, pip_name or import_name]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=VENV_SETUP_TIMEOUT)
    except subprocess.TimeoutExpired:
        return f"{import_name} install timed out"
    if r.returncode != 0:
        return f"{import_name} install failed: {r.stderr.strip()[:300]}"
    record_env_command(shlex.join(cmd))
    return None


def ensure_pytest(python: str) -> str | None:
    return ensure_pkg(python, "pytest")


async def check_venv_compat(venv_name: str = ".venv") -> dict:
    """Check compatibility by replaying the repo's own import statements in a venv.

    Scans the cloned repo's Python files with AST, collects every unique
    `import X` and `from X import Y` where X is an installed package, then
    executes each statement in the venv. This catches both ABI conflicts
    (numpy 1 vs 2) and removed-API errors (e.g. `from transformers import AdamW`
    removed in transformers>=4.38).

    Args:
        venv_name: Directory name of the venv to check, relative to the repo
                   output dir. Default ".venv" (the server venv). Pass
                   ".venv-repo" to check the repo-side venv in the two-venv
                   layout.

    Returns only failures; successful imports are omitted to keep output small.

    Example:
        check_venv_compat()
        check_venv_compat(venv_name=".venv-repo")
    """
    # Run on a worker thread — see bash()/bash_env() in shell.py for why.
    return await asyncio.to_thread(_check_venv_compat_sync, venv_name)


def _check_venv_compat_sync(venv_name: str) -> dict:
    out_dir  = output_dir().resolve()
    repo_dir = repo_path().resolve()
    venv_py  = out_dir / venv_name / "bin" / "python"
    if not venv_py.exists():
        return {"error": f"venv python not found at {venv_py}"}

    try:
        r = subprocess.run(
            [str(venv_py.absolute()), str(COMPAT_CHECK_SCRIPT), str(repo_dir)],
            capture_output=True, text=True, timeout=VENV_COMPAT_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"compat check timed out after {VENV_COMPAT_TIMEOUT}s"}
    if r.returncode != 0:
        return {"error": f"compat check script failed: {r.stderr.strip()[:500]}"}

    try:
        data = json.loads(r.stdout.strip())
    except Exception:
        return {"error": f"could not parse compat output: {r.stdout[:300]}"}

    conflicts = data.get("conflicts", {})
    return {
        "conflicts": conflicts,
        "checked": data.get("checked", 0),
        "has_conflicts": bool(conflicts),
    }
