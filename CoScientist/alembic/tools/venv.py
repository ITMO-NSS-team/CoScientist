"""Venv creation and dependency-compatibility checking."""
import asyncio
import json
import subprocess
from pathlib import Path

from alembic.tools.paths import COMPAT_CHECK_SCRIPT, output_dir, repo_path


def _pip_install(use_uv: bool, python: str, venv_dir: Path, *args: str) -> None:
    """Install into the venv via uv (when available) or the venv's own pip."""
    cmd = (["uv", "pip", "install", "--python", python, *args]
           if use_uv
           else [str(venv_dir / "bin" / "pip"), "install", *args])
    subprocess.run(cmd, check=True, capture_output=True, text=True)


async def setup_venv(repo_url: str, packages: list[str] | None = None,
               requirements_file: str | None = None,
               pyproject_toml: str | None = None,
               python_version: str | None = None) -> dict:
    """Create a .venv in the output directory and install dependencies.

    Uses `uv` when available, falls back to `python -m venv` + `pip`.
    Always installs `mcp` and `pytest` automatically.

    Args:
        repo_url:          Repository URL.
        packages:          Extra pip-installable package names.
        requirements_file: Path to requirements.txt relative to cloned repo root.
        pyproject_toml:    Path to pyproject.toml relative to cloned repo root.
        python_version:    Python version string, e.g. "3.11".

    Examples:
        setup_venv("https://github.com/Roestlab/massformer",
                   requirements_file="requirements.txt")
        setup_venv("https://github.com/Roestlab/massformer",
                   pyproject_toml="pyproject.toml", python_version="3.11")
    """
    # F23: run on a worker thread — see bash()/bash_env() in shell.py for why.
    return await asyncio.to_thread(
        _setup_venv_sync, repo_url, packages, requirements_file,
        pyproject_toml, python_version,
    )


def _setup_venv_sync(repo_url: str, packages: list[str] | None,
                      requirements_file: str | None,
                      pyproject_toml: str | None,
                      python_version: str | None) -> dict:
    out_dir  = output_dir(repo_url)
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
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        return {"success": False, "error": f"venv creation failed: {e.stderr.strip()}"}

    errors = []

    if requirements_file:
        req_path = repo_path(repo_url) / requirements_file
        if req_path.exists():
            try:
                _pip_install(use_uv, python, venv_dir, "-r", str(req_path))
            except subprocess.CalledProcessError as e:
                errors.append(f"requirements install failed: {e.stderr.strip()}")
        else:
            errors.append(f"requirements file not found: {req_path}")

    if pyproject_toml:
        proj_path = repo_path(repo_url) / pyproject_toml
        if proj_path.exists():
            try:
                _pip_install(use_uv, python, venv_dir, "-e", str(proj_path.parent))
            except subprocess.CalledProcessError as e:
                errors.append(f"pyproject.toml install failed: {e.stderr.strip()}")
        else:
            errors.append(f"pyproject.toml not found: {proj_path}")

    install_pkgs = ["fastmcp", "pytest", "mcp"] + (packages or [])
    try:
        _pip_install(use_uv, python, venv_dir, *install_pkgs)
    except subprocess.CalledProcessError as e:
        errors.append(f"package install failed: {e.stderr.strip()}")

    if errors:
        return {"success": False, "venv": str(venv_dir), "error": "; ".join(errors)}
    return {"success": True, "venv": str(venv_dir), "python": python}


async def check_venv_compat(repo_url: str, venv_name: str = ".venv") -> dict:
    """Check compatibility by replaying the repo's own import statements in a venv.

    Scans the cloned repo's Python files with AST, collects every unique
    `import X` and `from X import Y` where X is an installed package, then
    executes each statement in the venv.  This catches both ABI conflicts
    (numpy 1 vs 2) and removed-API errors (e.g. `from transformers import AdamW`
    removed in transformers>=4.38).

    Args:
        repo_url:  Repository URL.
        venv_name: Directory name of the venv to check, relative to the repo
                   output dir. Default ".venv" (the server venv). Pass
                   ".venv-repo" to check the repo-side venv when running the
                   two-venv layout.

    Returns only failures; successful imports are omitted to keep output small.

    Example:
        check_venv_compat("https://github.com/Roestlab/massformer")
        check_venv_compat("https://github.com/Roestlab/massformer", venv_name=".venv-repo")
    """
    # F23: run on a worker thread — see bash()/bash_env() in shell.py for why.
    return await asyncio.to_thread(_check_venv_compat_sync, repo_url, venv_name)


def _check_venv_compat_sync(repo_url: str, venv_name: str) -> dict:
    out_dir  = output_dir(repo_url).resolve()
    repo_dir = repo_path(repo_url).resolve()
    venv_py  = out_dir / venv_name / "bin" / "python"
    if not venv_py.exists():
        return {"error": f"venv python not found at {venv_py}"}

    r = subprocess.run(
        [str(venv_py.absolute()), str(COMPAT_CHECK_SCRIPT), str(repo_dir)],
        capture_output=True, text=True, timeout=240,
    )
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
