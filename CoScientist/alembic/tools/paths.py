"""Workdir layout: constants, per-repo path helpers, and subprocess scripts."""
import os
from pathlib import Path

from alembic.common import get_repo_name

WORKDIR = Path(os.environ.get("ALEMBIC_WORKDIR", ".alembic"))
MAX_BYTES = 40_000

# Standalone scripts run inside a repo's venv (see venv.py / invoke.py).
_SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
COMPAT_CHECK_SCRIPT = _SCRIPTS_DIR / "compat_check.py"
INVOKE_TOOL_SCRIPT  = _SCRIPTS_DIR / "invoke_tool.py"

IGNORE = {
    ".git", "__pycache__", ".eggs", "*.egg-info", "dist", "build",
    "node_modules", ".tox", ".mypy_cache", ".pytest_cache",
    "checkpoints", "wandb", "mlruns", ".ipynb_checkpoints",
}
IGNORE_EXTS = {
    ".pyc", ".pyo", ".so", ".dylib", ".dll", ".exe",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
    ".pdf", ".zip", ".tar", ".gz", ".h5", ".hdf5",
    ".pt", ".pth", ".ckpt", ".pkl", ".npy", ".npz", ".parquet",
}


def _repo_base(repo_url: str) -> Path:
    """Root dir for everything related to this repo: <WORKDIR>/<repo-name>/"""
    return WORKDIR / get_repo_name(repo_url)


def repo_path(repo_url: str) -> Path:
    """Where the repo is cloned: <WORKDIR>/<repo-name>/repos/"""
    return _repo_base(repo_url) / "repos"


def output_dir(repo_url: str) -> Path:
    """Where server.py, tests, .venv live: <WORKDIR>/<repo-name>/output/"""
    return _repo_base(repo_url) / "output"


def reports_dir(repo_url: str) -> Path:
    """Where .md reports live: <WORKDIR>/<repo-name>/reports/"""
    return _repo_base(repo_url) / "reports"


def venv_python(out_dir: Path) -> str:
    """Return the venv python path if it exists, else fall back to 'python'.

    Uses the venv symlink path directly — do NOT resolve(), as that follows
    the symlink to the bare uv Python binary which lacks the venv site-packages.
    """
    candidate = out_dir / ".venv" / "bin" / "python"
    return str(candidate.absolute()) if candidate.exists() else "python"


def rel_or_ignored(path: Path, root: Path) -> str | None:
    """Relative path string for an indexable file, or None if it's ignored."""
    if not path.is_file() or path.suffix in IGNORE_EXTS:
        return None
    rel = path.relative_to(root)
    if any(part in IGNORE for part in rel.parts):
        return None
    return str(rel)
