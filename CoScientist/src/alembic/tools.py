import subprocess
from pathlib import Path

WORKDIR = Path(
    __import__("os").environ.get("ALEMBIC_WORKDIR", ".alembic")
)
MAX_BYTES = 40_000

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

_ALLOWED_CMDS = ("ls", "grep", "head", "glob")
_ENV_ALLOWED_CMDS = (*_ALLOWED_CMDS, "pip", "pip3", "uv", "conda", "python", "python3", "which")


def _repo_name(repo_url: str) -> str:
    return repo_url.rstrip("/").split("/")[-1].removesuffix(".git")


def _repo_base(repo_url: str) -> Path:
    """Root dir for everything related to this repo: <WORKDIR>/<repo-name>/"""
    return WORKDIR / _repo_name(repo_url)


def _repo_path(repo_url: str) -> Path:
    """Where the repo is cloned: <WORKDIR>/<repo-name>/repos/"""
    return _repo_base(repo_url) / "repos"


def _output_dir(repo_url: str) -> Path:
    """Where server.py, tests, .venv live: <WORKDIR>/<repo-name>/output/"""
    return _repo_base(repo_url) / "output"


def _reports_dir(repo_url: str) -> Path:
    """Where .md reports live: <WORKDIR>/<repo-name>/reports/"""
    return _repo_base(repo_url) / "reports"


def _venv_python(out_dir: Path) -> str:
    """Return the venv python path if it exists, else fall back to 'python'."""
    candidate = out_dir / ".venv" / "bin" / "python"
    return str(candidate.resolve()) if candidate.exists() else "python"


def clone_repo(repo_url: str) -> dict:
    """Clone a GitHub repository to local disk.

    Returns the local path and a flat file list for you to select from.

    Example:
        clone_repo("https://github.com/Roestlab/massformer")
        # -> {"local_path": ".alembic/massformer/repos", "files": [...]}
    """
    dest = _repo_path(repo_url)
    if not dest.exists():
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth=1", repo_url, str(dest)],
            check=True, capture_output=True,
        )

    files = []
    for p in dest.rglob("*"):
        if p.is_file() and p.suffix not in IGNORE_EXTS:
            rel = p.relative_to(dest)
            if not any(part in IGNORE for part in rel.parts):
                files.append(str(rel))

    return {"local_path": str(dest), "files": sorted(files)}


def read_file(repo_url: str, path: str) -> dict:
    """Read a text file from the locally cloned repository.

    Returns up to 40 KB of content. Do NOT use this on data files (.csv,
    .parquet, .tsv, .json arrays) — use bash("head -n 20 <path>") instead.

    Example:
        read_file("https://github.com/Roestlab/massformer", "README.md")
    """
    full = _repo_path(repo_url) / path
    if not full.exists():
        return {"error": f"File not found: {path}."}
    if full.suffix in IGNORE_EXTS:
        return {"error": f"Binary/data file skipped: {path}."}
    raw = full.read_bytes()[:MAX_BYTES]
    return {"path": path, "content": raw.decode("utf-8", errors="replace")}


def bash(command: str) -> dict:
    """Run a restricted shell command. Only ls, grep, head, and glob are supported.

    Examples:
        bash("ls .alembic/massformer/repos")
        bash("grep -r 'def train' .alembic/massformer/repos -l")
        bash("head -n 30 .alembic/massformer/repos/README.md")
        bash("glob .alembic/massformer/repos/**/*.yaml")
        bash("python -m py_compile .alembic/massformer/output/server.py && echo OK")
    """
    stripped = command.strip()
    cmd_name = stripped.split()[0] if stripped else ""

    if cmd_name not in _ALLOWED_CMDS:
        # Allow "python -m py_compile ..." for syntax checks
        if not (cmd_name == "python" and "-m" in stripped and "py_compile" in stripped):
            return {
                "error": f"Command '{cmd_name}' is not allowed. "
                         f"Only {_ALLOWED_CMDS} are supported, plus "
                         f"'python -m py_compile <file> && echo OK'."
            }

    if cmd_name == "glob":
        parts = stripped.split(None, 1)
        if len(parts) < 2:
            return {"error": "glob requires a pattern argument."}
        pattern = parts[1]
        matched = sorted(str(p) for p in Path("/").glob(pattern.lstrip("/")))
        return {"matches": matched}

    try:
        result = subprocess.run(
            stripped,
            shell=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        output = result.stdout
        if result.returncode != 0 and result.stderr:
            output += "\n[stderr] " + result.stderr
        return {"output": output[:MAX_BYTES]}
    except subprocess.TimeoutExpired:
        return {"error": "Command timed out after 15 seconds."}


def bash_env(command: str) -> dict:
    """Run an environment setup command (pip, uv, conda, python, etc.).

    Extends bash with package-manager commands. Timeout is 300 s to
    accommodate slow installs.

    Examples:
        bash_env("uv venv .alembic/massformer/output/.venv --python 3.11")
        bash_env("uv pip install --python .alembic/massformer/output/.venv/bin/python torch torchvision")
        bash_env("pip install -r .alembic/massformer/repos/requirements.txt")
        bash_env("which python3")
    """
    stripped = command.strip()
    cmd_name = stripped.split()[0] if stripped else ""

    if cmd_name not in _ENV_ALLOWED_CMDS:
        return {
            "error": f"Command '{cmd_name}' is not allowed. "
                     f"Supported: {sorted(_ENV_ALLOWED_CMDS)}."
        }

    if cmd_name == "glob":
        parts = stripped.split(None, 1)
        if len(parts) < 2:
            return {"error": "glob requires a pattern argument."}
        pattern = parts[1]
        matched = sorted(str(p) for p in Path("/").glob(pattern.lstrip("/")))
        return {"matches": matched}

    try:
        result = subprocess.run(
            stripped,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = result.stdout
        if result.returncode != 0 and result.stderr:
            output += "\n[stderr] " + result.stderr
        return {"output": output[:MAX_BYTES]}
    except subprocess.TimeoutExpired:
        return {"error": "Command timed out after 300 seconds."}


def search(repo_url: str, pattern: str) -> dict:
    """Find files in the cloned repo matching a glob pattern.

    Examples:
        search("https://github.com/Roestlab/massformer", "**/*.yaml")
        search("https://github.com/Roestlab/massformer", "*.sh")
    """
    dest = _repo_path(repo_url)
    matched = []
    for p in dest.glob(pattern):
        if p.is_file() and p.suffix not in IGNORE_EXTS:
            rel = p.relative_to(dest)
            if not any(part in IGNORE for part in rel.parts):
                matched.append(str(rel))
    return {"pattern": pattern, "matches": sorted(matched)}


def read_report(repo_url: str, report_name: str) -> dict:
    """Read a Markdown report from this repo's reports directory.

    Args:
        repo_url:    Repository URL.
        report_name: Filename without the .md extension: "exploration", "server",
                     or "validation".

    Example:
        read_report("https://github.com/Roestlab/massformer", "exploration")
        # -> {"report_path": ".alembic/massformer/reports/exploration.md", ...}
    """
    path = _reports_dir(repo_url) / f"{report_name}.md"
    if not path.exists():
        return {"error": f"No report found at {path}."}
    return {"report_path": str(path), "content": path.read_text(encoding="utf-8")}


def write_file(repo_url: str, relative_path: str, content: str) -> dict:
    """Write a source file to the output directory for this repo.

    Output lives at .alembic/<repo-name>/output/<relative_path>.

    Examples:
        write_file("https://github.com/Roestlab/massformer", "server.py", "...")
        write_file("https://github.com/Roestlab/massformer", "tests/test_server.py", "...")
        write_file("https://github.com/Roestlab/massformer", "helpers/run_analysis.py", "...")
    """
    dest = _output_dir(repo_url) / relative_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")
    return {"written": str(dest)}


def read_output_file(repo_url: str, relative_path: str) -> dict:
    """Read a file from the output directory for this repo.

    Examples:
        read_output_file("https://github.com/Roestlab/massformer", "server.py")
        read_output_file("https://github.com/Roestlab/massformer", "tests/test_server.py")
    """
    full = _output_dir(repo_url) / relative_path
    if not full.exists():
        return {"error": f"File not found: {full}"}
    raw = full.read_bytes()[:MAX_BYTES]
    return {"path": str(full), "content": raw.decode("utf-8", errors="replace")}


def update_file(repo_url: str, relative_path: str, content: str) -> dict:
    """Overwrite a file in the output directory with corrected content.

    Always write the full file — not a patch.

    Examples:
        update_file("https://github.com/Roestlab/massformer", "server.py", "...")
        update_file("https://github.com/Roestlab/massformer", "tests/test_server.py", "...")
    """
    dest = _output_dir(repo_url) / relative_path
    if not dest.exists():
        return {"error": f"File not found: {dest}. Cannot update a file that does not exist."}
    dest.write_text(content, encoding="utf-8")
    return {"updated": str(dest)}


def setup_venv(repo_url: str, packages: list[str] | None = None,
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
    out_dir  = _output_dir(repo_url)
    out_dir.mkdir(parents=True, exist_ok=True)
    venv_dir = out_dir / ".venv"
    python   = str(venv_dir / "bin" / "python")

    use_uv = subprocess.run(["which", "uv"], capture_output=True).returncode == 0

    try:
        if use_uv:
            uv_venv_cmd = ["uv", "venv", str(venv_dir)]
            if python_version:
                uv_venv_cmd += ["--python", python_version]
            subprocess.run(uv_venv_cmd, check=True, capture_output=True, text=True)
        else:
            py_bin = f"python{python_version}" if python_version else "python"
            subprocess.run([py_bin, "-m", "venv", str(venv_dir)],
                           check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        return {"success": False, "error": f"venv creation failed: {e.stderr.strip()}"}

    errors = []

    if requirements_file:
        req_path = _repo_path(repo_url) / requirements_file
        if req_path.exists():
            try:
                if use_uv:
                    subprocess.run(
                        ["uv", "pip", "install", "--python", python, "-r", str(req_path)],
                        check=True, capture_output=True, text=True,
                    )
                else:
                    subprocess.run(
                        [str(venv_dir / "bin" / "pip"), "install", "-r", str(req_path)],
                        check=True, capture_output=True, text=True,
                    )
            except subprocess.CalledProcessError as e:
                errors.append(f"requirements install failed: {e.stderr.strip()}")
        else:
            errors.append(f"requirements file not found: {req_path}")

    if pyproject_toml:
        proj_path = _repo_path(repo_url) / pyproject_toml
        if proj_path.exists():
            try:
                if use_uv:
                    subprocess.run(
                        ["uv", "pip", "install", "--python", python, "-e", str(proj_path.parent)],
                        check=True, capture_output=True, text=True,
                    )
                else:
                    subprocess.run(
                        [str(venv_dir / "bin" / "pip"), "install", "-e", str(proj_path.parent)],
                        check=True, capture_output=True, text=True,
                    )
            except subprocess.CalledProcessError as e:
                errors.append(f"pyproject.toml install failed: {e.stderr.strip()}")
        else:
            errors.append(f"pyproject.toml not found: {proj_path}")

    install_pkgs = ["mcp", "pytest"] + (packages or [])
    try:
        if use_uv:
            subprocess.run(
                ["uv", "pip", "install", "--python", python] + install_pkgs,
                check=True, capture_output=True, text=True,
            )
        else:
            subprocess.run(
                [str(venv_dir / "bin" / "pip"), "install"] + install_pkgs,
                check=True, capture_output=True, text=True,
            )
    except subprocess.CalledProcessError as e:
        errors.append(f"package install failed: {e.stderr.strip()}")

    if errors:
        return {"success": False, "venv": str(venv_dir), "error": "; ".join(errors)}
    return {"success": True, "venv": str(venv_dir), "python": python}


def validate_syntax(repo_url: str) -> dict:
    """Check server.py for syntax errors and failed imports.

    Example:
        validate_syntax("https://github.com/Roestlab/massformer")
    """
    out_dir = _output_dir(repo_url)
    server  = out_dir / "server.py"
    python  = _venv_python(out_dir)
    if not server.exists():
        return {"passed": False, "stage": "syntax", "error": f"server.py not found at {server}"}

    r1 = subprocess.run(
        [python, "-m", "py_compile", str(server)],
        capture_output=True, text=True,
    )
    if r1.returncode != 0:
        return {"passed": False, "stage": "syntax", "error": r1.stderr.strip()}

    load_snippet = (
        "import importlib.util as _u, sys as _s; "
        f"_s.path.insert(0, '{server.parent}'); "
        f"_spec=_u.spec_from_file_location('server', r'{server}'); "
        "_mod=_u.module_from_spec(_spec); "
        "_spec.loader.exec_module(_mod)"
    )
    r2 = subprocess.run(
        [python, "-c", load_snippet],
        capture_output=True, text=True, timeout=30,
        cwd=str(server.parent),
    )
    if r2.returncode != 0:
        return {"passed": False, "stage": "imports", "error": r2.stderr.strip()}

    return {"passed": True}


def run_tests(repo_url: str) -> dict:
    """Run the pytest test suite for the generated MCP server.

    Example:
        run_tests("https://github.com/Roestlab/massformer")
    """
    out_dir  = _output_dir(repo_url)
    test_dir = out_dir / "tests"
    python   = _venv_python(out_dir)
    if not test_dir.exists():
        return {"passed": False, "output": f"Test directory not found: {test_dir}"}

    try:
        r = subprocess.run(
            [python, "-m", "pytest", str(test_dir), "-v", "--tb=short", "--no-header"],
            capture_output=True, text=True, timeout=120,
            cwd=str(out_dir),
        )
    except subprocess.TimeoutExpired:
        return {"passed": False, "output": "pytest timed out after 120 seconds."}

    output = (r.stdout + r.stderr)[:MAX_BYTES]
    return {"passed": r.returncode == 0, "output": output}


def write_report(repo_url: str, report_name: str, content: str) -> dict:
    """Write a Markdown report to this repo's reports directory.

    Args:
        repo_url:    Repository URL.
        report_name: Filename without the .md extension: "exploration", "server",
                     or "validation".
        content:     Full Markdown content to write.

    Example:
        write_report("https://github.com/Roestlab/massformer", "exploration", "# massformer...")
        # -> {"report_path": ".alembic/massformer/reports/exploration.md"}
    """
    reports = _reports_dir(repo_url)
    reports.mkdir(parents=True, exist_ok=True)
    out = reports / f"{report_name}.md"
    out.write_text(content, encoding="utf-8")
    return {"report_path": str(out)}
