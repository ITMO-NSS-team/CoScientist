import fnmatch
import re
import os
import subprocess
from pathlib import Path

DOCKER_IMAGE_MARKER = ".docker_image"
DOCKER_BUILD_FAIL_COUNTER = ".docker_build_failures"
DOCKER_BUILD_MAX_ATTEMPTS = 5

WORKDIR = Path(
    os.environ.get("ALEMBIC_WORKDIR", ".alembic")
)
REPO_DIR    = WORKDIR / "repos"
REPORTS_DIR = WORKDIR / "reports"
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


def _path_part_ignored(part: str) -> bool:
    return any(fnmatch.fnmatch(part, pattern) for pattern in IGNORE)


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


def _docker_safe_image_component(name: str) -> str:
    s = name.lower()
    s = re.sub(r"[^a-z0-9_.-]", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "repo"


def _python_exec_argv(workspace_root: Path, python_cli: list[str]) -> list[str]:
    """Argv prefix + python subcommand for either Docker (mounted workspace) or host Python."""
    marker_path = workspace_root / DOCKER_IMAGE_MARKER
    if marker_path.exists():
        image = marker_path.read_text(encoding="utf-8").strip()
        return [
            "docker", "run", "--rm",
            "-v", f"{workspace_root.resolve()}:/app",
            "-w", "/app",
            image,
            "python",
        ] + python_cli
    venv_python = workspace_root / ".venv" / "bin" / "python"
    if venv_python.exists():
        return [str(venv_python)] + python_cli
    return ["python"] + python_cli


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
            if not any(_path_part_ignored(part) for part in rel.parts):
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
            if not any(_path_part_ignored(part) for part in rel.parts):
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
    """Write a file inside the cloned repository (e.g. ``server.py``, ``tests/...``).

    Markdown reports belong in ``reports/`` via ``write_report`` only.

    Examples:
        write_file("https://github.com/Roestlab/massformer", "server.py", "...")
        write_file("https://github.com/Roestlab/massformer", "tests/test_server.py", "...")
        write_file("https://github.com/Roestlab/massformer", "helpers/run_analysis.py", "...")
    """
    dest = _repo_path(repo_url) / Path((relative_path or "").strip())
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")
    return {"written": str(dest)}


def update_file(repo_url: str, relative_path: str, content: str) -> dict:
    """Overwrite a file in the cloned repository (full file content, not a patch).

    Read the current file with ``read_file(repo_url, relative_path)`` first.

    Examples:
        update_file("https://github.com/Roestlab/massformer", "server.py", "...")
        update_file("https://github.com/Roestlab/massformer", "tests/test_server.py", "...")
    """
    dest = _repo_path(repo_url) / Path((relative_path or "").strip())
    if not dest.exists():
        return {"error": f"File not found: {dest}. Cannot update a file that does not exist."}
    dest.write_text(content, encoding="utf-8")
    return {"updated": str(dest)}


def build_docker_image(repo_url: str) -> dict:
    """Run ``docker build`` for project using ``Dockerfile`` at the repository root.

    On success, writes ``.docker_image`` in the repository root so ``validate_syntax`` /
    ``run_tests`` use this image and removes ``.docker_build_failures``.

    After ``DOCKER_BUILD_MAX_ATTEMPTS`` failed builds for the same repository, returns
    without running ``docker build`` (``max_attempts_reached: true``).

    Build output is captured so the returned ``error`` includes docker's stderr/stdout
    (truncated to ``MAX_BYTES``).

    Returns:
        {"success": True,  "image": "<tag>", "dockerfile": "<path>",
         "python": "<example docker run ...>"}
        {"success": False, "error": "<message>", "dockerfile": "<path>" if present,
         optional "failed_attempts", "max_attempts_reached"}

    Example:
        write_file(repo_url, "Dockerfile", "FROM python:3.11-slim\\n...")
        build_docker_image(repo_url)
    """
    dv = subprocess.run(
        ["docker", "version"],
        capture_output=True,
        text=True,
    )
    if dv.returncode != 0:
        err = (dv.stderr or dv.stdout or "").strip() or "unknown error"
        return {"success": False, "error": f"docker not available: {err}"}

    name = _repo_name(repo_url)
    repo_root = _repo_path(repo_url)
    dockerfile_path = repo_root / "Dockerfile"
    marker_path = repo_root / DOCKER_IMAGE_MARKER
    if not dockerfile_path.is_file():
        return {
            "success": False,
            "error": "No Dockerfile at repository root. Use write_file(repo_url, 'Dockerfile', <full content>) first.",
        }

    counter_path = repo_root / DOCKER_BUILD_FAIL_COUNTER
    prev_fails = 0
    if counter_path.is_file():
        try:
            prev_fails = int(counter_path.read_text(encoding="utf-8").strip() or 0)
        except ValueError:
            pass
    if prev_fails >= DOCKER_BUILD_MAX_ATTEMPTS:
        return {
            "success": False,
            "max_attempts_reached": True,
            "failed_attempts": prev_fails,
            "dockerfile": str(dockerfile_path),
            "error": (
                f"docker build already failed {DOCKER_BUILD_MAX_ATTEMPTS} times for this "
                "clone; not retrying. Stop calling build_docker_image and record the "
                "outcome in the server report."
            ),
        }

    image_tag = f"alembic-{_docker_safe_image_component(name)}:latest"
    docker_build_proc = subprocess.Popen(
        [
            "docker",
            "build",
            "-f",
            str(dockerfile_path),
            "-t",
            image_tag,
            str(repo_root),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    error_logs = []
    full_log = []
    for line in docker_build_proc.stdout:
        print(line, end="")  
        full_log.append(line)
    
    for line in docker_build_proc.stderr:
        print(line, end="")  
        error_logs.append(line)
    
    docker_build_proc.wait()
    returncode = docker_build_proc.returncode
    if returncode != 0:
        log = "\n".join(
            s for s in (error_logs) if s
        )
        if not log:
            log = "(no output captured from docker build)"
        fails = prev_fails + 1
        counter_path.write_text(f"{fails}\n", encoding="utf-8")
        err = f"docker build failed (exit {returncode}):\n{log}"[:MAX_BYTES]
        out: dict = {
            "success": False,
            "dockerfile": str(dockerfile_path),
            "failed_attempts": fails,
            "error": err,
        }
        if fails >= DOCKER_BUILD_MAX_ATTEMPTS:
            out["max_attempts_reached"] = True
        return out

    if counter_path.exists():
        counter_path.unlink()
    marker_path.write_text(image_tag + "\n", encoding="utf-8")
    run_hint = (
        f"docker run --rm -v {repo_root.resolve()}:/app -w /app "
        f"{image_tag} python server.py"
    )
    return {
        "success": True,
        "image": image_tag,
        "dockerfile": str(dockerfile_path),
        "python": run_hint,
    }


def validate_syntax(repo_url: str) -> dict:
    """Check server.py for syntax errors and failed imports.

    Example:
        validate_syntax("https://github.com/Roestlab/massformer")
    """
    workspace = _repo_path(repo_url)
    server = workspace / "server.py"
    if not server.exists():
        return {"passed": False, "stage": "syntax", "error": f"server.py not found at {server}"}

    # Syntax check
    syntax_check_proc = subprocess.Popen(
        _python_exec_argv(workspace, ["-m", "py_compile", "server.py"]),
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True, 
        cwd=str(workspace),
    )
    error_logs = []
    logs = []
    for line in syntax_check_proc.stdout:
        print(line, end="")  
        error_logs.append(line)
    
    for line in syntax_check_proc.stderr:
        print(line, end="")  
        logs.append(line)
    
    syntax_check_proc.wait()
    
    if syntax_check_proc.returncode != 0:
        return {"passed": False, "stage": "syntax", "error": "".join(error_logs)}
    
    return {"passed": True}


def run_tests(repo_url: str) -> dict:
    """Run the pytest test suite for the generated MCP server.

    Runs ``pytest`` in the clone root (``tests/`` under the repo).
    Returns pass/fail status and the full pytest output (stdout + stderr,
    truncated to 40 KB).

    Example:
        run_tests("https://github.com/Roestlab/massformer")
    """
    workspace = _repo_path(repo_url)
    test_dir = workspace / "tests"
    if not test_dir.exists():
        return {"passed": False, "output": f"Test directory not found: {test_dir}"}

    run_kw: dict = {
        "capture_output": True,
        "text": True,
        "timeout": 120,
    }
    run_kw["cwd"] = str(workspace)

    try:
        r = subprocess.run(
            _python_exec_argv(
                workspace,
                ["-m", "pytest", "tests", "-v", "--tb=short", "--no-header"],
            ),
            **run_kw,
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
