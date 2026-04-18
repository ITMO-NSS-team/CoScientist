import re
import shlex
import subprocess
from pathlib import Path

DEFAULT_PYTHON_VERSION = "3.11"
DOCKER_IMAGE_MARKER = ".docker_image"

_ALEMBIC_BASE = Path(
    __import__("os").environ.get("ALEMBIC_WORKDIR", "/var/tmp/alembic")
)
REPO_DIR    = _ALEMBIC_BASE / "repos"
REPORTS_DIR = _ALEMBIC_BASE / "reports"
OUTPUT_DIR  = _ALEMBIC_BASE / "output"
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


def _repo_name(repo_url: str) -> str:
    return repo_url.rstrip("/").split("/")[-1].removesuffix(".git")


def _repo_path(repo_url: str) -> Path:
    return REPO_DIR / _repo_name(repo_url)


def _docker_safe_image_component(name: str) -> str:
    s = name.lower()
    s = re.sub(r"[^a-z0-9_.-]", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "repo"


def _python_exec_argv(out_dir: Path, python_cli: list[str]) -> list[str]:
    """Argv prefix + python subcommand for either Docker (mounted workspace) or host Python."""
    marker_path = out_dir / DOCKER_IMAGE_MARKER
    if marker_path.exists():
        image = marker_path.read_text(encoding="utf-8").strip()
        return [
            "docker", "run", "--rm",
            "-v", f"{out_dir.resolve()}:/workspace",
            "-w", "/workspace",
            image,
            "python",
        ] + python_cli
    venv_python = out_dir / ".venv" / "bin" / "python"
    if venv_python.exists():
        return [str(venv_python)] + python_cli
    return ["python"] + python_cli


def clone_repo(repo_url: str) -> dict:
    """Clone a GitHub repository to local disk.

    Returns the local path and a flat file list for you to select from.

    Example:
        clone_repo("https://github.com/Roestlab/massformer")
        # -> {"local_path": "/tmp/repos/massformer", "files": [...]}
    """
    dest = _repo_path(repo_url)
    if not dest.exists():
        REPO_DIR.mkdir(parents=True, exist_ok=True)
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

    return {"local_path": str(dest), 
            "files": sorted(files)
            }


def read_file(repo_url: str, path: str) -> dict:
    """Read a text file from the locally cloned repository.

    Returns up to 40 KB of content. Do NOT use this on data files (.csv,
    .parquet, .tsv, .json arrays) — use bash("head -n 20 <path>") instead
    to peek at their structure.

    Example:
        read_file("https://github.com/Roestlab/massformer", "README.md")
        read_file("https://github.com/Roestlab/massformer", "src/train.py")
    """
    full = _repo_path(repo_url) / path
    if not full.exists():
        return {"error": f"File not found: {path}. It may be a data file not included in the shallow clone."}
    if full.suffix in IGNORE_EXTS:
        return {"error": f"Binary/data file skipped: {path}. Use bash('head -n 20 {full}') to peek if it is text-based."}
    raw = full.read_bytes()[:MAX_BYTES]
    return {"path": path, "content": raw.decode("utf-8", errors="replace")}


def bash(command: str) -> dict:
    """Run a restricted shell command. Only ls, grep, head, and glob are supported.

    - ls   : list directory contents
    - grep : search file contents
    - head : preview first N lines of a file
    - glob : list files matching a shell glob pattern (Python-interpreted)

    Examples:
        bash("ls /tmp/repos/massformer")
        bash("ls -la /tmp/repos/massformer/src")
        bash("ls -R /tmp/repos/massformer")                          # full tree
        bash("grep -r 'def train' /tmp/repos/massformer -l")         # find files containing pattern
        bash("grep -n 'ArgumentParser' /tmp/repos/massformer/train.py")
        bash("head -n 30 /tmp/repos/massformer/README.md")
        bash("head -n 5 /tmp/repos/massformer/data/sample.csv")      # peek data files
        bash("glob /tmp/repos/massformer/**/*.yaml")                  # find all yaml files
        bash("glob /tmp/repos/massformer/**/config*")
    """
    stripped = command.strip()
    cmd_name = stripped.split()[0] if stripped else ""

    if cmd_name not in _ALLOWED_CMDS:
        return {
            "error": f"Command '{cmd_name}' is not allowed. "
                     f"Only {_ALLOWED_CMDS} are supported."
        }

    if cmd_name == "glob":
        # glob <pattern>
        parts = stripped.split(None, 1)
        if len(parts) < 2:
            return {"error": "glob requires a pattern argument."}
        pattern = parts[1]
        matched = sorted(str(p) for p in Path("/").glob(pattern.lstrip("/")))
        return {"matches": matched}

    # For ls, grep, head — run via subprocess with a timeout
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


def search(repo_url: str, pattern: str) -> dict:
    """Find files in the cloned repo matching a glob pattern.

    Pattern is relative to the repo root. Ignores binary and generated files.

    Examples:
        search("https://github.com/Roestlab/massformer", "**/*.yaml")
        search("https://github.com/Roestlab/massformer", "**/config*")
        search("https://github.com/Roestlab/massformer", "*.sh")
        search("https://github.com/Roestlab/massformer", "**/*train*")
    """
    dest = _repo_path(repo_url)
    matched = []
    for p in dest.glob(pattern):
        if p.is_file() and p.suffix not in IGNORE_EXTS:
            rel = p.relative_to(dest)
            if not any(part in IGNORE for part in rel.parts):
                matched.append(str(rel))
    return {"pattern": pattern, "matches": sorted(matched)}


def read_report(report_name: str) -> dict:
    """Read a Markdown report from the shared reports directory (/tmp/alembic_reports/).

    Args:
        report_name: Filename without the .md extension, e.g. "massformer_exploration".

    Example:
        read_report("massformer_exploration")
        # -> {"report_path": "/tmp/alembic_reports/massformer_exploration.md", "content": "..."}
    """
    path = REPORTS_DIR / f"{report_name}.md"
    if not path.exists():
        return {"error": f"No report found at {path}."}
    return {"report_path": str(path), "content": path.read_text(encoding="utf-8")}


def write_file(repo_url: str, relative_path: str, content: str) -> dict:
    """Write a source file to the MCP server output directory for this repo.

    Output lives at /tmp/alembic_output/<repo-name>/<relative_path>.
    Call this to write the server file and test file.

    Args:
        repo_url:      Repository URL (used to namespace the output folder).
        relative_path: Path relative to the output folder, e.g. "server.py"
                       or "tests/test_server.py".
        content:       Full text content to write.

    Examples:
        write_file("https://github.com/Roestlab/massformer", "server.py", "...")
        write_file("https://github.com/Roestlab/massformer", "tests/test_server.py", "...")
    """
    name = _repo_name(repo_url)
    dest = OUTPUT_DIR / name / relative_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")
    return {"written": str(dest)}


def read_output_file(repo_url: str, relative_path: str) -> dict:
    """Read a file from the MCP server output directory for this repo.

    Use this to inspect generated server.py or test files before fixing them.
    Returns up to 40 KB of content.

    Args:
        repo_url:      Repository URL.
        relative_path: Path relative to the output folder, e.g. "server.py"
                       or "tests/test_server.py".

    Examples:
        read_output_file("https://github.com/Roestlab/massformer", "server.py")
        read_output_file("https://github.com/Roestlab/massformer", "tests/test_server.py")
    """
    name = _repo_name(repo_url)
    full = OUTPUT_DIR / name / relative_path
    if not full.exists():
        return {"error": f"File not found: {full}"}
    raw = full.read_bytes()[:MAX_BYTES]
    return {"path": str(full), "content": raw.decode("utf-8", errors="replace")}


def update_file(repo_url: str, relative_path: str, content: str) -> dict:
    """Overwrite a file in the MCP server output directory with corrected content.

    Read the file first with read_output_file, fix the issue, then call this
    with the complete corrected content. Always write the full file — not a patch.

    Args:
        repo_url:      Repository URL.
        relative_path: Path relative to the output folder, e.g. "server.py"
                       or "tests/test_server.py".
        content:       Complete corrected file content.

    Examples:
        update_file("https://github.com/Roestlab/massformer", "server.py", "...")
        update_file("https://github.com/Roestlab/massformer", "tests/test_server.py", "...")
    """
    name = _repo_name(repo_url)
    dest = OUTPUT_DIR / name / relative_path
    if not dest.exists():
        return {"error": f"File not found: {dest}. Cannot update a file that does not exist."}
    dest.write_text(content, encoding="utf-8")
    return {"updated": str(dest)}


def compose_dockerfile_body(
    py_minor: str,
    requirements_rel: str | None,
    editable_relpath: str | None,
    install_pkgs: list[str],
) -> str:
    """Dockerfile with build context = cloned repo root. Repo is copied to /app."""
    lines = [
        f"FROM python:{py_minor}-slim",
        "WORKDIR /app",
        "ENV PIP_DISABLE_PIP_VERSION_CHECK=1",
        "RUN pip install --no-cache-dir --upgrade pip",
        "COPY . .",
    ]
    if requirements_rel:
        rq = Path(requirements_rel).as_posix()
        lines.append(f"RUN pip install --no-cache-dir -r {shlex.quote(rq)}")
    if editable_relpath is not None:
        ep = Path(editable_relpath).as_posix()
        ed_arg = "." if ep in (".", "") else "./" + ep
        lines.append(f"RUN pip install --no-cache-dir -e {shlex.quote(ed_arg)}")
    quoted_pkgs = " ".join(shlex.quote(p) for p in install_pkgs)
    lines.append(f"RUN pip install --no-cache-dir {quoted_pkgs}")
    return "\n".join(lines) + "\n"


def setup_venv(repo_url: str, packages: list[str] | None = None,
               requirements_file: str | None = None,
               pyproject_toml: str | None = None,
               python_version: str | None = None) -> dict:
    """Build a Docker image from the cloned repo with the requested Python and deps.

    Writes ``Dockerfile`` under the output directory and runs ``docker build`` with
    the clone root as context. Records the image name in ``.docker_image`` for
    ``validate_syntax`` / ``run_tests``. Requires Docker on the host.

    Uses ``python:<version>-slim`` as the base image (not suitable for CUDA/GPU-only stacks).

    Args:
        repo_url:          Repository URL (used to namespace the output folder).
        packages:          Extra pip-installable package names,
                           e.g. ["numpy", "torch"].  May be None.
        requirements_file: Path to a requirements.txt file relative to the
                           cloned repo root, e.g. "requirements.txt". May be None.
        pyproject_toml:    Path to a pyproject.toml relative to the cloned repo
                           root. When provided, that directory is installed editable.
                           May be None.
        python_version:    Python minor series, e.g. "3.11" or "3.10". May be None
                           (defaults to DEFAULT_PYTHON_VERSION).

    Returns:
        {"success": True,  "image": "<tag>", "dockerfile": "<path>",
         "python": "<example docker run ...>"}
        {"success": False, "error": "<message>", "dockerfile": "<path>" if written}

    Examples:
        setup_venv("https://github.com/Roestlab/massformer",
                   requirements_file="requirements.txt")
        setup_venv("https://github.com/Roestlab/massformer",
                   pyproject_toml="pyproject.toml")
        setup_venv("https://github.com/Roestlab/massformer",
                   pyproject_toml="pyproject.toml", packages=["extra-pkg"])
        setup_venv("https://github.com/Roestlab/massformer",
                   pyproject_toml="pyproject.toml", python_version="3.11")
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
    out_dir = OUTPUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_root = _repo_path(repo_url)
    dockerfile_path = out_dir / "Dockerfile"
    marker_path = out_dir / DOCKER_IMAGE_MARKER
    if marker_path.exists():
        marker_path.unlink()

    errors: list[str] = []
    requirements_rel: str | None = None
    if requirements_file:
        req_path = repo_root / requirements_file
        if req_path.exists():
            requirements_rel = Path(requirements_file).as_posix()
        else:
            errors.append(f"requirements file not found: {req_path}")

    editable_relpath: str | None = None
    if pyproject_toml:
        proj_path = repo_root / pyproject_toml
        if proj_path.exists():
            try:
                rel = proj_path.parent.resolve().relative_to(repo_root.resolve())
                editable_relpath = "." if rel == Path(".") else rel.as_posix()
            except ValueError:
                errors.append(f"pyproject.toml not under repo root: {proj_path}")
        else:
            errors.append(f"pyproject.toml not found: {proj_path}")

    if errors:
        return {
            "success": False,
            "dockerfile": str(dockerfile_path),
            "error": "; ".join(errors),
        }

    py_minor = (python_version or DEFAULT_PYTHON_VERSION).lstrip("python").strip()
    if py_minor.startswith("v"):
        py_minor = py_minor[1:]

    install_pkgs = ["mcp", "pytest"] + (packages or [])
    body = compose_dockerfile_body(py_minor, requirements_rel, editable_relpath, install_pkgs)
    dockerfile_path.write_text(body, encoding="utf-8")

    image_tag = f"alembic-{_docker_safe_image_component(name)}:latest"
    try:
        subprocess.run(
            [
                "docker", "build",
                "-f", str(dockerfile_path),
                "-t", image_tag,
                str(repo_root),
            ],
            check=True,
            capture_output=False,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or e.stdout or "").strip()
        return {
            "success": False,
            "dockerfile": str(dockerfile_path),
            "error": f"docker build failed: {msg}"[:MAX_BYTES],
        }

    marker_path.write_text(image_tag + "\n", encoding="utf-8")
    run_hint = (
        f"docker run --rm -v {out_dir.resolve()}:/workspace -w /workspace "
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

    Runs two checks in sequence:
      1. py_compile  — catches SyntaxError before any code runs
      2. module load — imports the file to surface missing packages or
                       top-level NameError / ImportError

    Returns {"passed": True} on success, or
            {"passed": False, "stage": "syntax"|"imports", "error": "<traceback>"}

    Example:
        validate_syntax("https://github.com/Roestlab/massformer")
    """
    name = _repo_name(repo_url)
    out_dir = OUTPUT_DIR / name
    server  = out_dir / "server.py"
    if not server.exists():
        return {"passed": False, "stage": "syntax", "error": f"server.py not found at {server}"}

    run_kw: dict = {"capture_output": True, "text": True}
    run_kw["cwd"] = str(out_dir)

    # Stage 1: syntax check
    r1 = subprocess.run(
        _python_exec_argv(out_dir, ["-m", "py_compile", "server.py"]),
        **run_kw,
    )
    if r1.returncode != 0:
        return {"passed": False, "stage": "syntax", "error": r1.stderr.strip()}

    # Stage 2: import check (load module without running mcp.run())
    marker_path = out_dir / DOCKER_IMAGE_MARKER
    if marker_path.exists():
        load_snippet = (
            "import importlib.util as _u, sys as _s; "
            "_s.path.insert(0, '/workspace'); "
            "_spec=_u.spec_from_file_location('server', '/workspace/server.py'); "
            "_mod=_u.module_from_spec(_spec); "
            "_spec.loader.exec_module(_mod)"
        )
    else:
        load_snippet = (
            "import importlib.util as _u, sys as _s; "
            f"_s.path.insert(0, r'{server.parent}'); "
            f"_spec=_u.spec_from_file_location('server', r'{server}'); "
            "_mod=_u.module_from_spec(_spec); "
            "_spec.loader.exec_module(_mod)"
        )
    r2 = subprocess.run(
        _python_exec_argv(out_dir, ["-c", load_snippet]),
        capture_output=True, text=True, timeout=30,
        cwd=str(out_dir),
    )
    if r2.returncode != 0:
        return {"passed": False, "stage": "imports", "error": r2.stderr.strip()}

    return {"passed": True}


def run_tests(repo_url: str) -> dict:
    """Run the pytest test suite for the generated MCP server.

    Executes tests/test_server.py under /tmp/alembic_output/<repo-name>/.
    Returns pass/fail status and the full pytest output (stdout + stderr,
    truncated to 40 KB).

    Example:
        run_tests("https://github.com/Roestlab/massformer")
        # -> {"passed": True/False, "output": "...pytest output..."}
    """
    name = _repo_name(repo_url)
    out_dir  = OUTPUT_DIR / name
    test_dir = out_dir / "tests"
    if not test_dir.exists():
        return {"passed": False, "output": f"Test directory not found: {test_dir}"}

    run_kw: dict = {
        "capture_output": True,
        "text": True,
        "timeout": 120,
    }
    run_kw["cwd"] = str(out_dir)

    try:
        r = subprocess.run(
            _python_exec_argv(
                out_dir,
                ["-m", "pytest", "tests", "-v", "--tb=short", "--no-header"],
            ),
            **run_kw,
        )
    except subprocess.TimeoutExpired:
        return {"passed": False, "output": "pytest timed out after 120 seconds."}

    output = (r.stdout + r.stderr)[:MAX_BYTES]
    return {"passed": r.returncode == 0, "output": output}


def write_report(report_name: str, content: str) -> dict:
    """Write a Markdown report to the shared reports directory (/tmp/alembic_reports/).

    Args:
        report_name: Filename without the .md extension, e.g. "massformer_exploration".
        content:     Full Markdown content to write.

    Example:
        write_report("massformer_exploration", "# massformer\\n\\n## Description\\n...")
        # -> {"report_path": "/tmp/alembic_reports/massformer_exploration.md"}
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"{report_name}.md"
    out.write_text(content, encoding="utf-8")
    return {"report_path": str(out)}
