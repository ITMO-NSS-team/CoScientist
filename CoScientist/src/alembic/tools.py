import fnmatch
import json as _json
import os
import re
import subprocess
import textwrap
from pathlib import Path

DOCKER_IMAGE_MARKER = ".docker_image"
DOCKER_BUILD_FAIL_COUNTER = ".docker_build_failures"
DOCKER_BUILD_MAX_ATTEMPTS = 5

WORKDIR = Path(os.environ.get("ALEMBIC_WORKDIR", ".alembic"))
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


def _docker_safe_image_component(name: str) -> str:
    return re.sub(r"[^a-z0-9._-]", "-", name.lower())


_ALLOWED_CMDS = ("ls", "grep", "head", "glob")
_ENV_ALLOWED_CMDS = (*_ALLOWED_CMDS, "pip", "pip3", "uv", "conda", "python", "python3", "which")


def _repo_name(repo_url: str) -> str:
    return repo_url.rstrip("/").split("/")[-1].removesuffix(".git")


def _repo_base(repo_url: str) -> Path:
    """Root dir for everything related to this repo: <WORKDIR>/<repo-name>/"""
    return WORKDIR / _repo_name(repo_url)


def _repo_path(repo_url: str) -> Path:
    """Where the repo is cloned and generated code lives: <WORKDIR>/<repo-name>/repos/"""
    return _repo_base(repo_url) / "repos"


def _output_dir(repo_url: str) -> Path:
    """Where the venv and docker marker live: <WORKDIR>/<repo-name>/output/"""
    return _repo_base(repo_url) / "output"


def _reports_dir(repo_url: str) -> Path:
    """Where .md reports live: <WORKDIR>/<repo-name>/reports/"""
    return _repo_base(repo_url) / "reports"


def _venv_python(out_dir: Path) -> str:
    """Return the venv python path if it exists, else fall back to 'python'.

    Uses the venv symlink path directly — do NOT resolve(), as that follows
    the symlink to the bare uv Python binary which lacks the venv site-packages.
    """
    candidate = out_dir / ".venv" / "bin" / "python"
    return str(candidate.absolute()) if candidate.exists() else "python"


def _python_exec_argv(repo_url: str, python_cli: list) -> list:
    """Return argv to run python in the best available env: Docker > venv > host."""
    out_dir   = _output_dir(repo_url)
    repo_root = _repo_path(repo_url)
    marker    = out_dir / DOCKER_IMAGE_MARKER
    if marker.exists():
        image = marker.read_text(encoding="utf-8").strip()
        return [
            "docker", "run", "--rm",
            "-v", f"{repo_root.resolve()}:/app",
            "-w", "/app",
            image, "python",
        ] + python_cli
    venv_py = out_dir / ".venv" / "bin" / "python"
    if venv_py.exists():
        return [str(venv_py.absolute())] + python_cli
    return ["python"] + python_cli


# ── Public tools ──────────────────────────────────────────────────────────────

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


def read_output_file(repo_url: str, path: str) -> dict:
    """Read a file from the output directory (venv info, pip-freeze dumps, etc.).

    Example:
        read_output_file("https://github.com/Roestlab/massformer", "pip_freeze.txt")
    """
    full = _output_dir(repo_url) / path
    if not full.exists():
        return {"error": f"File not found in output/: {path}."}
    raw = full.read_bytes()[:MAX_BYTES]
    return {"path": str(full), "content": raw.decode("utf-8", errors="replace")}


def bash(command: str) -> dict:
    """Run a restricted shell command. Only ls, grep, head, and glob are supported.

    Examples:
        bash("ls .alembic/massformer/repos")
        bash("grep -r 'def train' .alembic/massformer/repos -l")
        bash("head -n 30 .alembic/massformer/repos/README.md")
        bash("glob .alembic/massformer/repos/**/*.yaml")
        bash("python -m py_compile .alembic/massformer/repos/server.py && echo OK")
    """
    stripped = command.strip()
    cmd_name = stripped.split()[0] if stripped else ""

    if cmd_name not in _ALLOWED_CMDS:
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
        bash_env("uv pip install --python .alembic/massformer/output/.venv/bin/python torch")
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
        report_name: Filename without the .md extension: "exploration", "environment",
                     "server", "docker", or "validation".

    Example:
        read_report("https://github.com/Roestlab/massformer", "exploration")
    """
    path = _reports_dir(repo_url) / f"{report_name}.md"
    if not path.exists():
        return {"error": f"No report found at {path}."}
    return {"report_path": str(path), "content": path.read_text(encoding="utf-8")}


def write_file(repo_url: str, relative_path: str, content: str) -> dict:
    """Write a generated file into the repos directory (server.py, tests/, helpers/, Dockerfile).

    Markdown reports belong in reports/ via write_report only.

    Examples:
        write_file("https://github.com/Roestlab/massformer", "server.py", "...")
        write_file("https://github.com/Roestlab/massformer", "tests/test_server.py", "...")
        write_file("https://github.com/Roestlab/massformer", "helpers/run_analysis.py", "...")
        write_file("https://github.com/Roestlab/massformer", "Dockerfile", "FROM python:3.10-slim\\n...")
    """
    dest = _repo_path(repo_url) / Path((relative_path or "").strip())
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")
    return {"written": str(dest)}


def update_file(repo_url: str, relative_path: str, content: str) -> dict:
    """Overwrite a generated file in the repos directory (full file content, not a patch).

    Read the current file with read_file(repo_url, relative_path) first.

    Examples:
        update_file("https://github.com/Roestlab/massformer", "server.py", "...")
        update_file("https://github.com/Roestlab/massformer", "tests/test_server.py", "...")
    """
    dest = _repo_path(repo_url) / Path((relative_path or "").strip())
    if not dest.exists():
        return {"error": f"File not found: {dest}. Cannot update a file that does not exist."}
    dest.write_text(content, encoding="utf-8")
    return {"updated": str(dest)}


def setup_venv(
    repo_url: str,
    requirements_file: str | None = None,
    pyproject_toml: str | None = None,
    packages: list[str] | None = None,
    python_version: str = "3.10",
) -> dict:
    """Create a virtual environment in output/.venv and install dependencies.

    Always installs fastmcp, pytest, and mcp in addition to any specified packages.
    Uses uv if available, falls back to python -m venv + pip.

    Args:
        repo_url:          Repository URL.
        requirements_file: Repo-relative path to requirements.txt, or None.
        pyproject_toml:    Repo-relative path to pyproject.toml for dep extraction, or None.
        packages:          Extra package names to install (no version pins needed).
        python_version:    Python version string, e.g. "3.10". Must be >= 3.10.

    Example:
        setup_venv("https://github.com/Roestlab/massformer",
                   requirements_file="requirements.txt", python_version="3.10")
        setup_venv("https://github.com/Roestlab/massformer",
                   packages=["torch", "numpy"], python_version="3.11")
    """
    out_dir = _output_dir(repo_url)
    out_dir.mkdir(parents=True, exist_ok=True)
    venv_dir = out_dir / ".venv"
    errors: list[str] = []
    use_uv = subprocess.run(["which", "uv"], capture_output=True).returncode == 0

    # Create venv
    if use_uv:
        r = subprocess.run(
            ["uv", "venv", str(venv_dir), "--python", python_version],
            capture_output=True, text=True,
        )
    else:
        r = subprocess.run(
            [f"python{python_version}", "-m", "venv", str(venv_dir)],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            r = subprocess.run(
                ["python3", "-m", "venv", str(venv_dir)],
                capture_output=True, text=True,
            )
    if r.returncode != 0:
        return {"success": False, "error": f"venv creation failed: {(r.stderr or r.stdout).strip()}"}

    python = _venv_python(out_dir)

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

    install_pkgs = ["fastmcp", "pytest", "mcp"] + (packages or [])
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
        return {"success": False, "venv": str(venv_dir), "python": python, "error": "; ".join(errors)}
    return {"success": True, "venv": str(venv_dir), "python": python}


def check_venv_compat(repo_url: str) -> dict:
    """Check compatibility by replaying the repo's own import statements in the venv.

    Scans the cloned repo's Python files with AST, collects every unique
    `import X` and `from X import Y` where X is an installed package, then
    executes each statement in the venv.  This catches both ABI conflicts
    (numpy 1 vs 2) and removed-API errors (e.g. `from transformers import AdamW`
    removed in transformers>=4.38).

    Returns only failures; successful imports are omitted to keep output small.

    Example:
        check_venv_compat("https://github.com/Roestlab/massformer")
        # -> {"has_conflicts": True,
        #     "conflicts": {"from transformers import AdamW":
        #                       {"error": "cannot import name 'AdamW' ..."}}}
    """
    out_dir  = _output_dir(repo_url).resolve()
    repo_dir = _repo_path(repo_url).resolve()
    python   = _venv_python(out_dir)

    script = textwrap.dedent("""\
        import sys, ast, importlib.metadata, json
        from pathlib import Path

        repo_path = Path(sys.argv[1])

        installed_roots = set()
        for dist in importlib.metadata.distributions():
            top = dist.read_text("top_level.txt")
            if top:
                for n in top.strip().splitlines():
                    n = n.strip()
                    if n and not n.startswith("_"):
                        installed_roots.add(n)
            else:
                record = dist.read_text("RECORD") or ""
                for line in record.splitlines():
                    part = line.split(",")[0].strip().split("/")[0]
                    if (not part
                            or part.endswith((".dist-info", ".data"))
                            or part.startswith(("_", "."))
                            or "." in part):
                        continue
                    installed_roots.add(part.removesuffix(".py"))

        stmts = {}
        for py_file in repo_path.rglob("*.py"):
            try:
                source = py_file.read_text(encoding="utf-8", errors="replace")
                tree   = ast.parse(source, filename=str(py_file))
            except Exception:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root = alias.name.split(".")[0]
                        if root in installed_roots:
                            stmt = f"import {alias.name}"
                            stmts[stmt] = stmt
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.level == 0:
                        root = node.module.split(".")[0]
                        if root in installed_roots:
                            names = ", ".join(a.name for a in node.names)
                            stmt  = f"from {node.module} import {names}"
                            key   = node.module + ":" + ",".join(
                                sorted(a.name for a in node.names)
                            )
                            stmts[key] = stmt

        conflicts = {}
        for stmt in stmts.values():
            try:
                exec(stmt)  # noqa: S102
            except ImportError as e:
                conflicts[stmt] = {"error": str(e)}
            except Exception as e:
                conflicts[stmt] = {"error": type(e).__name__ + ": " + str(e)}

        print(json.dumps({"conflicts": conflicts, "checked": len(stmts)}))
    """)

    check_file = out_dir / "_compat_check.py"
    check_file.write_text(script, encoding="utf-8")
    try:
        r = subprocess.run(
            [python, str(check_file), str(repo_dir)],
            capture_output=True, text=True, timeout=240,
        )
    finally:
        check_file.unlink(missing_ok=True)

    if r.returncode != 0:
        return {"error": f"compat check script failed: {r.stderr.strip()[:500]}"}

    try:
        data = _json.loads(r.stdout.strip())
    except Exception:
        return {"error": f"could not parse compat output: {r.stdout[:300]}"}

    conflicts = data.get("conflicts", {})
    return {
        "conflicts": conflicts,
        "checked": data.get("checked", 0),
        "has_conflicts": bool(conflicts),
    }


def build_docker_image(repo_url: str) -> dict:
    """Run ``docker build`` for project using ``Dockerfile`` at the repository root.

    On success, writes ``.docker_image`` in output/ so ``validate_syntax`` /
    ``run_tests`` use this image and removes ``.docker_build_failures``.

    After ``DOCKER_BUILD_MAX_ATTEMPTS`` failed builds for the same repository, returns
    without running ``docker build`` (``max_attempts_reached: true``).

    Returns:
        {"success": True,  "image": "<tag>", "dockerfile": "<path>",
         "build_log": "<stdout>", "python": "<example docker run ...>"}
        {"success": False, "error": "<message>", "dockerfile": "<path>" if present,
         optional "failed_attempts", "max_attempts_reached"}

    Example:
        write_file(repo_url, "Dockerfile", "FROM python:3.10-slim\\n...")
        build_docker_image(repo_url)
    """
    dv = subprocess.run(["docker", "version"], capture_output=True, text=True)
    if dv.returncode != 0:
        err = (dv.stderr or dv.stdout or "").strip() or "unknown error"
        return {"success": False, "error": f"docker not available: {err}"}

    name      = _repo_name(repo_url)
    repo_root = _repo_path(repo_url)
    out_dir   = _output_dir(repo_url)
    out_dir.mkdir(parents=True, exist_ok=True)

    dockerfile_path = repo_root / "Dockerfile"
    if not dockerfile_path.is_file():
        return {
            "success": False,
            "error": (
                "No Dockerfile at repository root. "
                "Use write_file(repo_url, 'Dockerfile', <full content>) first."
            ),
        }

    counter_path = out_dir / DOCKER_BUILD_FAIL_COUNTER
    marker_path  = out_dir / DOCKER_IMAGE_MARKER

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
                "outcome in the docker report."
            ),
        }

    image_tag = f"alembic-{_docker_safe_image_component(name)}:latest"
    proc = subprocess.Popen(
        ["docker", "build", "-f", str(dockerfile_path), "-t", image_tag, str(repo_root)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    for line in proc.stdout:
        print(line, end="")
        stdout_lines.append(line)
    for line in proc.stderr:
        print(line, end="")
        stderr_lines.append(line)
    proc.wait()

    if proc.returncode != 0:
        # Combine both streams so the agent sees the full picture
        combined = "".join(stdout_lines) + "".join(stderr_lines)
        if not combined.strip():
            combined = "(no output captured from docker build)"
        fails = prev_fails + 1
        counter_path.write_text(f"{fails}\n", encoding="utf-8")
        out: dict = {
            "success": False,
            "dockerfile": str(dockerfile_path),
            "failed_attempts": fails,
            "error": f"docker build failed (exit {proc.returncode}):\n{combined}"[:MAX_BYTES],
        }
        if fails >= DOCKER_BUILD_MAX_ATTEMPTS:
            out["max_attempts_reached"] = True
        return out

    counter_path.unlink(missing_ok=True)
    marker_path.write_text(image_tag + "\n", encoding="utf-8")
    build_log = "".join(stdout_lines)
    run_hint  = (
        f"docker run --rm -v {repo_root.resolve()}:/app -w /app "
        f"{image_tag} python server.py"
    )
    return {
        "success":   True,
        "image":     image_tag,
        "dockerfile": str(dockerfile_path),
        "build_log": build_log[:MAX_BYTES],
        "python":    run_hint,
    }


def validate_syntax(repo_url: str) -> dict:
    """Check server.py for syntax errors.

    Uses Docker if the image is available, the venv if setup, otherwise host Python.

    Example:
        validate_syntax("https://github.com/Roestlab/massformer")
    """
    repo_root = _repo_path(repo_url)
    server    = repo_root / "server.py"
    if not server.exists():
        return {"passed": False, "stage": "syntax", "error": f"server.py not found at {server}"}

    proc = subprocess.Popen(
        _python_exec_argv(repo_url, ["-m", "py_compile", "server.py"]),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(repo_root),
    )
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    for line in proc.stdout:
        print(line, end="")
        stdout_lines.append(line)
    for line in proc.stderr:
        print(line, end="")
        stderr_lines.append(line)
    proc.wait()

    if proc.returncode != 0:
        # py_compile writes errors to stderr
        error_msg = "".join(stderr_lines) or "".join(stdout_lines) or "(no error output)"
        return {"passed": False, "stage": "syntax", "error": error_msg}

    return {"passed": True}


def run_tests(repo_url: str) -> dict:
    """Run the pytest test suite for the generated MCP server.

    Runs pytest tests/ from the clone root.  Returns pass/fail status and
    the full pytest output (stdout + stderr, truncated to 40 KB).

    Requires a Docker image (output/.docker_image) — call build_docker_image first.

    Example:
        run_tests("https://github.com/Roestlab/massformer")
    """
    repo_root = _repo_path(repo_url)
    test_dir  = repo_root / "tests"
    if not test_dir.exists():
        return {"passed": False, "output": f"Test directory not found: {test_dir}"}

    marker = _output_dir(repo_url) / DOCKER_IMAGE_MARKER
    if not marker.exists():
        return {
            "passed": False,
            "output": (
                f"Docker image marker not found at {marker}. "
                "The docker agent must run build_docker_image successfully before tests can run."
            ),
        }

    try:
        r = subprocess.run(
            _python_exec_argv(
                repo_url,
                ["-m", "pytest", "tests", "-v", "--tb=short", "--no-header"],
            ),
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(repo_root),
        )
    except subprocess.TimeoutExpired:
        return {"passed": False, "output": "pytest timed out after 120 seconds."}

    output = (r.stdout + r.stderr)[:MAX_BYTES]
    return {"passed": r.returncode == 0, "output": output}


def write_report(repo_url: str, report_name: str, content: str) -> dict:
    """Write a Markdown report to this repo's reports directory.

    Args:
        repo_url:    Repository URL.
        report_name: Filename without the .md extension: "exploration", "environment",
                     "server", "docker", or "validation".
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
