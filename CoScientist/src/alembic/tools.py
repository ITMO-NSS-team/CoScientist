import subprocess
from pathlib import Path

REPO_DIR = Path("/tmp/repos")
REPORTS_DIR = Path("/tmp/alembic_reports")
OUTPUT_DIR = Path("/tmp/alembic_output")
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


def _repo_path(repo_url: str) -> Path:
    name = repo_url.rstrip("/").split("/")[-1].removesuffix(".git")
    return REPO_DIR / name


def clone_repo(repo_url: str) -> dict:
    """Clone a GitHub repository to local disk.

    Call this first — before any other tool. Returns the local path and
    a flat file list for you to select from.

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


def read_report(repo_url: str) -> dict:
    """Read the Markdown analysis report previously written by the explorer agent.

    Call this first — it gives you the repo description and MCP usage scenarios
    to implement as tools.

    Example:
        read_report("https://github.com/Roestlab/massformer")
        # -> {"report_path": "...", "content": "# massformer\\n..."}
    """
    name = repo_url.rstrip("/").split("/")[-1].removesuffix(".git")
    path = REPORTS_DIR / f"{name}.md"
    if not path.exists():
        return {"error": f"No report found at {path}. Run the explorer agent first."}
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
    name = repo_url.rstrip("/").split("/")[-1].removesuffix(".git")
    dest = OUTPUT_DIR / name / relative_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")
    return {"written": str(dest)}


def write_report(repo_url: str, content: str) -> dict:
    """Write the final Markdown analysis report for the repository.

    Call this as the last step after you have explored the repo.
    The report must contain:
      1. A concise description of what the repo does.
      2. 1–5 usage scenarios most likely to be useful as MCP tools.

    Args:
        repo_url: The repository URL (used to name the output file).
        content:  Full Markdown content of the report.

    Example:
        write_report(
            "https://github.com/Roestlab/massformer",
            "# massformer\\n\\n## Description\\n...\\n\\n## MCP Usage Scenarios\\n..."
        )
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    name = repo_url.rstrip("/").split("/")[-1].removesuffix(".git")
    out = REPORTS_DIR / f"{name}.md"
    out.write_text(content, encoding="utf-8")
    return {"report_path": str(out)}
