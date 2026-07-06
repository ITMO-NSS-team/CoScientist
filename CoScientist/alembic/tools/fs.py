"""Filesystem tools: clone/read/search the repo, read/write output and reports."""
import asyncio
import subprocess

from alembic.tools.paths import (
    IGNORE_EXTS, MAX_BYTES, output_dir, rel_or_ignored, repo_path,
    reports_dir,
)


async def clone_repo(repo_url: str) -> dict:
    """Clone a GitHub repository to local disk.

    Returns the local path and a flat file list for you to select from.

    Example:
        clone_repo("https://github.com/Roestlab/massformer")
        # -> {"local_path": ".alembic/massformer/repos", "files": [...]}
    """
    # F23: run on a worker thread — see bash()/bash_env() in shell.py for why.
    return await asyncio.to_thread(_clone_repo_sync, repo_url)


def _clone_repo_sync(repo_url: str) -> dict:
    dest = repo_path(repo_url)
    if not dest.exists():
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth=1", repo_url, str(dest)],
            check=True, capture_output=True,
        )

    files = [rel for p in dest.rglob("*") if (rel := rel_or_ignored(p, dest))]
    return {"local_path": str(dest), "files": sorted(files)}


def read_file(repo_url: str, path: str) -> dict:
    """Read a text file from the locally cloned repository.

    Returns up to 40 KB of content. Do NOT use this on data files (.csv,
    .parquet, .tsv, .json arrays) — use bash("head -n 20 <path>") instead.

    Example:
        read_file("https://github.com/Roestlab/massformer", "README.md")
    """
    full = repo_path(repo_url) / path
    if not full.exists():
        return {"error": f"File not found: {path}."}
    if full.suffix in IGNORE_EXTS:
        return {"error": f"Binary/data file skipped: {path}."}
    raw = full.read_bytes()[:MAX_BYTES]
    return {"path": path, "content": raw.decode("utf-8", errors="replace")}


def search(repo_url: str, pattern: str) -> dict:
    """Find files in the cloned repo matching a glob pattern.

    Examples:
        search("https://github.com/Roestlab/massformer", "**/*.yaml")
        search("https://github.com/Roestlab/massformer", "*.sh")
    """
    dest = repo_path(repo_url)
    matched = [rel for p in dest.glob(pattern) if (rel := rel_or_ignored(p, dest))]
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
    path = reports_dir(repo_url) / f"{report_name}.md"
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
    dest = output_dir(repo_url) / relative_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")
    return {"written": str(dest)}


def read_output_file(repo_url: str, relative_path: str) -> dict:
    """Read a file from the output directory for this repo.

    Examples:
        read_output_file("https://github.com/Roestlab/massformer", "server.py")
        read_output_file("https://github.com/Roestlab/massformer", "tests/test_server.py")
    """
    full = output_dir(repo_url) / relative_path
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
    dest = output_dir(repo_url) / relative_path
    if not dest.exists():
        return {"error": f"File not found: {dest}. Cannot update a file that does not exist."}
    dest.write_text(content, encoding="utf-8")
    return {"updated": str(dest)}


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
    reports = reports_dir(repo_url)
    reports.mkdir(parents=True, exist_ok=True)
    out = reports / f"{report_name}.md"
    out.write_text(content, encoding="utf-8")
    return {"report_path": str(out)}
