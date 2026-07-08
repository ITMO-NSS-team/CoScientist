"""Filesystem tools: clone/read/search the repo, read/write output and reports."""
import asyncio
import contextvars
import re
import subprocess

import yaml

from alembic.tools.paths import (
    IGNORE_EXTS, MAX_BYTES, output_dir, rel_or_ignored, repo_path,
    reports_dir,
)

# ── read_file de-dup (audit N1 follow-up) ──────────────────────────────────────
# The Explorer was observed cycling read_file over the same handful of files
# (eda.py, ppg.py, …) many times, thrashing to the step ceiling. This tracks the
# paths already read *within one agent invocation* and stubs a repeat so the
# re-read is free and the model is nudged to move on.
#
# Scoped, NOT global: agent_runtime.enable_read_dedup() installs a fresh set at
# the start of each agent turn only for the agent that loops (the Explorer), and
# None for every other agent. So a *different* agent (e.g. the Coder) that later
# reads a file the Explorer already read still gets its full content — the set is
# per-invocation, and disabled entirely outside the Explorer.
_read_seen: contextvars.ContextVar[set | None] = contextvars.ContextVar(
    "read_seen", default=None
)


def enable_read_dedup(enabled: bool) -> None:
    """Called by agent_runtime at each agent-invocation start: a fresh empty set
    when this agent should de-dup its own repeated reads, None to disable."""
    _read_seen.set(set() if enabled else None)


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
            # --recurse-submodules: some repos vendor real dependencies as git
            # submodules (e.g. auto-sklearn's autosklearn/automl_common) —
            # without this, the submodule directory clones empty, and any
            # import touching it fails with a misleading ModuleNotFoundError
            # that reads like a missing pip package, not a missing submodule.
            # --shallow-submodules keeps submodules shallow too, consistent
            # with the main --depth=1 (full submodule history isn't needed).
            ["git", "clone", "--depth=1", "--recurse-submodules",
             "--shallow-submodules", repo_url, str(dest)],
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
    if full.is_dir():
        return {"error": f"'{path}' is a directory, not a file. Use search() or bash('ls') to list its contents."}
    if full.suffix in IGNORE_EXTS:
        return {"error": f"Binary/data file skipped: {path}."}

    seen = _read_seen.get()
    if seen is not None:
        if path in seen:
            return {
                "path": path,
                "already_read": True,
                "note": (
                    f"You already read '{path}' earlier in this session and its "
                    "content has not changed. Do NOT read it again — use what you "
                    "already have. If you are cycling back to files you have "
                    "already read, you are done exploring: write the report now."
                ),
            }
        seen.add(path)

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
    if full.is_dir():
        return {"error": f"'{relative_path}' is a directory, not a file."}
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


_SAMPLES_FENCE_RE = re.compile(r"```ya?ml\s*\n(.*?)```", re.DOTALL)


def parse_samples_block(repo_url: str) -> dict:
    """F25: parse the coder report's ``## Sample invocations`` fenced YAML
    block (coder.py Step 6) into a plain ``{tool_name: args_dict | "SKIP"}``
    dict, so the SKIP/invoke split can be code-computed instead of trusted
    to the validator LLM re-reading the block correctly on every run.

    Not an agent tool — called directly by main.py before the Validator
    stage starts. Returns {} on any parse failure (missing report, no
    fenced block, malformed YAML) rather than raising: callers must treat
    that as "unknown" and fall back to the validator parsing the block
    itself, exactly as it always has.
    """
    server_md = reports_dir(repo_url) / "server.md"
    if not server_md.exists():
        return {}
    match = _SAMPLES_FENCE_RE.search(server_md.read_text(encoding="utf-8", errors="replace"))
    if not match:
        return {}
    try:
        parsed = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return {}
    samples = parsed.get("samples") if isinstance(parsed, dict) else None
    return samples if isinstance(samples, dict) else {}
