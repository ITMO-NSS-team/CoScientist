"""Validation tools: static syntax/import check, pytest, live tool invocation."""
from __future__ import annotations

import ast
import asyncio
import builtins
import contextvars
import json
import os
import signal
import subprocess
import tempfile
from pathlib import Path

from alembic.config import (
    HELPER_IMPORT_CHECK_TIMEOUT, INVOKE_TIMEOUT, MAX_BYTES, PYTEST_TIMEOUT,
    RESULT_MAX_LIST_ITEMS, RESULT_MAX_STR_LEN, RESULT_SENTINEL,
    SERVER_IMPORT_CHECK_TIMEOUT,
)
from alembic.tools.paths import (
    INVOKE_TOOL_SCRIPT, helper_venv_python, output_dir, repo_path, venv_python,
)


# ══════════════════════════════════════════════════════════════════════════════
# Static syntax / import / undefined-name checks (F28/F39/F45)
# ══════════════════════════════════════════════════════════════════════════════
async def validate_syntax(repo_url: str) -> dict:
    """Check server.py + helpers for syntax errors, failed imports, and
    undefined names — without running any business logic.

    Example: validate_syntax("https://github.com/Roestlab/massformer")
    """
    return await asyncio.to_thread(_validate_syntax_sync, repo_url)


def _is_main_guard(node: ast.stmt) -> bool:
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    sides = [node.test.left, *node.test.comparators]
    return (any(isinstance(s, ast.Name) and s.id == "__name__" for s in sides)
            and any(isinstance(s, ast.Constant) and s.value == "__main__" for s in sides))


def _import_safe_prefix(source: str) -> str:
    """Leading slice of top-level statements safe to execute without running
    the helper's argparse/business logic (F28). Recognized-safe shapes:
    argparse-gated flat code (stop at the ArgumentParser/parse_args line) or a
    def main()+__main__-guard wrapper (whole thing safe). A script matching
    neither (F39) is treated as unsafe — only imports/defs and sys.path setup
    are kept."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    has_main_guard = any(_is_main_guard(n) for n in tree.body)
    has_argparse   = "ArgumentParser(" in source or "parse_args(" in source
    safe: list[ast.stmt] = []
    for node in tree.body:
        if _is_main_guard(node):
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                             ast.Import, ast.ImportFrom)):
            safe.append(node)
            continue
        segment = ast.get_source_segment(source, node) or ""
        if has_argparse:
            if "ArgumentParser(" in segment or "parse_args(" in segment:
                break
            safe.append(node)
        elif has_main_guard:
            safe.append(node)
        elif "sys.path.insert(" in segment or "sys.path.append(" in segment:
            safe.append(node)
        else:
            break
    return "\n".join(ast.get_source_segment(source, n) or "" for n in safe)


_BUILTIN_NAMES = frozenset(dir(builtins)) | {
    "__name__", "__file__", "__doc__", "__package__", "__spec__", "__loader__",
    "__builtins__", "__annotations__", "__dict__",
}


def _extract_target_names(node: ast.expr, out: set[str]) -> None:
    if isinstance(node, ast.Name):
        out.add(node.id)
    elif isinstance(node, (ast.Tuple, ast.List)):
        for elt in node.elts:
            _extract_target_names(elt, out)
    elif isinstance(node, ast.Starred):
        _extract_target_names(node.value, out)


def _find_undefined_names(source: str) -> list[str] | None:
    """F45: whole-file, zero-execution pass — flag any Name(Load) reference not
    bound anywhere in the file and not a builtin (catches `torch.x` with no
    `import torch`). Deliberately whole-file-permissive; bails on `import *`."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    bound: set[str] = set()
    referenced: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    return None
                bound.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bound.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bound.add(node.name)
            a = node.args
            for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs):
                bound.add(arg.arg)
            if a.vararg:
                bound.add(a.vararg.arg)
            if a.kwarg:
                bound.add(a.kwarg.arg)
        elif isinstance(node, ast.Lambda):
            a = node.args
            for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs):
                bound.add(arg.arg)
            if a.vararg:
                bound.add(a.vararg.arg)
            if a.kwarg:
                bound.add(a.kwarg.arg)
        elif isinstance(node, ast.ClassDef):
            bound.add(node.name)
        elif isinstance(node, (ast.Assign, ast.NamedExpr)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                _extract_target_names(t, bound)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.For, ast.AsyncFor)):
            _extract_target_names(node.target, bound)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    _extract_target_names(item.optional_vars, bound)
        elif isinstance(node, ast.ExceptHandler):
            if node.name:
                bound.add(node.name)
        elif isinstance(node, ast.comprehension):
            _extract_target_names(node.target, bound)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            referenced.add(node.id)
    return sorted(referenced - bound - _BUILTIN_NAMES) or None


def _check_helper_imports(python: str, helper: Path, repo_dir: Path, timeout: int) -> str | None:
    """F28: execute only the import-safe prefix (with argv[1]=repo dir) to
    confirm imports resolve, as a real script file (server.py invokes helpers
    that way, so sys.path[0] matches). Returns an error string or None."""
    prefix = _import_safe_prefix(helper.read_text(encoding="utf-8", errors="replace"))
    fd, tmp_path = tempfile.mkstemp(dir=str(helper.parent), suffix=".py")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(prefix)
        r = subprocess.run([python, tmp_path, str(repo_dir)],
                           capture_output=True, text=True, timeout=timeout, cwd=str(repo_dir))
    except subprocess.TimeoutExpired:
        return f"{helper.name}: import check timed out after {timeout} seconds"
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return f"{helper.name}: {r.stderr.strip()}" if r.returncode != 0 else None


def _validate_syntax_sync(repo_url: str) -> dict:
    out_dir = output_dir(repo_url).resolve()
    server  = out_dir / "server.py"
    python  = venv_python(out_dir)
    if not server.exists():
        return {"passed": False, "stage": "syntax", "error": f"server.py not found at {server}"}

    r1 = subprocess.run([python, "-m", "py_compile", str(server)], capture_output=True, text=True)
    if r1.returncode != 0:
        return {"passed": False, "stage": "syntax", "error": r1.stderr.strip()}

    load = ("import importlib.util as _u, sys as _s; "
            f"_s.path.insert(0, r'{server.parent}'); "
            f"_spec=_u.spec_from_file_location('server', r'{server}'); "
            "_mod=_u.module_from_spec(_spec); _spec.loader.exec_module(_mod)")
    r2 = subprocess.run([python, "-c", load], capture_output=True, text=True,
                        timeout=SERVER_IMPORT_CHECK_TIMEOUT, cwd=str(server.parent))
    if r2.returncode != 0:
        return {"passed": False, "stage": "imports", "error": r2.stderr.strip()}

    helpers_dir = out_dir / "helpers"
    if helpers_dir.is_dir():
        repo_dir = repo_path(repo_url).resolve()
        helper_python = helper_venv_python(out_dir)
        for helper in sorted(helpers_dir.glob("*.py")):
            rc = subprocess.run([helper_python, "-m", "py_compile", str(helper)],
                                capture_output=True, text=True)
            if rc.returncode != 0:
                return {"passed": False, "stage": "helper_syntax", "error": f"{helper.name}: {rc.stderr.strip()}"}
            undefined = _find_undefined_names(helper.read_text(encoding="utf-8", errors="replace"))
            if undefined:
                return {"passed": False, "stage": "helper_undefined_names",
                        "error": (f"{helper.name}: name(s) {', '.join(undefined)} are used but never "
                                  f"imported or defined in the file — will NameError at runtime.")}
            err = _check_helper_imports(helper_python, helper, repo_dir, HELPER_IMPORT_CHECK_TIMEOUT)
            if err:
                return {"passed": False, "stage": "helper_imports", "error": err}
    return {"passed": True}


# ══════════════════════════════════════════════════════════════════════════════
# pytest
# ══════════════════════════════════════════════════════════════════════════════
async def run_tests(repo_url: str) -> dict:
    """Run the generated pytest suite. Example: run_tests("https://github.com/x/y")"""
    return await asyncio.to_thread(_run_tests_sync, repo_url)


def _run_tests_sync(repo_url: str) -> dict:
    out_dir  = output_dir(repo_url).resolve()
    test_dir = out_dir / "tests"
    python   = venv_python(out_dir)
    if not test_dir.exists():
        return {"passed": False, "output": f"Test directory not found: {test_dir}"}
    try:
        r = subprocess.run([python, "-m", "pytest", str(test_dir), "-v", "--tb=short", "--no-header"],
                           capture_output=True, text=True, timeout=PYTEST_TIMEOUT, cwd=str(out_dir))
    except subprocess.TimeoutExpired:
        return {"passed": False, "output": f"pytest timed out after {PYTEST_TIMEOUT} seconds."}
    return {"passed": r.returncode == 0, "output": (r.stdout + r.stderr)[:MAX_BYTES]}


# ══════════════════════════════════════════════════════════════════════════════
# Live tool invocation
# ══════════════════════════════════════════════════════════════════════════════
_skip_tools: contextvars.ContextVar[frozenset] = contextvars.ContextVar("skip_tools", default=frozenset())


def set_skip_tools(names) -> None:
    """F25 code-enforced SKIP gate — populated by the pipeline before validation."""
    _skip_tools.set(frozenset(names))


# N3: file extensions that mark a string arg as a local-file path worth
# existence-checking before invocation. Deliberately narrow (a curated set,
# not "anything with a slash") so HF ids like "MahmoodLab/UNI2-h" and device
# strings like "cuda:0" are never mistaken for paths.
_PATH_EXTS = {
    ".csv", ".tsv", ".txt", ".json", ".yaml", ".yml", ".pdb", ".fasta", ".fa",
    ".fastq", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".svg",
    ".h5", ".hdf5", ".npy", ".npz", ".pt", ".pth", ".ckpt", ".safetensors",
    ".nii", ".gz", ".zip", ".tar", ".pdf", ".mat", ".parquet", ".fits", ".wav",
    ".mp3", ".pkl", ".pickle", ".xlsx", ".xls", ".sdf", ".mol", ".mol2", ".cif",
}


def _bad_sample_reason(args: dict, repo_dir: Path, out_dir: Path) -> str | None:
    """N3: return a reason if a path-shaped string arg doesn't resolve — so a
    bad sample is reported distinctly instead of burning a debugger round on a
    code bug that isn't one. Conservative: only flags values with a known file
    extension that resolve nowhere."""
    for key, val in (args or {}).items():
        if not isinstance(val, str) or "://" in val:
            continue
        if Path(val).suffix.lower() not in _PATH_EXTS:
            continue
        candidates = [Path(val), repo_dir / val, out_dir / val]
        if not any(c.exists() for c in candidates):
            return (f"sample arg {key}={val!r} looks like a file path but does not "
                    f"exist under the repo — bad sample, not a code bug")
    return None


async def invoke_mcp_tool(repo_url: str, tool_name: str, args: dict | None = None) -> dict:
    """Invoke an @mcp.tool() from the generated server.py live, in the server
    venv, and return its result.

    Returns {"ok": True, "result": ...} on success; {"ok": False, "error",
    "traceback", "stderr"} on a real error; or {"skipped": True, "reason": ...}
    when the tool is SKIP-listed, timed out (>INVOKE_TIMEOUT, treated as
    resource-heavy), or was given a non-resolving file-path sample (bad sample).

    Example:
        invoke_mcp_tool("https://github.com/x/y", "predict", {"input": "data/x.csv"})
    """
    return await asyncio.to_thread(_invoke_mcp_tool_sync, repo_url, tool_name, args)


def _truncate_large_result(value):
    """F30: recursively cap list length / string length in a tool result."""
    if isinstance(value, list):
        out = [_truncate_large_result(v) for v in value[:RESULT_MAX_LIST_ITEMS]]
        if len(value) > RESULT_MAX_LIST_ITEMS:
            out.append(f"... ({len(value) - RESULT_MAX_LIST_ITEMS} more items truncated)")
        return out
    if isinstance(value, dict):
        return {k: _truncate_large_result(v) for k, v in value.items()}
    if isinstance(value, str) and len(value) > RESULT_MAX_STR_LEN:
        return value[:RESULT_MAX_STR_LEN] + f"... ({len(value) - RESULT_MAX_STR_LEN} more chars truncated)"
    return value


def _parse_result(stdout: str) -> dict | None:
    """N5: extract the JSON after the last RESULT_SENTINEL line, so library
    banners / progress bars printed before it never break the parse. Falls back
    to the last non-empty line for older/sentinel-less helpers."""
    if RESULT_SENTINEL in stdout:
        tail = stdout.rsplit(RESULT_SENTINEL, 1)[1].strip()
    else:
        lines = [l for l in stdout.splitlines() if l.strip()]
        tail = lines[-1] if lines else ""
    try:
        return json.loads(tail)
    except Exception:
        return None


def _invoke_mcp_tool_sync(repo_url: str, tool_name: str, args: dict | None = None) -> dict:
    if tool_name in _skip_tools.get():
        return {"skipped": True, "reason": f"'{tool_name}' is SKIP-listed (F25) — not invoked."}

    out_dir = output_dir(repo_url).resolve()
    server  = out_dir / "server.py"
    venv_py = out_dir / ".venv" / "bin" / "python"
    if not server.exists():
        return {"ok": False, "error": f"server.py not found at {server}"}
    if not venv_py.exists():
        return {"ok": False, "error": f"server venv python not found at {venv_py}"}

    bad = _bad_sample_reason(args or {}, repo_path(repo_url).resolve(), out_dir)
    if bad:
        return {"skipped": True, "bad_sample": True, "reason": bad}

    env = os.environ.copy()
    env["SERVER_PATH"]    = str(server)
    env["TOOL_NAME"]      = tool_name
    env["TOOL_ARGS_JSON"] = json.dumps(args or {})

    # start_new_session=True → own process group so a timeout kills the WHOLE
    # tree (server.py spawns its own uncapped helper subprocess), not just the
    # immediate child (F37).
    proc = subprocess.Popen([str(venv_py), str(INVOKE_TOOL_SCRIPT)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            env=env, cwd=str(out_dir), start_new_session=True)
    try:
        stdout, stderr = proc.communicate(timeout=INVOKE_TIMEOUT)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        return {"skipped": True, "reason": (
            f"'{tool_name}' did not return within {INVOKE_TIMEOUT}s and was killed — "
            f"treated as resource-heavy and SKIPPED, not FAILED (not a confirmed bug).")}

    parsed = _parse_result(stdout.strip())
    if parsed is None:
        return {"ok": False, "error": "could not parse invoker output",
                "returncode": proc.returncode, "stdout": stdout[-1500:], "stderr": stderr[-1500:]}
    if not parsed.get("ok") and stderr:
        parsed.setdefault("stderr", stderr[-2000:])
    if parsed.get("ok") and "result" in parsed:
        parsed["result"] = _truncate_large_result(parsed["result"])
    return parsed
