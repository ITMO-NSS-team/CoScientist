"""Validation tools: syntax/import check, pytest run, and live tool invocation."""
import ast
import asyncio
import json
import os
import subprocess
import tempfile
from pathlib import Path

from alembic.tools.paths import (
    INVOKE_TOOL_SCRIPT, MAX_BYTES, helper_venv_python, output_dir, repo_path, venv_python,
)


async def validate_syntax(repo_url: str) -> dict:
    """Check server.py for syntax errors and failed imports.

    Example:
        validate_syntax("https://github.com/Roestlab/massformer")
    """
    # F23: run on a worker thread — see bash()/bash_env() in shell.py for why.
    return await asyncio.to_thread(_validate_syntax_sync, repo_url)


def _is_main_guard(node: ast.stmt) -> bool:
    """True for ``if __name__ == "__main__":`` (or `.../ "== '__main__'"`)."""
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    test = node.test
    sides = [test.left, *test.comparators]
    return (
        any(isinstance(s, ast.Name) and s.id == "__name__" for s in sides)
        and any(isinstance(s, ast.Constant) and s.value == "__main__" for s in sides)
    )


def _import_safe_prefix(source: str) -> str:
    """Return the leading slice of ``source``'s top-level statements that is
    safe to execute without triggering the helper's real argparse/business
    logic (F28).

    Helper scripts (coder.py Step 3) come in two observed shapes: flat
    top-level code (``parser = argparse.ArgumentParser(); ...; result =
    obj.run(...)`` with no function wrapper — the documented template), or
    a `def main():` wrapper with real logic only run via an
    `if __name__ == "__main__":` guard. Function/class definitions and
    imports are always safe to execute (defining doesn't run the body);
    only a *top-level* statement that itself constructs/parses an argparse
    parser is where real execution would begin, and the `__main__` guard is
    never executed at all.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source  # let the real py_compile check below report this

    safe: list[ast.stmt] = []
    for node in tree.body:
        if _is_main_guard(node):
            continue  # never executed — this is exactly the real entrypoint
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                              ast.ClassDef, ast.Import, ast.ImportFrom)):
            safe.append(node)  # defining, not calling — always safe
            continue
        segment = ast.get_source_segment(source, node) or ""
        if "ArgumentParser(" in segment or "parse_args(" in segment:
            break  # top-level argparse construction/parsing — stop here
        safe.append(node)

    return "\n".join(ast.get_source_segment(source, n) or "" for n in safe)


def _check_helper_imports(python: str, helper: Path, repo_dir: Path) -> str | None:
    """F28: verify a helper script's imports actually resolve, without running
    its business logic. Helper scripts (coder.py Step 3) always accept
    REPO_PATH as sys.argv[1] and do their own sys.path manipulation with it
    before importing from the repo's own modules — so we can't just import
    the file naively, and we can't safely exec() the whole thing (the rest of
    the file is real argparse + tool logic, which would run for real).

    Instead, execute only the import-safe prefix computed by
    ``_import_safe_prefix`` (see there for the two helper-script shapes it
    handles), with sys.argv[1] set to the real repo clone dir so any inline
    ``sys.path.insert(0, sys.argv[1])``-style setup in that prefix runs
    exactly as it would at real invocation time. Returns an error string
    (stderr) on failure, or None if the prefix executed cleanly.

    The prefix is written to a temp file *in the same directory as the
    original helper* and run as a real script (``python tmp_file.py ...``),
    not via ``-c`` — server.py always invokes helpers this way, and Python
    sets ``sys.path[0]`` to the script's own directory only when run this
    way (``-c`` sets it to ``''``/cwd instead). Getting this wrong produces
    false positives: verified against `aizynthfinder`'s helpers, where a
    ``-c``-based check spuriously failed on a file that genuinely works at
    real invocation time.
    """
    source = helper.read_text(encoding="utf-8", errors="replace")
    prefix = _import_safe_prefix(source)

    fd, tmp_path = tempfile.mkstemp(dir=str(helper.parent), suffix=".py")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(prefix)
        r = subprocess.run(
            [python, tmp_path, str(repo_dir)],
            capture_output=True, text=True, timeout=30,
            cwd=str(repo_dir),
        )
    except subprocess.TimeoutExpired:
        return f"{helper.name}: import check timed out after 30 seconds"
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    if r.returncode != 0:
        return f"{helper.name}: {r.stderr.strip()}"
    return None


def _validate_syntax_sync(repo_url: str) -> dict:
    out_dir = output_dir(repo_url).resolve()
    server  = out_dir / "server.py"
    python  = venv_python(out_dir)
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
        f"_s.path.insert(0, r'{server.parent}'); "
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

    # F28: server.py's own import-exec above never touches helpers/*.py —
    # those are only ever invoked lazily via subprocess when a tool actually
    # runs, so a hallucinated import/class/function inside one is otherwise
    # invisible until it burns a live invoke_mcp_tool + debugger round-trip.
    helpers_dir = out_dir / "helpers"
    if helpers_dir.is_dir():
        repo_dir    = repo_path(repo_url).resolve()
        # Helpers import the repo's own packages, which in two-venv mode
        # live in .venv-repo, not .venv — using the wrong venv here produces
        # false ModuleNotFoundErrors (confirmed against `ase`'s real
        # two-venv layout, where numpy is repo-side only).
        helper_python = helper_venv_python(out_dir)
        for helper in sorted(helpers_dir.glob("*.py")):
            r3 = subprocess.run(
                [helper_python, "-m", "py_compile", str(helper)],
                capture_output=True, text=True,
            )
            if r3.returncode != 0:
                return {"passed": False, "stage": "helper_syntax",
                        "error": f"{helper.name}: {r3.stderr.strip()}"}
            err = _check_helper_imports(helper_python, helper, repo_dir)
            if err:
                return {"passed": False, "stage": "helper_imports", "error": err}

    return {"passed": True}


async def run_tests(repo_url: str) -> dict:
    """Run the pytest test suite for the generated MCP server.

    Example:
        run_tests("https://github.com/Roestlab/massformer")
    """
    # F23: run on a worker thread — see bash()/bash_env() in shell.py for why.
    return await asyncio.to_thread(_run_tests_sync, repo_url)


def _run_tests_sync(repo_url: str) -> dict:
    out_dir  = output_dir(repo_url).resolve()
    test_dir = out_dir / "tests"
    python   = venv_python(out_dir)
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


async def invoke_mcp_tool(repo_url: str, tool_name: str, args: dict | None = None) -> dict:
    """Actually invoke an @mcp.tool() function from the generated server.py.

    Runs server.py inside the server venv (where fastmcp is installed),
    looks up ``tool_name``, and calls it with ``args`` as kwargs. The
    server's module-level ``mcp.run()`` is monkey-patched to a no-op so
    the import does not block. FastMCP wraps tool functions in a
    FunctionTool — we unwrap to the original callable before calling.

    Use this in the validator/debugger flow to surface runtime errors
    that pytest with mocked subprocess cannot catch (missing apt
    packages, missing pip deps, helper-script bugs, wrong argv).

    Args:
        repo_url:  Repository URL.
        tool_name: Exact function name (the @mcp.tool() target).
        args:      Dict of keyword arguments to pass to the tool.

    Returns:
        On success: ``{"ok": True, "result": <whatever the tool returned>}``.
        On error:   ``{"ok": False, "error": "<ExcName: msg>",
                      "traceback": "<full traceback>",
                      "stderr": "<tail of stderr>"}``.

    Example:
        invoke_mcp_tool(
            "https://github.com/Roestlab/massformer",
            "run_inference",
            {"smiles_fp": "predictions/example_smiles.csv",
             "output_fp": "predictions/demo_out.csv",
             "custom_fp": "config/demo/demo_eval.yml",
             "checkpoint_name": "demo",
             "device": -1},
        )
    """
    # F23: run on a worker thread — see bash()/bash_env() in shell.py for why.
    return await asyncio.to_thread(_invoke_mcp_tool_sync, repo_url, tool_name, args)


def _invoke_mcp_tool_sync(repo_url: str, tool_name: str, args: dict | None = None) -> dict:
    out_dir = output_dir(repo_url).resolve()
    server  = out_dir / "server.py"
    venv_py = out_dir / ".venv" / "bin" / "python"
    if not server.exists():
        return {"ok": False, "error": f"server.py not found at {server}"}
    if not Path(venv_py).exists():
        return {"ok": False, "error": f"server venv python not found at {venv_py}"}

    env = os.environ.copy()
    env["SERVER_PATH"]    = str(server)
    env["TOOL_NAME"]      = tool_name
    env["TOOL_ARGS_JSON"] = json.dumps(args or {})

    try:
        r = subprocess.run(
            [str(venv_py), str(INVOKE_TOOL_SCRIPT)],
            capture_output=True, text=True, env=env, timeout=900,
            cwd=str(out_dir),
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "invocation timed out after 900 seconds"}

    # The invoker prints a single JSON line. If it crashed before printing,
    # surface stderr so the debugger can see what failed.
    stdout = r.stdout.strip()
    last_line = stdout.splitlines()[-1] if stdout else ""
    try:
        parsed = json.loads(last_line)
        if not parsed.get("ok") and r.stderr:
            parsed.setdefault("stderr", r.stderr[-2000:])
        return parsed
    except Exception:
        return {
            "ok": False,
            "error": "could not parse invoker output",
            "returncode": r.returncode,
            "stdout": stdout[-1500:],
            "stderr": r.stderr[-1500:],
        }
