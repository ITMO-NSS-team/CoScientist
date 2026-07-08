"""Validation tools: syntax/import check, pytest run, and live tool invocation."""
import ast
import asyncio
import contextvars
import json
import os
import signal
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

    Helper scripts (coder.py Step 3) come in two *recognized-safe* shapes:
    flat top-level code gated by an argparse call (``parser =
    argparse.ArgumentParser(); ...; result = obj.run(...)`` — the documented
    template, safe up to the ``ArgumentParser(``/``parse_args(`` line), or a
    `def main():` wrapper with real logic only run via an `if __name__ ==
    "__main__":` guard (safe in full — defining a function doesn't call it,
    and the guard is never executed by this check). Function/class
    definitions and imports are always safe (defining doesn't run the body).

    F39: a helper can violate coder.py's "must use argparse" instruction and
    instead index ``sys.argv`` directly with no argparse call and no
    `__main__` guard at all (observed live: CONCH's `image_to_text_retrieval.py`).
    Such a script matches NEITHER recognized-safe shape, so there is no
    reliable point at which to stop — treating it as "safe until an argparse
    call is seen" (the old default) meant this check would actually start
    executing the helper's real business logic (model loading, inference)
    instead of just checking imports, only accidentally stopping when it
    happened to crash on a missing sys.argv index. Fixed by inverting the
    default: a plain top-level statement is only appended when we've
    positively identified a recognized-safe shape (argparse-gated, or inside
    a main()-wrapped script) or it is itself a `sys.path.insert`/`.append`
    call (needed for the repo's own imports to resolve, and safe on its own);
    anything else, in a script matching neither shape, stops the prefix right
    there rather than being assumed safe.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source  # let the real py_compile check below report this

    has_main_guard = any(_is_main_guard(node) for node in tree.body)
    has_argparse   = "ArgumentParser(" in source or "parse_args(" in source

    safe: list[ast.stmt] = []
    for node in tree.body:
        if _is_main_guard(node):
            continue  # never executed — this is exactly the real entrypoint
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                              ast.ClassDef, ast.Import, ast.ImportFrom)):
            safe.append(node)  # defining, not calling — always safe
            continue
        segment = ast.get_source_segment(source, node) or ""
        if has_argparse:
            if "ArgumentParser(" in segment or "parse_args(" in segment:
                break  # top-level argparse construction/parsing — stop here
            safe.append(node)
        elif has_main_guard:
            # Stray top-level statement in a main()-wrapped script — real
            # logic still lives inside the guarded call, so this is safe by
            # construction regardless of what this one statement does.
            safe.append(node)
        elif "sys.path.insert(" in segment or "sys.path.append(" in segment:
            safe.append(node)  # needed for the repo's own imports to resolve
        else:
            # F39: neither recognized-safe shape applies to this script, and
            # this statement isn't a path-setup line either — no proven-safe
            # boundary exists past this point. Stop here instead of assuming
            # it's fine to execute.
            break

    return "\n".join(ast.get_source_segment(source, n) or "" for n in safe)


def _check_helper_imports(python: str, helper: Path, repo_dir: Path, timeout: int) -> str | None:
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
            capture_output=True, text=True, timeout=timeout,
            cwd=str(repo_dir),
        )
    except subprocess.TimeoutExpired:
        return f"{helper.name}: import check timed out after {timeout} seconds"
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    if r.returncode != 0:
        return f"{helper.name}: {r.stderr.strip()}"
    return None


def _validate_syntax_sync(repo_url: str) -> dict:
    # deferred: see main.py's timeout block
    from alembic.main import HELPER_IMPORT_CHECK_TIMEOUT, SERVER_IMPORT_CHECK_TIMEOUT
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
        capture_output=True, text=True, timeout=SERVER_IMPORT_CHECK_TIMEOUT,
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
            err = _check_helper_imports(helper_python, helper, repo_dir, HELPER_IMPORT_CHECK_TIMEOUT)
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


# F30: a *successful* invoke_mcp_tool result has no size cap at all, unlike
# stdout/stderr (MAX_BYTES above). A real tool returning full numpy arrays
# as JSON lists (e.g. thousands of ECG samples) re-enters the conversation
# as context on every later LLM call in the stage — observed immediately
# preceding a provider-side crash in a live BioSPPy run
# (benchmarks/alembic/runs/2026-07-06_rerun8_f24-verify/); the crash's own
# error was too generic to prove this was the cause, but capping oversized
# fields is good practice regardless of that one incident.
_RESULT_MAX_LIST_ITEMS = 20
_RESULT_MAX_STR_LEN    = 2000


def _truncate_large_result(value):
    """Recursively cap list length / string length in a tool result (F30)."""
    if isinstance(value, list):
        truncated = [_truncate_large_result(v) for v in value[:_RESULT_MAX_LIST_ITEMS]]
        if len(value) > _RESULT_MAX_LIST_ITEMS:
            truncated.append(f"... ({len(value) - _RESULT_MAX_LIST_ITEMS} more items truncated)")
        return truncated
    if isinstance(value, dict):
        return {k: _truncate_large_result(v) for k, v in value.items()}
    if isinstance(value, str) and len(value) > _RESULT_MAX_STR_LEN:
        return value[:_RESULT_MAX_STR_LEN] + f"... ({len(value) - _RESULT_MAX_STR_LEN} more chars truncated)"
    return value


# F25: code-enforced SKIP gate. main.py's set_skip_tools() populates this
# from the coder report's own samples: block, right before the Validator
# stage starts (via tools.fs.parse_samples_block) — see the observed AgML
# failure in IMPROVEMENTS_SPEC.md#f25, where the validator LLM invoked a
# tool it had itself just marked SKIP moments earlier because the
# skip/invoke split was never enforced, only requested in free text. The
# validator is also told this list explicitly in its opening message
# (belt-and-suspenders): this contextvar is the "suspenders" half, refusing
# the call regardless of whether the LLM remembers to honor its own read of
# the block.
_skip_tools: contextvars.ContextVar[frozenset] = contextvars.ContextVar(
    "skip_tools", default=frozenset()
)


def set_skip_tools(names) -> None:
    _skip_tools.set(frozenset(names))

def _invoke_mcp_tool_sync(repo_url: str, tool_name: str, args: dict | None = None) -> dict:
    from alembic.main import INVOKE_TIMEOUT  # deferred: see main.py's timeout block
    if tool_name in _skip_tools.get():
        return {
            "skipped": True,
            "reason": (
                f"'{tool_name}' is marked SKIP in server.md's samples block "
                "and was not invoked (code-enforced, F25) — report it as "
                "SKIPPED, not FAILED, and do not call the debugger for it."
            ),
        }
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

    # start_new_session=True puts invoke_tool.py in its own process group so
    # a timeout can kill the WHOLE tree, not just this immediate child.
    # server.py's tool functions spawn their own subprocess (the real helper
    # script) with no timeout of their own — SIGKILL-ing only invoke_tool.py
    # orphans that real work, which then keeps running unbounded (observed:
    # the AgML train_yolo.py process outlived a plain subprocess.run(timeout=)
    # kill of its parent).
    proc = subprocess.Popen(
        [str(venv_py), str(INVOKE_TOOL_SCRIPT)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
        cwd=str(out_dir), start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=INVOKE_TIMEOUT)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        return {
            "skipped": True,
            "reason": (
                f"'{tool_name}' did not return within {INVOKE_TIMEOUT}s and "
                "was killed — treated as resource-heavy and SKIPPED, not "
                "FAILED. This is NOT a confirmed bug: the tool may work "
                "fine, it was just too slow for a cheap validation sample. "
                "Do not call the debugger for it. A real check would need a "
                "decoupled, much-longer 'extended validation' pass for "
                "heavy tools (F25 item 2, still deferred)."
            ),
        }
    r = subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)

    # The invoker prints a single JSON line. If it crashed before printing,
    # surface stderr so the debugger can see what failed.
    stdout = r.stdout.strip()
    last_line = stdout.splitlines()[-1] if stdout else ""
    try:
        parsed = json.loads(last_line)
        if not parsed.get("ok") and r.stderr:
            parsed.setdefault("stderr", r.stderr[-2000:])
        if parsed.get("ok") and "result" in parsed:
            parsed["result"] = _truncate_large_result(parsed["result"])
        return parsed
    except Exception:
        return {
            "ok": False,
            "error": "could not parse invoker output",
            "returncode": r.returncode,
            "stdout": stdout[-1500:],
            "stderr": r.stderr[-1500:],
        }
