"""Validation tools: syntax/import check, pytest run, and live tool invocation."""
import json
import os
import subprocess
from pathlib import Path

from alembic.tools.paths import INVOKE_TOOL_SCRIPT, MAX_BYTES, output_dir, venv_python


def validate_syntax(repo_url: str) -> dict:
    """Check server.py for syntax errors and failed imports.

    Example:
        validate_syntax("https://github.com/Roestlab/massformer")
    """
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

    return {"passed": True}


def run_tests(repo_url: str) -> dict:
    """Run the pytest test suite for the generated MCP server.

    Example:
        run_tests("https://github.com/Roestlab/massformer")
    """
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


def invoke_mcp_tool(repo_url: str, tool_name: str, args: dict | None = None) -> dict:
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
