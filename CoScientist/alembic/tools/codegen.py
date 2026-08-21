"""Deterministic code generation: server.py, setup.sh, TM-Bench code.py.

The MCP wrapper is a pure function of the verified tool files (Q&A decision):
signatures and docstrings are extracted by AST and rendered into a FastMCP
server whose every tool shells through ``helpers/run_function.py`` — the same
runner the validator used, so serving and validation share one execution path
and the two-venv layout needs no special casing. An LLM touches server.py only
if the compile gate fails.
"""
from __future__ import annotations

import ast
import shutil
from pathlib import Path

from alembic.tools.paths import RUN_FUNCTION_SCRIPT, output_dir

_SAFE_TYPES = {"str", "int", "float", "bool", "dict", "list", "tuple", "set",
               "None", "Optional", "Union", "Any"}


def _safe_annotation(node: ast.expr | None) -> str | None:
    """Unparse an annotation only if every Name in it is a builtin/typing type
    the server venv is guaranteed to know — anything repo-specific is dropped."""
    if node is None:
        return None
    if any(isinstance(n, ast.Name) and n.id not in _SAFE_TYPES for n in ast.walk(node)):
        return None
    if any(isinstance(n, ast.Attribute) for n in ast.walk(node)):
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None


def _literal_default(node: ast.expr | None) -> str | None:
    """Unparse a default only when it is a literal; a non-literal default makes
    the wrapper param required (the coder is instructed to use literals)."""
    if node is None:
        return None
    try:
        ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None
    return ast.unparse(node)


def _find_def(source: str, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return n
    return None


def tool_signature(name: str, out_dir: Path | None = None) -> dict | None:
    """Extract {name, params: [(name, annotation|None, default|None)], doc}
    from tools/<name>.py. None if the file/function is missing or unparseable."""
    out = out_dir or output_dir()
    f = out / "tools" / f"{name}.py"
    if not f.exists():
        return None
    fn = _find_def(f.read_text(encoding="utf-8", errors="replace"), name)
    if fn is None:
        return None
    a = fn.args
    pos = [*a.posonlyargs, *a.args]
    defaults: list[ast.expr | None] = [None] * (len(pos) - len(a.defaults)) + list(a.defaults)
    params = [(arg.arg, _safe_annotation(arg.annotation), _literal_default(d))
              for arg, d in zip(pos, defaults)]
    params += [(arg.arg, _safe_annotation(arg.annotation),
                _literal_default(d) if d else None)
               for arg, d in zip(a.kwonlyargs, a.kw_defaults)]
    return {"name": name, "params": params,
            "doc": ast.get_docstring(fn) or f"Run {name}."}


def function_param_names(name: str, out_dir: Path | None = None) -> tuple[set[str] | None, bool]:
    """(accepted param names, has **kwargs) for the generated tools/<name>.py
    function — the ground truth for which kwargs it actually accepts. Returns
    (None, False) if the file/function is missing/unparseable. Used to filter
    sample args against the REAL wrapper signature (task tools rename the repo's
    params, so the repo symbol's params are the wrong thing to filter against)."""
    out = out_dir or output_dir()
    f = out / "tools" / f"{name}.py"
    if not f.exists():
        return None, False
    fn = _find_def(f.read_text(encoding="utf-8", errors="replace"), name)
    if fn is None:
        return None, False
    a = fn.args
    names = {p.arg for p in (*a.posonlyargs, *a.args, *a.kwonlyargs)}
    return names, a.kwarg is not None


def _render_param(p: tuple[str, str | None, str | None]) -> str:
    name, ann, default = p
    s = name + (f": {ann}" if ann else "")
    return s + (f" = {default}" if default is not None else "")


def _escape_doc(doc: str) -> str:
    return doc.replace("\\", "\\\\").replace('"""', r"\"\"\"")


def render_server(repo_name: str, signatures: list[dict]) -> str:
    """Render the FastMCP server from extracted tool signatures."""
    blocks = []
    for sig in signatures:
        # required params first — a literal-defaulted param may follow one
        # without a default in the original, which Python forbids; reorder.
        params = sorted(sig["params"], key=lambda p: p[2] is not None)
        args = ", ".join(_render_param(p) for p in params)
        payload = ", ".join(f'"{n}": {n}' for n, _, _ in sig["params"])
        blocks.append(
            f'@mcp.tool()\n'
            f'def {sig["name"]}({args}) -> dict:\n'
            f'    """{_escape_doc(sig["doc"])}"""\n'
            f'    return _call("{sig["name"]}", {{{payload}}})\n'
        )
    tools_src = "\n\n".join(blocks)
    return f'''"""FastMCP server for {repo_name} — generated by alembic.

Each tool shells through the tools venv via helpers/run_function.py — the same
runner that validated the tool functions, so serving and validation share one
execution path (two-venv layouts work unchanged).
"""
import json
import subprocess
from pathlib import Path

from fastmcp import FastMCP

_OUT = Path(__file__).resolve().parent
_PYTHON = str(_OUT / ".venv" / "bin" / "python")   # main venv: repo + deps
_RUNNER = str(_OUT / "helpers" / "run_function.py")
_SENTINEL = "<<<ALEMBIC_RESULT>>>"

mcp = FastMCP("{repo_name}")


def _call(tool: str, kwargs: dict) -> dict:
    r = subprocess.run([_PYTHON, _RUNNER, str(_OUT), tool, json.dumps(kwargs)],
                       cwd=str(_OUT), capture_output=True, text=True)
    parts = r.stdout.rsplit(_SENTINEL, 1)
    if len(parts) == 2:
        out = json.loads(parts[1].strip())
        if out.get("ok"):
            res = out.get("result")
            return res if isinstance(res, dict) else {{"result": res}}
        raise RuntimeError(out.get("error") or "tool failed")
    raise RuntimeError((r.stderr or r.stdout)[-2000:] or "runner produced no output")


{tools_src}

if __name__ == "__main__":
    mcp.run()
'''


def write_server(repo_name: str, tool_names: list[str]) -> dict:
    """Generate output/server.py + output/helpers/run_function.py for every
    tool whose signature extracts cleanly. Returns {written, tools, skipped}."""
    out = output_dir()
    sigs, skipped = [], []
    for name in tool_names:
        sig = tool_signature(name, out)
        (sigs if sig else skipped).append(sig or name)
    helpers = out / "helpers"
    helpers.mkdir(parents=True, exist_ok=True)
    shutil.copy(RUN_FUNCTION_SCRIPT, helpers / "run_function.py")
    server = out / "server.py"
    server.write_text(render_server(out.parent.name, sigs), encoding="utf-8")
    return {"written": str(server), "tools": [s["name"] for s in sigs], "skipped": skipped}


def render_setup_sh(commands: list[str]) -> str:
    """setup.sh from the recorded transcript of successful env-stage commands."""
    body = "\n".join(commands) if commands else "# (no environment commands were recorded)"
    return ("#!/usr/bin/env bash\n"
            "# Environment setup transcript — the commands that actually succeeded\n"
            "# during the alembic Environment stage, in order. Recorded by code.\n"
            "set -euo pipefail\n"
            "cd /work\n\n"
            f"{body}\n")


def write_setup_sh(commands: list[str]) -> Path:
    out = output_dir() / "setup.sh"
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = out.read_text(encoding="utf-8") if out.exists() else ""
    rendered = render_setup_sh(commands)
    if commands or not existing:
        out.write_text(rendered, encoding="utf-8")
        out.chmod(0o755)
    return out


def render_code_py(tool_name: str, out_dir: Path | None = None) -> str | None:
    """TM-Bench export: the tool function copied verbatim (self-contained by
    construction — imports live inside the function body)."""
    out = out_dir or output_dir()
    f = out / "tools" / f"{tool_name}.py"
    if not f.exists():
        return None
    source = f.read_text(encoding="utf-8", errors="replace")
    fn = _find_def(source, tool_name)
    if fn is None:
        return None
    segment = ast.get_source_segment(source, fn)
    return (segment + "\n") if segment else None
