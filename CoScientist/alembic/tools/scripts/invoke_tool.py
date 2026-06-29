"""Invoke a single tool function from server.py with kwargs and print JSON.

Run by ``invoke_mcp_tool`` with the *server venv's* python. Inputs come from
the environment: SERVER_PATH, TOOL_NAME, TOOL_ARGS_JSON. Prints one JSON line.

Names are underscore-prefixed to avoid clashing with anything the imported
server module pulls into scope.
"""
import importlib.util as _u
import json as _json
import os as _os
import sys as _sys
import traceback as _tb
from pathlib import Path as _P

_server   = _P(_os.environ["SERVER_PATH"]).resolve()
_toolname = _os.environ["TOOL_NAME"]
_args     = _json.loads(_os.environ.get("TOOL_ARGS_JSON", "{}"))

_sys.path.insert(0, str(_server.parent))

# Prevent server.py's module-level mcp.run() from blocking the import.
import fastmcp as _fastmcp
_orig_run = _fastmcp.FastMCP.run
_fastmcp.FastMCP.run = lambda *a, **k: None
try:
    _spec = _u.spec_from_file_location("server", _server)
    _mod  = _u.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
finally:
    _fastmcp.FastMCP.run = _orig_run

_fn = getattr(_mod, _toolname, None)
if _fn is None:
    print(_json.dumps({"ok": False, "error": f"Tool {_toolname!r} not found"}))
    raise SystemExit(0)
# FastMCP wraps @mcp.tool() into FunctionTool — unwrap to original callable.
for _attr in ("fn", "func", "__wrapped__"):
    _inner = getattr(_fn, _attr, None)
    if callable(_inner):
        _fn = _inner
        break

try:
    _result = _fn(**_args)
    print(_json.dumps({"ok": True, "result": _result}, default=str))
except Exception as _e:
    print(_json.dumps({
        "ok": False,
        "error": f"{type(_e).__name__}: {_e}",
        "traceback": _tb.format_exc(),
    }))
