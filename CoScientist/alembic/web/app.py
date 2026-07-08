"""FastAPI app: serves the dashboard and streams the alembic pipeline over ws.

The pipeline (``alembic.main.run_pipeline``) emits raw UI events through an
``on_event`` callback. This module registers a callback that:

  * forwards every raw event to the browser, and
  * *enriches* two kinds of tool results so the panels can render richly:
      - ``write_report``  -> reads the .md the agent just wrote, splits it into
        sections, and pushes a ``report`` event (fills left/right panels).
      - ``invoke_mcp_tool`` -> maps the result to pass/fail per tool and pushes
        a ``validation`` event (green/red badges, hover error).
      - ``validate_syntax`` / ``run_tests`` -> pushes a ``check`` event.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from alembic.main import run_pipeline

WEB_DIR = Path(__file__).parent
TEMPLATE_PATH = WEB_DIR / "templates" / "index.html"


# ---------------------------------------------------------------------------
# Report parsing
# ---------------------------------------------------------------------------
def _split_sections(md: str) -> dict[str, str]:
    """Split a report into ``{h2-title: body}`` on ``## `` headers.

    Text before the first ``## `` is stored under the top ``# `` title (or
    ``"_intro"``). Robust to free-form report content — no strict schema.
    """
    sections: dict[str, str] = {}
    current = "_intro"
    buf: list[str] = []
    for line in md.splitlines():
        if line.startswith("## "):
            sections[current] = "\n".join(buf).strip()
            current = line[3:].strip()
            buf = []
        elif line.startswith("# ") and current == "_intro" and not buf:
            sections["_title"] = line[2:].strip()
        else:
            buf.append(line)
    sections[current] = "\n".join(buf).strip()
    return {k: v for k, v in sections.items() if v or k == "_title"}


def _read_report(report_path: str) -> Optional[dict]:
    try:
        content = Path(report_path).read_text(encoding="utf-8")
    except OSError:
        return None
    return {"raw": content, "sections": _split_sections(content)}


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    app = FastAPI(title="Alembic Pipeline Dashboard", version="1.0.0")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return TEMPLATE_PATH.read_text(encoding="utf-8")

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await ws.accept()
        await ws.send_json({
            "type": "connected",
            "timestamp": datetime.now().isoformat(),
        })

        # active["task"] = the current pipeline task; active["run_id"] is bumped
        # on every run/stop so a stale run (possibly blocked in a sync subprocess
        # that asyncio cannot interrupt) unwinds itself the next time it emits.
        active: dict = {"task": None, "run_id": 0}
        # the most recent invoke_mcp_tool call (tool + args), awaiting its result
        pending_invoke: dict = {"tool": None, "args": None}

        async def send(msg: dict):
            try:
                await ws.send_json(msg)
            except (RuntimeError, WebSocketDisconnect):
                pass

        def make_on_event(my_run: int):
            async def on_event(msg: dict):
                """Pipeline -> browser bridge. Forwards raw + enriches results.

                Raises CancelledError once this run is no longer current, which
                unwinds a pipeline that was cancelled while blocked in a
                synchronous tool (git clone / pip / pytest) — the moment it
                reaches its next emit, it dies instead of continuing.
                """
                if my_run != active["run_id"]:
                    raise asyncio.CancelledError()
                await _forward(msg)
            return on_event

        async def _forward(msg: dict):
            await send(msg)

            mtype = msg.get("type")
            if mtype == "tool_call" and msg.get("name") == "invoke_mcp_tool":
                a = msg.get("args") or {}
                pending_invoke["tool"] = a.get("tool_name")
                pending_invoke["args"] = a.get("args") or {}

            elif mtype == "tool_result":
                name = msg.get("name")
                resp = msg.get("response") or {}

                if name == "write_report":
                    path = resp.get("report_path")
                    if path:
                        parsed = _read_report(path)
                        if parsed:
                            # report_name = filename stem (exploration/server/validation)
                            report_name = Path(path).stem
                            await send({
                                "type": "report",
                                "report": report_name,
                                "stage": msg.get("stage"),
                                **parsed,
                            })

                elif name == "invoke_mcp_tool":
                    ok = bool(resp.get("ok"))
                    err = None if ok else (
                        resp.get("error")
                        or resp.get("stderr")
                        or "invocation failed"
                    )
                    await send({
                        "type": "validation",
                        "tool": pending_invoke.get("tool"),
                        "passed": ok,
                        "input": pending_invoke.get("args"),
                        "output": resp.get("result") if ok else None,
                        "error": err,
                        "traceback": resp.get("traceback"),
                    })
                    pending_invoke["tool"] = None
                    pending_invoke["args"] = None

                elif name in ("validate_syntax", "run_tests"):
                    await send({
                        "type": "check",
                        "name": name,
                        "passed": bool(resp.get("passed")),
                        "detail": resp.get("error") or resp.get("output") or "",
                    })

        async def run(repo_url: str, resume_from: Optional[str], my_run: int):
            try:
                await run_pipeline(repo_url, resume_from=resume_from,
                                   on_event=make_on_event(my_run))
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — surface to UI
                if my_run == active["run_id"]:
                    await send({"type": "pipeline", "status": "error",
                                "message": str(exc)})

        try:
            while True:
                raw = await ws.receive_text()
                data = json.loads(raw)
                mt = data.get("type", "")

                if mt == "run":
                    repo_url = (data.get("repo_url") or "").strip()
                    if not repo_url:
                        await send({"type": "error", "message": "empty repo_url"})
                        continue
                    # bump first: invalidates any in-flight run before we replace it
                    active["run_id"] += 1
                    my_run = active["run_id"]
                    pending_invoke["tool"] = None
                    old = active["task"]
                    if old and not old.done():
                        old.cancel()   # do NOT await — it may be stuck in a subprocess
                    active["task"] = asyncio.create_task(
                        run(repo_url, data.get("resume_from"), my_run)
                    )

                elif mt == "stop":
                    active["run_id"] += 1   # invalidate current run
                    old = active["task"]
                    active["task"] = None
                    if old and not old.done():
                        old.cancel()
                    await send({"type": "pipeline", "status": "cancelled"})

                elif mt == "ping":
                    await send({"type": "pong"})

        except WebSocketDisconnect:
            active["run_id"] += 1
            if active["task"] and not active["task"].done():
                active["task"].cancel()
        except Exception as exc:  # noqa: BLE001
            print(f"[alembic-web] ws error: {exc}")
            active["run_id"] += 1
            if active["task"] and not active["task"].done():
                active["task"].cancel()

    return app
