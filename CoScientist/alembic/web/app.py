"""FastAPI app: serves the dashboard and streams an alembic run over a WebSocket.

The pipeline emits low-level events (``pipeline``/``stage``/``tool_call``/
``tool_result``/``validation``) through :mod:`alembic.events`. This module
installs a per-run sink that:

  * forwards every raw event to the browser (drives the rail + activity feed),
    and
  * **enriches** each stage boundary by reading the run's on-disk artifacts —
    which, in the remaster architecture, are the authoritative source of truth
    (R2: everything is structured data on disk):
      - ``write_report`` result / explorer done  -> ``report`` (exploration map);
      - explorer/coder/validator done            -> ``server`` (tool cards, with
                                                     live pass/fail once known);
      - environment done                         -> ``setup`` (recorded setup.sh);
      - coder/wrapper done                       -> ``files`` (generated output)
                                                     + ``examples`` (per-tool
                                                     sample invocations)
                                                     + ``check`` (syntax/tests).

Manual, on-demand tool calls from the UI run through ``invoke_tool_function`` —
the same execution path the validator uses (the MCP wrap is only the final
stage), so a tool can be exercised the instant the coder has written it.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from alembic import events
from alembic.contract import load_plan
from alembic.main import run_pipeline
from alembic.tools.invoke import invoke_tool_function
from alembic.tools.paths import output_dir, reports_dir

WEB_DIR = Path(__file__).parent
TEMPLATE_PATH = WEB_DIR / "templates" / "index.html"

_MAX_FILE_CHARS = 14_000


# ---------------------------------------------------------------------------
# Disk readers — turn the run's on-disk state into UI-shaped payloads
# ---------------------------------------------------------------------------
def _split_sections(md: str) -> dict[str, str]:
    """Split a markdown report into ``{h2-title: body}`` on ``## `` headers.

    Text before the first ``## `` is stored under the leading ``# `` title as
    ``_title``. Robust to free-form content — no strict schema."""
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


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_output_files(out: Path) -> list[dict]:
    """Collect the generated artefacts (server.py + tools/helpers/tests) as
    ``{path, lang, content}`` — setup.sh is surfaced separately (how-to-run)."""
    files: list[dict] = []

    def add(p: Path, lang: str) -> None:
        if not p.is_file():
            return
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        if len(txt) > _MAX_FILE_CHARS:
            txt = txt[:_MAX_FILE_CHARS] + "\n… (truncated)"
        files.append({"path": str(p.relative_to(out)), "lang": lang, "content": txt})

    add(out / "server.py", "python")
    for sub in ("tools", "helpers", "tests"):
        d = out / sub
        if d.is_dir():
            for f in sorted(d.glob("*.py")):
                add(f, "python")
    return files


def _read_setup(out: Path) -> Optional[str]:
    p = out / "setup.sh"
    if not p.is_file():
        return None
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _tools_payload() -> dict:
    """The right-panel tool cards, merged from plan.json (names + real params +
    purpose) and, once the validator has run, validation.json (verdicts)."""
    plan = load_plan()
    if not plan:
        return {"tools": [], "title": ""}
    validation = _read_json(reports_dir() / "validation.json") or {}
    by_name = {t.get("name"): t for t in validation.get("tools", [])}
    tools = []
    for t in plan.tools:
        v = by_name.get(t.name, {})
        status = v.get("status")            # perfect | passed | failed | untested
        badge = {"perfect": "pass", "passed": "pass",
                 "failed": "fail"}.get(status)   # None -> pending in the UI
        tools.append({
            "name": t.name,
            "sig": ", ".join(t.params),
            "ret": "dict",
            "desc": t.purpose,
            "target": t.target,
            "status": badge,                    # pass | fail | None(pending)
            "verdict": status,                  # richer label for the card
            "exec_ok": v.get("exec_ok"),
            "invoc_passed": v.get("invoc_passed"),
            "invoc_total": v.get("invoc_total"),
            "perfect": bool(v.get("perfect")),
            "error": v.get("error") or None,
        })
    return {"tools": tools, "title": f"{plan.repo_url.rstrip('/').split('/')[-1]} · MCP server"}


def _examples_payload() -> dict:
    """Per-tool invocation examples from the plan's sample_args + evidence."""
    plan = load_plan()
    if not plan:
        return {"examples": []}
    examples = []
    for t in plan.tools:
        if t.sample_args is None:
            continue
        examples.append({"name": t.name, "args": t.sample_args,
                         "evidence": t.evidence or ""})
    return {"examples": examples}


def _syntax_check() -> Optional[dict]:
    """The coder artefact gate (G3) result → a 'syntax' check badge."""
    status = _read_json(reports_dir() / "stage_status.json") or {}
    coder = status.get("coder")
    if not coder:
        return None
    passed = coder.get("status") == "passed"
    gate = coder.get("gate", {})
    detail = "" if passed else json.dumps(gate.get("errors", gate), ensure_ascii=False)[:2000]
    return {"name": "syntax", "passed": passed, "detail": detail}


def _tests_check() -> Optional[dict]:
    """The validator counts → a 'tests' check badge."""
    validation = _read_json(reports_dir() / "validation.json")
    if not validation:
        return None
    c = validation.get("counts", {})
    tp, tt = c.get("tests_passed") or 0, c.get("tests_total") or 0
    passed = bool(tt) and tp >= tt
    detail = f"smoke tests {tp}/{tt}; tools passed {c.get('tools_passed', 0)}/{c.get('tools_total', 0)}"
    return {"name": "tests", "passed": passed, "detail": detail}


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    app = FastAPI(title="Alembic Pipeline Dashboard", version="2.0.0")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return TEMPLATE_PATH.read_text(encoding="utf-8")

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await ws.accept()
        await ws.send_json({"type": "connected",
                            "timestamp": datetime.now().isoformat()})

        # active["run_id"] is bumped on every run/stop; a stale run (possibly
        # blocked in a sync subprocess asyncio can't interrupt) unwinds itself
        # the next time its sink is called and sees it is no longer current.
        active: dict = {"task": None, "run_id": 0, "repo_url": None}

        async def send(msg: dict) -> None:
            try:
                await ws.send_json(msg)
            except (RuntimeError, WebSocketDisconnect):
                pass

        # -- enrichment: raw event -> browser, plus derived panel events -------
        async def _forward(msg: dict) -> None:
            await send(msg)
            mtype = msg.get("type")

            if mtype == "tool_result" and msg.get("name") == "write_report":
                path = (msg.get("response") or {}).get("report_path")
                if path:
                    parsed = _read_report(path)
                    if parsed:
                        await send({"type": "report", "report": Path(path).stem,
                                    "stage": msg.get("stage"), **parsed})

            elif mtype == "stage" and msg.get("status") in ("done", "failed"):
                await _enrich_stage(msg.get("stage"))

        async def _enrich_stage(stage: Optional[str]) -> None:
            try:
                await _enrich_stage_inner(stage)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — enrichment must never break the stream
                print(f"[alembic-web] enrich({stage}) failed: {exc}")

        async def _enrich_stage_inner(stage: Optional[str]) -> None:
            out = output_dir()
            if stage == "explorer":
                rep = reports_dir() / "exploration.md"
                if rep.is_file():
                    parsed = _read_report(str(rep))
                    if parsed:
                        await send({"type": "report", "report": "exploration",
                                    "stage": "explorer", **parsed})
                await send({"type": "server", **_tools_payload()})
                await send({"type": "examples", **_examples_payload()})

            elif stage == "environment":
                setup = _read_setup(out)
                if setup is not None:
                    await send({"type": "setup", "content": setup})

            elif stage == "coder":
                await send({"type": "files", "files": _read_output_files(out)})
                await send({"type": "server", **_tools_payload()})
                await send({"type": "examples", **_examples_payload()})
                chk = _syntax_check()
                if chk:
                    await send({"type": "check", **chk})

            elif stage == "validator":
                await send({"type": "server", **_tools_payload()})
                chk = _tests_check()
                if chk:
                    await send({"type": "check", **chk})

            elif stage == "wrapper":
                await send({"type": "files", "files": _read_output_files(out)})
                await send({"type": "server", **_tools_payload()})

        def make_sink(my_run: int):
            async def sink(msg: dict) -> None:
                if my_run != active["run_id"]:
                    raise asyncio.CancelledError()
                await _forward(msg)
            return sink

        async def run(repo_url: str, resume_from: Optional[str], my_run: int) -> None:
            token = events.set_sink(make_sink(my_run))
            try:
                await run_pipeline(repo_url, resume_from=resume_from)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — surface to the UI
                if my_run == active["run_id"]:
                    await send({"type": "pipeline", "status": "error",
                                "message": str(exc)})
            finally:
                events.reset_sink(token)

        async def _do_invoke(tool: str, args: dict, call_id) -> None:
            """Invoke a generated tool-function with user args (blocking subprocess)."""
            try:
                res = await invoke_tool_function(tool, args)
            except Exception as exc:  # noqa: BLE001
                res = {"ok": False, "error": str(exc)}
            ok = bool(res.get("ok"))
            await send({
                "type": "invoke_result", "call_id": call_id, "tool": tool, "ok": ok,
                "output": res.get("result") if ok else None,
                "reason": res.get("reason"),
                "error": None if ok else (res.get("error") or res.get("stderr") or "call failed"),
                "traceback": res.get("traceback"),
            })

        try:
            while True:
                data = json.loads(await ws.receive_text())
                mt = data.get("type", "")

                if mt == "run":
                    repo_url = (data.get("repo_url") or "").strip()
                    if not repo_url:
                        await send({"type": "error", "message": "empty repo_url"})
                        continue
                    active["run_id"] += 1          # invalidate any in-flight run first
                    active["repo_url"] = repo_url
                    my_run = active["run_id"]
                    old = active["task"]
                    if old and not old.done():
                        old.cancel()               # do NOT await — may be in a subprocess
                    active["task"] = asyncio.create_task(
                        run(repo_url, data.get("resume_from"), my_run))

                elif mt == "stop":
                    active["run_id"] += 1
                    old = active["task"]
                    active["task"] = None
                    if old and not old.done():
                        old.cancel()
                    await send({"type": "pipeline", "status": "cancelled"})

                elif mt == "invoke":
                    tool = data.get("tool")
                    args = data.get("args") or {}
                    call_id = data.get("call_id")
                    if not active.get("repo_url") or not tool:
                        await send({"type": "invoke_result", "call_id": call_id,
                                    "tool": tool, "ok": False,
                                    "error": "no built tools yet — run a repo first"})
                    else:
                        asyncio.create_task(_do_invoke(tool, args, call_id))

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
