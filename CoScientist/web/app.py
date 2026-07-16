import asyncio
import json
import logging
import os
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from CoScientist.main import CoScientistManager
from CoScientist.web.handler import WebHITLHandler
from CoScientist.web.session_registry import LocalSessionRegistry
from CoScientist.agents import agent_system
from CoScientist.hitl.tool import hitl_toolset
from CoScientist.config import get_settings

from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.adk.agents.run_config import RunConfig
from google.adk.workflow.utils._workflow_hitl_utils import (
    has_request_input_function_call,
    get_request_input_interrupt_ids,
    create_request_input_response,
    REQUEST_INPUT_FUNCTION_CALL_NAME,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _json_safe(value):
    """Return a deeply JSON-serializable copy of ``value``.

    ws.send_json() calls json.dumps() WITHOUT a ``default`` hook, so any object
    it can't natively encode (e.g. a pydantic ``MCPServer`` nested inside a tool
    result) raises and kills the whole event stream. Round-tripping through
    json.dumps(default=str) coerces such leaves to strings while preserving the
    dict/list structure the frontend expects — matching the project's existing
    logging/graph serialisation convention.
    """
    try:
        return json.loads(json.dumps(value, default=str, ensure_ascii=False))
    except (TypeError, ValueError):
        return str(value)


# ---------------------------------------------------------------------------
# Runtime types and constants
# ---------------------------------------------------------------------------
WEB_DIR = Path(__file__).parent
TEMPLATE_PATH = WEB_DIR / "templates" / "index.html"
APP_NAME = "coscientist_app"
SessionKey = tuple[str, str]


class WebRuntime:
    """Process-local users, ADK sessions, managers, sockets, and event logs."""

    def __init__(self) -> None:
        self.session_service = InMemorySessionService()
        self.registry = LocalSessionRegistry()
        self.managers: dict[SessionKey, CoScientistManager] = {}
        self.manager_lock = asyncio.Lock()
        self.execution_locks: dict[SessionKey, asyncio.Lock] = {}
        self.agent_events: dict[SessionKey, list[dict[str, Any]]] = defaultdict(list)
        self.pending_hitl: dict[str, dict[str, Any]] = {}
        self.hitl_handler = WebHITLHandler()
        self.sockets: dict[SessionKey, list[WebSocket]] = defaultdict(list)
        self.active_runs: dict[SessionKey, asyncio.Task] = {}

    async def get_manager(self, user_id: str, session_id: str) -> CoScientistManager:
        self.registry.require_session(user_id, session_id)
        key = (user_id, session_id)
        manager = self.managers.get(key)
        if manager is not None:
            return manager
        async with self.manager_lock:
            manager = self.managers.get(key)
            if manager is None:
                manager = CoScientistManager(
                    app_name=APP_NAME,
                    user_id=user_id,
                    session_id=session_id,
                    session_service=self.session_service,
                )
                await manager.initialize()
                self.managers[key] = manager
                self.execution_locks[key] = asyncio.Lock()
        return manager

    def attach_socket(self, key: SessionKey, ws: WebSocket) -> None:
        if ws not in self.sockets[key]:
            self.sockets[key].append(ws)

    def detach_socket(self, key: SessionKey, ws: WebSocket) -> None:
        sockets = self.sockets.get(key)
        if not sockets:
            return
        try:
            sockets.remove(ws)
        except ValueError:
            pass
        if not sockets:
            self.sockets.pop(key, None)

    async def send(self, key: SessionKey, payload: dict[str, Any]) -> None:
        """Broadcast an event only to tabs viewing this session."""
        for socket in list(self.sockets.get(key, [])):
            try:
                await socket.send_json(payload)
            except Exception:
                self.detach_socket(key, socket)

    async def close(self) -> None:
        tasks = list(self.active_runs.values())
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.gather(
            *(manager.close() for manager in self.managers.values()),
            return_exceptions=True,
        )


def _wire_hitl(runtime: WebRuntime) -> None:
    """Wire the routing Web handler once for this application runtime."""

    # SessionAgent handlers are delegates because workflow assembly deep-copies
    # their references.
    wired = []
    for name, agent in agent_system.agents.items():
        handler = getattr(agent, "hitl_handler", None)
        if handler is not None and hasattr(handler, "set_delegate"):
            handler.set_delegate(runtime.hitl_handler)
            wired.append(name)

    if hasattr(hitl_toolset._handler, "set_delegate"):
        hitl_toolset._handler.set_delegate(runtime.hitl_handler)
    else:
        hitl_toolset._handler = runtime.hitl_handler

    # CoderToolset owns its approval handler separately from the agent-level
    # delegates. Preserve HITL__ENABLED semantics while routing Web approvals.
    from CoScientist.tools.coder_tools import coder_toolset
    if coder_toolset._hitl_handler is not None:
        coder_toolset._hitl_handler = runtime.hitl_handler

    logging.getLogger("CoScientist.web").info(
        "Session-routing WebHITLHandler wired into: %s", wired,
    )


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[CoScientist Web] Starting up …")
    yield
    print("[CoScientist Web] Shutting down …")
    await app.state.runtime.close()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    os.environ["COSCIENTIST_WEB_MODE"] = "true"
    runtime = WebRuntime()
    _wire_hitl(runtime)
    app = FastAPI(
        title="CoScientist Web UI",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.runtime = runtime

    # Vendored JS/CSS (e.g. vis-network for the live graph) so the UI works
    # offline / behind a VPN without any CDN.
    _static_dir = WEB_DIR / "static"
    if _static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

    # --- HTML endpoint ---
    @app.get("/", response_class=HTMLResponse)
    async def index():
        # no-store: a cached index.html silently serves an OLD frontend — HITL
        # review cards then render without controls/content.
        return HTMLResponse(
            TEMPLATE_PATH.read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-store"},
        )

    # --- Local users and sessions (process lifetime only) ---
    @app.get("/api/users")
    async def list_users():
        return JSONResponse({"users": runtime.registry.list_users()})

    @app.post("/api/users")
    async def create_user(data: dict):
        try:
            user = runtime.registry.create_user(data.get("nickname", ""))
        except ValueError as exc:
            status_code = 409 if "already registered" in str(exc) else 400
            raise HTTPException(status_code=status_code, detail=str(exc)) from exc
        return JSONResponse({"user": user}, status_code=201)

    @app.get("/api/users/{user_id}/sessions")
    async def list_user_sessions(user_id: str):
        try:
            sessions = runtime.registry.list_sessions(user_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse({"sessions": sessions})

    @app.post("/api/users/{user_id}/sessions")
    async def create_user_session(user_id: str, data: dict):
        try:
            runtime.registry.require_user(user_id)
            raw_title = data.get("title", "")
            if not isinstance(raw_title, str):
                raise ValueError("Session title must be a string.")
            title = " ".join(raw_title.strip().split()) or "New session"
            if len(title) > 120:
                raise ValueError("Session title must be at most 120 characters.")
            session_id = f"session_{uuid4().hex}"
            from CoScientist.graph.session_scope import (
                GRAPH_SCOPE_SESSION_KEY,
                GRAPH_SCOPE_USER_KEY,
            )
            await runtime.session_service.create_session(
                app_name=APP_NAME,
                user_id=user_id,
                session_id=session_id,
                state={
                    "active_tasks": [],
                    GRAPH_SCOPE_USER_KEY: user_id,
                    GRAPH_SCOPE_SESSION_KEY: session_id,
                },
            )
            session = runtime.registry.create_session(
                user_id,
                title,
                session_id=session_id,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse({"session": session}, status_code=201)

    @app.get("/api/users/{user_id}/sessions/{session_id}")
    async def get_user_session(user_id: str, session_id: str):
        try:
            session = runtime.registry.require_session(user_id, session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse({"session": session})

    @app.patch("/api/users/{user_id}/sessions/{session_id}")
    async def rename_user_session(user_id: str, session_id: str, data: dict):
        try:
            session = runtime.registry.rename_session(
                user_id, session_id, data.get("title", "")
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse({"session": session})

    # --- HITL diagnostics ---
    @app.get("/api/hitl-status")
    async def hitl_status():
        """Why am I (not) being asked questions — one glance."""
        from CoScientist.config import get_settings

        agents = {
            name: getattr(agent, "hitl_handler", None) is not None
            for name, agent in agent_system.agents.items()
            if hasattr(agent, "hitl_handler")
        }
        return JSONResponse({
            "hitl_enabled": get_settings().hitl.enabled,
            "websocket_connections": runtime.hitl_handler.connection_count(),
            "session_agents_with_handler": agents,
            "pending_requests": runtime.hitl_handler.pending_summary(),
            "auto_approve_timeout_seconds": runtime.hitl_handler.HITL_TIMEOUT_SECONDS,
        })

    # --- Knowledge graph (live view) ---
    @app.get("/graph", response_class=HTMLResponse)
    async def graph_page():
        return (WEB_DIR / "templates" / "graph.html").read_text(encoding="utf-8")

    def graph_payload(user_id: str, session_id: str, view: str):
        """Return one session's graphs (semantic memory is shared per user)."""
        runtime.registry.require_session(user_id, session_id)
        try:
            from CoScientist.graph.memory import get_knowledge_graph
            from CoScientist.graph.memory_store import get_knowledge_memory
            from CoScientist.graph.research.store import get_research_graph

            if view == "research":
                return get_research_graph(
                    user_id=user_id,
                    session_id=session_id,
                ).to_view()
            if view == "memory":
                return get_knowledge_memory(
                    user_id=user_id,
                    session_id=session_id,
                ).full()

            execution = get_knowledge_graph(
                user_id=user_id,
                session_id=session_id,
            ).full()
            if view == "knowledge":
                from CoScientist.graph.knowledge import to_knowledge_graph
                return to_knowledge_graph(
                    execution,
                    memory=get_knowledge_memory(
                        user_id=user_id,
                        session_id=session_id,
                    ),
                )
            return execution
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 — never break the UI
            return {"nodes": [], "edges": [], "error": str(exc)}

    @app.get("/api/users/{user_id}/sessions/{session_id}/graph")
    async def api_session_graph(
        user_id: str,
        session_id: str,
        view: str = "execution",
    ):
        try:
            payload = graph_payload(user_id, session_id, view)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse(payload)

    @app.get("/api/graph")
    async def api_graph(
        user_id: str = "",
        session_id: str = "",
        view: str = "execution",
    ):
        """Compatibility endpoint; an explicit session scope is mandatory."""
        if not user_id or not session_id:
            raise HTTPException(
                status_code=400,
                detail="user_id and session_id are required",
            )
        return await api_session_graph(user_id, session_id, view)

    # --- Roadmap endpoints ---
    @app.get("/api/users/{user_id}/sessions/{session_id}/roadmap")
    async def get_roadmap(user_id: str, session_id: str):
        try:
            runtime.registry.require_session(user_id, session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        adk_session = await runtime.session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )
        if adk_session is None:
            raise HTTPException(status_code=404, detail="ADK session not found.")
        tasks = adk_session.state.get("active_tasks", [])
        return JSONResponse({
            "content": json.dumps(tasks, ensure_ascii=False, indent=2),
            "tasks": _json_safe(tasks),
        })

    @app.post("/api/users/{user_id}/sessions/{session_id}/roadmap")
    async def save_roadmap(user_id: str, session_id: str, data: dict):
        try:
            runtime.registry.require_session(user_id, session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        content = data.get("content", "")
        try:
            tasks = json.loads(content) if isinstance(content, str) and content.strip() else []
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid roadmap JSON: {exc.msg}") from exc
        if not isinstance(tasks, list):
            raise HTTPException(status_code=400, detail="Roadmap must be a JSON list of tasks.")

        adk_session = await runtime.session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )
        if adk_session is None:
            raise HTTPException(status_code=404, detail="ADK session not found.")
        await runtime.session_service.append_event(
            adk_session,
            Event(
                invocation_id=f"roadmap_{uuid4().hex}",
                author="user",
                actions=EventActions(state_delta={"active_tasks": tasks}),
            ),
        )
        runtime.registry.touch_session(user_id, session_id)
        return JSONResponse({"status": "success", "tasks": _json_safe(tasks)})


    # --- ТЗ document (microfluidics profile) ---
    @app.get("/api/tz-document")
    async def get_tz_document(name: str = ""):
        """Serve a ТЗ document from tz_documents/ (the latest one by default).

        The TZSpecAgent announces the file in the chat; this endpoint lets the
        user open it in the browser.
        """
        from fastapi.responses import PlainTextResponse

        tz_dir = Path("tz_documents")
        if not tz_dir.is_dir():
            return JSONResponse({"error": "no ТЗ documents yet"}, status_code=404)
        if name:
            # Only bare file names inside tz_documents/ — no path traversal.
            candidate = tz_dir / Path(name).name
            if not candidate.is_file():
                return JSONResponse({"error": f"no such document: {name}"}, status_code=404)
        else:
            files = sorted(tz_dir.glob("TZ_*.md"))
            if not files:
                return JSONResponse({"error": "no ТЗ documents yet"}, status_code=404)
            candidate = files[-1]
        return PlainTextResponse(
            candidate.read_text(encoding="utf-8"),
            media_type="text/markdown; charset=utf-8",
        )

    # --- Agent info ---
    @app.get("/api/agents")
    async def get_agents():
        """Return list of registered agents."""
        return JSONResponse({
            "agents": [
                {"name": "OrchestratorAgent", "role": "orchestrator", "status": "idle"},
                {"name": "PlannerAgent", "role": "planner", "status": "idle"},
                {"name": "CoderAgent", "role": "coder", "status": "idle"},
                {"name": "HypothesesAgent", "role": "hypothesis", "status": "idle"},
                {"name": "ResearchAgent", "role": "research", "status": "idle"},
                {"name": "ToolRetrieverAgent", "role": "tool_retriever", "status": "idle"},
                {"name": "ToolRerankerAgent", "role": "tool_reranker", "status": "idle"},
                {"name": "ToolWebSearcherAgent", "role": "tool_websearcher", "status": "idle"},
                {"name": "ToolSearcherAgent", "role": "tool_searcher", "status": "idle"},
                {"name": "ExperimentAgent", "role": "experiment", "status": "idle"},
            ]
        })

    # --- Events log ---
    @app.get("/api/users/{user_id}/sessions/{session_id}/events")
    async def get_events(user_id: str, session_id: str):
        try:
            runtime.registry.require_session(user_id, session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse({"events": runtime.agent_events[(user_id, session_id)][-100:]})

    # --- WebSocket ---
    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        user_id = (ws.query_params.get("user_id") or "").strip()
        session_id = (ws.query_params.get("session_id") or "").strip()
        await ws.accept()
        try:
            user = runtime.registry.require_user(user_id)
            session_meta = runtime.registry.require_session(user_id, session_id)
        except KeyError as exc:
            await ws.send_json({"type": "error", "message": str(exc)})
            await ws.close(code=4404, reason="Unknown user or session")
            return

        key = (user_id, session_id)
        runtime.registry.touch_session(user_id, session_id)
        runtime.attach_socket(key, ws)
        await ws.send_json({
            "type": "connected",
            "timestamp": datetime.now().isoformat(),
            "message": f"Connected as {user['nickname']}",
        })
        adk_session = await runtime.session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )
        current_run = runtime.active_runs.get(key)
        await ws.send_json({
            "type": "session_snapshot",
            "user": user,
            "session": session_meta,
            "messages": runtime.agent_events[key],
            "active_tasks": (
                _json_safe(adk_session.state.get("active_tasks", []))
                if adk_session else []
            ),
            "status": "processing" if current_run and not current_run.done() else "idle",
        })
        # Re-deliver only HITL requests belonging to this session.
        await runtime.hitl_handler.attach_websocket(ws, key)
        delivered_interrupts = set()
        for pending in runtime.pending_hitl.values():
            payload = pending.get("payload")
            if pending.get("session_key") == key and payload and id(payload) not in delivered_interrupts:
                await ws.send_json(payload)
                delivered_interrupts.add(id(payload))

        try:
            while True:
                raw = await ws.receive_text()
                data = json.loads(raw)
                msg_type = data.get("type", "")

                if msg_type == "chat_message":
                    active_task = runtime.active_runs.get(key)
                    if active_task and not active_task.done():
                        active_task.cancel()
                        try:
                            await active_task
                        except asyncio.CancelledError:
                            pass
                    active_task = asyncio.create_task(_handle_chat(runtime, key, data))
                    runtime.active_runs[key] = active_task

                    def discard_finished(task, *, run_key=key):
                        if runtime.active_runs.get(run_key) is task:
                            runtime.active_runs.pop(run_key, None)

                    active_task.add_done_callback(discard_finished)
                elif msg_type == "stop_chat":
                    active_task = runtime.active_runs.get(key)
                    if active_task and not active_task.done():
                        active_task.cancel()
                        try:
                            await active_task
                        except asyncio.CancelledError:
                            pass
                    runtime.active_runs.pop(key, None)
                    _cancel_pending_hitl(runtime, key)
                    runtime.hitl_handler.reset(key)
                    runtime.registry.touch_session(user_id, session_id, status="idle")
                    await runtime.send(key, {
                        "type": "status",
                        "status": "idle",
                        "message": "Agent execution stopped. Session history was preserved.",
                    })
                    await runtime.send(key, {
                        "type": "final_response",
                        "content": "Stopped",
                    })
                elif msg_type == "hitl_response":
                    _handle_hitl_response(runtime, key, data)
                elif msg_type == "ping":
                    await ws.send_json({"type": "pong"})
                else:
                    await ws.send_json({
                        "type": "error",
                        "message": f"Unknown message type: {msg_type}",
                    })
        except WebSocketDisconnect:
            runtime.hitl_handler.detach_websocket(ws, key)
            runtime.detach_socket(key, ws)
            print("[WebSocket] Client disconnected")
        except Exception as exc:
            runtime.hitl_handler.detach_websocket(ws, key)
            runtime.detach_socket(key, ws)
            print(f"[WebSocket] Error: {exc}")

    return app


# ---------------------------------------------------------------------------
# HITL helpers
# ---------------------------------------------------------------------------
def _cancel_pending_hitl(runtime: WebRuntime, key: SessionKey):
    """Cancel ADK RequestInput waits belonging to one session."""
    for interrupt_id, info in list(runtime.pending_hitl.items()):
        if info.get("session_key") != key:
            continue
        info["event"].set()
        runtime.pending_hitl.pop(interrupt_id, None)


# ---------------------------------------------------------------------------
# Message handlers
# ---------------------------------------------------------------------------
async def _handle_chat(runtime: WebRuntime, key: SessionKey, data: dict):
    """Run user query through the agent pipeline, streaming events.
    
    Handles ADK RequestInput HITL: when the workflow pauses (interrupt event),
    this sends the HITL request to the browser, waits for the response, then
    resumes the workflow by calling run_async with a FunctionResponse message.
    """
    query = data.get("message", "").strip()
    if not query:
        await runtime.send(key, {"type": "error", "message": "Empty query"})
        return

    user_id, session_id = key

    # Echo user message
    runtime.agent_events[key].append({
        "type": "user_message",
        "message": query,
        "timestamp": datetime.now().isoformat(),
    })

    await runtime.send(key, {
        "type": "status",
        "status": "processing",
        "message": f"Processing query: {query}",
    })

    try:
        manager = await runtime.get_manager(user_id, session_id)
        runtime.registry.touch_session(user_id, session_id, status="processing")
        execution_lock = runtime.execution_locks[key]

        async with execution_lock:
            await _run_chat_invocation(runtime, key, manager, query)

    except asyncio.CancelledError:
        runtime.registry.touch_session(user_id, session_id, status="idle")
        raise
    except Exception as exc:
        runtime.registry.touch_session(user_id, session_id, status="idle")
        error_msg = f"Error processing query: {str(exc)}"
        error_event = {
            "type": "error",
            "message": error_msg,
            "timestamp": datetime.now().isoformat(),
        }
        await runtime.send(key, error_event)
        runtime.agent_events[key].append(error_event)


async def _run_chat_invocation(
    runtime: WebRuntime,
    key: SessionKey,
    manager: CoScientistManager,
    query: str,
) -> None:
    """Execute one serialized ADK invocation for a session."""
    user_id, session_id = key
    current_message = types.Content(
        role="user",
        parts=[types.Part(text=query)],
    )

    final_response = "No response"

    try:
        # Loop: run -> check for HITL interrupt -> wait for response -> resume
        while True:
            hitl_interrupt_event = None
            pending_wait_event = None

            async for event in manager.runner.run_async(
                user_id=manager.user_id,
                session_id=manager.session_id,
                new_message=current_message,
                # Lift ADK's 500-LLM-call default so a long autonomous run driven
                # by a single prompt isn't cut off mid-work.
                run_config=RunConfig(
                    max_llm_calls=get_settings().orchestrator.max_llm_calls
                ),
            ):
                # Stream each event to frontend
                event_data = {
                    "type": "agent_event",
                    "author": event.author or "system",
                    "is_final": event.is_final_response(),
                    "timestamp": datetime.now().isoformat(),
                }

                if event.content and event.content.parts:
                    text_parts = [p.text for p in event.content.parts if p.text]
                    if text_parts:
                        event_data["content"] = "\n".join(text_parts)

                    # Extract tool calls (function_call) and tool responses
                    # (function_response) so the frontend can show live
                    # tool activity for agents like ExperimentAgent.
                    tool_calls = []
                    tool_responses = []
                    for part in event.content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            fc = part.function_call
                            tool_calls.append({
                                "name": fc.name,
                                "args": _json_safe(dict(fc.args) if fc.args else {}),
                            })
                        if hasattr(part, 'function_response') and part.function_response:
                            fr = part.function_response
                            # Deep JSON-safe: a tool may return a dict that NESTS a
                            # non-serializable object (e.g. a pydantic MCPServer under
                            # "result"). A shallow isinstance(dict) check passes such a
                            # payload straight through and then ws.send_json crashes the
                            # whole stream — so sanitise recursively (objects -> str).
                            tool_responses.append({
                                "name": fr.name,
                                "response": _json_safe(fr.response),
                            })
                    if tool_calls:
                        event_data["tool_calls"] = tool_calls
                    if tool_responses:
                        event_data["tool_responses"] = tool_responses

                if event.actions and event.actions.escalate:
                    event_data["escalation"] = event.error_message or "Unknown error"

                # Check for HITL RequestInput interrupt
                if has_request_input_function_call(event):
                    hitl_interrupt_event = event
                    interrupt_ids = get_request_input_interrupt_ids(event)
                    
                    # Extract the message and schema from the function call args
                    hitl_message = ""
                    hitl_schema = None
                    for part in event.content.parts:
                        if (part.function_call 
                            and part.function_call.name == REQUEST_INPUT_FUNCTION_CALL_NAME):
                            args = part.function_call.args or {}
                            hitl_message = args.get("message", "")
                            hitl_schema = args.get("responseSchema") or args.get("response_schema")
                    
                    # Send HITL request to browser
                    hitl_payload = {
                        "type": "hitl_request",
                        "request_id": interrupt_ids[0] if interrupt_ids else "",
                        "interrupt_id": interrupt_ids[0] if interrupt_ids else "",
                        "interrupt_ids": interrupt_ids,
                        "message": hitl_message,
                        "response_schema": hitl_schema,
                        "agent_name": event.author or "system",
                        "timestamp": datetime.now().isoformat(),
                    }
                    event_data["hitl_request"] = hitl_payload
                    # Register before delivery so an immediate browser answer
                    # cannot race ahead of the pending-request table.
                    pending_wait_event = asyncio.Event()
                    for iid in interrupt_ids:
                        runtime.pending_hitl[iid] = {
                            "event": pending_wait_event,
                            "response": None,
                            "session_key": key,
                            "payload": hitl_payload,
                        }
                    await runtime.send(key, hitl_payload)

                runtime.agent_events[key].append(event_data)
                await runtime.send(key, event_data)

                if event.is_final_response() and not hitl_interrupt_event:
                    if event.content and event.content.parts:
                        final_response = event.content.parts[0].text or ""
                    elif event.actions and event.actions.escalate:
                        final_response = f"Escalation: {event.error_message or 'Unknown error'}"

            # If there was a HITL interrupt, wait for the browser response
            if hitl_interrupt_event:
                interrupt_ids = get_request_input_interrupt_ids(hitl_interrupt_event)
                
                wait_event = pending_wait_event or asyncio.Event()
                for iid in interrupt_ids:
                    runtime.pending_hitl.setdefault(iid, {
                        "event": wait_event,
                        "response": None,
                        "session_key": key,
                    })
                
                print(f"[HITL] Waiting for browser response for interrupts: {interrupt_ids}")
                
                # Wait for ALL interrupt responses (with timeout)
                try:
                    await asyncio.wait_for(wait_event.wait(), timeout=600)
                except asyncio.TimeoutError:
                    print(f"[HITL] Timeout waiting for response, auto-approving")
                    for iid in interrupt_ids:
                        if iid in runtime.pending_hitl and runtime.pending_hitl[iid]["response"] is None:
                            runtime.pending_hitl[iid]["response"] = {"approved": True}

                # Build FunctionResponse message for resume
                response_parts = []
                for iid in interrupt_ids:
                    info = runtime.pending_hitl.pop(iid, None)
                    response_data = (info["response"] if info and info["response"] else {"approved": True})
                    response_parts.append(
                        create_request_input_response(iid, response_data)
                    )

                # Resume: send FunctionResponse back to run_async
                current_message = types.Content(
                    role="user",
                    parts=response_parts,
                )
                
                await runtime.send(key, {
                    "type": "status",
                    "status": "processing",
                    "message": "Resuming workflow after HITL response...",
                })
                
                # Continue the while loop to call run_async again with the FR message
                continue
            else:
                # No interrupt, we're done
                break

        runtime.registry.touch_session(user_id, session_id, status="idle")
        await runtime.send(key, {
            "type": "final_response",
            "content": final_response,
            "timestamp": datetime.now().isoformat(),
        })

    except asyncio.CancelledError:
        runtime.registry.touch_session(user_id, session_id, status="idle")
        raise
    except Exception as exc:
        runtime.registry.touch_session(user_id, session_id, status="idle")
        error_msg = f"Error processing query: {str(exc)}"
        await runtime.send(key, {
            "type": "error",
            "message": error_msg,
            "timestamp": datetime.now().isoformat(),
        })
        runtime.agent_events[key].append({
            "type": "error",
            "message": error_msg,
            "timestamp": datetime.now().isoformat(),
        })


def _handle_hitl_response(runtime: WebRuntime, key: SessionKey, data: dict):
    """Resolve a pending HITL request from the browser.
    
    Routes responses to either:
    1. WebHITLHandler (for SessionAgent's custom HITL, e.g. PlannerAgent)
    2. _pending_hitl dict (for ADK RequestInput workflow interrupts)
    
    The browser sends back:
        {
            "type": "hitl_response",
            "request_id": "<id>",   // for WebHITLHandler
            "interrupt_id": "<id>", // for ADK RequestInput
            "approved": true/false,
            "feedback": "..."
        }
    """
    request_id = data.get("request_id")
    interrupt_id = data.get("interrupt_id")

    # 1) Try WebHITLHandler (SessionAgent / PlannerAgent HITL)
    if request_id:
        resolved = runtime.hitl_handler.resolve_request(request_id, data, key)
        if resolved:
            return
        # If it was resolved there, no need to check _pending_hitl
        if request_id not in runtime.pending_hitl:
            return

    # 2) Try ADK RequestInput mechanism
    lookup_id = interrupt_id or request_id
    if not lookup_id:
        print("[HITL] No interrupt_id or request_id in hitl_response, ignoring")
        return

    info = runtime.pending_hitl.get(lookup_id)
    if not info:
        # Already handled by WebHITLHandler or unknown
        return
    if info.get("session_key") != key:
        logging.getLogger("CoScientist.web").warning(
            "Ignoring RequestInput response from the wrong session"
        )
        return
    
    # Store the response data
    response = {
        "approved": data.get("approved", False),
        "feedback": data.get("feedback"),
        "instructions": data.get("instructions"),
        "free_input": data.get("free_input"),
    }
    info["response"] = response
    
    # Check if all interrupt IDs sharing this wait_event have responses
    wait_event = info["event"]
    for pending in runtime.pending_hitl.values():
        if pending["event"] is wait_event and pending.get("session_key") == key:
            pending["response"] = response
    all_resolved = all(
        v["response"] is not None
        for v in runtime.pending_hitl.values()
        if v["event"] is wait_event and v.get("session_key") == key
    )
    if all_resolved:
        wait_event.set()  # Unblock the _handle_chat loop
