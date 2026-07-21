import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from CoScientist.main import CoScientistManager
from CoScientist.web.handler import WebHITLHandler
from CoScientist.agents import agent_system
from CoScientist.hitl.tool import hitl_toolset
from CoScientist.tools.coder_tools import coder_toolset
from CoScientist.tools.task_tracker import task_tracker_instance
from CoScientist.config import get_settings

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
# Globals
# ---------------------------------------------------------------------------
WEB_DIR = Path(__file__).parent
TEMPLATE_PATH = WEB_DIR / "templates" / "index.html"

# Manager will be lazily created so the import doesn't trigger heavy init
_manager = None
_manager_lock = asyncio.Lock()

# Store agent events for the frontend
_agent_events: list[dict] = []

# Pending HITL requests: interrupt_id -> { "event": asyncio.Event, "response": dict }
_pending_hitl: dict[str, dict] = {}

# WebHITLHandler for SessionAgent's custom HITL (used by PlannerAgent)
_web_hitl_handler = WebHITLHandler()


def _apply_frontend_settings(frontend: dict) -> None:
    """Map the frontend JS ``appSettings`` object to ``settings.web``.

    Called before every ``_get_manager()`` invocation so the config singleton
    is always up-to-date when the system is (re)built.
    """
    from CoScientist.config import get_settings
    web = get_settings().web

    general = frontend.get("general", {})
    if "startMode" in general:
        web.start_mode = general["startMode"]
    if "maxRetries" in general:
        web.max_retries = int(general["maxRetries"])
    if "hitlEnabled" in general:
        val = bool(general["hitlEnabled"])
        web.hitl_enabled = val
        get_settings().hitl.enabled = val
    if "usePlanner" in general:
        web.use_planner = bool(general["usePlanner"])
    if "opikEnabled" in general:
        web.opik_enabled = bool(general["opikEnabled"])

    research = frontend.get("researchAgent", {})
    if "maxSearches" in research:
        val = int(research["maxSearches"])
        if val >= 0:
            web.max_searches = val

    task_exec = frontend.get("taskExecutorAgent", {})
    if "keepScore" in task_exec:
        web.executor_tool_keep_score = float(task_exec["keepScore"])
    if "abstainScore" in task_exec:
        web.executor_tool_abstain_score = float(task_exec["abstainScore"])

    coder = frontend.get("coderAgent", {})
    if "sandboxUrl" in coder:
        web.sandbox_url = coder["sandboxUrl"]
    if "workspaceId" in coder:
        val = coder["workspaceId"]
        web.coder_workspace_id = val if val else None


# Track which start_mode the current manager was built with so we can
# detect changes and rebuild (the source of truth is settings.web but we
# need to compare to the value at build time).
_built_start_mode: Optional[str] = None


async def _get_manager():
    """Lazy-init CoScientistManager for the current settings.

    If ``settings.web.start_mode`` changed since the last init, tear down
    the old manager first and rebuild.
    """
    from CoScientist.config import get_settings
    global _manager, _built_start_mode

    current_mode = get_settings().web.start_mode

    if _manager is not None and _built_start_mode == current_mode:
        return _manager

    async with _manager_lock:
        current_mode = get_settings().web.start_mode
        # Double-check inside the lock.
        if _manager is not None and _built_start_mode == current_mode:
            return _manager

        # Tear down a stale manager whose mode no longer matches.
        if _manager is not None:
            await _manager.close()
            _manager = None

        _manager = CoScientistManager()
        await _manager.initialize()
        _built_start_mode = current_mode

        # Wire WebHITLHandler into every session agent with a HITL review loop
        # (PlannerAgent, and the ТЗ agents of the microfluidics profile) and
        # into the HITL toolset. We use set_delegate because the workflow
        # deepcopies references at init.
        wired = []
        for _name, _agent in agent_system.agents.items():
            handler = getattr(_agent, 'hitl_handler', None)
            if handler is not None and hasattr(handler, 'set_delegate'):
                handler.set_delegate(_web_hitl_handler)
                wired.append(_name)
        logging.getLogger("CoScientist.web").info(
            "WebHITLHandler wired into session agents: %s "
            "(empty list => HITL disabled via HITL__ENABLED)", wired,
        )

        # Also update the hitl_toolset if it's delegating
        if hasattr(hitl_toolset._handler, 'set_delegate'):
            hitl_toolset._handler.set_delegate(_web_hitl_handler)
        else:
            hitl_toolset._handler = _web_hitl_handler

        # Update the coder_toolset if it's delegating
        if getattr(coder_toolset, "_hitl_handler", None) is not None:
            if hasattr(coder_toolset._hitl_handler, 'set_delegate'):
                coder_toolset._hitl_handler.set_delegate(_web_hitl_handler)
            else:
                coder_toolset._hitl_handler = _web_hitl_handler

        return _manager


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[CoScientist Web] Starting up …")
    yield
    print("[CoScientist Web] Shutting down …")
    if _manager:
        await _manager.close()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    os.environ["COSCIENTIST_WEB_MODE"] = "true"
    app = FastAPI(
        title="CoScientist Web UI",
        version="1.0.0",
        lifespan=lifespan,
    )

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
            "hitl_enabled": get_settings().web.hitl_enabled,
            "websocket_connections": len(_web_hitl_handler._sockets),
            "session_agents_with_handler": agents,
            "pending_requests": _web_hitl_handler.pending_summary(),
            "auto_approve_timeout_seconds": _web_hitl_handler.HITL_TIMEOUT_SECONDS,
        })

    # --- Knowledge graph (live view) ---
    @app.get("/graph", response_class=HTMLResponse)
    async def graph_page():
        return (WEB_DIR / "templates" / "graph.html").read_text(encoding="utf-8")

    @app.get("/api/graph")
    async def api_graph(view: str = "execution"):
        """Current graph (the /graph page polls this).

        view=research → the typed research context graph (the shared blackboard);
        view=execution → the raw agent-activity log graph;
        view=memory → the cross-run knowledge memory.
        """
        try:
            if view == "research":
                # The typed research context graph (the blackboard agents write).
                from CoScientist.graph.research.store import research_graph
                return JSONResponse(research_graph.to_view())
            if view == "memory":
                from CoScientist.graph.memory_store import knowledge_memory
                return JSONResponse(knowledge_memory.full())
            from CoScientist.graph.memory import knowledge_graph
            return JSONResponse(knowledge_graph.full())
        except Exception as e:  # noqa: BLE001 — never break the UI
            return JSONResponse({"nodes": [], "edges": [], "error": str(e)}, status_code=500)

    # --- Roadmap endpoints ---
    @app.get("/api/roadmap")
    async def get_roadmap():
        try:
            content = json.dumps(task_tracker_instance.tasks, indent=2, ensure_ascii=False)
            return JSONResponse({"content": content})
        except Exception as e:
            return JSONResponse({"content": "", "error": str(e)}, status_code=500)

    @app.post("/api/roadmap")
    async def save_roadmap(data: dict):
        content = data.get("content", "")
        try:
            parsed = json.loads(content)
            if not isinstance(parsed, list):
                return JSONResponse({"status": "error", "error": "Content must be a JSON array"}, status_code=400)
            task_tracker_instance.tasks = parsed
            return JSONResponse({"status": "success"})
        except json.JSONDecodeError as e:
            return JSONResponse({"status": "error", "error": f"Invalid JSON: {e}"}, status_code=400)
        except Exception as e:
            return JSONResponse({"status": "error", "error": str(e)}, status_code=500)


    # --- Settings endpoints ---
    @app.get("/api/settings")
    async def get_settings_api():
        """Return current WebSettings."""
        from CoScientist.config import get_settings
        web = get_settings().web
        return JSONResponse({
            "general": {
                "startMode": web.start_mode,
                "maxRetries": web.max_retries,
                "hitlEnabled": web.hitl_enabled,
                "usePlanner": web.use_planner,
                "opikEnabled": web.opik_enabled,
            },
            "researchAgent": {"maxSearches": web.max_searches},
            "taskExecutorAgent": {
                "keepScore": web.executor_tool_keep_score,
                "abstainScore": web.executor_tool_abstain_score,
            },
            "coderAgent": {
                "sandboxUrl": web.sandbox_url,
                "workspaceId": web.coder_workspace_id or "",
            },
        })

    @app.post("/api/settings")
    async def save_settings_api(data: dict):
        """Update WebSettings from the frontend."""
        from CoScientist.config import get_settings
        _apply_frontend_settings(data)
        web = get_settings().web
        return JSONResponse({
            "status": "success",
            "general": {
                "startMode": web.start_mode,
                "maxRetries": web.max_retries,
                "hitlEnabled": web.hitl_enabled,
                "usePlanner": web.use_planner,
                "opikEnabled": web.opik_enabled,
            },
            "researchAgent": {"maxSearches": web.max_searches},
            "taskExecutorAgent": {
                "keepScore": web.executor_tool_keep_score,
                "abstainScore": web.executor_tool_abstain_score,
            },
            "coderAgent": {
                "sandboxUrl": web.sandbox_url,
                "workspaceId": web.coder_workspace_id or "",
            },
        })
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
    @app.get("/api/events")
    async def get_events():
        return JSONResponse({"events": _agent_events[-100:]})

    # --- WebSocket ---
    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await ws.accept()

        # Bind the socket AND re-deliver any pending HITL requests: a review
        # raised during a reconnect (or bound to a closed tab) must reappear
        # instead of silently auto-approving on timeout.
        await _web_hitl_handler.attach_websocket(ws)

        # Send initial connection confirmation
        await ws.send_json({
            "type": "connected",
            "timestamp": datetime.now().isoformat(),
            "message": "Connected to CoScientist backend",
        })

        active_task: Optional[asyncio.Task] = None

        try:
            while True:
                raw = await ws.receive_text()
                data = json.loads(raw)
                msg_type = data.get("type", "")

                if msg_type == "chat_message":
                    if active_task and not active_task.done():
                        active_task.cancel()
                        try:
                            await active_task
                        except asyncio.CancelledError:
                            pass
                    active_task = asyncio.create_task(_handle_chat(ws, data))
                elif msg_type == "stop_chat":
                    if active_task and not active_task.done():
                        active_task.cancel()
                        try:
                            await active_task
                        except asyncio.CancelledError:
                            pass
                        active_task = None
                    
                    # Cancel all pending HITL requests
                    _cancel_pending_hitl()
                    _web_hitl_handler.reset()

                    # Erase manager memory
                    global _manager, _built_start_mode
                    async with _manager_lock:
                        if _manager:
                            await _manager.close()
                            _manager = None
                            _built_start_mode = None
                    
                    # Clear events log + per-session state (tasks + execution
                    # graph) so a new run starts clean. Cross-run memory is kept.
                    _agent_events.clear()
                    try:
                        from CoScientist.main import reset_session_state
                        reset_session_state()
                    except Exception:  # noqa: BLE001
                        pass

                    await ws.send_json({
                        "type": "status",
                        "status": "idle",
                        "message": "Agent execution stopped, memory cleared.",
                    })
                    await ws.send_json({
                        "type": "final_response",
                        "content": "Stopped",
                    })
                elif msg_type == "hitl_response":
                    _handle_hitl_response(data)
                elif msg_type == "ping":
                    await ws.send_json({"type": "pong"})
                else:
                    await ws.send_json({
                        "type": "error",
                        "message": f"Unknown message type: {msg_type}",
                    })
        except WebSocketDisconnect:
            # Only drop THIS socket. Pending HITL reviews stay alive: they are
            # re-delivered when a tab reconnects, and auto-approve on timeout.
            _web_hitl_handler.detach_websocket(ws)
            if active_task and not active_task.done():
                active_task.cancel()
            print("[WebSocket] Client disconnected")
        except Exception as exc:
            _web_hitl_handler.detach_websocket(ws)
            if active_task and not active_task.done():
                active_task.cancel()
            print(f"[WebSocket] Error: {exc}")

    return app


# ---------------------------------------------------------------------------
# HITL helpers
# ---------------------------------------------------------------------------
def _cancel_pending_hitl():
    """Cancel all pending HITL requests."""
    for info in _pending_hitl.values():
        info["event"].set()  # unblock any waiters
    _pending_hitl.clear()


# ---------------------------------------------------------------------------
# Message handlers
# ---------------------------------------------------------------------------
async def _handle_chat(ws: WebSocket, data: dict):
    """Run user query through the agent pipeline, streaming events.
    
    Handles ADK RequestInput HITL: when the workflow pauses (interrupt event),
    this sends the HITL request to the browser, waits for the response, then
    resumes the workflow by calling run_async with a FunctionResponse message.
    """
    query = data.get("message", "").strip()
    if not query:
        await ws.send_json({"type": "error", "message": "Empty query"})
        return

    # Echo user message
    _agent_events.append({
        "type": "user_message",
        "message": query,
        "timestamp": datetime.now().isoformat(),
    })

    await ws.send_json({
        "type": "status",
        "status": "processing",
        "message": f"Processing query: {query}",
    })

    try:
        manager = await _get_manager()

        # The message to send for this invocation (initially the user query)
        current_message = types.Content(
            role="user",
            parts=[types.Part(text=query)],
        )

        final_response = "No response"

        # Loop: run -> check for HITL interrupt -> wait for response -> resume
        while True:
            hitl_interrupt_event = None

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
                        "interrupt_ids": interrupt_ids,
                        "message": hitl_message,
                        "response_schema": hitl_schema,
                        "agent_name": event.author or "system",
                        "timestamp": datetime.now().isoformat(),
                    }
                    event_data["hitl_request"] = hitl_payload
                    await ws.send_json(hitl_payload)

                _agent_events.append(event_data)
                await ws.send_json(event_data)

                if event.is_final_response() and not hitl_interrupt_event:
                    if event.content and event.content.parts:
                        final_response = event.content.parts[0].text or ""
                    elif event.actions and event.actions.escalate:
                        final_response = f"Escalation: {event.error_message or 'Unknown error'}"

            # If there was a HITL interrupt, wait for the browser response
            if hitl_interrupt_event:
                interrupt_ids = get_request_input_interrupt_ids(hitl_interrupt_event)
                
                # Register pending HITL for each interrupt_id
                wait_event = asyncio.Event()
                for iid in interrupt_ids:
                    _pending_hitl[iid] = {"event": wait_event, "response": None}
                
                print(f"[HITL] Waiting for browser response for interrupts: {interrupt_ids}")
                
                # Wait for ALL interrupt responses (with timeout)
                try:
                    await asyncio.wait_for(wait_event.wait(), timeout=600)
                except asyncio.TimeoutError:
                    print(f"[HITL] Timeout waiting for response, auto-approving")
                    for iid in interrupt_ids:
                        if iid in _pending_hitl and _pending_hitl[iid]["response"] is None:
                            _pending_hitl[iid]["response"] = {"approved": True}

                # Build FunctionResponse message for resume
                response_parts = []
                for iid in interrupt_ids:
                    info = _pending_hitl.pop(iid, None)
                    response_data = (info["response"] if info and info["response"] else {"approved": True})
                    response_parts.append(
                        create_request_input_response(iid, response_data)
                    )

                # Resume: send FunctionResponse back to run_async
                current_message = types.Content(
                    role="user",
                    parts=response_parts,
                )
                
                await ws.send_json({
                    "type": "status",
                    "status": "processing",
                    "message": "Resuming workflow after HITL response...",
                })
                
                # Continue the while loop to call run_async again with the FR message
                continue
            else:
                # No interrupt, we're done
                break

        await ws.send_json({
            "type": "final_response",
            "content": final_response,
            "timestamp": datetime.now().isoformat(),
        })

    except asyncio.CancelledError:
        # Propagate task cancellation cleanly
        raise
    except Exception as exc:
        error_msg = f"Error processing query: {str(exc)}"
        await ws.send_json({
            "type": "error",
            "message": error_msg,
            "timestamp": datetime.now().isoformat(),
        })
        _agent_events.append({
            "type": "error",
            "message": error_msg,
            "timestamp": datetime.now().isoformat(),
        })


def _handle_hitl_response(data: dict):
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
        _web_hitl_handler.resolve_request(request_id, data)
        # If it was resolved there, no need to check _pending_hitl
        if request_id not in {k for k in _pending_hitl}:
            return

    # 2) Try ADK RequestInput mechanism
    lookup_id = interrupt_id or request_id
    if not lookup_id:
        print("[HITL] No interrupt_id or request_id in hitl_response, ignoring")
        return

    info = _pending_hitl.get(lookup_id)
    if not info:
        # Already handled by WebHITLHandler or unknown
        return
    
    # Store the response data
    info["response"] = {
        "approved": data.get("approved", False),
        "feedback": data.get("feedback"),
        "instructions": data.get("instructions"),
        "free_input": data.get("free_input"),
    }
    
    # Check if all interrupt IDs sharing this wait_event have responses
    wait_event = info["event"]
    all_resolved = all(
        v["response"] is not None
        for v in _pending_hitl.values()
        if v["event"] is wait_event
    )
    if all_resolved:
        wait_event.set()  # Unblock the _handle_chat loop