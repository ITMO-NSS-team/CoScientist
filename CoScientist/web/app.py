"""
FastAPI application for CoScientist web interface.

Provides:
  - HTML UI served from templates
  - WebSocket for real-time agent events and HITL
  - REST endpoints for session management
"""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from CoScientist.web.handler import WebHITLHandler
from CoScientist.main import CoScientistManager

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
WEB_DIR = Path(__file__).parent
TEMPLATE_PATH = WEB_DIR / "templates" / "index.html"

web_hitl_handler = WebHITLHandler()

# Manager will be lazily created so the import doesn't trigger heavy init
_manager = None
_manager_lock = asyncio.Lock()

# Store agent events for the frontend
_agent_events: list[dict] = []


async def _get_manager():
    """Lazy-init CoScientistManager with web HITL handler."""
    global _manager
    if _manager is not None:
        return _manager

    async with _manager_lock:
        if _manager is not None:
            return _manager

        _manager = CoScientistManager(hitl_handler=web_hitl_handler)
        await _manager.initialize()
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
    app = FastAPI(
        title="CoScientist Web UI",
        version="1.0.0",
        lifespan=lifespan,
    )

    # --- HTML endpoint ---
    @app.get("/", response_class=HTMLResponse)
    async def index():
        return TEMPLATE_PATH.read_text(encoding="utf-8")

    # --- Roadmap endpoints ---
    @app.get("/api/roadmap")
    async def get_roadmap():
        # Look for roadmap.txt in the current directory or parent directories
        path = Path("roadmap.txt")
        if not path.exists():
            # Try workspace directory CoScientist/roadmap.txt
            path = Path(__file__).parent.parent.parent / "roadmap.txt"
        
        if not path.exists():
            return JSONResponse({"content": "", "error": "roadmap.txt not found"}, status_code=404)
        
        try:
            content = path.read_text(encoding="utf-8")
            return JSONResponse({"content": content})
        except Exception as e:
            return JSONResponse({"content": "", "error": str(e)}, status_code=500)

    @app.post("/api/roadmap")
    async def save_roadmap(data: dict):
        content = data.get("content", "")
        path = Path("roadmap.txt")
        if not path.exists():
            path = Path(__file__).parent.parent.parent / "roadmap.txt"
            
        try:
            path.write_text(content, encoding="utf-8")
            return JSONResponse({"status": "success"})
        except Exception as e:
            return JSONResponse({"status": "error", "error": str(e)}, status_code=500)


    # --- Agent info ---
    @app.get("/api/agents")
    async def get_agents():
        """Return list of registered agents."""
        return JSONResponse({
            "agents": [
                {"name": "OrchestratorAgent", "role": "orchestrator", "status": "idle"},
                {"name": "PlannerAgent", "role": "planner", "status": "idle"},
                {"name": "HypothesesAgent", "role": "hypothesis", "status": "idle"},
                {"name": "ResearchAgent", "role": "research", "status": "idle"},
                {"name": "ToolRetrieverAgent", "role": "tool_retriever", "status": "idle"},
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
        web_hitl_handler.set_websocket(ws)

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
                    
                    # Reset HITL handler
                    web_hitl_handler.reset()

                    # Erase manager memory
                    global _manager
                    async with _manager_lock:
                        if _manager:
                            await _manager.close()
                            _manager = None
                    
                    # Clear events log
                    _agent_events.clear()
                    
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
            web_hitl_handler.set_websocket(None)
            if active_task and not active_task.done():
                active_task.cancel()
            print("[WebSocket] Client disconnected")
        except Exception as exc:
            web_hitl_handler.set_websocket(None)
            if active_task and not active_task.done():
                active_task.cancel()
            print(f"[WebSocket] Error: {exc}")

    return app


# ---------------------------------------------------------------------------
# Message handlers
# ---------------------------------------------------------------------------
async def _handle_chat(ws: WebSocket, data: dict):
    """Run user query through the agent pipeline, streaming events."""
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

        # Use ADK runner to stream events
        from google.genai import types

        content = types.Content(
            role="user",
            parts=[types.Part(text=query)],
        )

        final_response = "No response"

        async for event in manager.runner.run_async(
            user_id=manager.user_id,
            session_id=manager.session_id,
            new_message=content,
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

            if event.actions and event.actions.escalate:
                event_data["escalation"] = event.error_message or "Unknown error"

            _agent_events.append(event_data)
            await ws.send_json(event_data)

            if event.is_final_response():
                if event.content and event.content.parts:
                    final_response = event.content.parts[0].text or ""
                elif event.actions and event.actions.escalate:
                    final_response = f"Escalation: {event.error_message or 'Unknown error'}"

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
    """Resolve a pending HITL request from the browser."""
    request_id = data.get("request_id")
    if not request_id:
        return

    web_hitl_handler.resolve_request(request_id, {
        "action": data.get("action", "approve"),
        "approved": data.get("approved", False),
        "selected_option": data.get("selected_option"),
        "instructions": data.get("instructions"),
        "free_input": data.get("free_input"),
    })
