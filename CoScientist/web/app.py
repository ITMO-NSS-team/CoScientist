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

from CoScientist.main import CoScientistManager
from CoScientist.web.handler import WebHITLHandler
from CoScientist.agents import planner_agent
from CoScientist.hitl.tool import hitl_toolset

from google.genai import types
from google.adk.workflow.utils._workflow_hitl_utils import (
    has_request_input_function_call,
    get_request_input_interrupt_ids,
    create_request_input_response,
    REQUEST_INPUT_FUNCTION_CALL_NAME,
)

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


async def _get_manager():
    """Lazy-init CoScientistManager."""
    global _manager
    if _manager is not None:
        return _manager

    async with _manager_lock:
        if _manager is not None:
            return _manager

        _manager = CoScientistManager()
        await _manager.initialize()

        # Wire WebHITLHandler into PlannerAgent (SessionAgent) and HITL toolset
        # We use set_delegate because the workflow deepcopies references at init
        if hasattr(planner_agent, 'hitl_handler') and hasattr(planner_agent.hitl_handler, 'set_delegate'):
            planner_agent.hitl_handler.set_delegate(_web_hitl_handler)
        
        # Also update the hitl_toolset if it's delegating
        if hasattr(hitl_toolset._handler, 'set_delegate'):
            hitl_toolset._handler.set_delegate(_web_hitl_handler)
        else:
            hitl_toolset._handler = _web_hitl_handler

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

    # --- HTML endpoint ---
    @app.get("/", response_class=HTMLResponse)
    async def index():
        return TEMPLATE_PATH.read_text(encoding="utf-8")

    # --- Roadmap endpoints ---
    @app.get("/api/roadmap")
    async def get_roadmap():
        path = Path("task_tracker_data.json")
        if not path.exists():
            path = Path(__file__).parent.parent.parent / "task_tracker_data.json"
        
        if not path.exists():
            return JSONResponse({"content": "", "error": "task_tracker_data.json not found"}, status_code=404)
        
        try:
            content = path.read_text(encoding="utf-8")
            return JSONResponse({"content": content})
        except Exception as e:
            return JSONResponse({"content": "", "error": str(e)}, status_code=500)

    @app.post("/api/roadmap")
    async def save_roadmap(data: dict):
        content = data.get("content", "")
        path = Path("task_tracker_data.json")
        if not path.exists():
            path = Path(__file__).parent.parent.parent / "task_tracker_data.json"
            
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

        # Set websocket on the WebHITLHandler so it can send HITL requests
        _web_hitl_handler.set_websocket(ws)

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
            if _web_hitl_handler._websocket == ws:
                _cancel_pending_hitl()
                _web_hitl_handler.reset()
                _web_hitl_handler.set_websocket(None)
            if active_task and not active_task.done():
                active_task.cancel()
            print("[WebSocket] Client disconnected")
        except Exception as exc:
            if _web_hitl_handler._websocket == ws:
                _cancel_pending_hitl()
                _web_hitl_handler.reset()
                _web_hitl_handler.set_websocket(None)
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
                                "args": dict(fc.args) if fc.args else {},
                            })
                        if hasattr(part, 'function_response') and part.function_response:
                            fr = part.function_response
                            # Safely serialise – response can be dict, str,
                            # Pydantic model, etc.
                            try:
                                resp_payload = (
                                    fr.response
                                    if isinstance(fr.response, (dict, list, str, int, float, bool, type(None)))
                                    else str(fr.response)
                                )
                            except Exception:
                                resp_payload = str(fr.response)
                            tool_responses.append({
                                "name": fr.name,
                                "response": resp_payload,
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