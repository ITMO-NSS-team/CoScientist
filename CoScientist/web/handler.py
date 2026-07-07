import asyncio
import logging
import time
import uuid

from CoScientist.hitl.handler import AbstractHITLHandler
from CoScientist.hitl.models import HITLRequest, HITLResponse, HITLAction

logger = logging.getLogger("CoScientist.web.hitl")


class WebHITLHandler(AbstractHITLHandler):

    def __init__(self):
        # request_id -> {"future": asyncio.Future, "payload": dict, "created": float}
        self._pending: dict[str, dict] = {}
        # ALL connected browser sockets. HITL requests are BROADCAST to every
        # tab (chat events are per-connection, but a review bound to a single
        # "last" socket silently vanishes when another tab/window is open).
        self._sockets: list = []
        # event log that the frontend can poll
        self._event_log: list[dict] = []

    def __deepcopy__(self, memo):
        return self

    @property
    def _websocket(self):
        """Back-compat: the most recently attached socket (None when empty)."""
        return self._sockets[-1] if self._sockets else None

    def set_websocket(self, ws):
        """Legacy setter: make `ws` the only known socket (None clears all)."""
        self._sockets = [] if ws is None else [ws]

    def has_connections(self) -> bool:
        return bool(self._sockets)

    async def attach_websocket(self, ws):
        """Register a (re)connected browser socket and re-deliver pending
        requests to it, so a review raised while this tab was away reappears
        instead of silently auto-approving on timeout."""
        if ws not in self._sockets:
            self._sockets.append(ws)
        logger.info("HITL websocket attached (%d connection(s))", len(self._sockets))
        for request_id, entry in list(self._pending.items()):
            try:
                await ws.send_json(entry["payload"])
                logger.info(
                    "HITL request %s (%s) re-delivered to the new connection",
                    request_id[:8], entry["payload"].get("agent_name"),
                )
            except Exception as exc:  # noqa: BLE001 — delivery is best-effort
                logger.warning("HITL re-delivery of %s failed: %s", request_id[:8], exc)

    def detach_websocket(self, ws):
        try:
            self._sockets.remove(ws)
        except ValueError:
            pass
        logger.info("HITL websocket detached (%d connection(s) left)", len(self._sockets))

    async def _broadcast(self, payload) -> int:
        """Send to every connected tab, pruning dead sockets. Returns count."""
        delivered = 0
        for ws in list(self._sockets):
            try:
                await ws.send_json(payload)
                delivered += 1
            except Exception as exc:  # noqa: BLE001 — prune and continue
                logger.warning("HITL delivery to a socket failed (%s) — pruning", exc)
                self.detach_websocket(ws)
        return delivered

    # Seconds to wait for a HITL response before auto-approving (prevents infinite hang)
    HITL_TIMEOUT_SECONDS: int = 300

    def pending_summary(self) -> list[dict]:
        """Diagnostics for /api/hitl-status."""
        now = time.time()
        return [
            {
                "request_id": rid[:8],
                "agent_name": entry["payload"].get("agent_name"),
                "age_seconds": round(now - entry["created"], 1),
            }
            for rid, entry in self._pending.items()
        ]

    async def handle_request(self, request: HITLRequest) -> HITLResponse:
        request_id = str(uuid.uuid4())

        payload = {
            "type": "hitl_request",
            "request_id": request_id,
            "agent_name": request.agent_name,
            "action_type": request.action_type.value,
            "message": request.message,
            "options": request.options,
            "context": request.context,
            "invoked_via": request.invoked_via,
        }

        self._event_log.append(payload)

        # Register BEFORE sending so a reconnect between send and await
        # re-delivers instead of losing the request.
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending[request_id] = {
            "future": future, "payload": payload, "created": time.time(),
        }

        delivered = await self._broadcast(payload)
        if delivered:
            logger.info(
                "HITL request %s (%s) sent to %d browser tab(s)",
                request_id[:8], request.agent_name, delivered,
            )
        else:
            logger.warning(
                "HITL request %s (%s): NO live browser connection — will "
                "re-deliver on reconnect; auto-approve in %ss",
                request_id[:8], request.agent_name, self.HITL_TIMEOUT_SECONDS,
            )

        try:
            response_data = await asyncio.wait_for(
                asyncio.shield(future),
                timeout=self.HITL_TIMEOUT_SECONDS,
            )
            logger.info(
                "HITL response for %s (%s): %s",
                request_id[:8], request.agent_name,
                {k: response_data.get(k) for k in ("action", "approved")},
            )
        except asyncio.TimeoutError:
            self._pending.pop(request_id, None)
            logger.warning(
                "HITL TIMEOUT for %s (%s): no browser response in %ss — AUTO-APPROVING",
                request_id[:8], request.agent_name, self.HITL_TIMEOUT_SECONDS,
            )
            response_data = {"action": "approve", "approved": True}
            await self._broadcast({
                "type": "hitl_timeout",
                "request_id": request_id,
                "agent_name": request.agent_name,
                "timeout_seconds": self.HITL_TIMEOUT_SECONDS,
            })

        action = HITLAction(response_data.get("action", "approve"))
        return HITLResponse(
            action=action,
            approved=response_data.get("approved", False),
            selected_option=response_data.get("selected_option"),
            instructions=response_data.get("instructions"),
            free_input=response_data.get("free_input"),
        )

    def resolve_request(self, request_id: str, response_data: dict):
        """Called when the browser sends a HITL response."""
        entry = self._pending.pop(request_id, None)
        if entry and not entry["future"].done():
            entry["future"].set_result(response_data)

    def reset(self):
        """Cancel all pending requests and clear state."""
        for entry in list(self._pending.values()):
            if not entry["future"].done():
                entry["future"].cancel()
        self._pending.clear()
        self._event_log.clear()

    def get_event_log(self) -> list[dict]:
        return list(self._event_log)

    def clear_event_log(self):
        self._event_log.clear()
