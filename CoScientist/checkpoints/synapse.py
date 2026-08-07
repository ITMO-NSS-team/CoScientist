"""Synapse v1 contract bridge for the checkpoint subsystem.

Off unless SYNAPSE__ENABLED. Holds a process-local registry mapping an A2A
contextId (== ADK session id) to the platform-issued run_id + traceparent, so
capture stamps the platform's run_id instead of the self-generated one. Also
builds the snapshot_ref and fires the outbound "snapshot ready" callback.

Every function is best-effort: a bridge failure never breaks the science run.
"""
from __future__ import annotations

import logging
import threading
from typing import Dict, Optional

import httpx

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_RUN_CONTEXT: Dict[str, Dict[str, Optional[str]]] = {}   # context_id -> {run_id, traceparent}


def _bundle_base_url() -> Optional[str]:
    from CoScientist.config import get_settings
    return get_settings().synapse.bundle_base_url


def register_run(context_id: str, run_id: str, traceparent: Optional[str] = None) -> None:
    with _LOCK:
        _RUN_CONTEXT[context_id] = {"run_id": run_id, "traceparent": traceparent}


def run_id_for(context_id: str) -> Optional[str]:
    with _LOCK:
        entry = _RUN_CONTEXT.get(context_id)
        return entry["run_id"] if entry else None


def traceparent_for(context_id: str) -> Optional[str]:
    with _LOCK:
        entry = _RUN_CONTEXT.get(context_id)
        return entry["traceparent"] if entry else None


def clear_runs() -> None:
    with _LOCK:
        _RUN_CONTEXT.clear()


def snapshot_ref_for(checkpoint_id: str) -> Optional[str]:
    base = _bundle_base_url()
    if not base:
        return None
    return f"{base.rstrip('/')}/api/checkpoints/{checkpoint_id}/bundle"


def _synapse_cfg():
    from CoScientist.config import get_settings
    return get_settings().synapse


def notify_snapshot_saved(manifest) -> None:
    """Tell the platform a snapshot is ready (best-effort). v1 §5: one call,
    the platform records the point itself — no separate 'snapshot created' event."""
    cfg = _synapse_cfg()
    if not cfg.enabled or not cfg.callback_url:
        return
    body = {
        "point_id": manifest.checkpoint_id,
        "run_id": manifest.run_id,
        "time": manifest.created_at,
        "label": manifest.label,
        "snapshot_ref": manifest.snapshot_ref,
    }
    try:
        httpx.post(f"{cfg.callback_url.rstrip('/')}/points", json=body, timeout=5.0)
    except Exception as exc:  # noqa: BLE001 — never break the run
        logger.warning("synapse: snapshot-ready callback failed: %s", exc)


# ── minimal OTel: hang our steps under the platform's traceparent ────────────
from opentelemetry import trace as _ot_trace
from opentelemetry.trace import (
    NonRecordingSpan, SpanContext, TraceFlags, set_span_in_context,
)

_TRACER = None
_OTEL_READY = False


def setup_otel() -> None:
    """Install an OTLP exporter once (best-effort). No-op if unconfigured."""
    global _TRACER, _OTEL_READY
    if _OTEL_READY:
        return
    _OTEL_READY = True
    cfg = _synapse_cfg()
    try:
        if cfg.otlp_endpoint:
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            provider = TracerProvider()
            provider.add_span_processor(
                SimpleSpanProcessor(OTLPSpanExporter(endpoint=cfg.otlp_endpoint)))
            _ot_trace.set_tracer_provider(provider)
        _TRACER = _ot_trace.get_tracer("coscientist.synapse")
    except Exception as exc:  # noqa: BLE001
        logger.warning("synapse: OTel setup failed: %s", exc)


def _parent_ctx(traceparent: Optional[str]):
    """W3C traceparent -> an OTel context whose current span is the remote parent."""
    if not traceparent:
        return None
    try:
        _v, trace_id, span_id, flags = traceparent.split("-")
        sc = SpanContext(
            trace_id=int(trace_id, 16), span_id=int(span_id, 16),
            is_remote=True, trace_flags=TraceFlags(int(flags, 16)),
        )
        return set_span_in_context(NonRecordingSpan(sc))
    except Exception:  # noqa: BLE001
        return None


def _start_run_span(context_id: str, agent_name: str, run_id: str):
    if _TRACER is None:
        return None
    ctx = _parent_ctx(traceparent_for(context_id))
    return _TRACER.start_span(
        "invoke_agent", context=ctx,
        attributes={"gen_ai.operation.name": "invoke_agent",
                    "gen_ai.agent.name": agent_name, "run_id": run_id},
    )


def _end_run_span(handle) -> None:
    if handle is not None:
        try:
            handle.end()
        except Exception:  # noqa: BLE001
            pass


from google.adk.plugins.base_plugin import BasePlugin as _BasePlugin


class SynapseTracePlugin(_BasePlugin):
    """Hangs one invoke_agent span per run under the platform's traceparent."""

    def __init__(self) -> None:
        super().__init__(name="synapse_trace")
        setup_otel()
        self._spans: Dict[str, object] = {}

    async def before_run_callback(self, *, invocation_context):
        s = invocation_context.session
        rid = run_id_for(s.id) or f"{s.app_name}__{s.id}"
        self._spans[invocation_context.invocation_id] = _start_run_span(
            s.id, invocation_context.agent.name, rid)
        return None

    async def after_run_callback(self, *, invocation_context):
        _end_run_span(self._spans.pop(invocation_context.invocation_id, None))
        return None
