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
