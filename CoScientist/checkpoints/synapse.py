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
