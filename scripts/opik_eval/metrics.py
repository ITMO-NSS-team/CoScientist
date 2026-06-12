"""Per-trace reliability metrics from Opik spans.

Quantifies the three failure modes recorded in DEVGRAPH F014 / measured by F015h:
  - empty/blank LLM responses (OpenRouter flakiness),
  - hallucinated/unregistered tool names ("Tool 'X' not found"),
  - runaway orchestration loops (LLM-call count / wall-clock ceiling).
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter


def search_spans_retry(client, project: str, trace_id: str, max_results: int = 500,
                       attempts: int = 7):
    """client.search_spans with backoff on HTTP 429 (Opik rate-limits span search)."""
    delay = 5
    for i in range(attempts):
        try:
            return client.search_spans(project_name=project, trace_id=trace_id,
                                       max_results=max_results)
        except Exception as exc:  # opik ApiError carries .status_code / .headers
            status = getattr(exc, "status_code", None)
            if status == 429 and i < attempts - 1:
                reset = None
                try:
                    reset = int((getattr(exc, "headers", {}) or {}).get("ratelimit-reset"))
                except Exception:
                    reset = None
                wait = min((reset or delay) + 1, 60)
                time.sleep(wait)
                delay = min(delay * 2, 60)
                continue
            raise
    return []

# A run is "runaway" if it makes this many LLM calls or hits this wall-clock.
RUNAWAY_LLM_CALLS = 25
RUNAWAY_SECONDS = 690.0  # observed 700s ceiling in F014

_NOTFOUND = re.compile(r"Tool '([^']+)' not found", re.I)
_EXC_TYPE = re.compile(r"exception_type='?([A-Za-z_][A-Za-z0-9_]*)'?")


def _is_empty(content) -> bool:
    if content is None:
        return True
    if isinstance(content, str):
        return content.strip() == ""
    if isinstance(content, list):
        return len(content) == 0
    return False


def _query(trace) -> str:
    inp = getattr(trace, "input", None)
    try:
        if isinstance(inp, dict):
            parts = inp.get("parts")
            if parts:
                return str(parts[0].get("text", ""))[:90]
        return str(inp)[:90]
    except Exception:
        return ""


def trace_metrics(client, trace, project: str) -> dict:
    """Compute reliability metrics for one Opik trace (fetches its spans)."""
    spans = search_spans_retry(client, project, trace.id, max_results=500)
    llm = [s for s in spans if getattr(s, "type", None) == "llm"]
    tool = [s for s in spans if getattr(s, "type", None) == "tool"]
    general = [s for s in spans if getattr(s, "type", None) == "general"]

    empty = 0
    finish: Counter = Counter()
    for s in llm:
        out = s.output if isinstance(s.output, dict) else {}
        if _is_empty(out.get("content")):
            empty += 1
        finish[out.get("finish_reason")] += 1

    notfound: Counter = Counter()
    for s in spans:
        ei = getattr(s, "error_info", None)
        if ei:
            for name in _NOTFOUND.findall(str(ei)):
                notfound[name] += 1

    calls: Counter = Counter()
    for s in tool:
        nm = getattr(s, "name", None)
        inp = s.input if isinstance(s.input, dict) else {}
        key = (nm, json.dumps(inp, ensure_ascii=False, sort_keys=True)[:200])
        calls[key] += 1
    max_repeat = max(calls.values()) if calls else 0

    md = trace.metadata or {}
    model = (md.get("main_model") or "?").split("/")[-1]
    dur = float(trace.duration) / 1000 if trace.duration else 0.0
    n_llm = len(llm)

    err = getattr(trace, "error_info", None)
    err_type = None
    if err:
        m = _EXC_TYPE.search(str(err))
        err_type = m.group(1) if m else "error"

    usage = trace.usage or {}
    return {
        "id": trace.id,
        "start": str(getattr(trace, "start_time", ""))[:19],
        "model": model,
        "duration_s": round(dur, 1),
        "span_count": getattr(trace, "span_count", len(spans)) or len(spans),
        "n_llm": n_llm,
        "n_tool": len(tool),
        "n_general": len(general),
        "empty_llm": empty,
        "finish": dict(finish),
        "tool_not_found": sum(notfound.values()),
        "notfound_names": dict(notfound),
        "max_repeat_toolcall": max_repeat,
        "runaway": bool(n_llm >= RUNAWAY_LLM_CALLS or dur >= RUNAWAY_SECONDS),
        "errored": bool(err),
        "error_type": err_type,
        "completion_tokens": usage.get("completion_tokens"),
        "query": _query(trace),
    }
