import os

from CoScientist.config import get_settings

settings = get_settings()

# Opik tracing is gated behind a single env flag (OPIK__ENABLED). When it is
# off we make opik's own ``@track`` decorators no-ops process-wide and skip all
# tracer setup, so nothing is ever shipped to the Opik backend (avoids the free
# account span-limit 402s) and the app stays fully functional without it.
if not settings.opik.enabled:
    os.environ.setdefault("OPIK_TRACK_DISABLE", "true")
    multi_agent_tracer = None
else:
    # Only set the env var when a key is actually present — assigning None raises
    # TypeError and would crash importing this module (and most of the app).
    if settings.opik.api_key:
        os.environ["OPIK_API_KEY"] = settings.opik.api_key

    import opik

    # Don't let an opik misconfiguration (no key, no network) take down the app
    # on import — tracing is best-effort.
    try:
        opik.configure(use_local=False)
    except Exception as e:  # pragma: no cover - best-effort tracing setup
        print(f"[opik] configure failed, tracing may be disabled: {e!r}")

    from opik.integrations.adk import OpikTracer

    # Avoid dumping the full settings (which include API keys/passwords) into
    # trace metadata — only expose non-secret descriptors.
    _safe_metadata = {
        "main_model": settings.llm.main_model,
        "coder_model": settings.llm.coder_model or settings.llm.main_model,
    }

    multi_agent_tracer = OpikTracer(
        name="multi-agent-orchestrator",
        metadata=_safe_metadata,
        project_name=settings.opik.opik_project_name or "adk-coscientist",
    )
