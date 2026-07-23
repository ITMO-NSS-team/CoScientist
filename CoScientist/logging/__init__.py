"""Logging helpers, loaded lazily to keep plugin imports side-effect free."""
from typing import Any

__all__ = ["get_logger", "multi_agent_tracer"]


def __getattr__(name: str) -> Any:
    if name == "get_logger":
        from CoScientist.logging.logger import get_logger

        globals()[name] = get_logger
        return get_logger
    if name == "multi_agent_tracer":
        from CoScientist.logging.opik_tracer import multi_agent_tracer

        globals()[name] = multi_agent_tracer
        return multi_agent_tracer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
