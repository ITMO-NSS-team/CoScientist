"""Lazy A2A package exports."""
from importlib import import_module
from typing import Any

_EXPORTS = {
    "AGENT_PORTS": ("CoScientist.a2a.config", "AGENT_PORTS"),
    "AGENT_URLS": ("CoScientist.a2a.config", "AGENT_URLS"),
    "AGENT_CARD_URLS": ("CoScientist.a2a.config", "AGENT_CARD_URLS"),
    "make_a2a_app": ("CoScientist.a2a.server", "make_a2a_app"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
