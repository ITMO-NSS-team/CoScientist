"""YAML-driven assembly of the CoScientist multi-agent system.

The single source of truth for the system layout is ``CoScientist/agents/system.yaml``:
which agents exist, who is subordinate to whom, which tools / callbacks / prompts
each agent uses, whether it uses HITL, and how it is exposed over A2A.

Modules:
  * registry   — name -> object registries (tools, callbacks, prompts, classes, ...)
  * schema     — pydantic models for the YAML + loading/validation
  * prompting  — PromptContext: renders the unified prompt sections (<<TOOLS>>, ...)
  * bindings   — registers every concrete tool/callback/class/schema in the registries
  * assembler  — builds the agent tree from the validated config

Typical use:
    from CoScientist.assembly import build_system
    system = build_system()                       # in-process sub-agents
    system = build_system(remote_subagents=True)  # sub-agents over A2A
"""
from importlib import import_module
from typing import Any

_EXPORTS = {
    "AgentSystem": ("CoScientist.assembly.assembler", "AgentSystem"),
    "build_system": ("CoScientist.assembly.assembler", "build_system"),
    "delegatable_agent_names": (
        "CoScientist.assembly.assembler",
        "delegatable_agent_names",
    ),
    "load_config": ("CoScientist.assembly.schema", "load_config"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
