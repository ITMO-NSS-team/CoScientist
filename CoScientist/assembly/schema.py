"""Pydantic schema + loader for ``system.yaml``.

The YAML declares every agent of the system in one place. Per agent:

  class:        llm | sequential | parallel | custom:<registered name>
  enabled:      bool, or "${settings.path}" resolved against app settings —
                a disabled agent is still BUILT (so it can be served standalone
                over A2A) but is not attached to / advertised by its parents
  model:        "main" | "coder" | a literal litellm model string
  prompt:       name of a registered prompt template
  tools:        registered tool names
  subordinates: agents attached as AgentTool (and rendered into <<AGENTS>>/<<ROUTING>>)
  children:     composite children (sequential/parallel execution order)
  callbacks:    {before_model|after_model|before_tool|after_tool|before_agent|after_agent: [names]}
  hitl:         whether the agent uses human-in-the-loop (tools + prompt section
                for llm agents, review-loop handler for session agents)
  output_key / output_schema / planner / options: passthrough constructor config
  a2a:          how the agent is exposed as an A2A service (key, port, skill, env)
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from CoScientist.config import get_settings

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "agents" / "system.yaml"


def _resolve_setting_ref(value: Union[bool, str]) -> bool:
    """Resolve an ``enabled`` value: a bool, or "${dotted.settings.path}"."""
    if isinstance(value, bool):
        return value
    text = value.strip()
    if not (text.startswith("${") and text.endswith("}")):
        raise ValueError(
            f"enabled must be a bool or '${{settings.path}}', got {value!r}"
        )
    obj: Any = get_settings()
    for part in text[2:-1].split("."):
        obj = getattr(obj, part)
    return bool(obj)


class SkillConfig(BaseModel):
    """One A2A AgentSkill advertised on the agent card."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    description: str
    tags: List[str] = Field(default_factory=list)


class A2AConfig(BaseModel):
    """How the agent is exposed as a standalone A2A service."""

    model_config = ConfigDict(extra="forbid")

    key: str  # snake key: env prefix ("<KEY>_PORT") and serve-module argument
    port: int  # default port; overridable via the <KEY>_PORT env var
    skill: SkillConfig
    # Env defaults applied (setdefault) before the serving process builds the
    # system — for settings that must exist before tool modules import.
    env: Dict[str, str] = Field(default_factory=dict)


class CallbacksConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    before_model: List[str] = Field(default_factory=list)
    after_model: List[str] = Field(default_factory=list)
    before_tool: List[str] = Field(default_factory=list)
    after_tool: List[str] = Field(default_factory=list)
    before_agent: List[str] = Field(default_factory=list)
    after_agent: List[str] = Field(default_factory=list)

    def items(self):
        return self.model_dump().items()


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str = ""  # filled from the mapping key
    cls: str = Field("llm", alias="class")
    enabled: Union[bool, str] = True
    root: bool = False
    model: Optional[str] = None
    description: str = ""
    # How a PARENT's prompt routes work to this agent (one routing bullet).
    routing: str = ""
    # How the planner's roster describes this agent (defaults to description).
    planning: str = ""
    prompt: Optional[str] = None
    tools: List[str] = Field(default_factory=list)
    subordinates: List[str] = Field(default_factory=list)
    children: List[str] = Field(default_factory=list)
    callbacks: CallbacksConfig = Field(default_factory=CallbacksConfig)
    hitl: bool = False
    include_contents: Optional[str] = "default"
    mode: Optional[str] = None
    output_key: Optional[str] = None
    output_schema: Optional[str] = None
    planner: Optional[str] = None
    # Extra constructor kwargs for custom agent classes (e.g. plan_file_path).
    options: Dict[str, Any] = Field(default_factory=dict)
    a2a: Optional[A2AConfig] = None

    @field_validator("cls")
    @classmethod
    def _known_class(cls, v: str) -> str:
        if v in ("llm", "sequential", "parallel") or v.startswith("custom:"):
            return v
        raise ValueError(
            f"class must be llm | sequential | parallel | custom:<name>, got {v!r}"
        )

    @model_validator(mode="after")
    def _shape(self) -> "AgentConfig":
        composite = self.cls in ("sequential", "parallel")
        if composite:
            if not self.children:
                raise ValueError(f"{self.cls} agent needs non-empty children")
            for forbidden in ("tools", "subordinates", "prompt", "model"):
                if getattr(self, forbidden):
                    raise ValueError(
                        f"{self.cls} agent cannot have {forbidden} (got {getattr(self, forbidden)!r})"
                    )
        elif self.children:
            raise ValueError(f"{self.cls} agent cannot have children")
        return self

    def is_enabled(self) -> bool:
        return _resolve_setting_ref(self.enabled)


class DefaultsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "main"


class SystemConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    agents: Dict[str, AgentConfig]

    @model_validator(mode="after")
    def _validate_graph(self) -> "SystemConfig":
        for key, agent in self.agents.items():
            if agent.name and agent.name != key:
                raise ValueError(f"Agent key {key!r} != name {agent.name!r}")
            agent.name = key

        roots = [a.name for a in self.agents.values() if a.root]
        if len(roots) != 1:
            raise ValueError(f"Exactly one agent must have root: true, got {roots}")

        for agent in self.agents.values():
            for ref in agent.subordinates + agent.children:
                if ref not in self.agents:
                    raise ValueError(f"{agent.name}: unknown agent reference {ref!r}")
            dupes = {r for r in agent.subordinates if agent.subordinates.count(r) > 1}
            if dupes:
                raise ValueError(f"{agent.name}: duplicate subordinates {sorted(dupes)}")

        # No cycles through children/subordinates (also guarantees a build order).
        self.build_order()

        keys = [a.a2a.key for a in self.agents.values() if a.a2a]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate a2a keys: {sorted(keys)}")
        ports = [a.a2a.port for a in self.agents.values() if a.a2a]
        if len(ports) != len(set(ports)):
            raise ValueError(f"Duplicate a2a ports: {sorted(ports)}")
        return self

    # ── queries ──────────────────────────────────────────────────────────────
    @property
    def root(self) -> AgentConfig:
        return next(a for a in self.agents.values() if a.root)

    def agent(self, name: str) -> AgentConfig:
        if name not in self.agents:
            raise KeyError(f"Unknown agent {name!r}")
        return self.agents[name]

    def deps(self, name: str) -> List[str]:
        agent = self.agent(name)
        return agent.children + agent.subordinates

    def build_order(self) -> List[str]:
        """Dependency-first topological order over children + subordinates."""
        order: List[str] = []
        state: Dict[str, int] = {}  # 0 visiting, 1 done

        def visit(name: str, chain: tuple) -> None:
            if state.get(name) == 1:
                return
            if state.get(name) == 0:
                cycle = " -> ".join(chain + (name,))
                raise ValueError(f"Agent dependency cycle: {cycle}")
            state[name] = 0
            for dep in self.deps(name):
                visit(dep, chain + (name,))
            state[name] = 1
            order.append(name)

        for name in self.agents:
            visit(name, ())
        return order

    def enabled_subordinates(self, name: str) -> List[AgentConfig]:
        return [
            self.agent(s) for s in self.agent(name).subordinates
            if self.agent(s).is_enabled()
        ]

    def parents_of(self, name: str) -> List[AgentConfig]:
        return [a for a in self.agents.values() if name in a.subordinates]

    def delegatable_names(self) -> set:
        """Names of every agent that some agent delegates to via AgentTool."""
        return {s for a in self.agents.values() for s in a.subordinates}

    def a2a_agents(self) -> List[AgentConfig]:
        return [a for a in self.agents.values() if a.a2a]

    def a2a_agent_by_key(self, key: str) -> AgentConfig:
        for a in self.agents.values():
            if a.a2a and a.a2a.key == key:
                return a
        known = ", ".join(sorted(x.a2a.key for x in self.a2a_agents()))
        raise KeyError(f"No agent with a2a key {key!r}. Known: {known}")


def load_config(path: Optional[Path] = None) -> SystemConfig:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return SystemConfig.model_validate(raw)


@lru_cache(maxsize=1)
def get_config() -> SystemConfig:
    """The default system config, loaded once per process."""
    return load_config()


__all__ = [
    "A2AConfig",
    "AgentConfig",
    "CallbacksConfig",
    "DEFAULT_CONFIG_PATH",
    "DefaultsConfig",
    "SkillConfig",
    "SystemConfig",
    "get_config",
    "load_config",
]
