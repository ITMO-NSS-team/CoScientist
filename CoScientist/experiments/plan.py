"""Structured experiment-plan schema for the experiments module (АМ).

DEVGRAPH: feature F015a, roadmap step R05. The orchestrator decomposes a research
task into an ``ExperimentPlan`` — a DAG of steps, each naming the exact tools it
needs and the artifacts it produces — instead of handing FEDOT.MAS one big query.

Borrows:
  - Routine (S013): ``required_tools`` are exact tool/server names.
  - ReWOO / HuggingGPT (S014): steps form a DAG via ``deps`` + artifact ids that
    later steps reference as ``{artifact_id}`` placeholders.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


class Artifact(BaseModel):
    """An output a step produces; later steps reference it by ``{id}``."""

    id: str = Field(..., description="stable id, referenced downstream as {id}, e.g. 'mols_v1'")
    description: str = Field(..., description="what this artifact holds")
    kind: str = Field("data", description="table|molecules|score|figure|text|model|dataset|other")


class StepProvenance(BaseModel):
    """Where a step's intent / required capability comes from (carried into F015d)."""

    hypothesis: Optional[str] = None
    source: Optional[str] = Field(
        None, description="citing paper / repo URL, so a tool built for this step is traceable"
    )


class ServerTools(BaseModel):
    """The MCP server that provides a step's tools, and which tools are needed.

    This is the unit handed to FEDOT.MAS (server-granular dispatch, see F015g.D1):
    ``servers_payload = {server: HttpMCPServer(url, ...)}``. The planner fills
    ``server`` + ``tools`` from the live tool inventory (F015c / retrieve_tools);
    ``url`` is resolved at dispatch/registration time (left None at plan time).
    """

    server: str = Field(..., description="exact MCP server name from the inventory")
    tools: list[str] = Field(..., description="exact tool names needed from THIS server")
    url: Optional[str] = Field(None, description="resolved at dispatch (registry/presigned); None at plan time")


class ExperimentStep(BaseModel):
    id: str = Field(..., description="unique step id, e.g. 's1'")
    subtask: str = Field(..., description="single, concrete computational sub-task (imperative)")
    tool_servers: list[ServerTools] = Field(
        default_factory=list,
        description="tools this step needs, GROUPED BY the MCP server that provides them",
    )
    run_params: dict[str, Any] = Field(
        default_factory=dict,
        description="concrete run parameters; may reference upstream artifacts as {artifact_id}",
    )
    expected_artifacts: list[Artifact] = Field(default_factory=list)
    deps: list[str] = Field(default_factory=list, description="ids of prerequisite steps")
    provenance: Optional[StepProvenance] = None


class PlanError(ValueError):
    """Raised when a plan is structurally invalid (bad ids / deps / cycle)."""


class ExperimentPlan(BaseModel):
    goal: str = Field(..., description="one-line restatement of what the plan achieves")
    steps: list[ExperimentStep]

    @model_validator(mode="after")
    def _validate_structure(self) -> "ExperimentPlan":
        ids = [s.id for s in self.steps]
        if not ids:
            raise PlanError("plan has no steps")
        if len(ids) != len(set(ids)):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise PlanError(f"duplicate step ids: {dupes}")
        idset = set(ids)
        for s in self.steps:
            for d in s.deps:
                if d not in idset:
                    raise PlanError(f"step {s.id!r} depends on unknown step {d!r}")
                if d == s.id:
                    raise PlanError(f"step {s.id!r} depends on itself")
        self._topo_ids()  # raises PlanError on a cycle
        return self

    # ── graph helpers ────────────────────────────────────────────────────────
    def _topo_ids(self) -> list[str]:
        """Kahn topological sort; raises PlanError if the DAG has a cycle."""
        order_index = {s.id: i for i, s in enumerate(self.steps)}
        remaining = {s.id: set(s.deps) for s in self.steps}
        order: list[str] = []
        while remaining:
            ready = sorted((sid for sid, d in remaining.items() if not d),
                           key=order_index.get)
            if not ready:
                raise PlanError(f"dependency cycle among steps: {sorted(remaining)}")
            nid = ready[0]
            order.append(nid)
            del remaining[nid]
            for d in remaining.values():
                d.discard(nid)
        return order

    def topological_order(self) -> list[ExperimentStep]:
        by_id = {s.id: s for s in self.steps}
        return [by_id[i] for i in self._topo_ids()]

    def artifact_ids(self) -> set[str]:
        return {a.id for s in self.steps for a in s.expected_artifacts}

    def required_servers(self) -> list[str]:
        """Distinct MCP server names the plan needs (order-preserving)."""
        seen: dict[str, None] = {}
        for s in self.steps:
            for ts in s.tool_servers:
                seen.setdefault(ts.server, None)
        return list(seen)

    def required_tool_names(self) -> list[str]:
        """Distinct tool names across all servers (order-preserving)."""
        seen: dict[str, None] = {}
        for s in self.steps:
            for ts in s.tool_servers:
                for t in ts.tools:
                    seen.setdefault(t, None)
        return list(seen)
