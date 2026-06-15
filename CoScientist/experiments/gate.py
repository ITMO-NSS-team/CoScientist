"""Deterministic plan gate (R12 / F015b) — model-free structural + capability check.

Runs BEFORE any LLM critic on a submit_plan(ExperimentPlan). The plan is already
structurally valid by construction (ExperimentPlan enforces unique ids / resolvable
deps / acyclicity — plan.py:_validate_structure). This gate adds the two checks the
constructor does NOT do, which are exactly the wave-1 'gap'/'wrong' hazards the
LLM-only critic was needed to catch:

  * tool-resolvability — every (server, tool) the plan names must exist in the
    live/frozen MCP inventory; an out-of-inventory capability is a gap to BUILD
    (F015c/F015d), surfaced as a reject, not a silent pass.
  * empty-compute-step — a step with kind == 'compute' MUST name >=1 tool_server.
    Without this, an honestly-empty step for an out-of-inventory capability
    (e.g. "run a full clinical-trial simulation") sails through the constructor
    (tool_servers defaults to []), so the deterministic gate would have NO teeth
    on the gap hazard — the exact hole the wave-1 adversarial review found.

Returns a GateResult with a machine-readable code; never raises. The caller (the
submit_plan handler) decides reject->re-emit vs surface-to-HITL.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence, Union

from CoScientist.experiments.plan import ExperimentPlan

# inventory shape: {server: [tool_name, ...]} OR {server: [(tool_name, desc), ...]}
Inventory = Mapping[str, Sequence[Union[str, tuple, list]]]

# Step kinds that REQUIRE an MCP tool to run. Other kinds (research / hypothesize /
# code_exec) legitimately carry no tool_server and are not flagged as empty.
_TOOL_REQUIRING_KINDS = {"compute"}


@dataclass
class GateResult:
    ok: bool
    code: str            # pass | reject:empty_compute_step | reject:unknown_server | reject:unresolvable_tool
    detail: str = ""
    offending: list = field(default_factory=list)


def _normalize_inventory(inventory: Inventory) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for server, items in (inventory or {}).items():
        names: set[str] = set()
        for it in items or []:
            names.add(it[0] if isinstance(it, (tuple, list)) else it)
        out[server] = names
    return out


def deterministic_gate(plan: ExperimentPlan, inventory: Inventory) -> GateResult:
    """Model-free gate over a (constructed, structurally-valid) ExperimentPlan."""
    inv = _normalize_inventory(inventory)
    empty: list[str] = []
    unknown_servers: list[str] = []
    unresolvable: list[str] = []

    for s in plan.steps:
        if s.kind in _TOOL_REQUIRING_KINDS and not s.tool_servers:
            empty.append(s.id)
            continue
        if not inv:
            continue  # no inventory snapshot -> can't check resolvability; only empty-compute is enforced
        for ts in s.tool_servers:
            if ts.server not in inv:
                unknown_servers.append(f"{s.id}:{ts.server}")
                continue
            for t in ts.tools:
                if t not in inv[ts.server]:
                    unresolvable.append(f"{s.id}:{ts.server}.{t}")

    # Order matters: an empty compute step is the most fundamental gap.
    if empty:
        return GateResult(False, "reject:empty_compute_step",
                          f"compute step(s) name no tool_server (capability gap to build): {empty}", empty)
    if unknown_servers:
        return GateResult(False, "reject:unknown_server",
                          f"step(s) name an MCP server not in the inventory: {unknown_servers}", unknown_servers)
    if unresolvable:
        return GateResult(False, "reject:unresolvable_tool",
                          f"step(s) name a tool not in the inventory: {unresolvable}", unresolvable)
    return GateResult(True, "pass", "structurally valid; all (server, tool) resolve against the inventory")
