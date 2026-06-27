"""Node/edge schema for the Dynamic Execution Graph (see docs/execution_graph.md).

Raw agent-call granularity for the MVP; the optional `semantic` slot is filled
later by the step-abstraction layer.
"""
from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field

NodeKind = Literal["goal", "agent_call", "tool_call", "decision", "reflection"]
NodeStatus = Literal["running", "success", "failed", "pruned"]
EdgeType = Literal[
    "caused_by",
    "delegated_to",
    "produced",
    "failed_into",
    "depends_on",
    "validated_by",
    "branches_to",
]


class Semantic(BaseModel):
    type: Optional[str] = None
    goal: Optional[str] = None
    entity: Optional[str] = None


class Node(BaseModel):
    id: str
    run_id: str
    kind: NodeKind
    label: str = ""
    executor_agent: Optional[str] = None
    status: NodeStatus = "running"
    parent_ids: List[str] = Field(default_factory=list)
    input: Optional[Any] = None
    output: Optional[str] = None
    verdict: Optional[str] = None  # critic verdict — the reward signal (Fact 1)
    t_start: Optional[float] = None
    t_end: Optional[float] = None
    semantic: Optional[Semantic] = None


class Edge(BaseModel):
    run_id: str
    src: str
    dst: str
    type: EdgeType = "caused_by"


class StatusUpdate(BaseModel):
    run_id: str
    status: Optional[NodeStatus] = None
    output: Optional[str] = None
    verdict: Optional[str] = None
    t_end: Optional[float] = None
