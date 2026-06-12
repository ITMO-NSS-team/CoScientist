"""Experiments module (АМ) — plan → critic → bounded execution (DEVGRAPH F015)."""
from CoScientist.experiments.plan import (
    Artifact,
    ExperimentPlan,
    ExperimentStep,
    PlanError,
    StepProvenance,
)
from CoScientist.experiments.planner import (
    PlanGenerationError,
    generate_plan,
)

__all__ = [
    "Artifact",
    "ExperimentPlan",
    "ExperimentStep",
    "StepProvenance",
    "PlanError",
    "generate_plan",
    "PlanGenerationError",
]
