"""Fill empty ExperimentTask.design cells from facts already on the task.

Does not invent science, does not change route, and never writes a leftover
inventory tool into analysis_artifacts.path_or_tool.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

from CoScientist.experiments.schemas import (
    DesignAnalysisArtifact,
    DesignBaseline,
    DesignDataset,
    DesignMetric,
    ExperimentPlan,
    ExperimentTask,
    is_design_placeholder,
)

_ROUTE_PREPARE = {
    "fedot_mas": "mcp",
    "react_tools": "mcp",
    "alembic_build": "mcp",
    "coder": "coder",
    "research": "research",
    "medical": "medical",
}
_ARTIFACT_ROLE = {
    "data": "metrics_table",
    "plot": "report",
    "report": "report",
    "log": "report",
    "code": "code",
    "model": "config",
    "mcp_server": "config",
}


def _bound_tool_names(task: ExperimentTask) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for server in task.mcp_servers:
        for tool in server.tools:
            name = str(tool.name or "").strip()
            if name and name not in seen:
                seen.add(name)
                names.append(name)
    return names


def _operation_statement(task: ExperimentTask, ops_index: Mapping[str, Mapping[str, Any]]) -> str:
    ref = str(task.design.operation_ref or "").strip().upper()
    if ref and ref in ops_index:
        return str((ops_index[ref] or {}).get("statement") or "").strip()
    return ""


def _dataset_from_inputs(task: ExperimentTask) -> str:
    for ref in task.input_data:
        text = str(ref.description or ref.data_id or "").strip()
        if text and not is_design_placeholder(text):
            return text
    return ""


def _metric_from_criteria(task: ExperimentTask) -> DesignMetric | None:
    for crit in task.success_criteria:
        if crit.kind != "threshold" or not crit.metric:
            continue
        name = str(crit.metric).strip()
        if is_design_placeholder(name):
            continue
        op = crit.operator or "=="
        direction = "minimize" if op in {"<", "<="} else "maximize" if op in {">", ">="} else "compare"
        return DesignMetric(name=name, direction=direction, threshold=crit.target, test=None)
    return None


def _analysis_from_expected(task: ExperimentTask) -> list[DesignAnalysisArtifact]:
    prepare = _ROUTE_PREPARE.get(task.route.value, "coder")
    bound = _bound_tool_names(task)
    # Only the tool already bound on THIS task — never leftover inventory.
    path = bound[0] if prepare == "mcp" and bound else None
    out: list[DesignAnalysisArtifact] = []
    for item in task.expected_artifacts:
        name = str(item.name or "").strip()
        if not name or is_design_placeholder(name):
            continue
        role = _ARTIFACT_ROLE.get(item.role, "report")
        out.append(DesignAnalysisArtifact(
            name=name, role=role, prepare_via=prepare, path_or_tool=path,
        ))
    return out


def _fill_task(
    task: ExperimentTask,
    *,
    ops_index: Mapping[str, Mapping[str, Any]],
) -> ExperimentTask:
    design = task.design
    question = "" if is_design_placeholder(design.experiment_question) else design.experiment_question.strip()
    if not question:
        question = _operation_statement(task, ops_index) or str(task.description or "").strip()

    dataset_name = "" if is_design_placeholder(design.dataset.name) else design.dataset.name.strip()
    if not dataset_name:
        tools = _bound_tool_names(task)
        dataset_name = _dataset_from_inputs(task) or (
            f"output of {tools[0]}" if tools else ""
        )
    dataset = design.dataset.model_copy(update={"name": dataset_name})

    baselines = list(design.baselines)
    if not baselines:
        tools = _bound_tool_names(task)
        if tools:
            baselines = [DesignBaseline(name=tools[0], kind="method", ref=None)]

    metrics = list(design.metrics)
    if not metrics:
        if filled := _metric_from_criteria(task):
            metrics = [filled]

    artifacts = list(design.analysis_artifacts)
    if not artifacts:
        artifacts = _analysis_from_expected(task)

    return task.model_copy(update={
        "design": design.model_copy(update={
            "experiment_question": question,
            "dataset": dataset,
            "baselines": baselines,
            "metrics": metrics,
            "analysis_artifacts": artifacts,
        }),
    })


def fill_experiment_design(
    plan: ExperimentPlan,
    *,
    operations: Iterable[Any] = (),
) -> ExperimentPlan:
    """Copy known facts into empty design cells. Safe to run after MCP repair."""
    ops_index = {
        str(op.get("operation_id") or "").strip().upper(): op
        for op in operations or []
        if isinstance(op, dict) and str(op.get("operation_id") or "").strip()
    }
    return plan.model_copy(update={
        "tasks": [_fill_task(task, ops_index=ops_index) for task in plan.tasks],
    })


__all__ = ["fill_experiment_design"]
