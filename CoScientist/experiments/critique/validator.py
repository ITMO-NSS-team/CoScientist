"""Deterministic plan validation and critique for Experiment Module v1b."""
from __future__ import annotations

import json
import re
from typing import Any, Iterable
from uuid import uuid4

from pydantic import ValidationError

from CoScientist.config.settings import ExperimentsSettings
from CoScientist.experiments.capabilities.inventory import (
    inventory_pairs,
    is_paper_demo_tool,
    request_capabilities,
    tool_capabilities,
)
from CoScientist.experiments.schemas import (
    CritiqueIssue,
    ExecutionRoute,
    ExperimentPlan,
    PlanCritique,
    utc_now,
)

_MCP = {ExecutionRoute.FEDOT_MAS, ExecutionRoute.REACT_TOOLS}
_ALT = re.compile(r"\b(otherwise|else|либо|иначе|alternativ)\b|/", re.I)


class PlanValidationError(ValueError):
    def __init__(self, message: str, *, errors: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.errors = errors or []


def _issue(n: int, *, category: str, severity: str, message: str, suggestion: str, task_id: str | None = None) -> CritiqueIssue:
    return CritiqueIssue(
        issue_id=f"DET-{n:03d}", category=category, severity=severity,
        task_id=task_id, message=message, suggestion=suggestion,
    )


def _tool_output_blob(task: Any) -> str:
    return " ".join(
        str(getattr(tool, a, "") or "")
        for s in getattr(task, "mcp_servers", None) or []
        for tool in getattr(s, "tools", None) or []
        for a in ("name", "description", "input_schema")
    ).lower()


def _is_image_artifact(a: Any) -> bool:
    return getattr(a, "role", "") == "plot" or str(getattr(a, "media_type", "") or "").startswith("image/")


def _normalize_hypothesis_ids(refs: Iterable[Any]) -> list[str]:
    out, seen = [], set()
    for item in refs or []:
        if isinstance(item, dict):
            hid = str(item.get("hypothesis_id") or item.get("id") or "").strip().upper()
        else:
            hid = str(item or "").strip().upper()
        if hid and hid not in seen:
            seen.add(hid)
            out.append(hid)
    return out


def _iter_inventory(tools: Iterable[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    out, seen = [], set()
    for item in tools:
        if not isinstance(item, dict):
            continue
        name = str(item.get("tool") or item.get("name") or "").strip()
        if name and name not in seen:
            seen.add(name)
            out.append((name, item))
    return out


def _planned_tools(plan: ExperimentPlan, *, mcp_only: bool = False) -> set[str]:
    return {
        t.name for task in plan.tasks if not mcp_only or task.route in _MCP
        for s in task.mcp_servers for t in s.tools
    }


def _explicit_tool_requirement(request: str, tool: str) -> bool:
    e = re.escape(tool.lower())
    return any(re.search(p, request, re.I | re.DOTALL) for p in (
        rf"(?:use|call|run|invoke|via|through|with|использу[йя]|вызови|через)\s+[`'\"]?{e}",
        rf"[`'\"]?{e}[`'\"]?\s+(?:tool|mcp|server|инструмент)",
        rf"(?:tool|mcp)\s+[`'\"]?{e}",
        rf"(?:must|should|required|обязательн\w*)\s+.{0,40}{e}",
    ))


def _unused_matched_inventory_tools(plan: ExperimentPlan, *, available_tools: Iterable[dict[str, Any]]) -> list[str]:
    """Option A: unused capability-matched inventory → minor. Skips paper-demo."""
    needed = request_capabilities(plan.source_request)
    if not needed:
        return []
    planned = _planned_tools(plan, mcp_only=True)
    unused = []
    for tool, item in _iter_inventory(available_tools):
        desc = str(item.get("description") or "")
        if is_paper_demo_tool(tool, desc):
            continue
        caps = tool_capabilities(tool, desc)
        if not (caps & needed) or tool in planned:
            continue
        if any(tool_capabilities(o) & caps & needed for o in planned):
            continue
        unused.append(tool)
    return sorted(unused)


def _named_inventory_tools_missing(plan: ExperimentPlan, *, available_tools: Iterable[dict[str, Any]]) -> list[str]:
    """Explicitly required inventory tools omitted from the plan."""
    request = plan.source_request.lower()
    planned = _planned_tools(plan)
    by_name, named = {}, []
    for tool, item in _iter_inventory(available_tools):
        by_name[tool] = item
        if re.search(rf"(?<![\w-]){re.escape(tool.lower())}(?![\w-])", request) and _explicit_tool_requirement(request, tool):
            named.append(tool)
    if not named:
        return []
    missing = [t for t in named if t not in planned]
    if not missing:
        return []
    if any(t in planned for t in named) and _ALT.search(request):
        return []
    still = []
    for tool in missing:
        item = by_name.get(tool) or {}
        schema = item.get("input_schema")
        unusable = not isinstance(schema, dict) or not schema
        caps = tool_capabilities(tool, str(item.get("description") or ""))
        if unusable and any(tool_capabilities(o) & caps for o in planned):
            continue
        still.append(tool)
    return sorted(still)


_PLACEHOLDER_HOST_RE = re.compile(
    r"(?i)\b(?:https?://)?(?:www\.)?(?:example\.(?:com|org|net)|localhost|127\.0\.0\.1)\b"
)
_FAKE_S3_RE = re.compile(r"(?i)\bs3://artifacts(?:/|\b)")
_PLACEHOLDER_TOKEN_RE = re.compile(
    r"(?i)\b(?:placeholder[_-]?url|dummy[_-]?url|fake[_-]?dataset|"
    r"public_ld50_data\.csv|your[_-]bucket|insert[_-]url[_-]here)\b"
)


def _iter_plan_url_strings(plan: ExperimentPlan) -> list[tuple[str, str]]:
    """Collect (task_id_or_plan, text) candidates that may contain fake URLs."""
    out: list[tuple[str, str]] = []
    for blob in (plan.goal, plan.hypothesis or "", *(plan.assumptions or []), *(plan.risks or [])):
        if str(blob).strip():
            out.append(("plan", str(blob)))
    for task in plan.tasks:
        tid = task.id
        for blob in (task.description, task.repo_url or "", *(task.warnings or [])):
            if str(blob).strip():
                out.append((tid, str(blob)))
        d = task.design
        for blob in (d.experiment_question, d.dataset.name, d.dataset.notes or ""):
            if str(blob).strip():
                out.append((tid, str(blob)))
        if d.dataset.ref is not None:
            ref = d.dataset.ref
            for blob in (
                getattr(ref, "url", None),
                getattr(ref, "workspace_path", None),
                getattr(ref, "prepare_instruction", None),
                getattr(ref, "description", None),
            ):
                if blob and str(blob).strip():
                    out.append((tid, str(blob)))
        for ref in task.input_data:
            for blob in (
                ref.url, ref.workspace_path, ref.prepare_instruction, ref.description,
                f"s3://{ref.bucket}/{ref.s3_key}" if ref.bucket and ref.s3_key else "",
            ):
                if blob and str(blob).strip():
                    out.append((tid, str(blob)))
        lp = task.launch_params
        if lp is None:
            continue
        if isinstance(lp, str):
            out.append((tid, lp))
        else:
            try:
                out.append((tid, json.dumps(lp, ensure_ascii=False)))
            except Exception:
                out.append((tid, str(lp)))
    return out


def _placeholder_url_hits(text: str) -> list[str]:
    hits: list[str] = []
    if _PLACEHOLDER_HOST_RE.search(text):
        hits.append("example/localhost host")
    if _FAKE_S3_RE.search(text):
        hits.append("s3://artifacts placeholder")
    if _PLACEHOLDER_TOKEN_RE.search(text):
        hits.append("placeholder dataset/url token")
    return hits


def critique_plan(
    plan: ExperimentPlan,
    *,
    settings: ExperimentsSettings,
    available_tools: Iterable[dict[str, Any]] = (),
    preferred_tools: Iterable[dict[str, Any]] | None = None,
    previous_plan: ExperimentPlan | None = None,
    hypothesis_refs: Iterable[Any] = (),
    repo_candidates: Iterable[Any] = (),
) -> PlanCritique:
    """Routes, registry refs, scientific design, revision invariants."""
    issues: list[CritiqueIssue] = []

    def add(**kw: Any) -> None:
        issues.append(_issue(len(issues) + 1, **kw))

    def fe(tid: str, sev: str, msg: str, sug: str) -> None:
        add(category="feasibility", severity=sev, task_id=tid, message=msg, suggestion=sug)

    def co(sev: str, msg: str, sug: str, tid: str | None = None) -> None:
        add(category="completeness", severity=sev, task_id=tid, message=msg, suggestion=sug)

    for scope, text in _iter_plan_url_strings(plan):
        if hits := _placeholder_url_hits(text):
            tid = None if scope == "plan" else scope
            fe(
                tid or "plan",
                "blocker",
                f"{scope}: fabricated/placeholder URL or data locator ({', '.join(hits)}).",
                "Use a real inventory/tool-backed dataset, an upstream task_artifact, "
                "or omit the URL — never example.com / s3://artifacts / dummy hosts.",
            )

    cand = {
        str(i.get("url") or "").strip().rstrip("/").lower()
        for i in (repo_candidates or [])
        if isinstance(i, dict) and str(i.get("url") or "").strip()
    }
    n = len(plan.tasks)
    if n > settings.max_plan_tasks:
        add(category="complexity", severity="blocker",
            message=f"Plan has {n} tasks; max is {settings.max_plan_tasks}.",
            suggestion="Merge tasks without losing dependencies or deliverables.")
    if n >= settings.complexity_warning_tasks:
        add(category="complexity", severity="minor",
            message=f"Plan has {n} tasks and may be expensive for v0.",
            suggestion="Confirm each task is an unavoidable execution unit.")

    enabled = {ExecutionRoute.REACT_TOOLS, ExecutionRoute.CODER}
    if settings.route_fedot:
        enabled.add(ExecutionRoute.FEDOT_MAS)
    if settings.route_alembic:
        enabled.add(ExecutionRoute.ALEMBIC_BUILD)

    inv_list = list(available_tools)
    inventory = inventory_pairs(inv_list)
    completeness = list(preferred_tools) if preferred_tools is not None else list(available_tools)

    if settings.require_task_design:
        for task in plan.tasks:
            d, tid = task.design, task.id
            for field, sug in (
                ("baselines", "List ≥1 baseline method/model/prior_result/external comparator."),
                ("metrics", "List ≥1 evaluation metric with direction."),
                ("analysis_artifacts", "Stub ≥1 code/config/metrics_table/report analysis artifact."),
            ):
                if not getattr(d, field):
                    co("blocker", f"{tid} design.{field} is empty.", sug, tid)
            if not d.dataset.name.strip():
                co("major", f"{tid} design.dataset.name is missing.",
                   "Name the dataset/benchmark used to test this hypothesis.", tid)
            for field, sug in (
                ("baselines", "Replace placeholder with a concrete baseline."),
                ("metrics", "Replace placeholder with a concrete metric and direction."),
            ):
                if any(x.name.strip().lower().startswith("unspecified") for x in getattr(d, field)):
                    co("major", f"{tid} design.{field} still has an unspecified placeholder.", sug, tid)

        ctx = _normalize_hypothesis_ids(hypothesis_refs)
        plan_h = _normalize_hypothesis_ids([{"hypothesis_id": h.hypothesis_id} for h in plan.hypotheses])
        req, any_c = set(), set()
        for task in plan.tasks:
            ids = task.design.covered_hypothesis_ids()
            any_c |= ids
            if not task.optional:
                req |= ids
        if ctx:
            if miss := [h for h in ctx if h not in req]:
                co("major", f"Context hypothesis_refs uncovered by non-optional task design: {', '.join(miss)}.",
                   "Map each context hypothesis id via design.hypothesis_ref on a non-optional task.")
            if plan_h and (mp := [h for h in ctx if h not in plan_h]):
                add(category="consistency", severity="major",
                    message=f"plan.hypotheses omits context hypothesis ids: {', '.join(mp)}.",
                    suggestion="Copy every context hypothesis_ref into plan.hypotheses.")
            if plan_h and (extra := [h for h in plan_h if h not in ctx]):
                co("major", f"plan.hypotheses invents ids absent from hypothesis_refs: {', '.join(extra)}.",
                   "Copy only hypothesis_refs; do not invent additional hypothesis ids.")
        elif not plan.hypotheses:
            co("major", "No hypothesis_refs in context and plan.hypotheses is empty.",
               "HypothesesAgent should populate hypothesis_refs; copy them "
               "into plan.hypotheses (or one H1 from source_request) and link tasks.")
        if plan_h and (orphan := [h for h in plan_h if h not in any_c]):
            co("major", f"plan.hypotheses ids not linked from tasks: {', '.join(orphan)}.",
               "Each plan hypothesis must appear as design.hypothesis_ref (or also_tests) on ≥1 task.")

    for task in plan.tasks:
        tid = task.id
        if task.route == ExecutionRoute.ALEMBIC_BUILD:
            if not settings.route_alembic:
                fe(tid, "blocker", "Route 'alembic_build' is disabled by profile settings.",
                   "Use a ready MCP route or coder, or enable EXPERIMENTS__ROUTE_ALEMBIC.")
            elif not task.repo_url:
                fe(tid, "blocker", f"{tid} uses alembic_build but repo_url is missing.",
                   "Set repo_url to an exact URL from experiment_context.repo_candidates.")
            elif not cand:
                fe(tid, "blocker",
                   f"{tid} uses alembic_build but experiment_context.repo_candidates is empty "
                   "(no extracted git URL to build).",
                   "Only schedule alembic_build when a repo candidate from the ask fits.")
            elif (task.repo_url or "").strip().rstrip("/").lower() not in cand:
                fe(tid, "blocker",
                   f"{tid} alembic repo_url {task.repo_url!r} is not in experiment_context.repo_candidates.",
                   "Copy repo_url from a listed candidate, or drop alembic_build.")
            elif not task.post_build_route:
                fe(tid, "blocker", f"{tid} uses alembic_build but post_build_route is missing.",
                   "Set post_build_route to fedot_mas or react_tools after MCP build.")
            if task.mcp_servers:
                fe(tid, "blocker", f"{tid} alembic_build must keep mcp_servers empty at plan time.",
                   "Set mcp_servers to [] — runtime injects the built MCP URL after Alembic.")
        elif task.route not in enabled:
            fe(tid, "blocker", f"Route {task.route.value!r} is disabled by profile settings.",
               "Choose an enabled route.")

        if task.route in _MCP and not inventory:
            fe(tid, "blocker",
               f"{tid} uses {task.route.value} but the MCP capability inventory is empty.",
               "Use route=coder when no exact ready MCP covers the task.")

        if task.route == ExecutionRoute.CODER and task.mcp_servers and not settings.route_coder_mcp:
            fe(tid, "major", "Direct MCP-to-Coder mode is disabled.",
               "Remove MCP refs from the coder task or enable EXPERIMENTS__ROUTE_CODER_MCP.")

        for server in task.mcp_servers:
            if server.source == "registry":
                for tool in server.tools:
                    if (str(server.server_id), tool.name) not in inventory:
                        fe(tid, "blocker",
                           f"Registry tool {tool.name!r} on server {server.server_id!r} "
                           "is absent from the capability inventory.",
                           "Use an exact retrieved tool/server pair, or switch to coder.")

        if task.route in _MCP and "image/" not in _tool_output_blob(task):
            for art in task.expected_artifacts:
                if art.required and _is_image_artifact(art):
                    fe(tid, "minor",
                       f"Required artifact {art.name!r} is an image/plot "
                       "(role=plot or image/* media_type), but selected MCP tools do not "
                       "document image/* outputs.",
                       "Prefer required=false for viz extras; keep a required role=data artifact.")

    if miss := _named_inventory_tools_missing(plan, available_tools=completeness):
        co("major",
           f"source_request explicitly requires inventory tools absent from the plan: {', '.join(miss)}.",
           "Add tasks for each explicitly required tool, or drop those requirements.")

    # Option A: unused thematic inventory is advisory (minor).
    if unused := _unused_matched_inventory_tools(plan, available_tools=completeness):
        co("minor",
           "Request mentions capabilities that ready inventory tools cover, "
           f"but no MCP-route task uses them: {', '.join(unused)} "
           "(non-blocking; inventory is availability, not a checklist).",
           "Optional: bind matching inventory tools on fedot_mas/react_tools when required.")

    if previous_plan is not None:
        if plan.plan_id != previous_plan.plan_id:
            add(category="consistency", severity="blocker", message="plan_id changed between revisions.",
                suggestion="Keep the original plan_id and increment revision.")
        if plan.revision <= previous_plan.revision:
            add(category="consistency", severity="blocker", message="A revised plan must increment revision.",
                suggestion=f"Use revision >= {previous_plan.revision + 1}.")
        if removed := sorted({t.id for t in previous_plan.tasks} - {t.id for t in plan.tasks}):
            add(category="consistency", severity="major",
                message=f"Revised plan dropped task ids: {', '.join(removed)}.",
                suggestion="Keep stable task ids; mark obsolete work optional instead of deleting.")

    return PlanCritique(
        schema_version="plan-critique/0.1",
        critique_id=f"CRIT-{uuid4().hex}",
        plan_id=plan.plan_id,
        plan_revision=plan.revision,
        critic_type="deterministic",
        verdict="revise" if any(i.is_blocking for i in issues) else "approve",
        issues=issues,
        checked_at=utc_now(),
    )


def validate_and_critique_plan(
    payload: Any,
    *,
    settings: ExperimentsSettings,
    available_tools: Iterable[dict[str, Any]] = (),
    preferred_tools: Iterable[dict[str, Any]] | None = None,
    previous_plan: ExperimentPlan | None = None,
    hypothesis_refs: Iterable[Any] = (),
    repo_candidates: Iterable[Any] = (),
    **_kwargs: Any,
) -> tuple[ExperimentPlan, PlanCritique]:
    """Strict schema validation, then deterministic policy checks."""
    from CoScientist.experiments.critique.mcp_repair import repair_plan_mcp_bindings
    from CoScientist.experiments.schemas.models import reset_lenient_planner, set_lenient_planner

    inventory = list(available_tools)
    repaired = repair_plan_mcp_bindings(payload, inventory)
    token = set_lenient_planner(settings.lenient_planner)
    try:
        plan = ExperimentPlan.model_validate(repaired)
    except ValidationError as exc:
        raise PlanValidationError(
            "ExperimentPlan schema validation failed", errors=exc.errors(include_url=False)
        ) from exc
    finally:
        reset_lenient_planner(token)
    return plan, critique_plan(
        plan, settings=settings, available_tools=inventory,
        preferred_tools=None if preferred_tools is None else list(preferred_tools),
        previous_plan=previous_plan, hypothesis_refs=hypothesis_refs, repo_candidates=repo_candidates,
    )


__all__ = ["PlanValidationError", "critique_plan", "validate_and_critique_plan"]
