"""Background hypothesis validator — spec Module 4, fully asynchronous.

NOT an ADK sub-agent and NOT dependent on the orchestrator. It is a plugin that
watches the graph: whenever evidence lands on a hypothesis, it fires a
fire-and-forget asyncio task that judges the hypothesis (confirmed / refuted /
postponed) and writes the Conclusion. The main orchestration loop never awaits
it — on the web's persistent event loop the verdict lands in the graph a moment
later and the live viewer shows it. Judgment runs in a SMALL focused context
(one hypothesis' slice), not the orchestrator's.

Best-effort throughout: any failure leaves the graph untouched and never breaks
the run. Commits are attributed to source "ValidatorAgent" (its write
permissions live in schema.AGENT_PERMISSIONS).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional

from google.adk.plugins.base_plugin import BasePlugin

from CoScientist.graph.research import queries
from CoScientist.graph.research.store import get_research_graph, research_graph

logger = logging.getLogger(__name__)

SOURCE = "ValidatorAgent"
# Module-level refs so fire-and-forget tasks are not garbage-collected mid-flight.
_TASKS: set = set()

_SYSTEM = (
    "You are a rigorous scientific hypothesis validator. You are given ONE "
    "hypothesis, its confirmation criteria, and the evidence attached to it. Some "
    "evidence is UNCLASSIFIED (its polarity is not yet set) — decide for EACH "
    "evidence item whether it supports, refutes or refines the hypothesis (or is "
    "irrelevant). Then weigh it against the criteria and return a verdict. Be "
    "conservative: confirm only when supporting evidence is strong, consistent and "
    "meets the criteria; refute when there is decisive contradicting evidence; "
    "otherwise postpone. If the hypothesis claims to be THE dominant/leading/"
    "most-frequent option, a test against a theoretical/uniform baseline is NOT "
    "enough — that only shows it beats chance, not that it beats its actual "
    "runner-up. Do not mark such criteria 'met' unless the evidence contains a "
    "head-to-head comparison against the specific runner-up; a close runner-up "
    "with no such comparison is grounds to postpone, not confirm. When you "
    "postpone, additionally judge whether the "
    "hypothesis is CLOSE: evidence direction is consistent/supportive and exactly "
    "ONE specific, nameable piece is missing (e.g. a quantitative value, an "
    "untested sub-claim, a named comparison) — as opposed to genuinely weak or "
    "too vague to refine. Only recommend evolving it when the missing piece is "
    "concrete enough that a targeted literature search could plausibly find it — "
    "name that missing piece exactly, don't restate the whole hypothesis. Reply "
    "with STRICT JSON only, no prose:\n"
    '{"evidence":{"<evidence_id>":"supports|refutes|refines|irrelevant"},'
    '"verdict":"confirmed|refuted|postponed",'
    '"criteria":{"<criteria_id>":"met|not_met"},'
    '"conclusion":"one-paragraph synthesis of the finding",'
    '"validity_bounds":"limits of validity",'
    '"reason":"one sentence justifying the verdict",'
    '"evolve":{"recommended":true|false,'
    '"gap":"one sentence naming the SPECIFIC missing piece; empty if not recommended"}}'
)


def _enabled() -> bool:
    try:
        from CoScientist.config import get_settings
        return get_settings().research_graph.enabled
    except Exception:  # noqa: BLE001
        return False


async def _complete(system: str, user: str) -> str:
    """One small LLM call via litellm (same config as the semantic layer)."""
    import litellm
    from CoScientist.config import get_settings
    s = get_settings().llm
    model = os.getenv("RESEARCH_VALIDATOR_MODEL") or s.main_model
    resp = await litellm.acompletion(
        model=model, api_base=s.main_url, api_key=s.openai_api_key,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0,
    )
    return resp.choices[0].message.content or ""


def _parse_json(raw: str) -> Optional[Dict[str, Any]]:
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return None


def _research_id(graph: Any) -> str:
    """Return the current research generation for validator isolation."""
    full = graph.full()
    if not isinstance(full, dict):
        return ""
    return str(full.get("research_id") or "")


def _evidence_text(e: Dict[str, Any]) -> str:
    """`content`, or a fallback serialization of other attrs (e.g. bibliographic
    fields, metrics) when `content` is empty — so populated evidence doesn't
    read as blank to the judge."""
    attrs = e.get("attrs") or {}
    content = str(attrs.get("content", "")).strip()
    if content:
        return content
    rest = {k: v for k, v in attrs.items() if k not in ("content", "subtype")}
    if not rest:
        return ""
    return "; ".join(f"{k}={v}" for k, v in rest.items())


_EVIDENCE_EDGE_TYPES = ("supports", "refutes", "refines")


_RESTATEMENT_CUES = (
    r"interpreting", r"interpretation of", r"interprets",
    r"reinterpreting", r"reinterpretation of",
    r"re-?analysis of", r"re-?analyzing",
    r"re-?comput(?:ed|ing) from",
    r"restating", r"restatement of",
    r"same (?:number|result|data|finding) as",
    r"duplicate of",
)
_RESTATEMENT_PATTERN = (
    r"(?:" + "|".join(_RESTATEMENT_CUES) + r")\s+(?:the\s+|this\s+|that\s+)?"
)


def _cites_as_restatement(text: str, node_id: str) -> bool:
    """True if `text` explicitly frames itself as reinterpreting/re-running
    `node_id` (e.g. "re-analysis of E19"), as opposed to merely citing it as
    background. Narrow on purpose — under-flagging is the safe failure mode,
    since this feeds a check that can block a genuine confirmation."""
    if not node_id:
        return False
    return re.search(_RESTATEMENT_PATTERN + re.escape(node_id) + r"\b",
                     text or "", re.IGNORECASE) is not None


def _is_restatement(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """True if evidence `a` looks like a restatement of `b`: `a` names `b` as
    a reinterpretation (see `_cites_as_restatement`), or both share the same
    `source_ref`. A cheap text-level signal, not semantic dedup — catches an
    agent citing the same number twice under two evidence ids as if that
    were two confirmations."""
    attrs_a, attrs_b = a.get("attrs") or {}, b.get("attrs") or {}
    text_a = f"{attrs_a.get('content', '')} {attrs_a.get('source_ref', '')}"
    if _cites_as_restatement(text_a, str(b.get("id", ""))):
        return True
    ref_a = str(attrs_a.get("source_ref", "")).strip().lower()
    ref_b = str(attrs_b.get("source_ref", "")).strip().lower()
    return bool(ref_a) and ref_a == ref_b


def _independent_evidence_ids(by_id: Dict[str, Any], ev_ids) -> set:
    """Collapse text-level restatements (`_is_restatement`) among `ev_ids` so
    citing the same underlying number/source twice counts once. Evidence ids
    not present in `by_id` are dropped silently (defensive)."""
    keep: List[str] = []
    for eid in sorted(ev_ids):
        node = by_id.get(eid)
        if node is None:
            continue
        node = {**node, "id": eid}
        if any(_is_restatement(node, {**by_id[k], "id": k}) or
               _is_restatement({**by_id[k], "id": k}, node) for k in keep):
            continue
        keep.append(eid)
    return set(keep)


_SUPERLATIVE_RE = re.compile(
    r"\b(?:dominant|dominates?|most (?:frequent|common)|leading|"
    r"outperforms? all|highest[- ]frequency)\b"
    r"|\b(?:сам\w*\s+част\w*|доминир\w*|преоблада\w*)",
    re.IGNORECASE,
)
_COMPARATOR_RE = re.compile(
    r"\b(vs\.?|versus|compared to|against the|second[- ]most|runner-?up|"
    r"против|чем у|второе место|ближайш\w* конкурент)\b",
    re.IGNORECASE,
)
_BASELINE_COMPARISON_RE = re.compile(
    r"\b(?:vs\.?|versus|compared to|against(?: the)?)\s+(?:an?\s+|the\s+)?"
    r"(?:uniform|random|baseline|chance|theoretical|expected|null)\b[\w\s%().,-]*",
    re.IGNORECASE,
)


def _claims_superlative(text: str) -> bool:
    """True if a hypothesis's own wording claims to be THE dominant/leading/
    most-frequent option — a claim that is only meaningful relative to
    whatever the actual runner-up is, not to a theoretical/uniform baseline."""
    return bool(_SUPERLATIVE_RE.search(text or ""))


def _has_named_comparator(texts: List[str]) -> bool:
    """True if any evidence text frames a head-to-head comparison (e.g. "vs
    indole") rather than only a theoretical/uniform baseline. Baseline
    phrases are stripped first so a bare "vs uniform baseline" doesn't
    self-match. Fuzzy text signal — a miss only soft-flags the Conclusion,
    never downgrades the verdict (see judge_hypothesis)."""
    for t in texts:
        stripped = _BASELINE_COMPARISON_RE.sub(" ", t or "")
        if _COMPARATOR_RE.search(stripped):
            return True
    return False


def _build_user(slice_: Dict[str, Any], hid: str) -> str:
    """Render the focused judging prompt from the hypothesis' context slice."""
    by_id = {n["id"]: n for n in slice_.get("nodes", [])}
    edges = slice_.get("edges", [])
    h = by_id.get(hid, {})
    lines = [f"HYPOTHESIS {hid}: {(h.get('attrs') or {}).get('formulation', '')}", ""]
    crit = [by_id[e["from"]] for e in edges
            if e["type"] == "formulated_for" and e["to"] == hid and e["from"] in by_id]
    lines.append("CONFIRMATION CRITERIA:")
    lines += [f"- {c['id']}: {(c.get('attrs') or {}).get('threshold', c.get('attrs'))}"
              for c in crit] or ["- (none defined)"]
    lines.append("")
    for kind, label in (("supports", "SUPPORTING"), ("refutes", "REFUTING"),
                        ("refines", "REFINING"), ("relates_to", "UNCLASSIFIED (assign polarity)")):
        evs = [by_id[e["from"]] for e in edges
               if e["type"] == kind and e["to"] == hid and e["from"] in by_id
               and by_id[e["from"]].get("type") == "Evidence"]
        if evs:
            lines.append(f"{label} EVIDENCE:")
            lines += [f"- {e['id']} ({(e.get('attrs') or {}).get('subtype','')}): "
                      f"{_evidence_text(e)}" for e in evs]
            lines.append("")
    return "\n".join(lines)


async def judge_hypothesis(
    hid: str,
    complete: Optional[Callable] = None,
    graph=None,
    expected_research_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Judge one hypothesis and commit the verdict + Conclusion. Best-effort;
    returns the CommitResult dict, or None if it could not judge/commit.
    `complete` is injectable for tests (bypasses the real LLM)."""
    try:
        graph = graph or research_graph
        if expected_research_id is not None \
                and _research_id(graph) != expected_research_id:
            return None
        sl = graph.get_context_slice(hid, depth=2)
        if "error" in sl:
            return None
        by_id = {n["id"]: n for n in sl.get("nodes", [])}
        h = by_id.get(hid)
        if not h or h.get("type") != "Hypothesis":
            return None
        if h.get("status") != "under_verification":
            return None  # only judge branches that are actually under verification

        raw = await (complete or _complete)(_SYSTEM, _build_user(sl, hid))
        data = _parse_json(raw)
        if not data:
            return None
        verdict = str(data.get("verdict", "")).strip().lower()
        if verdict not in ("confirmed", "refuted", "postponed"):
            return None

        # The LLM call yields control for long enough that the same store may
        # have been reset to a new research. Never commit an old verdict into
        # that new research generation.
        if expected_research_id is not None \
                and _research_id(graph) != expected_research_id:
            logger.info(
                "[validator] discarding stale judgment for %s from research %s",
                hid,
                expected_research_id,
            )
            return None

        status_updates: List[Dict[str, Any]] = [
            {"id": hid, "status": verdict, "reason": str(data.get("reason", ""))[:300]}]
        # criteria the model judged, restricted to this hypothesis' actual criteria
        crit_ids = {e["from"] for e in sl.get("edges", [])
                    if e["type"] == "formulated_for" and e["to"] == hid}
        for ccid, met in (data.get("criteria") or {}).items():
            if ccid in crit_ids:
                is_met = str(met).strip().lower() in ("met", "true", "yes", "1")
                cur = by_id.get(ccid, {}).get("status")
                if (is_met and cur == "not_met") or (not is_met and cur == "met"):
                    status_updates.append({"id": ccid, "status": "met" if is_met else "not_met"})

        # Evidence attached to the hypothesis, and its CURRENT polarity edge (if any).
        cur_pol = {}
        for e in sl.get("edges", []):
            if e["to"] == hid and e["type"] in ("supports", "refutes", "refines", "relates_to") \
                    and by_id.get(e["from"], {}).get("type") == "Evidence":
                # a real polarity edge wins over the neutral relates_to
                if e["type"] != "relates_to" or e["from"] not in cur_pol:
                    cur_pol[e["from"]] = e["type"]

        edges: List[Dict[str, Any]] = []
        polarity = {}  # ev_id -> final supports/refutes/refines
        for eid, pol in (data.get("evidence") or {}).items():
            pol = str(pol).strip().lower()
            if eid not in cur_pol or pol not in ("supports", "refutes", "refines"):
                continue
            polarity[eid] = pol
            # add the polarity edge only if it isn't already present as such
            if cur_pol[eid] != pol:
                edges.append({"type": pol, "from": eid, "to": hid})
        # evidence that already had a polarity keeps it
        for eid, et in cur_pol.items():
            if et in ("supports", "refutes", "refines"):
                polarity.setdefault(eid, et)

        circular_reason = ""
        duplicate_evidence: List[str] = []
        supporting_ids = {eid for eid, pol in polarity.items()
                          if pol in ("supports", "refines")}
        if supporting_ids:
            independent = _independent_evidence_ids(by_id, supporting_ids)
            duplicate_evidence = sorted(supporting_ids - independent)
            if verdict == "confirmed":
                if not independent:
                    circular_reason = (
                        "confirmed verdict rejected: every supporting evidence "
                        "item is self-referential — born with the hypothesis "
                        "and/or a restatement of the same source/number under "
                        "a different id. No independent confirmation yet."
                    )
                    verdict = "postponed"
                    status_updates[0]["status"] = verdict
                    status_updates[0]["reason"] = circular_reason[:300]

        comparator_missing = False
        if verdict == "confirmed" and _claims_superlative(
                (h.get("attrs") or {}).get("formulation", "")):
            ev_texts = [_evidence_text(by_id[eid]) for eid in supporting_ids if eid in by_id]
            comparator_missing = not _has_named_comparator(ev_texts)

        shared_evidence_hyps: Dict[str, List[str]] = {}

        nodes: List[Dict[str, Any]] = []
        concl = str(data.get("conclusion", "")).strip()
        if concl:
            concl_attrs: Dict[str, Any] = {
                "synthesis": concl[:2000],
                "validity_bounds": str(data.get("validity_bounds", ""))[:500],
            }
            if duplicate_evidence:
                concl_attrs["duplicate_evidence"] = duplicate_evidence
            if comparator_missing:
                concl_attrs["comparator_check"] = "missing"
            if shared_evidence_hyps:
                concl_attrs["shares_evidence_with"] = {
                    k: sorted(v) for k, v in sorted(shared_evidence_hyps.items())
                }
            if verdict == "postponed":
                evolve = data.get("evolve")
                evolve = evolve if isinstance(evolve, dict) else {}
                gap = str(evolve.get("gap", "")).strip()[:300]
                concl_attrs["evolve_recommended"] = bool(evolve.get("recommended")) and bool(gap)
                if gap:
                    concl_attrs["evolve_gap"] = gap
                if circular_reason:
                    concl_attrs["evolve_recommended"] = True
                    concl_attrs["evolve_gap"] = (
                        "independent confirmation: a source for this claim "
                        "that is not already cited on it"
                    )
                    concl_attrs["independence_check"] = "failed"
            nodes.append({"type": "Conclusion", "ref": "cl", "attrs": concl_attrs})
            edges += [{"type": "based_on", "from": "#cl", "to": eid}
                      for eid in sorted(polarity)]
            edges += [{"type": "determines_sufficiency", "from": ccid, "to": "#cl"}
                      for ccid in sorted(crit_ids)]

        result = graph.commit(
            source=SOURCE,
            nodes=nodes,
            edges=edges,
            status_updates=status_updates,
        )
        if result.ok:
            logger.info("[validator] %s → %s%s", hid, verdict,
                        " (+Conclusion)" if concl else "")
        else:
            logger.info("[validator] %s commit rejected: %s", hid, result.errors)
        return result.model_dump(exclude_none=True)
    except Exception as exc:  # noqa: BLE001 — never break a run
        logger.warning("[validator] judging %s failed: %s", hid, exc)
        return None


class BackgroundValidatorPlugin(BasePlugin):
    """Fires fire-and-forget validations when a hypothesis gains evidence.

    Never awaits the LLM — schedules `judge_hypothesis` on the event loop and
    returns immediately. Dedup keys on research generation + exact evidence
    identities/polarities, so a hypothesis is re-judged when its evidence
    changes, without leaking state into a later research."""

    def __init__(self) -> None:
        super().__init__(name="background_validator")
        self._completed: set[tuple] = set()
        self._inflight: set[tuple] = set()
        self._research_by_graph: Dict[int, str] = {}

    @staticmethod
    def _key(graph: Any, research_id: str, item: Dict[str, Any]) -> tuple:
        evidence = tuple(
            tuple(sorted(str(value) for value in (item.get(kind) or [])))
            for kind in ("supporting", "refuting", "related")
        )
        return id(graph), research_id, str(item["hypothesis"]), evidence

    def _activate_research(self, graph: Any, research_id: str) -> None:
        """Drop completed dedup entries when a store starts a new research."""
        graph_id = id(graph)
        previous = self._research_by_graph.get(graph_id)
        if previous == research_id:
            return
        self._research_by_graph[graph_id] = research_id
        self._completed = {key for key in self._completed if key[0] != graph_id}

    async def _run_validation(
        self,
        *,
        key: tuple,
        graph: Any,
        hypothesis: str,
        research_id: str,
    ) -> None:
        try:
            result = await judge_hypothesis(
                hypothesis,
                graph=graph,
                expected_research_id=research_id,
            )
            if result and result.get("ok") and _research_id(graph) == research_id:
                self._completed.add(key)
        except Exception as exc:  # noqa: BLE001 -- background work is best-effort
            logger.warning("[validator] background judgment for %s failed: %s",
                           hypothesis, exc)
        finally:
            # A failed or rejected validation remains retryable on the next
            # research_commit callback.
            self._inflight.discard(key)

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result) -> None:
        if not _enabled() or getattr(tool, "name", "") != "research_commit":
            return None
        try:
            graph = get_research_graph(tool_context)
            research_id = _research_id(graph)
            self._activate_research(graph, research_id)
            for item in queries.unresolved_hypotheses(graph)["items"]:
                key = self._key(graph, research_id, item)
                if key in self._completed or key in self._inflight:
                    continue
                self._inflight.add(key)
                logger.info("[validator] scheduling background judgment for %s",
                            item["hypothesis"])
                task = asyncio.create_task(
                    self._run_validation(
                        key=key,
                        graph=graph,
                        hypothesis=item["hypothesis"],
                        research_id=research_id,
                    )
                )
                _TASKS.add(task)
                task.add_done_callback(_TASKS.discard)
        except Exception:  # noqa: BLE001
            pass
        return None


background_validator_plugin = BackgroundValidatorPlugin()


_SETTLE_TIMEOUT = 60.0
_SETTLE_POLL = 2.0

READINESS_STATE_KEY = "research_graph_readiness"


def _graph_readiness(graph: Any) -> Dict[str, Any]:
    """Explicit READY/INCOMPLETE verdict for the graph, evaluated fresh on
    every poll instead of inferred from a bare timer.

    READY requires BOTH: no hypothesis still `under_verification`, AND none
    carrying unresolved/unclassified evidence (`queries.unresolved_hypotheses`
    — a hypothesis can hold evidence with no verdict yet without being
    `under_verification`, e.g. its status flipped before the evidence edge
    landed; the old check only looked at the status count and missed that
    case). Anything else is INCOMPLETE, named by hypothesis id so the caller
    can log/report exactly what is still open."""
    unresolved = sorted(
        item["hypothesis"] for item in queries.unresolved_hypotheses(graph)["items"]
    )
    under_verification = (
        graph.overview().get("counts", {}).get("Hypothesis", {}).get("under_verification", 0)
    )
    ready = not unresolved and not under_verification
    return {
        "state": "ready" if ready else "incomplete",
        "unresolved_hypotheses": unresolved,
        "under_verification": under_verification,
    }


async def wait_for_validator_settle(callback_context: Any) -> None:
    """before_agent callback for ResultAggregatorAgent: gate the report on an
    EXPLICIT readiness verdict (`_graph_readiness`) instead of hoping a fixed
    timeout was enough.

    `after_tool_callback` above only reschedules on the NEXT `research_commit`
    — but ResultAggregatorAgent's tools are read-only, so a hypothesis left
    `under_verification` by the last agent to touch the graph is orphaned,
    never re-triggered. This re-schedules those stragglers through the same
    plugin instance (reusing its dedup, so nothing double-judges), polling up
    to `_SETTLE_TIMEOUT`. Whichever way the loop exits, the verdict — READY or
    INCOMPLETE, plus which hypotheses are still open — is written to
    `callback_context.state[READINESS_STATE_KEY]` and logged explicitly, so a
    report that goes out INCOMPLETE is a recorded decision, not a silent
    timeout. Best-effort: never raises, never blocks past the timeout; the
    result_aggregator prompt is expected to flag an INCOMPLETE graph as
    preliminary."""
    def _record(readiness: Dict[str, Any]) -> None:
        try:
            callback_context.state[READINESS_STATE_KEY] = readiness
        except Exception:  # noqa: BLE001 -- state unavailable (e.g. bare test doubles)
            pass

    if not _enabled():
        return None
    try:
        graph = get_research_graph(callback_context)
        research_id = _research_id(graph)
        background_validator_plugin._activate_research(graph, research_id)
        deadline = time.monotonic() + _SETTLE_TIMEOUT
        while True:
            readiness = _graph_readiness(graph)
            if readiness["state"] == "ready":
                logger.info("[validator] readiness gate: READY — all hypotheses resolved")
                _record(readiness)
                return None
            for item in queries.unresolved_hypotheses(graph)["items"]:
                key = background_validator_plugin._key(graph, research_id, item)
                if (key in background_validator_plugin._completed
                        or key in background_validator_plugin._inflight):
                    continue
                background_validator_plugin._inflight.add(key)
                logger.info(
                    "[validator] result_aggregator settle: rescheduling stale %s",
                    item["hypothesis"],
                )
                task = asyncio.create_task(
                    background_validator_plugin._run_validation(
                        key=key, graph=graph,
                        hypothesis=item["hypothesis"], research_id=research_id,
                    )
                )
                _TASKS.add(task)
                task.add_done_callback(_TASKS.discard)
            if time.monotonic() >= deadline:
                logger.warning(
                    "[validator] readiness gate: INCOMPLETE after %.0fs — "
                    "%d hypothesis(es) still unresolved: %s",
                    _SETTLE_TIMEOUT, len(readiness["unresolved_hypotheses"]),
                    ", ".join(readiness["unresolved_hypotheses"]) or "none named",
                )
                _record(readiness)
                return None
            await asyncio.sleep(_SETTLE_POLL)
    except Exception:  # noqa: BLE001
        logger.warning("[validator] wait_for_validator_settle failed", exc_info=True)
        _record({"state": "incomplete", "unresolved_hypotheses": [],
                  "under_verification": None, "error": "readiness check raised"})
        return None
