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
    "otherwise postpone. When you postpone, additionally judge whether the "
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
    """The evidence's `content`, or — when an agent recorded its findings under
    other attr keys instead (e.g. bibliographic title/authors/journal/year with
    no `content`, or computational metrics like mean_macro_f1) — a fallback
    serialization of those attrs, so the judge never sees a blank line for
    evidence that actually has data. Without this, populated Evidence nodes
    render as empty and the validator postpones for "no data" even when the
    graph has real findings attached."""
    attrs = e.get("attrs") or {}
    content = str(attrs.get("content", "")).strip()
    if content:
        return content
    rest = {k: v for k, v in attrs.items() if k not in ("content", "subtype")}
    if not rest:
        return ""
    return "; ".join(f"{k}={v}" for k, v in rest.items())


_EVIDENCE_EDGE_TYPES = ("supports", "refutes", "refines")


def _has_evolved_from(g: Any, hid: str) -> bool:
    """True if `hid` is itself an evolved hypothesis (points to a parent via
    `evolved_from`) — the only case the independence checks below apply to."""
    return any(k == "evolved_from" for _, _, k in g.out_edges(hid, keys=True))


def _seed_evidence_ids(g: Any, hid: str):
    """Evidence NODES created at the exact same instant as `hid` itself
    (identical `created_at`) — i.e. evidence fabricated in the same breath as
    the hypothesis it supports, as opposed to evidence that already existed.
    Deliberately keyed on the EVIDENCE NODE's own timestamp, not the edge's:
    the edge linking `hid` to a pre-existing piece of evidence is always drawn
    in the same commit that writes `hid` (that's just how a single commit
    works), so an edge-timestamp check would wrongly flag genuinely old,
    independently-found evidence — e.g. a literature fact ResearchAgent
    recorded long before this hypothesis existed — as "seed" merely because
    EvolutionAgent cited it while writing the hypothesis. `commit()` stamps
    one shared timestamp on every node/edge it writes (store.py), so matching
    on the evidence node's created_at is exact, not heuristic: only evidence
    EvolutionAgent minted AT THE MOMENT it wrote `hid` counts as seed. For an
    evolved hypothesis, support limited to that has never been checked against
    anything independent."""
    genesis = g.nodes[hid].get("created_at")
    if genesis is None:
        return set()
    return {u for u, _, k in g.in_edges(hid, keys=True)
            if k in _EVIDENCE_EDGE_TYPES and g.nodes[u].get("created_at") == genesis}


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
    """True if `text` explicitly frames itself as a reinterpretation/re-run of
    `node_id` (e.g. "interpreting E19 data", "re-analysis of E19") — as
    opposed to merely citing it alongside other ids as background context
    (e.g. "structural rationale from E2 ... and E7"), which is ordinary,
    non-circular citation and must NOT be flagged. Deliberately narrow: it
    will miss phrasings outside this cue list (a cheap heuristic, not
    semantic dedup) — under-flagging is the safe failure mode here, since
    this feeds a check that can block a genuine confirmation."""
    if not node_id:
        return False
    return re.search(_RESTATEMENT_PATTERN + re.escape(node_id) + r"\b",
                     text or "", re.IGNORECASE) is not None


def _is_restatement(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """True if evidence `a` looks like a restatement of `b` rather than an
    independent source: `a`'s own text explicitly frames itself as
    reinterpreting `b` (see `_cites_as_restatement`), or both cite the
    identical source_ref (same underlying dataset/script/document). A cheap
    text-level signal, not semantic dedup — it catches an agent citing the
    same computed number twice under two evidence ids as if that were two
    confirmations (the failure mode actually observed: H7 confirmed on E19 +
    E21, where E21 was EvolutionAgent's own restatement of E19's number,
    captioned "interpreting E19 data" and written the instant H7 was
    created)."""
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

        # Independence check: collapse evidence that merely restates another
        # cited item (same underlying source/number under a different id) so
        # it can't pad the evidence count, and — for an EVOLVED hypothesis —
        # also drop evidence born in the same commit as the hypothesis itself
        # (it never faced anything beyond what motivated EvolutionAgent to
        # write it). A "confirmed" verdict resting on nothing outside that is
        # not independent confirmation: downgrade it and route it back through
        # EvolutionAgent (evolve_recommended) for a genuinely separate source.
        circular_reason = ""
        duplicate_evidence: List[str] = []
        supporting_ids = {eid for eid, pol in polarity.items()
                          if pol in ("supports", "refines")}
        if supporting_ids:
            independent = _independent_evidence_ids(by_id, supporting_ids)
            duplicate_evidence = sorted(supporting_ids - independent)
            if verdict == "confirmed":
                fg = graph.full_graph()
                if _has_evolved_from(fg, hid):
                    independent -= _seed_evidence_ids(fg, hid)
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

        nodes: List[Dict[str, Any]] = []
        concl = str(data.get("conclusion", "")).strip()
        if concl:
            concl_attrs: Dict[str, Any] = {
                "synthesis": concl[:2000],
                "validity_bounds": str(data.get("validity_bounds", ""))[:500],
            }
            if duplicate_evidence:
                concl_attrs["duplicate_evidence"] = duplicate_evidence
            # The gap EvolutionAgent acts on (graph/research/evolution.py). Only
            # meaningful on a postponed verdict; only trusted "recommended" if a
            # concrete gap actually came with it (a bare true with no gap text
            # is treated as the model not really having one to name).
            if verdict == "postponed":
                evolve = data.get("evolve")
                evolve = evolve if isinstance(evolve, dict) else {}
                gap = str(evolve.get("gap", "")).strip()[:300]
                concl_attrs["evolve_recommended"] = bool(evolve.get("recommended")) and bool(gap)
                if gap:
                    concl_attrs["evolve_gap"] = gap
                if circular_reason:
                    # Overrides whatever the model said — it reasoned under
                    # "confirmed", so its own evolve judgment (if any) isn't
                    # about this gap. The next EvolutionAgent pass needs an
                    # INDEPENDENT source for the SAME claim, not a new sub-claim.
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

    The callback NEVER awaits the LLM — it schedules `judge_hypothesis` on the
    event loop and returns immediately, so the orchestration loop is never
    blocked (full asynchrony). Dedup includes the research generation and exact
    evidence identities/polarities, so a hypothesis is re-judged when its
    evidence changes without leaking state into a later research."""

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
