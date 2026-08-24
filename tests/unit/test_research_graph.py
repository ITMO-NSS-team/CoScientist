"""Unit tests for the Research Context Graph (CoScientist.graph.research).

Exercise the store/schema/queries directly with source= strings (no ADK
contexts needed) against a tmp_path store, plus a couple of assembly-level
checks that the tools are wired into the right agents' prompts.

Run from the repo root:  pytest tests/unit/test_research_graph.py -q
"""
import pytest
from dotenv import load_dotenv

load_dotenv()

from CoScientist.graph.research import queries, schema  # noqa: E402
from CoScientist.graph.research.store import ResearchGraphStore  # noqa: E402


@pytest.fixture
def store(tmp_path):
    return ResearchGraphStore(directory=str(tmp_path))


def _init(store):
    return store.init_research(
        source="OrchestratorAgent",
        question="Does compound X inhibit target Y?",
        constraints=[{"subtype": "ethics", "content": "no animal testing"}],
        tools=[{"name": "AutoDock", "tool_type": "computational"}],
        resources=[{"resource_type": "GPU-hours", "remaining": 100, "limit": 100}],
        empirical_bases=[{"base_type": "dataset", "volume": "12k"}],
    )


# ── schema invariants ─────────────────────────────────────────────────────────

def test_edge_pairs_reference_known_types():
    for edge, pairs in schema.EDGE_TYPES.items():
        for f, t in pairs:
            assert f in schema.NODE_TYPES, f"{edge}: unknown from-type {f}"
            assert t in schema.NODE_TYPES, f"{edge}: unknown to-type {t}"


def test_transitions_reference_declared_statuses():
    for typ, pairs in schema.STATUS_TRANSITIONS.items():
        declared = set(schema.NODE_TYPES[typ].statuses)
        for f, t in pairs:
            assert f in declared and t in declared, f"{typ}: {f}->{t} not declared"


def test_node_prefixes_unique():
    prefixes = [s.prefix for s in schema.NODE_TYPES.values()]
    assert len(prefixes) == len(set(prefixes))


def test_permissions_reference_known_types_edges_transitions():
    for agent, perm in schema.AGENT_PERMISSIONS.items():
        for t in perm.create | perm.update_attrs:
            assert t in schema.NODE_TYPES, f"{agent}: unknown create type {t}"
        for edge, f, t in perm.edges:
            assert (f, t) in schema.EDGE_TYPES[edge], f"{agent}: bad edge {edge} {f}->{t}"
        for typ, f, t in perm.transitions:
            assert (f, t) in schema.STATUS_TRANSITIONS[typ], f"{agent}: bad transition {typ} {f}->{t}"


def test_permission_agents_exist_in_system(request):
    """Every AGENT_PERMISSIONS key must be a real agent in system.yaml, EXCEPT
    the virtual write-sources: 'human' (writes via HITL) and 'ValidatorAgent'
    (writes via the fully-async background validator plugin, not a sub-agent)."""
    from CoScientist.assembly.schema import get_config
    agents = set(get_config().agents)
    virtual = {"human", "ValidatorAgent"}
    for name in schema.AGENT_PERMISSIONS:
        if name in virtual:
            continue
        assert name in agents, f"AGENT_PERMISSIONS has unknown agent {name!r}"


# ── init + happy-path commit ────────────────────────────────────────────────────

def test_init_creates_root_and_star(store):
    r = _init(store)
    assert r["ok"] and r["root_id"] == "Q1"
    types = {n["type"] for n in r["committed"]["nodes"]}
    assert {"ResearchQuestion", "Constraint", "Tool", "Resource", "EmpiricalBase"} <= types
    ov = store.overview()
    assert ov["root"] == "Q1"
    # context star edges exist
    full = store.full()
    edge_types = {e["type"] for e in full["edges"]}
    assert "contextualizes" in edge_types and "defines_scope" in edge_types


def test_commit_with_refs(store):
    _init(store)
    r = store.commit(
        source="HypothesesAgent",
        nodes=[{"type": "Hypothesis", "ref": "h", "attrs": {"formulation": "X binds Y"}},
               {"type": "VerificationMethod", "ref": "vm", "attrs": {"method_type": "computational"}}],
        edges=[{"type": "motivates", "from": "Q1", "to": "#h"},
               {"type": "tested_by", "from": "#h", "to": "#vm"}],
    )
    assert r.ok, r.errors
    ids = {n["type"]: n["id"] for n in r.committed["nodes"]}
    assert ids["Hypothesis"] == "H1" and ids["VerificationMethod"] == "VM1"


# ── transactionality ────────────────────────────────────────────────────────────

def test_commit_is_atomic_on_any_error(store):
    _init(store)
    before = store.full()["nodes"]
    r = store.commit(
        source="HypothesesAgent",
        nodes=[{"type": "Hypothesis", "ref": "h", "attrs": {"formulation": "ok"}},
               {"type": "Evidence", "attrs": {"subtype": "literature"}}],  # not allowed for Hypotheses
    )
    assert not r.ok
    assert any("may not create 'Evidence'" in e for e in r.errors)
    assert store.full()["nodes"] == before, "partial write on a failed commit"


# ── permission / transition / edge rejections ──────────────────────────────────

def test_orchestrator_cannot_create_context_star_mid_run(store):
    """The context-star types are seeded only via research_init. The orchestrator
    must NOT be able to invent them through a normal commit (selective-context:
    Tools come from Hypotheses/Coder, not the orchestrator mid-run)."""
    _init(store)
    for t in ("Tool", "Resource", "EmpiricalBase"):
        r = store.commit(source="OrchestratorAgent",
                         nodes=[{"type": t, "attrs": {"name": "x", "base_type": "d",
                                                      "resource_type": "gpu"}}])
        assert not r.ok, f"orchestrator should not create {t} mid-run"
        assert any("may not create" in e for e in r.errors)


def test_init_still_seeds_context_star_privileged(store):
    """research_init seeds Tool/Resource/EmpiricalBase/Constraint despite the
    orchestrator's narrowed general create-set (privileged path)."""
    r = _init(store)  # declares a Tool, Resource, EmpiricalBase, Constraint
    assert r["ok"], r.get("errors")
    types = {n["type"] for n in r["committed"]["nodes"]}
    assert {"Tool", "Resource", "EmpiricalBase", "Constraint"} <= types
    # but a structurally-invalid seed is still rejected even when privileged
    bad = store.init_research(source="OrchestratorAgent", question="Q?",
                              constraints=[{"content": "no subtype"}])  # Constraint needs subtype
    assert not bad["ok"] and any("subtype" in e for e in bad["errors"])


def test_orchestrator_can_wire_constraints(store):
    """regulates/constrains had no creator before; the orchestrator now wires
    seeded Constraints to the methods/hypotheses that appear later."""
    store.init_research(source="OrchestratorAgent", question="Q?",
                        constraints=[{"subtype": "ethics", "content": "no 3R breach"}])
    store.commit(source="HypothesesAgent",
                 nodes=[{"type": "Hypothesis", "ref": "h", "attrs": {"formulation": "x"}},
                        {"type": "VerificationMethod", "ref": "vm", "attrs": {"method_type": "lab"}}],
                 edges=[{"type": "tested_by", "from": "#h", "to": "#vm"}])
    r = store.commit(source="OrchestratorAgent",
                     edges=[{"type": "constrains", "from": "C1", "to": "H1"},
                            {"type": "regulates", "from": "C1", "to": "VM1"}])
    assert r.ok, r.errors


def test_permission_rejection_lists_allowed_types(store):
    _init(store)
    r = store.commit(source="ResearchAgent",
                     nodes=[{"type": "Tool", "attrs": {"name": "foo"}}])
    assert not r.ok
    assert any("ResearchAgent" in e and "Tool" in e for e in r.errors)


def test_bad_initial_status_rejected(store):
    _init(store)
    r = store.commit(source="HypothesesAgent",
                     nodes=[{"type": "Hypothesis", "status": "confirmed",
                             "attrs": {"formulation": "x"}}])
    assert not r.ok
    assert any("not a valid initial status" in e for e in r.errors)


def test_missing_subtype_rejected(store):
    _init(store)
    r = store.commit(source="ResearchAgent",
                     nodes=[{"type": "Evidence", "attrs": {"content": "no subtype"}}])
    assert not r.ok
    assert any("requires attrs.subtype" in e for e in r.errors)


def test_wrong_edge_pair_rejected(store):
    _init(store)
    store.commit(source="HypothesesAgent",
                 nodes=[{"type": "Hypothesis", "ref": "h", "attrs": {"formulation": "x"}}])
    # supports must go Evidence->Hypothesis; here Hypothesis->? is wrong direction
    r = store.commit(source="ResearchAgent",
                     nodes=[{"type": "Evidence", "ref": "e", "attrs": {"subtype": "literature"}}],
                     edges=[{"type": "supports", "from": "H1", "to": "#e"}])
    assert not r.ok
    assert any("must connect Evidence → Hypothesis" in e for e in r.errors)


def test_unknown_endpoint_lists_ids(store):
    _init(store)
    r = store.commit(source="ResearchAgent",
                     nodes=[{"type": "Evidence", "ref": "e", "attrs": {"subtype": "literature"}}],
                     edges=[{"type": "relates_to", "from": "#e", "to": "H99"}])
    assert not r.ok
    assert any("no node 'H99'" in e for e in r.errors)


def test_illegal_transition_rejected(store):
    _init(store)
    store.commit(source="HypothesesAgent",
                 nodes=[{"type": "Hypothesis", "ref": "h", "attrs": {"formulation": "x"}}])
    # refuted is not reachable from formulated in one step, and Hypotheses can't do it
    r = store.commit(source="OrchestratorAgent",
                     status_updates=[{"id": "H1", "status": "confirmed"}])
    assert not r.ok
    assert any("cannot go 'formulated' → 'confirmed'" in e for e in r.errors)


# ── RU aliases, enrichment, locking, persistence ────────────────────────────────

def test_russian_aliases_accepted(store):
    _init(store)
    r = store.commit(source="HypothesesAgent",
                     nodes=[{"type": "Гипотеза", "status": "сформулирована",
                             "attrs": {"formulation": "гипотеза"}}])
    assert r.ok, r.errors
    assert store.full()["nodes"][-1]["type"] == "Hypothesis"


def test_attrs_merge_enrichment(store):
    _init(store)
    r = store.commit(source="ResearchAgent",
                     nodes=[{"id": "EB1", "attrs": {"volume": "20k", "note": "cleaned"}}])
    assert r.ok, r.errors
    eb = next(n for n in store.full()["nodes"] if n["id"] == "EB1")
    assert eb["attrs"]["volume"] == "20k" and eb["attrs"]["note"] == "cleaned"


def test_hypothesis_branch_lock(store):
    _init(store)
    store.commit(source="HypothesesAgent",
                 nodes=[{"type": "Hypothesis", "ref": "h", "attrs": {"formulation": "x"}}])
    assert store.commit(source="OrchestratorAgent",
                        status_updates=[{"id": "H1", "status": "under_verification"}]).ok
    r = store.commit(source="OrchestratorAgent",
                     status_updates=[{"id": "H1", "status": "under_verification"}])
    assert not r.ok
    assert any("already under verification" in e for e in r.errors)


def test_ref_and_id_matching_is_case_insensitive(store):
    """LLMs define ref 'E4' then cite '#e4', and may write 'h1' for node 'H1'.
    Both must resolve (to the canonical stored key) instead of erroring."""
    _init(store)
    # ref defined uppercase ("Ev"), cited lowercase ("#ev"); bare id "q1" cited
    # lowercase. The node created here is assigned id E1 (ref is just a handle).
    r = store.commit(source="ResearchAgent",
                     nodes=[{"type": "Evidence", "ref": "Ev", "attrs": {"subtype": "literature"}}],
                     edges=[{"type": "relates_to", "from": "#ev", "to": "q1"}])
    assert r.ok, r.errors
    edges = store.full()["edges"]
    assert any(e["type"] == "relates_to" and e["from"] == "E1" and e["to"] == "Q1"
               for e in edges)
    # status update citing the node id in the wrong case ("e1" for "E1")
    assert store.commit(source="ResearchAgent",
                        status_updates=[{"id": "e1", "status": "validated"}]).ok
    # case-variant refs in one commit are a duplicate, not two nodes
    dup = store.commit(source="ResearchAgent",
                       nodes=[{"type": "Evidence", "ref": "x1", "attrs": {"subtype": "meta"}},
                              {"type": "Evidence", "ref": "X1", "attrs": {"subtype": "meta"}}])
    assert not dup.ok and any("duplicate ref" in e for e in dup.errors)


def test_duplicate_edge_is_idempotent(store):
    _init(store)
    store.commit(source="HypothesesAgent",
                 nodes=[{"type": "Hypothesis", "ref": "h", "attrs": {"formulation": "x"}}],
                 edges=[{"type": "motivates", "from": "Q1", "to": "#h"}])
    r = store.commit(source="HypothesesAgent",
                     edges=[{"type": "motivates", "from": "Q1", "to": "H1"}])
    assert r.ok
    assert any("already exists" in w for w in r.warnings)
    motivates = [e for e in store.full()["edges"] if e["type"] == "motivates"]
    assert len(motivates) == 1


def test_persistence_roundtrip_and_archive(tmp_path):
    s = ResearchGraphStore(directory=str(tmp_path))
    _init(s)
    s.commit(source="HypothesesAgent",
             nodes=[{"type": "Hypothesis", "attrs": {"formulation": "x"}}])
    before = s.full()
    s2 = ResearchGraphStore(directory=str(tmp_path))  # reload from disk
    after = s2.full()
    assert len(after["nodes"]) == len(before["nodes"])
    assert len(after["edges"]) == len(before["edges"])
    assert s2.root_id() == "Q1"
    # re-init archives the old graph and starts fresh
    r = s2.init_research(source="OrchestratorAgent", question="New question?")
    assert r["ok"] and "archived" in r
    assert list(tmp_path.glob("research_Q1_*.json")), "old graph not archived"


def test_research_generation_ids_are_unique_within_same_second(store, monkeypatch):
    import CoScientist.graph.research.store as store_module

    original_datetime = store_module.datetime

    class FixedDatetime:
        @staticmethod
        def now():
            return original_datetime(2024, 1, 1, 12, 0, 0)

    monkeypatch.setattr(store_module, "datetime", FixedDatetime)
    first = store.init_research(
        source="OrchestratorAgent",
        question="First question?",
    )
    first_id = store.full()["research_id"]
    second = store.init_research(
        source="OrchestratorAgent",
        question="Second question?",
    )
    second_id = store.full()["research_id"]

    assert first["ok"] and second["ok"]
    assert first_id != second_id


def test_no_delete_api():
    for attr in ("delete_node", "delete_edge", "remove_node", "remove_edge"):
        assert not hasattr(ResearchGraphStore, attr)


def test_search_limiter_ignores_research_tools():
    """Regression: the web-search limiter matched "search" as a SUBSTRING, so
    "re-search" tools (research_commit, …) were counted as searches and blocked
    once the cap was hit — which stopped agents recording anything in the graph.
    It must match "search" as a name token instead."""
    from CoScientist.agents.callbacks.tool_callbacks import SearchLimiter

    class _Tool:
        def __init__(self, name): self.name = name

    class _Ctx:
        def __init__(self): self.state = {}

    lim, ctx = SearchLimiter(max_searches=2), _Ctx()
    research_tools = ["research_commit", "research_context_slice", "research_overview",
                      "research_provenance", "research_init", "research_triggers",
                      "research_set_focus"]
    for _ in range(5):
        for name in research_tools:
            assert lim.limit_searches(_Tool(name), {}, ctx) is None
    assert ctx.state.get("_search_limiter_count", 0) == 0  # none counted as a search

    # Real search tools are still capped.
    lim2, ctx2 = SearchLimiter(max_searches=2), _Ctx()
    assert lim2.limit_searches(_Tool("tavily_search"), {}, ctx2) is None
    assert lim2.limit_searches(_Tool("download_papers_from_search"), {}, ctx2) is None
    blocked = lim2.limit_searches(_Tool("search_papers"), {}, ctx2)
    assert blocked is not None and "limit" in blocked["result"].lower()


# ── triggers ────────────────────────────────────────────────────────────────────

def _build_verifiable(store):
    _init(store)
    store.commit(
        source="HypothesesAgent",
        nodes=[{"type": "Hypothesis", "ref": "h", "attrs": {"formulation": "X binds Y"}},
               {"type": "VerificationMethod", "ref": "vm", "attrs": {"method_type": "computational"}},
               {"type": "ConfirmationCriteria", "ref": "cc", "attrs": {"threshold": "<-8"}}],
        edges=[{"type": "motivates", "from": "Q1", "to": "#h"},
               {"type": "tested_by", "from": "#h", "to": "#vm"},
               {"type": "formulated_for", "from": "#cc", "to": "#h"},
               {"type": "requires", "from": "#h", "to": "T1"},
               {"type": "uses", "from": "#vm", "to": "T1"}],
    )


def test_ready_trigger(store):
    _build_verifiable(store)
    ready = queries.ready_hypotheses(store)["items"]
    assert [i["hypothesis"] for i in ready] == ["H1"]


def test_blocked_trigger(store):
    _build_verifiable(store)
    # a hypothesis requiring a tool that is still being created is blocked, not ready
    store.commit(source="CoderAgent", nodes=[{"type": "Tool", "ref": "t2",
                 "status": "being_created", "attrs": {"name": "custom"}}])
    store.commit(source="HypothesesAgent",
                 nodes=[{"type": "Hypothesis", "ref": "h2", "attrs": {"formulation": "needs custom"}}],
                 edges=[{"type": "requires", "from": "#h2", "to": "T2"}])
    blocked = [i["hypothesis"] for i in queries.blocked_hypotheses(store)["items"]]
    assert "H2" in blocked
    assert "H1" not in blocked


def test_closable_and_missing_criteria(store):
    _build_verifiable(store)
    store.commit(source="OrchestratorAgent",
                 status_updates=[{"id": "H1", "status": "under_verification"}])
    store.commit(source="ExperimentAgent",
                 nodes=[{"type": "Evidence", "ref": "e", "attrs": {"subtype": "computational",
                         "content": "docking -9"}}],
                 edges=[{"type": "supports", "from": "#e", "to": "H1"}])
    # CC not met yet → awaiting, not closable
    res = queries.closable_hypotheses(store)
    assert not res["items"]
    assert [i["hypothesis"] for i in res["awaiting_criteria"]] == ["H1"]
    # meet CC → closable. Criteria transitions belong to the ValidatorAgent now,
    # not the orchestrator (verdict/criteria are the judge's job).
    assert not store.commit(source="OrchestratorAgent",
                            status_updates=[{"id": "CC1", "status": "met"}]).ok
    store.commit(source="ValidatorAgent", status_updates=[{"id": "CC1", "status": "met"}])
    res2 = queries.closable_hypotheses(store)
    assert [i["hypothesis"] for i in res2["items"]] == ["H1"]


def test_refuting_evidence_trigger(store):
    _build_verifiable(store)
    store.commit(source="OrchestratorAgent",
                 status_updates=[{"id": "H1", "status": "under_verification"}])
    store.commit(source="ResearchAgent",
                 nodes=[{"type": "Evidence", "ref": "e", "attrs": {"subtype": "literature",
                         "content": "contradicts"}}],
                 edges=[{"type": "refutes", "from": "#e", "to": "H1"}])
    items = queries.refuting_evidence(store)["items"]
    assert items and items[0]["hypothesis"] == "H1"


def test_resources_low_trigger(store):
    _init(store)
    store.commit(source="OrchestratorAgent",
                 nodes=[{"id": "R1", "attrs": {"remaining": 5, "limit": 100}}])
    low = [i["resource"] for i in queries.resources_low(store)["items"]]
    assert "R1" in low


def test_provenance_to_root(store):
    _build_verifiable(store)
    prov = store.get_provenance("VM1")
    assert prov["root"] == "Q1"
    ids = [n["id"] for n in prov["chain"]]
    assert ids[0] == "VM1" and ids[-1] == "Q1"


# ── deterministic maintainer + async background validator ───────────────────────

def test_maintainer_auto_advances_hypothesis_on_evidence(store):
    """Store invariant (deterministic, no LLM): attaching supporting/refuting
    evidence to a `formulated` hypothesis auto-moves it to `under_verification`,
    attributed to the graph-maintainer — so the graph looks live for free."""
    _init(store)
    store.commit(source="HypothesesAgent",
                 nodes=[{"type": "Hypothesis", "ref": "h", "attrs": {"formulation": "x"}}])
    r = store.commit(source="ResearchAgent",
                     nodes=[{"type": "Evidence", "ref": "e", "attrs": {"subtype": "literature",
                             "content": "backs it"}}],
                     edges=[{"type": "supports", "from": "#e", "to": "H1"}])
    assert r.ok, r.errors
    h1 = next(n for n in store.full()["nodes"] if n["id"] == "H1")
    assert h1["status"] == "under_verification"
    assert any(u.get("auto") for u in r.committed["status_updates"])
    assert h1["status_history"][-1]["source"] == "graph-maintainer"


def test_evidence_text_falls_back_to_structured_attrs():
    """Bug #2 regression: an Evidence node with no `content` (e.g. bibliographic
    title/authors/journal/year, or computational mean_macro_f1-style metrics)
    must not render as blank to the validator's LLM prompt."""
    import CoScientist.graph.research.validator as V

    with_content = {"attrs": {"subtype": "literature", "content": "strong support"}}
    assert V._evidence_text(with_content) == "strong support"

    bibliographic = {"attrs": {"subtype": "literature", "title": "A2A knockout study",
                               "authors": "Huang et al.", "year": 2005}}
    text = V._evidence_text(bibliographic)
    assert text and text != ""
    assert "A2A knockout study" in text and "Huang et al." in text and "2005" in text
    assert "subtype" not in text

    computational = {"attrs": {"subtype": "computational", "mean_macro_f1": 0.9664}}
    assert "0.9664" in V._evidence_text(computational)

    empty = {"attrs": {"subtype": "literature"}}
    assert V._evidence_text(empty) == ""


def test_build_user_surfaces_evidence_without_content_field(store):
    """End-to-end through _build_user: an Evidence node written with only
    structured fields (no `content`) still shows up with real text in the
    prompt the judge reads — not as an empty evidence line."""
    import CoScientist.graph.research.validator as V

    _build_verifiable(store)
    store.commit(source="ResearchAgent",
                 nodes=[{"type": "Evidence", "ref": "e", "attrs": {
                     "subtype": "computational", "mean_macro_f1": 0.9664,
                     "approach": "MiniROCKET + Ridge"}}],
                 edges=[{"type": "supports", "from": "#e", "to": "H1"}])
    sl = store.get_context_slice("H1", depth=2)
    prompt = V._build_user(sl, "H1")
    assert "0.9664" in prompt
    assert "MiniROCKET" in prompt


def test_background_validator_judges_hypothesis(store):
    """The async validator (with an injected fake LLM) turns an under_verification
    hypothesis with evidence into a verdict + Conclusion, committed as ValidatorAgent."""
    import asyncio
    import CoScientist.graph.research.validator as V

    _build_verifiable(store)
    store.commit(source="ResearchAgent",
                 nodes=[{"type": "Evidence", "ref": "e", "attrs": {"subtype": "literature",
                         "content": "strong support"}}],
                 edges=[{"type": "supports", "from": "#e", "to": "H1"}])
    assert next(n for n in store.full()["nodes"] if n["id"] == "H1")["status"] == "under_verification"

    async def fake_complete(system, user):
        return ('{"verdict":"confirmed","criteria":{"CC1":"met"},'
                '"conclusion":"X binds Y","validity_bounds":"in vitro","reason":"E1 meets CC1"}')

    orig = V.research_graph
    V.research_graph = store
    try:
        res = asyncio.run(V.judge_hypothesis("H1", complete=fake_complete))
    finally:
        V.research_graph = orig

    assert res and res["ok"], res
    nodes = {n["id"]: n for n in store.full()["nodes"]}
    assert nodes["H1"]["status"] == "confirmed"
    assert nodes["CC1"]["status"] == "met"
    assert any(n["type"] == "Conclusion" and n["source"] == "ValidatorAgent"
               for n in nodes.values())


def test_focus_autolink_relates_evidence_to_hypothesis(store):
    """Option A: Evidence committed with a focus hypothesis but no explicit link
    is auto-attached (relates_to) to that hypothesis, and the maintainer then
    advances the hypothesis to under_verification."""
    _build_verifiable(store)  # H1 formulated
    r = store.commit(source="ResearchAgent",
                     nodes=[{"type": "Evidence", "ref": "e",
                             "attrs": {"subtype": "literature", "content": "a finding"}}],
                     autolink_focus="H1")
    assert r.ok, r.errors
    edges = store.full()["edges"]
    assert any(e["type"] == "relates_to" and e["from"] == "E1" and e["to"] == "H1"
               for e in edges)
    h1 = next(n for n in store.full()["nodes"] if n["id"] == "H1")
    assert h1["status"] == "under_verification"


def test_validator_assigns_polarity_to_autolinked_evidence(store):
    """The validator turns an auto-linked (polarity-unknown) relates_to evidence
    into a supports/refutes edge as part of its verdict."""
    import asyncio
    import CoScientist.graph.research.validator as V

    _build_verifiable(store)
    store.commit(source="ResearchAgent",
                 nodes=[{"type": "Evidence", "ref": "e",
                         "attrs": {"subtype": "literature", "content": "supports it"}}],
                 autolink_focus="H1")

    async def fake(system, user):
        return ('{"evidence":{"E1":"supports"},"verdict":"confirmed",'
                '"criteria":{"CC1":"met"},"conclusion":"c","validity_bounds":"b","reason":"r"}')

    orig = V.research_graph
    V.research_graph = store
    try:
        res = asyncio.run(V.judge_hypothesis("H1", complete=fake))
    finally:
        V.research_graph = orig

    assert res and res["ok"], res
    full = store.full()
    assert any(e["type"] == "supports" and e["from"] == "E1" and e["to"] == "H1"
               for e in full["edges"])
    nodes = {n["id"]: n for n in full["nodes"]}
    assert nodes["H1"]["status"] == "confirmed" and nodes["CC1"]["status"] == "met"
    assert any(n["type"] == "Conclusion" for n in full["nodes"])


def test_validator_surfaces_evolve_gap_on_postponed(store):
    """A postponed verdict that names a concrete gap gets it recorded on the
    Conclusion (evolve_recommended/evolve_gap) — the signal EvolutionAgent
    acts on, per graph/research/evolution.py's design."""
    import asyncio
    import CoScientist.graph.research.validator as V

    _build_verifiable(store)
    store.commit(source="ResearchAgent",
                 nodes=[{"type": "Evidence", "ref": "e", "attrs": {"subtype": "literature",
                         "content": "direction is right, no numbers"}}],
                 edges=[{"type": "supports", "from": "#e", "to": "H1"}])

    async def fake_complete(system, user):
        return ('{"verdict":"postponed","criteria":{"CC1":"not_met"},'
                '"conclusion":"supportive but no IC50 value","validity_bounds":"in vitro",'
                '"reason":"missing quantitative threshold",'
                '"evolve":{"recommended":true,"gap":"IC50 value for X against Y"}}')

    orig = V.research_graph
    V.research_graph = store
    try:
        res = asyncio.run(V.judge_hypothesis("H1", complete=fake_complete))
    finally:
        V.research_graph = orig

    assert res and res["ok"], res
    nodes = {n["id"]: n for n in store.full()["nodes"]}
    assert nodes["H1"]["status"] == "postponed"
    concl = next(n for n in nodes.values() if n["type"] == "Conclusion")
    assert concl["attrs"]["evolve_recommended"] is True
    assert concl["attrs"]["evolve_gap"] == "IC50 value for X against Y"


def test_validator_ignores_recommended_flag_without_a_named_gap(store):
    """Defensive parsing: recommended=true with an empty gap string is NOT
    trusted — there is nothing concrete for EvolutionAgent to search for."""
    import asyncio
    import CoScientist.graph.research.validator as V

    _build_verifiable(store)
    store.commit(source="ResearchAgent",
                 nodes=[{"type": "Evidence", "ref": "e", "attrs": {"subtype": "literature",
                         "content": "weak, vague"}}],
                 edges=[{"type": "supports", "from": "#e", "to": "H1"}])

    async def fake_complete(system, user):
        return ('{"verdict":"postponed","criteria":{"CC1":"not_met"},'
                '"conclusion":"too vague to evolve","validity_bounds":"—",'
                '"reason":"insufficient evidence",'
                '"evolve":{"recommended":true,"gap":""}}')

    orig = V.research_graph
    V.research_graph = store
    try:
        res = asyncio.run(V.judge_hypothesis("H1", complete=fake_complete))
    finally:
        V.research_graph = orig

    assert res and res["ok"], res
    concl = next(n for n in store.full()["nodes"] if n["type"] == "Conclusion")
    assert concl["attrs"]["evolve_recommended"] is False
    assert "evolve_gap" not in concl["attrs"]


def test_evolvable_hypotheses_trigger(store):
    """queries.evolvable_hypotheses surfaces a postponed hypothesis whose
    Conclusion named a gap, and stops surfacing it once a child hypothesis
    has evolved_from it (don't re-fire on an already-rescued branch)."""
    import asyncio
    import CoScientist.graph.research.validator as V

    _build_verifiable(store)
    store.commit(source="ResearchAgent",
                 nodes=[{"type": "Evidence", "ref": "e", "attrs": {"subtype": "literature",
                         "content": "direction is right, no numbers"}}],
                 edges=[{"type": "supports", "from": "#e", "to": "H1"}])

    async def fake_complete(system, user):
        return ('{"verdict":"postponed","criteria":{"CC1":"not_met"},'
                '"conclusion":"c","validity_bounds":"b","reason":"r",'
                '"evolve":{"recommended":true,"gap":"IC50 value for X against Y"}}')

    orig = V.research_graph
    V.research_graph = store
    try:
        asyncio.run(V.judge_hypothesis("H1", complete=fake_complete))
    finally:
        V.research_graph = orig

    items = queries.evolvable_hypotheses(store)["items"]
    assert [i["hypothesis"] for i in items] == ["H1"]
    assert items[0]["gap"] == "IC50 value for X against Y"

    r = store.commit(
        source="EvolutionAgent",
        nodes=[{"type": "Hypothesis", "ref": "h2", "attrs": {"formulation": "narrower claim"}},
               {"type": "ConfirmationCriteria", "ref": "cc2", "attrs": {"threshold": "qualitative"}}],
        edges=[{"type": "motivates", "from": "Q1", "to": "#h2"},
               {"type": "evolved_from", "from": "#h2", "to": "H1"},
               {"type": "formulated_for", "from": "#cc2", "to": "#h2"}],
    )
    assert r.ok, r.errors
    assert queries.evolvable_hypotheses(store)["items"] == []


def test_evolution_agent_permissions(store):
    """EvolutionAgent may create Hypothesis/ConfirmationCriteria/Evidence and
    link evolved_from — but not judge (no ValidatorAgent-only transitions)."""
    errs = schema.validate_edge("EvolutionAgent", "evolved_from", "Hypothesis", "Hypothesis")
    assert errs == []
    errs = schema.validate_transition("EvolutionAgent", "Hypothesis",
                                      "under_verification", "confirmed")
    assert errs


def test_is_restatement_matches_on_shared_source_ref_even_without_id_mention():
    """Two evidence items citing the exact same source_ref are a restatement
    of each other even when neither text names the other's node id — the
    other detection path (explicit id citation) is covered by
    test_duplicate_evidence_flagged_but_does_not_block_ordinary_hypothesis."""
    import CoScientist.graph.research.validator as V

    a = {"id": "E5", "attrs": {"content": "Scaffold X is most frequent.",
                                "source_ref": "gsk3b_classified.csv"}}
    b = {"id": "E6", "attrs": {"content": "Scaffold X dominates the set.",
                                "source_ref": "gsk3b_classified.csv"}}
    assert V._is_restatement(a, b)
    assert V._is_restatement(b, a)

    c = {"id": "E7", "attrs": {"content": "Unrelated finding.",
                                "source_ref": "other_file.csv"}}
    assert not V._is_restatement(a, c)


def test_is_restatement_does_not_flag_ordinary_citation_among_several():
    """Citing another evidence id in passing, alongside other citations, as
    background context is ordinary scholarly citation — not a restatement.
    Regression for a false positive found replaying real data: an evidence
    item's write-up mentioned "and E7" among its cited rationale and was
    wrongly flagged as duplicating E7 before the cue-based narrowing (a bare
    "any mention of the id" check can't tell a citation list from a redo)."""
    import CoScientist.graph.research.validator as V

    a = {"id": "E12", "attrs": {
        "content": "New RDKit SMARTS scaffold counts for the potent subset.",
        "source_ref": "ChEMBL CHEMBL262 SMARTS analysis; structural rationale "
                       "from E2 (product data) and E7 (purine taxonomy)."}}
    b = {"id": "E7", "attrs": {"content": "CHIR-99021 is aminopyrimidine, not purine.",
                                "source_ref": "YeasenBio product description"}}
    assert not V._is_restatement(a, b)
    assert not V._is_restatement(b, a)


def test_confirmed_verdict_downgraded_when_evolved_hypothesis_has_only_birth_evidence(store):
    """Independence check: an evolved hypothesis whose ONLY evidence was
    committed in the SAME batch as the hypothesis itself (i.e. it has never
    faced anything beyond what motivated EvolutionAgent to write it) must not
    be confirmed. The verdict is downgraded to postponed and routed back
    through EvolutionAgent (evolve_recommended) for a genuinely separate
    source — this is the H7/E19/E21 failure mode from a live run."""
    import asyncio
    import CoScientist.graph.research.validator as V

    _build_verifiable(store)
    r = store.commit(
        source="EvolutionAgent",
        nodes=[{"type": "Hypothesis", "ref": "h2", "attrs": {"formulation": "narrower claim"}},
               {"type": "ConfirmationCriteria", "ref": "cc2", "attrs": {"threshold": "qualitative"}},
               {"type": "Evidence", "ref": "e2", "attrs": {"subtype": "computational",
                       "content": "self-computed re-analysis backing the narrower claim"}}],
        edges=[{"type": "motivates", "from": "Q1", "to": "#h2"},
               {"type": "evolved_from", "from": "#h2", "to": "H1"},
               {"type": "formulated_for", "from": "#cc2", "to": "#h2"},
               {"type": "supports", "from": "#e2", "to": "#h2"}],
    )
    assert r.ok, r.errors
    h2_id = next(n["id"] for n in r.committed["nodes"] if n.get("ref") == "h2")
    e2_id = next(n["id"] for n in r.committed["nodes"] if n.get("ref") == "e2")
    cc2_id = next(n["id"] for n in r.committed["nodes"] if n.get("ref") == "cc2")

    async def fake_complete(system, user):
        return ('{"evidence":{"%s":"supports"},"verdict":"confirmed",'
                '"criteria":{"%s":"met"},"conclusion":"looks solid",'
                '"validity_bounds":"b","reason":"evidence supports it"}' % (e2_id, cc2_id))

    orig = V.research_graph
    V.research_graph = store
    try:
        res = asyncio.run(V.judge_hypothesis(h2_id, complete=fake_complete))
    finally:
        V.research_graph = orig

    assert res and res["ok"], res
    nodes = {n["id"]: n for n in store.full()["nodes"]}
    assert nodes[h2_id]["status"] == "postponed"
    concl = next(n for n in nodes.values() if n["type"] == "Conclusion")
    assert concl["attrs"]["independence_check"] == "failed"
    assert concl["attrs"]["evolve_recommended"] is True
    assert "independent" in concl["attrs"]["evolve_gap"].lower()


def test_confirmed_verdict_stays_when_evolved_hypothesis_gains_independent_evidence(store):
    """Once an evolved hypothesis has evidence attached AFTER its own creation
    — a separate commit, unrelated content/source — a confirmed verdict is
    let through unchanged: independent confirmation did happen."""
    import asyncio
    import CoScientist.graph.research.validator as V

    _build_verifiable(store)
    r = store.commit(
        source="EvolutionAgent",
        nodes=[{"type": "Hypothesis", "ref": "h2", "attrs": {"formulation": "narrower claim"}},
               {"type": "ConfirmationCriteria", "ref": "cc2", "attrs": {"threshold": "qualitative"}},
               {"type": "Evidence", "ref": "eb", "attrs": {"subtype": "computational",
                       "content": "birth evidence motivating the narrower claim"}}],
        edges=[{"type": "motivates", "from": "Q1", "to": "#h2"},
               {"type": "evolved_from", "from": "#h2", "to": "H1"},
               {"type": "formulated_for", "from": "#cc2", "to": "#h2"},
               {"type": "supports", "from": "#eb", "to": "#h2"}],
    )
    assert r.ok, r.errors
    h2_id = next(n["id"] for n in r.committed["nodes"] if n.get("ref") == "h2")
    cc2_id = next(n["id"] for n in r.committed["nodes"] if n.get("ref") == "cc2")

    r2 = store.commit(
        source="ResearchAgent",
        nodes=[{"type": "Evidence", "ref": "ei", "attrs": {"subtype": "literature",
                "content": "an unrelated published study reporting the same effect",
                "source_ref": "Smith et al. 2024, J. Med. Chem."}}],
        edges=[{"type": "supports", "from": "#ei", "to": h2_id}],
    )
    assert r2.ok, r2.errors
    ei_id = next(n["id"] for n in r2.committed["nodes"] if n.get("ref") == "ei")

    async def fake_complete(system, user):
        return ('{"evidence":{"%s":"supports"},"verdict":"confirmed",'
                '"criteria":{"%s":"met"},"conclusion":"independently corroborated",'
                '"validity_bounds":"b","reason":"literature confirms it"}' % (ei_id, cc2_id))

    orig = V.research_graph
    V.research_graph = store
    try:
        res = asyncio.run(V.judge_hypothesis(h2_id, complete=fake_complete))
    finally:
        V.research_graph = orig

    assert res and res["ok"], res
    nodes = {n["id"]: n for n in store.full()["nodes"]}
    assert nodes[h2_id]["status"] == "confirmed"
    concl = next(n for n in nodes.values() if n["type"] == "Conclusion")
    assert "independence_check" not in concl["attrs"]


def test_pre_existing_evidence_cited_at_birth_does_not_count_as_seed(store):
    """A hypothesis EvolutionAgent evolves by citing an OLD, already-existing
    Evidence node (found independently, before this hypothesis existed) must
    NOT be treated as circular just because the graph edge linking it was
    necessarily drawn in the same commit as the hypothesis itself — only
    evidence minted at that exact moment (test above) is seed. This is the
    real E7 pattern from a live run: a pre-existing literature fact EvolutionAgent
    cites as structural rationale for a new, narrower claim."""
    import asyncio
    import CoScientist.graph.research.validator as V

    _build_verifiable(store)
    r0 = store.commit(source="ResearchAgent",
                      nodes=[{"type": "Evidence", "ref": "eold", "attrs": {"subtype": "literature",
                              "content": "an independently found background fact",
                              "source_ref": "some 2019 paper"}}],
                      edges=[{"type": "relates_to", "from": "#eold", "to": "H1"}])
    assert r0.ok, r0.errors
    eold_id = next(n["id"] for n in r0.committed["nodes"] if n.get("ref") == "eold")

    r = store.commit(
        source="EvolutionAgent",
        nodes=[{"type": "Hypothesis", "ref": "h2", "attrs": {"formulation": "narrower claim"}},
               {"type": "ConfirmationCriteria", "ref": "cc2", "attrs": {"threshold": "qualitative"}}],
        edges=[{"type": "motivates", "from": "Q1", "to": "#h2"},
               {"type": "evolved_from", "from": "#h2", "to": "H1"},
               {"type": "formulated_for", "from": "#cc2", "to": "#h2"},
               {"type": "refines", "from": eold_id, "to": "#h2"}],
    )
    assert r.ok, r.errors
    h2_id = next(n["id"] for n in r.committed["nodes"] if n.get("ref") == "h2")
    cc2_id = next(n["id"] for n in r.committed["nodes"] if n.get("ref") == "cc2")

    async def fake_complete(system, user):
        return ('{"evidence":{"%s":"refines"},"verdict":"confirmed",'
                '"criteria":{"%s":"met"},"conclusion":"backed by prior finding",'
                '"validity_bounds":"b","reason":"consistent with known fact"}'
                % (eold_id, cc2_id))

    orig = V.research_graph
    V.research_graph = store
    try:
        res = asyncio.run(V.judge_hypothesis(h2_id, complete=fake_complete))
    finally:
        V.research_graph = orig

    assert res and res["ok"], res
    nodes = {n["id"]: n for n in store.full()["nodes"]}
    assert nodes[h2_id]["status"] == "confirmed"
    concl = next(n for n in nodes.values() if n["type"] == "Conclusion")
    assert "independence_check" not in concl["attrs"]


def test_duplicate_evidence_flagged_but_does_not_block_ordinary_hypothesis(store):
    """A restated evidence item (its text names the earlier evidence's id) is
    flagged as duplicate_evidence on the Conclusion for transparency, but the
    hard independence block only applies to EVOLVED hypotheses — an ordinary
    (non-evolved) hypothesis can still be confirmed despite the duplicate."""
    import asyncio
    import CoScientist.graph.research.validator as V

    _build_verifiable(store)
    r1 = store.commit(source="ResearchAgent",
                      nodes=[{"type": "Evidence", "ref": "e1", "attrs": {"subtype": "computational",
                              "content": "IC50 = 5 nM, assay run on batch A",
                              "source_ref": "run_A.csv"}}],
                      edges=[{"type": "supports", "from": "#e1", "to": "H1"}])
    assert r1.ok, r1.errors
    e1_id = next(n["id"] for n in r1.committed["nodes"] if n.get("ref") == "e1")

    r2 = store.commit(source="ResearchAgent",
                      nodes=[{"type": "Evidence", "ref": "e2", "attrs": {"subtype": "computational",
                              "content": f"Re-analysis of {e1_id}: IC50 = 5 nM, same batch A run",
                              "source_ref": "run_A.csv"}}],
                      edges=[{"type": "supports", "from": "#e2", "to": "H1"}])
    assert r2.ok, r2.errors
    e2_id = next(n["id"] for n in r2.committed["nodes"] if n.get("ref") == "e2")

    async def fake_complete(system, user):
        return ('{"evidence":{"%s":"supports","%s":"supports"},"verdict":"confirmed",'
                '"criteria":{"CC1":"met"},"conclusion":"confirmed by assay",'
                '"validity_bounds":"in vitro","reason":"IC50 meets threshold"}'
                % (e1_id, e2_id))

    orig = V.research_graph
    V.research_graph = store
    try:
        res = asyncio.run(V.judge_hypothesis("H1", complete=fake_complete))
    finally:
        V.research_graph = orig

    assert res and res["ok"], res
    nodes = {n["id"]: n for n in store.full()["nodes"]}
    assert nodes["H1"]["status"] == "confirmed"
    concl = next(n for n in nodes.values() if n["type"] == "Conclusion")
    assert concl["attrs"]["duplicate_evidence"] == [e2_id]


def _build_superlative(store, formulation):
    """Like _build_verifiable but with a caller-supplied H1 formulation, for
    the comparator-check tests (needs superlative wording: "dominant" etc.)."""
    _init(store)
    store.commit(
        source="HypothesesAgent",
        nodes=[{"type": "Hypothesis", "ref": "h", "attrs": {"formulation": formulation}},
               {"type": "VerificationMethod", "ref": "vm", "attrs": {"method_type": "computational"}},
               {"type": "ConfirmationCriteria", "ref": "cc", "attrs": {"threshold": "p<0.05"}}],
        edges=[{"type": "motivates", "from": "Q1", "to": "#h"},
               {"type": "tested_by", "from": "#h", "to": "#vm"},
               {"type": "formulated_for", "from": "#cc", "to": "#h"},
               {"type": "requires", "from": "#h", "to": "T1"},
               {"type": "uses", "from": "#vm", "to": "T1"}],
    )


def test_claims_superlative_and_has_named_comparator():
    import CoScientist.graph.research.validator as V

    assert V._claims_superlative("Scaffold X is the dominant heteroaromatic scaffold")
    assert V._claims_superlative("X является самым частым скэффолдом")
    assert not V._claims_superlative("X binds Y with high affinity")

    assert V._has_named_comparator(["X (418) vs runner-up Y (400): p=0.001"])
    assert V._has_named_comparator(["X is the second-most frequent scaffold"])
    assert not V._has_named_comparator(
        ["X: 418/2979, binomial test vs uniform baseline (10%): p=1.97e-12"])


def test_shares_evidence_with_flags_confirmed_sibling_overlap(store):
    """Two nominally independent hypotheses (H1, H2) that both cite the SAME
    evidence node get `shares_evidence_with` on H1's Conclusion when H1 is
    confirmed — regression for a live case: a mechanistic paper (MRAS-SHOC2
    -> MAPK reactivation) cited as supporting BOTH a YAP/TAZ hypothesis and a
    separate NF1/SHOC2 hypothesis, presented in the final report as two
    independent confirmations when they partly rest on the same source."""
    import asyncio
    import CoScientist.graph.research.validator as V

    _build_verifiable(store)
    store.commit(
        source="HypothesesAgent",
        nodes=[{"type": "Hypothesis", "ref": "h2", "attrs": {"formulation": "Y regulates Z"}},
               {"type": "VerificationMethod", "ref": "vm2", "attrs": {"method_type": "computational"}},
               {"type": "ConfirmationCriteria", "ref": "cc2", "attrs": {"threshold": "<-8"}}],
        edges=[{"type": "motivates", "from": "Q1", "to": "#h2"},
               {"type": "tested_by", "from": "#h2", "to": "#vm2"},
               {"type": "formulated_for", "from": "#cc2", "to": "#h2"},
               {"type": "requires", "from": "#h2", "to": "T1"},
               {"type": "uses", "from": "#vm2", "to": "T1"}],
    )
    store.commit(
        source="ResearchAgent",
        nodes=[{"type": "Evidence", "ref": "e", "attrs": {"subtype": "literature",
                "content": "shared mechanistic finding cited by both branches"}}],
        edges=[{"type": "supports", "from": "#e", "to": "H1"},
               {"type": "supports", "from": "#e", "to": "H2"}],
    )

    async def fake_complete(system, user):
        return ('{"verdict":"confirmed","criteria":{"CC1":"met"},'
                '"conclusion":"X binds Y","validity_bounds":"in vitro","reason":"r"}')

    orig = V.research_graph
    V.research_graph = store
    try:
        res = asyncio.run(V.judge_hypothesis("H1", complete=fake_complete))
    finally:
        V.research_graph = orig

    assert res and res["ok"], res
    nodes = {n["id"]: n for n in store.full()["nodes"]}
    concl = next(n for n in nodes.values() if n["type"] == "Conclusion")
    assert concl["attrs"]["shares_evidence_with"] == {"H2": ["E1"]}


def test_shares_evidence_with_absent_when_no_overlap(store):
    """The ordinary case (no shared evidence across hypotheses) gets no flag —
    regression against false positives on every confirmed hypothesis."""
    import asyncio
    import CoScientist.graph.research.validator as V

    _build_verifiable(store)
    store.commit(source="ResearchAgent",
                 nodes=[{"type": "Evidence", "ref": "e", "attrs": {"subtype": "literature",
                         "content": "evidence used by nobody else"}}],
                 edges=[{"type": "supports", "from": "#e", "to": "H1"}])

    async def fake_complete(system, user):
        return ('{"verdict":"confirmed","criteria":{"CC1":"met"},'
                '"conclusion":"X binds Y","validity_bounds":"in vitro","reason":"r"}')

    orig = V.research_graph
    V.research_graph = store
    try:
        res = asyncio.run(V.judge_hypothesis("H1", complete=fake_complete))
    finally:
        V.research_graph = orig

    assert res and res["ok"], res
    concl = next(n for n in store.full()["nodes"] if n["type"] == "Conclusion")
    assert "shares_evidence_with" not in concl["attrs"]


def test_comparator_check_flags_superlative_confirmed_without_head_to_head(store):
    """A hypothesis claiming to be THE dominant scaffold, confirmed only via a
    uniform-baseline test with no head-to-head comparison against the actual
    runner-up, gets comparator_check='missing' on its Conclusion — flagged,
    NOT blocked (verdict stays confirmed; this signal is too fuzzy to hard-
    block on, see _has_named_comparator). Regression for a live case: a
    scaffold "confirmed dominant" via binomial-vs-uniform (p=1.97e-12) whose
    real runner-up was a percentage point behind — a proper head-to-head test
    on that pair came back p≈0.5, not significant at all."""
    import asyncio
    import CoScientist.graph.research.validator as V

    _build_superlative(store, "Scaffold X is the dominant heteroaromatic scaffold")
    store.commit(source="CoderAgent",
                 nodes=[{"type": "Evidence", "ref": "e", "attrs": {"subtype": "computational",
                         "content": "Scaffold X: 418/2979 (14.0%), binomial test vs uniform "
                                    "baseline (10%): p=1.97e-12"}}],
                 edges=[{"type": "supports", "from": "#e", "to": "H1"}])

    async def fake_complete(system, user):
        return ('{"verdict":"confirmed","criteria":{"CC1":"met"},'
                '"conclusion":"X is dominant","validity_bounds":"ChEMBL","reason":"p<0.05"}')

    orig = V.research_graph
    V.research_graph = store
    try:
        res = asyncio.run(V.judge_hypothesis("H1", complete=fake_complete))
    finally:
        V.research_graph = orig

    assert res and res["ok"], res
    nodes = {n["id"]: n for n in store.full()["nodes"]}
    assert nodes["H1"]["status"] == "confirmed"
    concl = next(n for n in nodes.values() if n["type"] == "Conclusion")
    assert concl["attrs"]["comparator_check"] == "missing"


def test_comparator_check_silent_when_head_to_head_is_present(store):
    """The same superlative claim, but evidence DOES frame a head-to-head
    comparison against the named runner-up — no flag."""
    import asyncio
    import CoScientist.graph.research.validator as V

    _build_superlative(store, "Scaffold X is the dominant heteroaromatic scaffold")
    store.commit(source="CoderAgent",
                 nodes=[{"type": "Evidence", "ref": "e", "attrs": {"subtype": "computational",
                         "content": "Scaffold X (418) vs runner-up Scaffold Y (200), "
                                    "Fisher's exact test: p=0.001"}}],
                 edges=[{"type": "supports", "from": "#e", "to": "H1"}])

    async def fake_complete(system, user):
        return ('{"verdict":"confirmed","criteria":{"CC1":"met"},'
                '"conclusion":"X beats its closest competitor","validity_bounds":"ChEMBL",'
                '"reason":"p<0.05 vs runner-up"}')

    orig = V.research_graph
    V.research_graph = store
    try:
        res = asyncio.run(V.judge_hypothesis("H1", complete=fake_complete))
    finally:
        V.research_graph = orig

    assert res and res["ok"], res
    concl = next(n for n in store.full()["nodes"] if n["type"] == "Conclusion")
    assert "comparator_check" not in concl["attrs"]


def test_comparator_check_does_not_fire_on_non_superlative_hypothesis(store):
    """A plain (non-superlative) confirmed hypothesis never gets this flag,
    regardless of what its evidence says."""
    import asyncio
    import CoScientist.graph.research.validator as V

    _build_verifiable(store)
    store.commit(source="ResearchAgent",
                 nodes=[{"type": "Evidence", "ref": "e", "attrs": {"subtype": "literature",
                         "content": "strong support, no baseline language at all"}}],
                 edges=[{"type": "supports", "from": "#e", "to": "H1"}])

    async def fake_complete(system, user):
        return ('{"verdict":"confirmed","criteria":{"CC1":"met"},'
                '"conclusion":"X binds Y","validity_bounds":"in vitro","reason":"r"}')

    orig = V.research_graph
    V.research_graph = store
    try:
        res = asyncio.run(V.judge_hypothesis("H1", complete=fake_complete))
    finally:
        V.research_graph = orig

    assert res and res["ok"], res
    concl = next(n for n in store.full()["nodes"] if n["type"] == "Conclusion")
    assert "comparator_check" not in concl["attrs"]


def test_background_validator_retries_after_failed_judgment(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    import CoScientist.graph.research.validator as V

    class FakeGraph:
        def full(self):
            return {"research_id": "research-1"}

    graph = FakeGraph()
    item = {
        "hypothesis": "H1",
        "supporting": ["E1"],
        "refuting": [],
        "related": [],
    }
    outcomes = [None, {"ok": True}]
    calls = []

    async def fake_judge(hypothesis, *, graph, expected_research_id):
        calls.append((hypothesis, expected_research_id))
        return outcomes.pop(0)

    monkeypatch.setattr(V, "_enabled", lambda: True)
    monkeypatch.setattr(V, "get_research_graph", lambda context: graph)
    monkeypatch.setattr(
        V.queries,
        "unresolved_hypotheses",
        lambda selected_graph: {"items": [item]},
    )
    monkeypatch.setattr(V, "judge_hypothesis", fake_judge)
    plugin = V.BackgroundValidatorPlugin()
    tool = SimpleNamespace(name="research_commit")

    async def trigger():
        before = set(V._TASKS)
        await plugin.after_tool_callback(
            tool=tool,
            tool_args={},
            tool_context=object(),
            result={},
        )
        scheduled = set(V._TASKS) - before
        if scheduled:
            await asyncio.gather(*scheduled)
        return len(scheduled)

    async def scenario():
        # The duplicate callback while the first task is still in flight must
        # not schedule another LLM call.
        before = set(V._TASKS)
        await plugin.after_tool_callback(
            tool=tool,
            tool_args={},
            tool_context=object(),
            result={},
        )
        await plugin.after_tool_callback(
            tool=tool,
            tool_args={},
            tool_context=object(),
            result={},
        )
        first_tasks = set(V._TASKS) - before
        assert len(first_tasks) == 1
        await asyncio.gather(*first_tasks)

        assert await trigger() == 1  # retry after the failed first result
        assert await trigger() == 0  # successful signature is now completed

    asyncio.run(scenario())
    assert calls == [("H1", "research-1"), ("H1", "research-1")]


def test_wait_for_validator_settle_reschedules_orphaned_hypothesis(monkeypatch):
    """A hypothesis stuck `under_verification` with no further research_commit
    to re-trigger it (see wait_for_validator_settle's docstring — the real
    case this is a regression test for: status_history stopping dead at
    "auto: evidence attached") gets picked up and resolved by the
    ResultAggregatorAgent's settle callback, without waiting for the full
    timeout."""
    import asyncio
    import CoScientist.graph.research.validator as V

    class FakeGraph:
        def __init__(self):
            self.settled = False

        def full(self):
            return {"research_id": "research-1"}

        def overview(self):
            return {"counts": {"Hypothesis": {"under_verification": 0 if self.settled else 1}}}

    graph = FakeGraph()
    item = {"hypothesis": "H6", "supporting": ["E1"], "refuting": [], "related": []}

    async def fake_judge(hypothesis, *, graph, expected_research_id):
        graph.settled = True
        return {"ok": True}

    monkeypatch.setattr(V, "_enabled", lambda: True)
    monkeypatch.setattr(V, "get_research_graph", lambda ctx: graph)
    monkeypatch.setattr(V.queries, "unresolved_hypotheses", lambda g: {"items": [item]})
    monkeypatch.setattr(V, "judge_hypothesis", fake_judge)
    monkeypatch.setattr(V, "_SETTLE_POLL", 0.01)
    monkeypatch.setattr(V, "_SETTLE_TIMEOUT", 2.0)
    monkeypatch.setattr(V, "background_validator_plugin", V.BackgroundValidatorPlugin())

    asyncio.run(asyncio.wait_for(V.wait_for_validator_settle(object()), timeout=1.0))
    assert graph.settled is True


def test_wait_for_validator_settle_gives_up_after_timeout(monkeypatch):
    """A hypothesis that never settles (e.g. the LLM judgment keeps failing)
    must not hang the report forever — the callback gives up at the bound
    and lets the (stricter) result_aggregator prompt flag it as preliminary."""
    import asyncio
    import CoScientist.graph.research.validator as V

    class FakeGraph:
        def full(self):
            return {"research_id": "research-1"}

        def overview(self):
            return {"counts": {"Hypothesis": {"under_verification": 1}}}

    graph = FakeGraph()

    monkeypatch.setattr(V, "_enabled", lambda: True)
    monkeypatch.setattr(V, "get_research_graph", lambda ctx: graph)
    monkeypatch.setattr(V.queries, "unresolved_hypotheses", lambda g: {"items": []})
    monkeypatch.setattr(V, "_SETTLE_POLL", 0.01)
    monkeypatch.setattr(V, "_SETTLE_TIMEOUT", 0.03)
    monkeypatch.setattr(V, "background_validator_plugin", V.BackgroundValidatorPlugin())

    asyncio.run(asyncio.wait_for(V.wait_for_validator_settle(object()), timeout=1.0))


def test_background_validator_dedup_tracks_related_evidence_and_research_id(
    monkeypatch,
):
    import asyncio
    from types import SimpleNamespace
    import CoScientist.graph.research.validator as V

    class FakeGraph:
        research_id = "research-1"

        def full(self):
            return {"research_id": self.research_id}

    graph = FakeGraph()
    item = {
        "hypothesis": "H1",
        "supporting": ["E1"],
        "refuting": [],
        "related": [],
    }
    calls = []

    async def fake_judge(hypothesis, *, graph, expected_research_id):
        calls.append((hypothesis, expected_research_id, tuple(item["related"])))
        return {"ok": True}

    monkeypatch.setattr(V, "_enabled", lambda: True)
    monkeypatch.setattr(V, "get_research_graph", lambda context: graph)
    monkeypatch.setattr(
        V.queries,
        "unresolved_hypotheses",
        lambda selected_graph: {"items": [item]},
    )
    monkeypatch.setattr(V, "judge_hypothesis", fake_judge)
    plugin = V.BackgroundValidatorPlugin()
    tool = SimpleNamespace(name="research_commit")

    async def trigger():
        before = set(V._TASKS)
        await plugin.after_tool_callback(
            tool=tool,
            tool_args={},
            tool_context=object(),
            result={},
        )
        scheduled = set(V._TASKS) - before
        if scheduled:
            await asyncio.gather(*scheduled)

    async def scenario():
        await trigger()
        await trigger()
        item["related"] = ["E2"]
        await trigger()
        graph.research_id = "research-2"
        await trigger()

    asyncio.run(scenario())
    assert calls == [
        ("H1", "research-1", ()),
        ("H1", "research-1", ("E2",)),
        ("H1", "research-2", ("E2",)),
    ]


def test_validator_discards_result_if_research_changes_during_llm_call():
    import asyncio
    import CoScientist.graph.research.validator as V

    class FakeGraph:
        research_id = "research-1"
        committed = False

        def full(self):
            return {"research_id": self.research_id}

        def get_context_slice(self, hypothesis, depth):
            return {
                "nodes": [{
                    "id": "H1",
                    "type": "Hypothesis",
                    "status": "under_verification",
                    "attrs": {"formulation": "x"},
                }],
                "edges": [],
            }

        def commit(self, **kwargs):
            self.committed = True
            raise AssertionError("a stale verdict must not be committed")

    graph = FakeGraph()

    async def fake_complete(system, user):
        graph.research_id = "research-2"
        return '{"verdict":"postponed","reason":"insufficient evidence"}'

    result = asyncio.run(V.judge_hypothesis(
        "H1",
        complete=fake_complete,
        graph=graph,
        expected_research_id="research-1",
    ))

    assert result is None
    assert not graph.committed
