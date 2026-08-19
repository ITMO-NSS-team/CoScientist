"""graph_bridge: approved plan → VM/tested_by; result → Evidence/GeneratedData."""
from __future__ import annotations

from types import SimpleNamespace

from CoScientist.graph.research.store import ResearchGraphStore

from .helpers import _approved_state, _plan, _task


def _seeded_store(tmp_path) -> ResearchGraphStore:
    """A research graph with a root question and one Hypothesis node (H1)."""
    store = ResearchGraphStore(directory=str(tmp_path))
    store.init_research(source="ContextInitAgent", question="Root question?")
    root = store.root_id()
    store.commit(
        source="HypothesesAgent",
        nodes=[{"type": "Hypothesis", "ref": "h1",
                "attrs": {"formulation": "Compound X inhibits target Y."}}],
        edges=[{"type": "motivates", "from": root, "to": "#h1"}],
        enforce_permissions=False,
    )
    return store


def _nodes_by_type(store: ResearchGraphStore) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for node in store.full()["nodes"]:
        out.setdefault(node["type"], []).append(node["id"])
    return out


def test_publish_plan_creates_vm_and_tested_by(tmp_path):
    from CoScientist.experiments.runtime.graph_bridge import publish_plan_to_graph

    store = _seeded_store(tmp_path)
    state = _approved_state(_plan(_task("EXP-1", hypothesis_ref="H1")))

    publish_plan_to_graph(store, state)

    by_type = _nodes_by_type(store)
    assert len(by_type.get("VerificationMethod", [])) == 1
    vm_id = by_type["VerificationMethod"][0]
    assert state["experiment_graph_vm_ids"]["EXP-1"] == vm_id
    edges = store.full()["edges"]
    assert any(e["type"] == "tested_by" and e["from"] == "H1" and e["to"] == vm_id
               for e in edges)


def test_publish_plan_is_idempotent(tmp_path):
    from CoScientist.experiments.runtime.graph_bridge import publish_plan_to_graph

    store = _seeded_store(tmp_path)
    state = _approved_state(_plan(_task("EXP-1", hypothesis_ref="H1")))

    publish_plan_to_graph(store, state)
    first_vm = dict(state["experiment_graph_vm_ids"])
    publish_plan_to_graph(store, state)

    assert _nodes_by_type(store).get("VerificationMethod") == [first_vm["EXP-1"]]
    assert state["experiment_graph_vm_ids"] == first_vm


def test_publish_result_writes_evidence_and_vm_status(tmp_path):
    from CoScientist.experiments.runtime.graph_bridge import (
        publish_plan_to_graph,
        publish_result_to_graph,
    )

    store = _seeded_store(tmp_path)
    state = _approved_state(_plan(_task("EXP-1", hypothesis_ref="H1")))
    publish_plan_to_graph(store, state)
    vm_id = state["experiment_graph_vm_ids"]["EXP-1"]

    task_result = {
        "result_id": "RES-1",
        "status": "success",
        "summary": "EXP-1 produced a managed result.",
        "artifacts": [{"name": "metrics.json", "bucket": "b", "s3_key": "k/metrics.json"}],
    }
    publish_result_to_graph(store, state, "EXP-1", task_result)

    by_type = _nodes_by_type(store)
    assert len(by_type.get("Evidence", [])) == 1
    assert len(by_type.get("GeneratedData", [])) == 1
    evidence_id = by_type["Evidence"][0]
    gd_id = by_type["GeneratedData"][0]
    edges = store.full()["edges"]
    assert any(e["type"] == "produces" and e["from"] == vm_id and e["to"] == evidence_id
               for e in edges)
    assert any(e["type"] == "derived_from" and e["from"] == gd_id and e["to"] == evidence_id
               for e in edges)
    vm_status = {n["id"]: n["status"] for n in store.full()["nodes"]}[vm_id]
    assert vm_status == "done"


def test_publish_result_failure_marks_vm_failed(tmp_path):
    from CoScientist.experiments.runtime.graph_bridge import (
        publish_plan_to_graph,
        publish_result_to_graph,
    )

    store = _seeded_store(tmp_path)
    state = _approved_state(_plan(_task("EXP-1", hypothesis_ref="H1")))
    publish_plan_to_graph(store, state)
    vm_id = state["experiment_graph_vm_ids"]["EXP-1"]

    publish_result_to_graph(store, state, "EXP-1",
                            {"result_id": "RES-2", "status": "failure", "summary": "boom"})

    by_type = _nodes_by_type(store)
    assert by_type.get("Evidence") is None
    vm_status = {n["id"]: n["status"] for n in store.full()["nodes"]}[vm_id]
    assert vm_status == "failed"


def test_publish_result_skips_when_no_vm(tmp_path):
    from CoScientist.experiments.runtime.graph_bridge import publish_result_to_graph

    store = _seeded_store(tmp_path)
    state: dict = {}
    publish_result_to_graph(store, state, "EXP-1",
                            {"result_id": "RES-3", "status": "success", "summary": "x"})
    assert _nodes_by_type(store).get("Evidence") is None


def test_publish_plan_swallows_store_errors():
    from CoScientist.experiments.runtime.graph_bridge import (
        publish_plan_to_graph,
        publish_result_to_graph,
    )

    class _Boom:
        def overview(self):
            raise RuntimeError("graph down")

        def commit(self, *a, **k):
            raise RuntimeError("graph down")

    state = _approved_state(_plan(_task("EXP-1", hypothesis_ref="H1")))
    # Must not raise — best-effort contract.
    publish_plan_to_graph(_Boom(), state)
    state["experiment_graph_vm_ids"] = {"EXP-1": "VM1"}
    publish_result_to_graph(_Boom(), state, "EXP-1",
                            {"result_id": "RES-4", "status": "success", "summary": "x"})
