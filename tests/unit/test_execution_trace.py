"""The execution graph regrouped as a chronological trace.

The call graph shows what is connected to what; these pin down what the trace
adds — one entry per user prompt, its calls in the order they ran, and the
input, output and duration of each.
"""
from CoScientist.graph.projection import turns


def _full():
    return {"nodes": [
        {"id": "goal:a", "kind": "goal", "turn_id": "a", "label": "first ask",
         "status": "success", "t_start": 100.0, "t_end": 140.0},
        {"id": "tool:2", "kind": "tool_call", "turn_id": "a", "label": "execute_bash",
         "executor_agent": "CoderAgent", "status": "failed", "input": "ls",
         "output": "boom", "t_start": 110.0, "t_end": 110.4},
        {"id": "tool:1", "kind": "tool_call", "turn_id": "a", "label": "tavily_search",
         "executor_agent": "ResearchAgent", "status": "success", "input": "{}",
         "output": "12 papers", "t_start": 101.0, "t_end": 103.5},
        {"id": "result:a", "kind": "result", "turn_id": "a", "output": "done",
         "t_start": 140.0, "t_end": 140.0},
        {"id": "goal:b", "kind": "goal", "turn_id": "b", "label": "second ask",
         "status": "running", "t_start": 200.0},
        {"id": "agent:CoderAgent", "kind": "agent", "status": "success"},
    ]}


def test_turns_are_separated_and_ordered_by_time():
    result = turns(_full())
    assert [t["turn_id"] for t in result["turns"]] == ["a", "b"]
    assert result["turns"][0]["prompt"] == "first ask"
    # Calls come back in the order they ran, not the order they were recorded.
    assert [c["tool"] for c in result["turns"][0]["calls"]] == [
        "tavily_search", "execute_bash"]


def test_a_call_carries_its_agent_arguments_result_and_duration():
    call = turns(_full())["turns"][0]["calls"][0]
    assert call["agent"] == "ResearchAgent"
    assert call["input"] == "{}" and call["output"] == "12 papers"
    assert call["status"] == "success" and call["duration"] == 2.5


def test_agent_nodes_are_not_turns():
    # One agent node serves the whole session, so it belongs to no single turn
    # and must not become one.
    assert all(t["turn_id"] != "agent:CoderAgent" for t in turns(_full())["turns"])


def test_snapshots_written_before_turn_id_still_group():
    legacy = {"nodes": [
        {"id": "goal:old", "kind": "goal", "label": "before the field existed",
         "status": "success", "t_start": 10.0, "t_end": 20.0},
        {"id": "result:old", "kind": "result", "output": "answer", "t_start": 20.0},
    ]}
    result = turns(legacy)
    assert result["count"] == 1
    assert result["turns"][0]["turn_id"] == "old"
    assert result["turns"][0]["answer"] == "answer"


def test_a_call_with_no_turn_is_kept_not_dropped():
    orphan = {"nodes": [{"id": "tool:x", "kind": "tool_call", "label": "t",
                         "status": "success", "t_start": 1.0}]}
    result = turns(orphan)
    assert result["count"] == 1 and result["turns"][0]["turn_id"] == "untagged"


def test_trace_view_is_served_and_grouped(tmp_path, monkeypatch):
    """The page and its data source exist and agree on the shape."""
    import time
    import uuid

    monkeypatch.setenv("GRAPH_SNAPSHOT_DIR", str(tmp_path / "graphs"))
    monkeypatch.setenv("WEB_STATE_DIR", str(tmp_path / "web"))
    from starlette.testclient import TestClient

    from CoScientist.graph.memory import get_knowledge_graph
    from CoScientist.web.app import create_app

    with TestClient(create_app()) as client:
        assert "Sessions" in client.get("/trace").text

        user = client.post(
            "/api/users", json={"nickname": f"t-{uuid.uuid4().hex[:8]}"}
        ).json()["user"]
        session = client.post(
            f"/api/users/{user['id']}/sessions", json={"title": "run"}
        ).json()["session"]

        graph = get_knowledge_graph(user_id=user["id"], session_id=session["id"])
        now = time.time()
        graph.add_node(id="goal:i1", kind="goal", turn_id="i1", label="do the thing",
                       status="success", t_start=now, t_end=now + 9)
        graph.add_node(id="tool:1", kind="tool_call", turn_id="i1", label="search",
                       executor_agent="ResearchAgent", status="success",
                       input="{}", output="hits", t_start=now + 1, t_end=now + 3)

        payload = client.get(
            f"/api/users/{user['id']}/sessions/{session['id']}/graph",
            params={"view": "trace"},
        ).json()

    assert payload["count"] == 1
    turn = payload["turns"][0]
    assert turn["prompt"] == "do the thing"
    assert [c["tool"] for c in turn["calls"]] == ["search"]
    assert turn["calls"][0]["agent"] == "ResearchAgent"


def test_graph_opens_empty_and_grows_only_with_what_ran(tmp_path, monkeypatch):
    """No roster in the trace: an agent node exists because that agent acted."""
    monkeypatch.setenv("GRAPH_SNAPSHOT_DIR", str(tmp_path))
    from CoScientist.graph.memory import KnowledgeGraph

    graph = KnowledgeGraph(run_id="execution")

    fresh = graph.full()
    assert [n["id"] for n in fresh["nodes"]] == ["system:root"]
    assert fresh["edges"] == []
    # The roster still resolves for prompts and for get_agents_info.
    assert len(graph.agents_info()) > 1

    graph.add_node(id="agent:CoderAgent", kind="agent", label="CoderAgent",
                   executor_agent="CoderAgent", status="success")
    acted = {n["id"] for n in graph.full()["nodes"] if n["kind"] == "agent"}
    assert acted == {"agent:CoderAgent"}


def test_legacy_snapshots_group_by_goal_and_by_time():
    """Older records carry no turn_id: namespaced ids and chronology recover it."""
    from CoScientist.graph.projection import turns as trace_by_turn

    full = {"nodes": [
        # Namespaced ids, the form used before turn_id existed.
        {"id": "goal:inv1", "kind": "goal", "label": "first", "t_start": 100.0,
         "t_end": 130.0, "status": "success"},
        {"id": "goal:inv1::tool:a", "kind": "tool_call", "label": "search",
         "t_start": 101.0, "t_end": 104.0},
        # No marker at all: it ran while the first request was open.
        {"id": "tool:loose", "kind": "tool_call", "label": "read",
         "t_start": 110.0, "t_end": 111.0},
        {"id": "goal:inv2", "kind": "goal", "label": "second", "t_start": 200.0,
         "t_end": 210.0, "status": "success"},
        {"id": "tool:later", "kind": "tool_call", "label": "write",
         "t_start": 205.0, "t_end": 206.0},
    ]}

    turns = trace_by_turn(full)["turns"]
    assert [t["prompt"] for t in turns] == ["first", "second"]
    assert [c["tool"] for c in turns[0]["calls"]] == ["search", "read"]
    assert [c["tool"] for c in turns[1]["calls"]] == ["write"]


def test_execution_tree_drops_the_roster_and_measures_depth():
    """request -> agent -> its calls -> answer, with nothing that never ran."""
    from CoScientist.graph.projection import execution_tree

    full = {"nodes": [
        {"id": "system:root", "kind": "system", "label": "the system"},
        {"id": "agent:NeverCalled", "kind": "agent", "label": "NeverCalled"},
        {"id": "goal:i1", "kind": "goal", "label": "do it", "t_start": 10.0},
        {"id": "goal:i1::agent:Research", "kind": "agent_call",
         "executor_agent": "Research", "t_start": 11.0},
        {"id": "tool:a", "kind": "tool_call", "label": "search", "t_start": 12.0},
        {"id": "tool:b", "kind": "tool_call", "label": "extract", "t_start": 13.0},
        {"id": "result:i1", "kind": "result", "output": "done", "t_start": 20.0},
    ], "edges": [
        {"src": "system:root", "dst": "agent:NeverCalled", "type": "has_member"},
        {"src": "system:root", "dst": "goal:i1", "type": "caused_by"},
        {"src": "goal:i1", "dst": "goal:i1::agent:Research", "type": "caused_by"},
        {"src": "goal:i1::agent:Research", "dst": "tool:a", "type": "caused_by"},
        {"src": "goal:i1::agent:Research", "dst": "tool:b", "type": "caused_by"},
        {"src": "goal:i1::agent:Research", "dst": "result:i1", "type": "produced"},
    ]}

    tree = execution_tree(full)
    ids = {n["id"] for n in tree["nodes"]}
    assert "agent:NeverCalled" not in ids, "an agent nothing called is roster"
    assert "system:root" not in ids, "the hub carries no information"

    level = {n["id"]: n["level"] for n in tree["nodes"]}
    assert level["goal:i1"] == 0
    assert level["goal:i1::agent:Research"] == 1
    assert level["tool:a"] == level["tool:b"] == 2
    # The answer ends the request, to the right of everything it did.
    assert level["result:i1"] > level["tool:a"]
