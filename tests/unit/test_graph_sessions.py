import json
from uuid import uuid4

from fastapi.testclient import TestClient

from CoScientist.graph.memory import get_knowledge_graph
from CoScientist.graph.memory_store import get_knowledge_memory
from CoScientist.graph.models import Edge, Node
from CoScientist.graph.research import store as research_store
from CoScientist.graph.semantic import Entity, Extraction, Relation
from CoScientist.graph.store import GraphStore


def _scope_ids():
    token = uuid4().hex
    return f"user_{token}", f"session_{token}_a", f"session_{token}_b"


def _create_user(client: TestClient, nickname: str) -> dict:
    response = client.post("/api/users", json={"nickname": nickname})
    assert response.status_code == 201
    return response.json()["user"]


def _create_session(client: TestClient, user_id: str, title: str) -> dict:
    response = client.post(
        f"/api/users/{user_id}/sessions",
        json={"title": title},
    )
    assert response.status_code == 201
    return response.json()["session"]


def test_graph_store_loads_snapshot_before_first_write(tmp_path):
    snapshot_dir = tmp_path / "execution"
    original = GraphStore(snapshot_dir=str(snapshot_dir))
    original.add_node(Node(
        id="old-goal",
        run_id="execution",
        kind="goal",
        label="persisted goal",
    ))
    original.add_node(Node(
        id="old-result",
        run_id="execution",
        kind="result",
        label="persisted result",
        status="success",
    ))
    original.add_edge(Edge(
        run_id="execution",
        src="old-goal",
        dst="old-result",
        type="produced",
    ))

    # The first operation on a new object is a write: it must merge with the
    # existing snapshot instead of replacing it with a one-node graph.
    restored = GraphStore(snapshot_dir=str(snapshot_dir))
    restored.add_node(Node(
        id="new-goal",
        run_id="execution",
        kind="goal",
        label="new goal",
    ))

    reloaded = GraphStore(snapshot_dir=str(snapshot_dir)).full("execution")
    assert {node["id"] for node in reloaded["nodes"]} == {
        "old-goal",
        "old-result",
        "new-goal",
    }
    assert {(
        edge["src"], edge["dst"], edge["type"]
    ) for edge in reloaded["edges"]} == {
        ("old-goal", "old-result", "produced")
    }


def test_graph_store_does_not_overwrite_unreadable_snapshot(tmp_path):
    snapshot_dir = tmp_path / "execution"
    snapshot_dir.mkdir()
    snapshot = snapshot_dir / "execution.json"
    damaged_contents = '{"nodes": ['
    snapshot.write_text(damaged_contents, encoding="utf-8")

    store = GraphStore(snapshot_dir=str(snapshot_dir))
    store.add_node(Node(
        id="new-goal",
        run_id="execution",
        kind="goal",
    ))

    assert snapshot.read_text(encoding="utf-8") == damaged_contents

    # An explicit reset remains available and replaces the broken snapshot.
    store.clear("execution")
    assert json.loads(snapshot.read_text(encoding="utf-8"))["nodes"] == []


def test_execution_graphs_are_isolated_by_session(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPH_SNAPSHOT_DIR", str(tmp_path / "execution"))
    user_id, first_session, second_session = _scope_ids()

    first = get_knowledge_graph(user_id=user_id, session_id=first_session)
    second = get_knowledge_graph(user_id=user_id, session_id=second_session)
    first.add_node(id="goal:first-session", kind="goal", label="first")
    second.add_node(id="goal:second-session", kind="goal", label="second")

    first_ids = {node["id"] for node in first.full()["nodes"]}
    second_ids = {node["id"] for node in second.full()["nodes"]}

    assert first is get_knowledge_graph(
        user_id=user_id,
        session_id=first_session,
    )
    assert first is not second
    assert "goal:first-session" in first_ids
    assert "goal:first-session" not in second_ids
    assert "goal:second-session" in second_ids
    assert "goal:second-session" not in first_ids


def test_research_graphs_are_isolated_by_session(tmp_path, monkeypatch):
    monkeypatch.setattr(
        research_store,
        "_default_dir",
        lambda: str(tmp_path / "research"),
    )
    user_id, first_session, second_session = _scope_ids()

    first = research_store.get_research_graph(
        user_id=user_id,
        session_id=first_session,
    )
    second = research_store.get_research_graph(
        user_id=user_id,
        session_id=second_session,
    )
    first_result = first.init_research(
        source="OrchestratorAgent",
        question="Question visible only in the first session?",
    )
    second_result = second.init_research(
        source="OrchestratorAgent",
        question="Question visible only in the second session?",
    )

    assert first_result["ok"] and second_result["ok"]
    assert first is not second
    assert first.full_graph().nodes["Q1"]["attrs"]["formulation"].startswith(
        "Question visible only in the first"
    )
    assert second.full_graph().nodes["Q1"]["attrs"]["formulation"].startswith(
        "Question visible only in the second"
    )


def test_user_memory_is_shared_across_sessions_but_not_users(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "KG_MEMORY_PATH",
        str(tmp_path / "memory" / "knowledge_memory.json"),
    )
    first_user, first_session, second_session = _scope_ids()
    second_user = f"user_{uuid4().hex}"

    first = get_knowledge_memory(
        user_id=first_user,
        session_id=first_session,
    )
    same_user_other_session = get_knowledge_memory(
        user_id=first_user,
        session_id=second_session,
    )
    other_user = get_knowledge_memory(
        user_id=second_user,
        session_id=first_session,
    )

    marker = uuid4().hex
    molecule_key = f"molecule:{marker}"
    target_key = f"target:{marker}"
    first.ingest(
        Extraction(
            entities=[
                Entity(key=molecule_key, type="molecule", name=molecule_key),
                Entity(key=target_key, type="target", name=target_key),
            ],
            relations=[
                Relation(src=molecule_key, dst=target_key, type="inhibits"),
            ],
        ),
        source="session-one",
    )

    assert first is same_user_other_session
    assert first is not other_user
    assert marker in str(same_user_other_session.full())
    assert marker not in str(other_user.full())


def test_web_graph_api_is_scoped_to_registered_session(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPH_SNAPSHOT_DIR", str(tmp_path / "execution"))
    monkeypatch.setattr(
        research_store,
        "_default_dir",
        lambda: str(tmp_path / "research"),
    )

    # Imported after the temporary graph paths are configured.
    from CoScientist.web.app import create_app

    app = create_app()
    with TestClient(app) as client:
        user = _create_user(client, f"graph-user-{uuid4().hex}")
        first_session = _create_session(client, user["id"], "First")
        second_session = _create_session(client, user["id"], "Second")

        first_graph = get_knowledge_graph(
            user_id=user["id"],
            session_id=first_session["id"],
        )
        second_graph = get_knowledge_graph(
            user_id=user["id"],
            session_id=second_session["id"],
        )
        first_graph.add_node(id="goal:web-first", kind="goal", label="first")
        second_graph.add_node(id="goal:web-second", kind="goal", label="second")

        first_response = client.get(
            f"/api/users/{user['id']}/sessions/{first_session['id']}/graph",
            params={"view": "execution"},
        )
        second_response = client.get(
            f"/api/users/{user['id']}/sessions/{second_session['id']}/graph",
            params={"view": "execution"},
        )
        assert first_response.status_code == 200
        assert second_response.status_code == 200
        first_ids = {node["id"] for node in first_response.json()["nodes"]}
        second_ids = {node["id"] for node in second_response.json()["nodes"]}
        assert "goal:web-first" in first_ids
        assert "goal:web-first" not in second_ids
        assert "goal:web-second" in second_ids
        assert "goal:web-second" not in first_ids

        assert client.get("/api/graph").status_code == 400
        compatibility_response = client.get(
            "/api/graph",
            params={
                "user_id": user["id"],
                "session_id": first_session["id"],
                "view": "execution",
            },
        )
        assert compatibility_response.status_code == 200
        assert "goal:web-first" in {
            node["id"] for node in compatibility_response.json()["nodes"]
        }

        other_user = _create_user(client, f"other-user-{uuid4().hex}")
        other_session = _create_session(client, other_user["id"], "Other")
        wrong_owner_response = client.get(
            f"/api/users/{user['id']}/sessions/{other_session['id']}/graph"
        )
        assert wrong_owner_response.status_code == 404


def test_graph_ui_uses_active_user_and_session_in_url():
    from CoScientist.web.app import create_app

    app = create_app()
    with TestClient(app) as client:
        index_html = client.get("/").text
        graph_html = client.get("/graph").text

    assert 'id="graph-link"' in index_html
    assert (
        "/graph?user_id=${encodeURIComponent(user.id)}"
        "&session_id=${encodeURIComponent(session.id)}"
    ) in index_html
    assert "new URLSearchParams(location.search)" in graph_html
    assert (
        "/api/users/${encodeURIComponent(userId)}"
        "/sessions/${encodeURIComponent(sessionId)}/graph"
    ) in graph_html
    assert 'fetch("/api/graph?view="' not in graph_html
