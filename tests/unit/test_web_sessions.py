import json

from fastapi.testclient import TestClient

from CoScientist.web.app import APP_NAME, create_app


def _create_user(client, nickname):
    response = client.post("/api/users", json={"nickname": nickname})
    assert response.status_code == 201
    return response.json()["user"]


def _create_session(client, user_id, title):
    response = client.post(
        f"/api/users/{user_id}/sessions",
        json={"title": title},
    )
    assert response.status_code == 201
    return response.json()["session"]


def test_local_users_sessions_and_roadmaps_are_isolated():
    app = create_app()
    with TestClient(app) as client:
        assert client.get("/api/users").json() == {"users": []}
        gleb = _create_user(client, "Gleb")
        assert client.post("/api/users", json={"nickname": "gleb"}).status_code == 409

        first = _create_session(client, gleb["id"], "First")
        second = _create_session(client, gleb["id"], "Second")
        first_tasks = [{"id": "TASK-1", "title": "Only first"}]

        saved = client.post(
            f"/api/users/{gleb['id']}/sessions/{first['id']}/roadmap",
            json={"content": json.dumps(first_tasks)},
        )
        assert saved.status_code == 200

        first_result = client.get(
            f"/api/users/{gleb['id']}/sessions/{first['id']}/roadmap"
        ).json()
        second_result = client.get(
            f"/api/users/{gleb['id']}/sessions/{second['id']}/roadmap"
        ).json()
        assert first_result["tasks"] == first_tasks
        assert second_result["tasks"] == []

        adk_session = app.state.runtime.session_service.sessions[APP_NAME][gleb["id"]][first["id"]]
        assert adk_session.state["active_tasks"] == first_tasks


def test_websocket_reconnect_receives_same_session_snapshot():
    app = create_app()
    with TestClient(app) as client:
        user = _create_user(client, "Gleb")
        session = _create_session(client, user["id"], "Work")
        key = (user["id"], session["id"])
        app.state.runtime.agent_events[key].append({
            "type": "user_message",
            "message": "previous question",
            "timestamp": "2026-01-01T10:00:00",
        })

        url = f"/ws?user_id={user['id']}&session_id={session['id']}"
        with client.websocket_connect(url) as websocket:
            assert websocket.receive_json()["type"] == "connected"
            snapshot = websocket.receive_json()
            assert snapshot["type"] == "session_snapshot"
            assert snapshot["user"]["nickname"] == "Gleb"
            assert snapshot["messages"][0]["message"] == "previous question"

        with client.websocket_connect(url) as websocket:
            websocket.receive_json()
            snapshot = websocket.receive_json()
            assert snapshot["session"]["id"] == session["id"]
            assert snapshot["messages"][0]["message"] == "previous question"

