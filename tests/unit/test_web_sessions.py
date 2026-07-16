import asyncio
import importlib
import json

from fastapi.testclient import TestClient

from CoScientist.web.app import APP_NAME, create_app

web_app = importlib.import_module("CoScientist.web.app")


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


def test_runtime_rejects_a_second_run_and_stop_cleans_only_its_session(monkeypatch):
    async def scenario():
        runtime = web_app.WebRuntime()
        first_key = ("user_a", "session_a")
        second_key = ("user_b", "session_b")
        started = asyncio.Event()
        release = asyncio.Event()

        async def blocked_chat(_runtime, key, data):
            assert key == first_key
            assert data["message"] == "first"
            started.set()
            await release.wait()

        monkeypatch.setattr(web_app, "_handle_chat", blocked_chat)
        assert await runtime.start_run(first_key, {"message": "first"})
        await started.wait()
        owner = runtime.active_runs[first_key]

        assert not await runtime.start_run(first_key, {"message": "second"})
        assert runtime.active_runs[first_key] is owner

        first_wait = asyncio.Event()
        second_wait = asyncio.Event()
        runtime.pending_hitl["first"] = {
            "event": first_wait,
            "response": None,
            "session_key": first_key,
        }
        runtime.pending_hitl["second"] = {
            "event": second_wait,
            "response": None,
            "session_key": second_key,
        }

        assert await runtime.stop_run(first_key)
        assert first_key not in runtime.active_runs
        assert first_wait.is_set()
        assert "first" not in runtime.pending_hitl
        assert not second_wait.is_set()
        assert "second" in runtime.pending_hitl

        web_app._cancel_pending_hitl(runtime)

    asyncio.run(scenario())


def test_finished_run_cannot_discard_a_new_owner():
    async def scenario():
        runtime = web_app.WebRuntime()
        key = ("user_a", "session_a")
        old_owner = asyncio.create_task(asyncio.sleep(0))
        await old_owner
        release = asyncio.Event()
        new_owner = asyncio.create_task(release.wait())
        runtime.active_runs[key] = new_owner

        assert not await runtime.discard_run(key, old_owner)
        assert runtime.active_runs[key] is new_owner

        new_owner.cancel()
        await asyncio.gather(new_owner, return_exceptions=True)

    asyncio.run(scenario())


def test_chat_controls_follow_server_status_broadcasts():
    html = web_app.TEMPLATE_PATH.read_text(encoding="utf-8")
    submit_handler = html.split(
        "document.getElementById('chat-form').addEventListener", 1
    )[1].split("function stopChat", 1)[0]
    stop_handler = html.split("function stopChat", 1)[1].split(
        "function clearChat", 1
    )[0]

    assert "function applyRunStatus(status, version = null)" in html
    assert "parsedVersion < runStatusVersion" in html
    assert "case 'status':" in html
    assert "applyRunStatus(data.status, data.run_status_version);" in html
    assert "case 'chat_accepted':" in html
    assert "addUserMsg(msg);" not in submit_handler
    assert "input.value = '';" not in submit_handler
    assert "send-btn').disabled" not in submit_handler
    assert "send-btn').disabled" not in stop_handler
    assert "stop-btn').classList" not in stop_handler


def test_initial_snapshot_is_ordered_before_live_socket_broadcasts():
    class SlowSocket:
        def __init__(self):
            self.messages = []
            self.first_send_started = asyncio.Event()
            self.release_first_send = asyncio.Event()

        async def send_json(self, payload):
            if not self.messages:
                self.first_send_started.set()
                await self.release_first_send.wait()
            self.messages.append(payload)

    async def scenario():
        runtime = web_app.WebRuntime()
        key = ("user_a", "session_a")
        socket = SlowSocket()
        attach = asyncio.create_task(runtime.attach_with_snapshot(
            key,
            socket,
            user={"id": key[0], "nickname": "A"},
            session={"id": key[1]},
            active_tasks=[],
        ))
        await socket.first_send_started.wait()
        live = asyncio.create_task(runtime.send(
            key,
            runtime.status_payload(key, "processing", "newer status"),
        ))
        await asyncio.sleep(0)
        socket.release_first_send.set()
        await asyncio.gather(attach, live)

        assert [message["type"] for message in socket.messages] == [
            "connected",
            "session_snapshot",
            "status",
        ]

    asyncio.run(scenario())


def test_stop_releases_control_lock_before_slow_socket_delivery():
    async def scenario():
        runtime = web_app.WebRuntime()
        key = ("user_a", "session_a")
        owner = asyncio.create_task(asyncio.Event().wait())
        runtime.active_runs[key] = owner
        send_started = asyncio.Event()
        release_send = asyncio.Event()

        async def slow_send(_key, _payload):
            send_started.set()
            await release_send.wait()

        runtime.send = slow_send
        stopping = asyncio.create_task(runtime.stop_run(key))
        await send_started.wait()

        lock = runtime.control_lock(key)
        await asyncio.wait_for(lock.acquire(), timeout=0.1)
        lock.release()
        release_send.set()
        assert await stopping

    asyncio.run(scenario())


def test_runtime_rejects_new_runs_after_shutdown_begins():
    async def scenario():
        runtime = web_app.WebRuntime()
        runtime._closing = True
        assert not await runtime.start_run(
            ("user_a", "session_a"),
            {"message": "must not start"},
        )
        assert runtime.active_runs == {}

    asyncio.run(scenario())
