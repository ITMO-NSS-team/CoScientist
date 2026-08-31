import asyncio
import importlib
import io
import zipfile

import pytest

from CoScientist.tools.coder_tools import openhands_sandbox as sandbox
from CoScientist.web import session_bundle

web_app = importlib.import_module("CoScientist.web.app")


def _build_session(sandbox_id):
    """A WebRuntime with one user/session, its ADK state bound to sandbox_id."""
    runtime = web_app.WebRuntime()
    user = runtime.registry.create_user("Nika")
    session = runtime.registry.create_session(user["id"], "Trajectory export")
    key = (user["id"], session["id"])

    async def _create():
        state = {}
        if sandbox_id is not None:
            state[sandbox.SESSION_STATE_KEY] = sandbox_id
        await runtime.session_service.create_session(
            app_name=web_app.APP_NAME,
            user_id=key[0],
            session_id=key[1],
            state=state,
        )

    asyncio.run(_create())
    return runtime, key


def _member_names(bundle_bytes):
    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as zf:
        return set(zf.namelist())


def test_export_includes_sandbox_trajectory_when_fetch_succeeds(monkeypatch):
    runtime, key = _build_session("task-123")

    async def fake_fetch(*, sandbox_id=None, **_kwargs):
        assert sandbox_id == "task-123"
        return {"task_id": sandbox_id, "events": [{"action": "execute_bash"}]}

    monkeypatch.setattr(sandbox, "aget_sandbox_trajectory", fake_fetch)

    bundle_bytes = asyncio.run(session_bundle.export_session(runtime, key))
    names = _member_names(bundle_bytes)

    assert session_bundle._SANDBOX_TRAJECTORY in names
    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as zf:
        import json
        payload = json.loads(zf.read(session_bundle._SANDBOX_TRAJECTORY))
    assert payload["task_id"] == "task-123"


def test_export_skips_trajectory_without_a_sandbox_binding():
    runtime, key = _build_session(None)

    bundle_bytes = asyncio.run(session_bundle.export_session(runtime, key))
    names = _member_names(bundle_bytes)

    assert session_bundle._SANDBOX_TRAJECTORY not in names
    # everything else is still there
    assert session_bundle._MANIFEST in names
    assert session_bundle._AGENT_EVENTS in names


def test_a_failing_trajectory_fetch_never_sinks_the_export(monkeypatch):
    runtime, key = _build_session("task-456")

    async def failing_fetch(*, sandbox_id=None, **_kwargs):
        raise TimeoutError("trajectory too large / server too slow")

    monkeypatch.setattr(sandbox, "aget_sandbox_trajectory", failing_fetch)

    bundle_bytes = asyncio.run(session_bundle.export_session(runtime, key))
    names = _member_names(bundle_bytes)

    assert session_bundle._SANDBOX_TRAJECTORY not in names
    assert session_bundle._MANIFEST in names
    assert session_bundle._AGENT_EVENTS in names
