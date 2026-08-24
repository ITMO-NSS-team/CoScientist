"""A built MCP server has to reach two places: the durable catalogue (so later
runs find it) and this run's state (so the executor can call it now). No
network, no database — the rag_tools manager is injected.
"""

import asyncio

import pytest

from CoScientist.tools.registry_bridge import (
    register_and_resolve,
    register_mcp_server,
    resolve_into_state,
)


class _Server:
    def __init__(self, url, name):
        self.url = url
        self.name = name
        self.status = "ok"


class _Manager:
    """Records what was registered; stands in for a live rag_tools manager."""

    def __init__(self):
        self.added = []
        self.closed = False

    async def add_server(self, *, name, protocol, url, description, headers, sync_tools):
        self.added.append(
            {"name": name, "url": url, "sync_tools": sync_tools, "headers": headers}
        )
        return _Server(url, name)

    async def close(self):
        self.closed = True


def test_registering_indexes_the_server_and_its_tools():
    manager = _Manager()

    server = asyncio.run(
        register_mcp_server("http://host:8000/mcp", "medsam", manager=manager)
    )

    assert manager.added == [
        {
            "name": "medsam",
            "url": "http://host:8000/mcp",
            "sync_tools": True,  # indexing the tools is the point of registering
            "headers": None,
        }
    ]
    assert server.url == "http://host:8000/mcp"
    assert not manager.closed  # an injected manager is the caller's to close


def test_a_missing_url_is_refused():
    with pytest.raises(ValueError):
        asyncio.run(register_mcp_server("", "medsam", manager=_Manager()))


def test_resolving_makes_the_tool_callable_this_run():
    state = {}

    resolve_into_state(state, _Server("http://host:8000/mcp", "medsam"))

    assert state["deployed_mcps"] == [
        {"url": "http://host:8000/mcp", "name": "medsam"}
    ]


def test_resolving_the_same_server_twice_adds_one_entry():
    """Retrieval may add the same server, so this has to compose with it."""
    state = {"deployed_mcps": [{"url": "http://host:8000/mcp", "name": "medsam"}]}

    resolve_into_state(state, "http://host:8000/mcp", "medsam")

    assert len(state["deployed_mcps"]) == 1


def test_resolving_keeps_servers_that_are_already_there():
    state = {"deployed_mcps": [{"url": "http://other:9000/mcp", "name": "other"}]}

    resolve_into_state(state, _Server("http://host:8000/mcp", "medsam"))

    assert [d["name"] for d in state["deployed_mcps"]] == ["other", "medsam"]


def test_one_call_does_both_halves():
    manager, state = _Manager(), {}

    server, entry = asyncio.run(
        register_and_resolve(
            "http://host:8000/mcp", "medsam", state, manager=manager
        )
    )

    assert manager.added[0]["url"] == "http://host:8000/mcp"  # in the catalogue
    assert state["deployed_mcps"] == [entry]  # and callable right now
    assert server.name == "medsam"


# ── the call site: a finished build must reach the catalogue ─────────────────


class _Ctx:
    def __init__(self):
        self.state = {}


def _finished_job(monkeypatch, tmp_path, calls):
    """A done build in alembic_tools' job registry, with the bridge stubbed."""
    tmp_log = tmp_path / "build.log"
    tmp_log.write_text("MCP server: http://host:8000/mcp\n", encoding="utf-8")
    from CoScientist.tools import alembic_tools

    async def _register_and_resolve(mcp_url, name, state, description="", **kw):
        calls.append({"url": mcp_url, "name": name})
        state.setdefault("deployed_mcps", []).append({"url": mcp_url, "name": name})
        return object(), state["deployed_mcps"][-1]

    monkeypatch.setattr(
        "CoScientist.tools.registry_bridge.register_and_resolve", _register_and_resolve
    )
    rec = {
        "job_id": "medsam-abc123",
        "repo_url": "https://github.com/org/medsam.git",
        "status": "done",
        "started_at": 0.0,
        "finished_at": 1.0,
        "mcp_url": "http://host:8000/mcp",
        "image": "alembic-tool:medsam",
        "log_file": str(tmp_log),
    }
    monkeypatch.setattr(alembic_tools, "_JOBS", {"medsam-abc123": rec})
    return alembic_tools, rec


def test_a_finished_build_is_registered_and_made_callable(monkeypatch, tmp_path):
    calls = []
    alembic_tools, _ = _finished_job(monkeypatch, tmp_path, calls)
    ctx = _Ctx()

    out = asyncio.run(alembic_tools.check_mcp_build("medsam-abc123", ctx))

    assert calls == [{"url": "http://host:8000/mcp", "name": "medsam"}]
    assert ctx.state["deployed_mcps"] == [
        {"url": "http://host:8000/mcp", "name": "medsam"}
    ]
    assert out["registered"] is True


def test_polling_the_same_build_registers_it_once(monkeypatch, tmp_path):
    calls = []
    alembic_tools, _ = _finished_job(monkeypatch, tmp_path, calls)
    ctx = _Ctx()

    asyncio.run(alembic_tools.check_mcp_build("medsam-abc123", ctx))
    asyncio.run(alembic_tools.check_mcp_build("medsam-abc123", ctx))

    assert len(calls) == 1


def test_a_registry_outage_does_not_fail_the_build(monkeypatch, tmp_path):
    """The build succeeded. A catalogue that is down is reported, not raised."""
    calls = []
    alembic_tools, _ = _finished_job(monkeypatch, tmp_path, calls)

    async def _boom(*a, **kw):
        raise ConnectionError("registry unreachable")

    monkeypatch.setattr(
        "CoScientist.tools.registry_bridge.register_and_resolve", _boom
    )

    out = asyncio.run(alembic_tools.check_mcp_build("medsam-abc123", _Ctx()))

    assert out["status"] == "done"
    assert out["mcp_url"] == "http://host:8000/mcp"
    assert out["registered"] is False
    assert "registry unreachable" in out["registration_error"]
