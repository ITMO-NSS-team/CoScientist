"""Tests for the AlembicAgent A2A integration.

The agent wraps the alembic pipeline (a whole scientific GitHub repo -> a
running MCP server) as a CoScientist sub-agent exposed over A2A. These assert
the wiring (config + prompt + tools + a2a) and the toolset's offline behaviour.

Run from the repo root:  pytest tests/unit/test_alembic_agent.py -q
"""
import asyncio

import pytest
from dotenv import load_dotenv

load_dotenv()

from CoScientist.assembly import build_system  # noqa: E402
from CoScientist.assembly.schema import get_config  # noqa: E402
from CoScientist.tools.alembic_tools import (  # noqa: E402
    AlembicToolset,
    _extract_tool_names,
    alembic_toolset_instance,
)


@pytest.fixture(scope="module")
def config():
    return get_config()


@pytest.fixture(scope="module")
def system(config):
    return build_system(config)


# ── wiring ───────────────────────────────────────────────────────────────────

def test_alembic_is_an_orchestrator_subordinate(config):
    assert "AlembicAgent" in config.agent("OrchestratorAgent").subordinates
    assert config.agent("AlembicAgent").tools == ["alembic"]


def test_alembic_exposed_over_a2a_with_unique_port(config):
    cfg = config.agent("AlembicAgent")
    assert cfg.a2a is not None
    assert cfg.a2a.key == "alembic"
    ports = [a.a2a.port for a in config.a2a_agents()]
    assert ports.count(cfg.a2a.port) == 1  # no collision
    assert config.a2a_agent_by_key("alembic").name == "AlembicAgent"


def test_prompt_documents_exactly_the_attached_tools(system):
    instruction = system.agent("AlembicAgent").instruction
    assert "build_mcp_server" in instruction
    assert "stop_mcp_server" in instruction
    # Built agent's function tools match what the prompt advertises.
    names = {
        getattr(t, "name", None) or getattr(t, "__name__", None)
        for t in system.agent("AlembicAgent").tools
    }
    assert {"build_mcp_server", "stop_mcp_server"} <= names


def test_alembic_named_in_orchestrator_prompt(system):
    assert "AlembicAgent" in system.agent("OrchestratorAgent").instruction


# ── toolset behaviour (offline) ──────────────────────────────────────────────

def test_toolset_instance_exposes_two_named_tools():
    names = [t.__name__ for t in alembic_toolset_instance]
    assert names == ["build_mcp_server", "stop_mcp_server"]


def test_build_returns_error_when_docker_missing(monkeypatch):
    monkeypatch.setattr(AlembicToolset, "_docker_available", staticmethod(lambda: False))
    ts = AlembicToolset()
    out = asyncio.run(ts.build_mcp_server("https://github.com/owner/repo"))
    assert out["status"] == "error"
    assert "Docker" in out["error"]


@pytest.mark.parametrize(
    "report, expected",
    [
        ("samples:\n  predict:\n    args: {}\n  embed:\n    args: {}", ["predict", "embed"]),
        ("```yaml\nsamples:\n  - tool: run\n  - tool: score\n```", ["run", "score"]),
        ("no tool block at all", []),
        ("", []),
    ],
)
def test_extract_tool_names(report, expected):
    assert _extract_tool_names(report) == expected
