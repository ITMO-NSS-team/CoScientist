"""Tests for A2A serve-mode decisions that must not import heavy agent deps."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys


def _load_serve_module():
    path = Path(__file__).resolve().parents[2] / "CoScientist" / "a2a" / "serve.py"
    spec = importlib.util.spec_from_file_location("a2a_serve_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _agent_cfg(*, root=False, subordinates=()):
    return SimpleNamespace(root=root, subordinates=list(subordinates))


def test_agents_with_a2a_subordinates_use_remote_mode():
    serve = _load_serve_module()

    assert serve._should_use_remote_subagents(
        _agent_cfg(subordinates=["ResearchAgent"])
    )


def test_root_and_leaf_serve_modes_are_preserved():
    serve = _load_serve_module()

    assert serve._should_use_remote_subagents(_agent_cfg(root=True))
    assert not serve._should_use_remote_subagents(_agent_cfg())


def test_serve_module_import_has_no_a2a_package_side_effects():
    for name in list(sys.modules):
        if name == "CoScientist" or name.startswith("CoScientist."):
            del sys.modules[name]

    serve = importlib.import_module("CoScientist.a2a.serve")

    assert hasattr(serve, "main")
    assert "CoScientist.a2a.config" not in sys.modules


def test_assembly_schema_import_does_not_build_agents_package():
    for name in list(sys.modules):
        if name == "CoScientist" or name.startswith("CoScientist."):
            del sys.modules[name]

    schema = importlib.import_module("CoScientist.assembly.schema")

    assert hasattr(schema, "get_config")
    assert "CoScientist.agents" not in sys.modules
    assert "CoScientist.config" not in sys.modules
