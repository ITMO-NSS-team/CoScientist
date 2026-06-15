"""Unit tests for the JSON-repair shim (F015a.A4 failure #1 — qwen malformed tool-call JSON)."""
from CoScientist.agents.llm_repair import repair_json_loads, _RepairJsonModule


def test_valid_json_unchanged():
    assert repair_json_loads('{"a": 1, "b": [2, 3]}') == {"a": 1, "b": [2, 3]}


def test_missing_comma():
    # the wave-1 "Expecting ',' delimiter" family
    out = repair_json_loads('{"a": 1 "b": 2}')
    assert isinstance(out, dict) and out.get("a") == 1


def test_truncated_object():
    # the "truncated big payload" family (char 441 / 2122)
    out = repair_json_loads('{"goal": "x", "steps": [{"id": "s1", "subtask": "gen"')
    assert isinstance(out, dict) and out.get("goal") == "x"


def test_extra_data_prefix_salvage():
    # "Extra data" — valid object then trailing garbage
    out = repair_json_loads('{"id": "s1"} blah blah not json')
    assert isinstance(out, dict) and out.get("id") == "s1"


def test_total_garbage_returns_empty():
    assert repair_json_loads("@@@ not json at all @@@") == {}


def test_shim_loads_passthrough_and_repair():
    shim = _RepairJsonModule()
    assert shim.loads('{"ok": true}') == {"ok": True}        # success path unchanged
    assert isinstance(shim.loads('{"a":1 "b":2}'), dict)      # repair path
    assert shim.JSONDecodeError is __import__("json").JSONDecodeError
