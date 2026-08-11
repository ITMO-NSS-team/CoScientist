"""HITL over A2A: the tools must not block the server, and must keep the names
the prompts/docs/unknown-tool guard expect."""
import asyncio
import importlib
import os

import pytest

from CoScientist.assembly.bindings import HITL_TOOL_DOCS
from CoScientist.hitl.handler import ConsoleHITLHandler
from CoScientist.hitl.models import HITLAction, HITLRequest


def _tools(a2a: bool):
    prev = os.environ.get("COSCIENTIST_A2A_MODE")
    if a2a:
        os.environ["COSCIENTIST_A2A_MODE"] = "1"
    else:
        os.environ.pop("COSCIENTIST_A2A_MODE", None)
    try:
        import CoScientist.hitl.tool as tool_mod
        importlib.reload(tool_mod)
        return tool_mod.get_hitl_tools()
    finally:
        if prev is None:
            os.environ.pop("COSCIENTIST_A2A_MODE", None)
        else:
            os.environ["COSCIENTIST_A2A_MODE"] = prev
        import CoScientist.hitl.tool as tool_mod
        importlib.reload(tool_mod)


def test_a2a_mode_attaches_long_running_hitl_tools():
    """Over A2A the HITL tools must be long-running: ADK then reports the call to
    the client and the A2A task goes to `input-required` instead of the server
    blocking on a console it does not have."""
    tools = _tools(a2a=True)
    assert {t.name for t in tools} == {"request_approval", "request_selection"}
    assert all(getattr(t, "is_long_running", False) for t in tools)


def test_in_process_mode_keeps_blocking_tools():
    """The in-process/web path is unchanged (handler-driven, not long-running)."""
    tools = _tools(a2a=False)
    assert {t.name for t in tools} == {"request_approval", "request_selection"}
    assert not any(getattr(t, "is_long_running", False) for t in tools)


def test_tool_names_match_docs_so_the_guard_accepts_them():
    """Names must equal HITL_TOOL_DOCS, else guard_unknown_tools would block a
    legitimate HITL call (the guard builds its valid set from the docs)."""
    doc_names = {d.name for d in HITL_TOOL_DOCS}
    assert {t.name for t in _tools(a2a=True)} == doc_names
    assert {t.name for t in _tools(a2a=False)} == doc_names


def test_a2a_tools_return_immediately_without_blocking():
    from CoScientist.hitl.a2a_tools import request_approval, request_selection

    approval = asyncio.run(asyncio.wait_for(request_approval("A", "Approve?"), timeout=5))
    assert approval["status"] == "pending_human_input"

    selection = asyncio.run(
        asyncio.wait_for(request_selection("A", "Pick", ["x", "y"]), timeout=5))
    assert selection["options"] == ["x", "y"]


def test_console_handler_does_not_hang_without_a_tty():
    """Safety net: any headless path (A2A/uvicorn/cron) must get an answer
    instead of blocking on stdin."""
    req = HITLRequest(agent_name="A", action_type=HITLAction.APPROVE, message="proceed?")
    resp = asyncio.run(asyncio.wait_for(ConsoleHITLHandler().handle_request(req), timeout=5))
    assert resp.action in (HITLAction.APPROVE, HITLAction.REJECT)
    assert resp.instructions and "headless" in resp.instructions.lower()


def test_console_headless_policy_can_reject(monkeypatch):
    monkeypatch.setenv("HITL_HEADLESS_POLICY", "reject")
    req = HITLRequest(agent_name="A", action_type=HITLAction.APPROVE, message="proceed?")
    resp = asyncio.run(asyncio.wait_for(ConsoleHITLHandler().handle_request(req), timeout=5))
    assert resp.approved is False
