"""Unit tests for critic _apply_revisions (no network required).

The pre-action critic may REVISE a proposed tool call's args. Sub-agent
delegation goes through AgentTool, which needs a single 'request' string; the
critic sometimes rewrites such a call into a domain arg shape and drops
'request', crashing AgentTool with KeyError. _apply_revisions must preserve the
'request' key in that case while still honouring legitimate revisions.
"""
from types import SimpleNamespace

from dotenv import load_dotenv

load_dotenv()

from CoScientist.agents.critic_agent import _apply_revisions


def _pending(tool, args):
    fc = SimpleNamespace(args=dict(args))
    part = SimpleNamespace(function_call=fc)
    return {"tool": tool, "args": dict(args), "_part": part}, fc


def test_preserves_request_when_revision_drops_it():
    call, fc = _pending("TaskExecutorAgent", {"request": "do the research"})
    # critic mis-rewrites delegation into search-style args (no 'request')
    _apply_revisions([call], [{"args": {"keywords": "CRISPR", "year": "2025", "limit": 3}}])
    assert fc.args.get("request") == "do the research"


def test_honours_legitimate_request_revision():
    call, fc = _pending("ResearchAgent", {"request": "old request"})
    _apply_revisions([call], [{"args": {"request": "new refined request"}}])
    assert fc.args["request"] == "new refined request"


def test_non_delegation_call_is_fully_revised():
    call, fc = _pending("list_available_tools", {"query": "x"})
    _apply_revisions([call], [{"args": {"query": "y", "limit": 5}}])
    assert fc.args == {"query": "y", "limit": 5}


def test_no_revision_entry_leaves_call_untouched():
    call, fc = _pending("ResearchAgent", {"request": "keep me"})
    _apply_revisions([call], [])  # fewer revisions than pending -> break
    assert fc.args == {"request": "keep me"}
