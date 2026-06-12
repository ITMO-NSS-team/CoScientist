"""Unit tests for ConsoleHITLHandler headless safety (no network required).

In a non-interactive session (no TTY) the handler must NOT call input() — that
raises EOFError and kills the whole agent run. Instead it auto-rejects so the
orchestrator can continue and finalize. These tests pin that behaviour.
"""
import asyncio
from unittest.mock import patch

from dotenv import load_dotenv

load_dotenv()

from CoScientist.hitl.handler import ConsoleHITLHandler
from CoScientist.hitl.models import HITLRequest, HITLAction


def _request(action=HITLAction.APPROVE, invoked_via="callback"):
    return HITLRequest(
        agent_name="CoderAgent",
        action_type=action,
        message="wants to run an outward-facing command",
        invoked_via=invoked_via,
    )


def _handle(req):
    return asyncio.run(ConsoleHITLHandler().handle_request(req))


def test_headless_auto_rejects_without_calling_input():
    def _boom(*a, **k):
        raise AssertionError("input() must not be called in a non-interactive session")

    with patch("sys.stdin") as stdin, patch("builtins.input", _boom):
        stdin.isatty.return_value = False
        resp = _handle(_request())
    assert resp.approved is False
    assert resp.action == HITLAction.REJECT


def test_headless_rejects_non_toggle_request():
    with patch("sys.stdin") as stdin:
        stdin.isatty.return_value = False
        resp = _handle(_request(invoked_via="tool"))
    assert resp.approved is False
    assert resp.action == HITLAction.REJECT


def test_eoferror_during_prompt_is_handled():
    # stdin claims to be a TTY but input() still EOFs (e.g. closed pipe) -> reject.
    def _eof(*a, **k):
        raise EOFError("EOF when reading a line")

    with patch("sys.stdin") as stdin, patch("builtins.input", _eof):
        stdin.isatty.return_value = True
        resp = _handle(_request())
    assert resp.approved is False
    assert resp.action == HITLAction.REJECT
