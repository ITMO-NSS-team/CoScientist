"""The critic's own LLM call: it must always come back, and always with a verdict.

The three critics bypass the agent tree, so none of `RetryingLiteLlm`'s
protections apply to them: no wall-clock deadline, no retry. They are awaited
inline — the pre/post critics inside the orchestrator's event loop, the plan
critic inside SessionAgent's review loop — so a call that never returns parks
the whole run. What has to hold: the call is bounded, a transient fault gets one
retry, and every failure path still yields a permissive verdict.

Run from the repo root:  pytest tests/unit/test_critic_llm_call.py -q
"""
import asyncio

import pytest

from CoScientist.agents.callbacks import critic as critic_module
from CoScientist.agents.common import RetryingLiteLlm


def _resp(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def _invoke():
    return asyncio.run(critic_module._invoke_critic_llm("SYSTEM", "USER"))


@pytest.fixture(autouse=True)
def _fast_critic(monkeypatch):
    """Keep the deadline, the retry backoff and the proxy probe out of the
    test's wall clock — the probe is a real network round-trip."""
    monkeypatch.setattr(critic_module, "_CRITIC_TIMEOUT_S", 0.05)
    monkeypatch.setattr(critic_module, "_CRITIC_HTTP_TIMEOUT_S", 0.04)
    monkeypatch.setattr(critic_module, "_CRITIC_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(critic_module.asyncio, "sleep", _noop_sleep)
    monkeypatch.setattr(
        RetryingLiteLlm, "_verify_proxy_reachable", staticmethod(_reachable)
    )


async def _noop_sleep(_delay):
    return None


async def _reachable():
    return None


def test_a_provider_that_never_answers_does_not_park_the_run(monkeypatch):
    """The failure this exists for: the per-read timeout only bounds a peer that
    goes silent, so a call that keeps the socket busy is capped by nothing."""
    calls = []

    async def hangs(**kwargs):
        calls.append(1)
        await asyncio.Event().wait()

    monkeypatch.setattr(critic_module.litellm, "acompletion", hangs)

    assert _invoke() == {"verdict": "approve"}
    # A blown deadline is not retried: the second attempt would buy the same
    # wait again on a route that just failed to answer within it.
    assert len(calls) == 1


def test_the_call_carries_a_deadline_and_no_constrained_decode(monkeypatch):
    """`response_format` is the other half of the hang — OpenRouter forwards it
    to the provider as a constrained decode (cf. ToolReranker in system.yaml)."""
    seen = {}

    async def capture(**kwargs):
        seen.update(kwargs)
        return _resp('{"verdict": "approve"}')

    monkeypatch.setattr(critic_module.litellm, "acompletion", capture)
    _invoke()

    assert "response_format" not in seen
    # Under the wall-clock deadline, so a stalled transport reports itself
    # rather than losing the race to a nameless `TimeoutError()`.
    assert seen["timeout"] == 0.04
    assert seen["timeout"] < critic_module._CRITIC_TIMEOUT_S
    assert seen["num_retries"] == 0
    # A verdict is a small object; unbounded generation is latency with no upside.
    assert seen["max_tokens"] == critic_module._CRITIC_MAX_TOKENS


def test_verdict_is_parsed_out_of_fences_and_prose(monkeypatch):
    """Without `response_format` nothing forces bare JSON, so parsing tolerates
    what a model actually emits."""
    async def fenced(**kwargs):
        return _resp('Here is my verdict:\n```json\n{"verdict": "revise", '
                     '"feedback": "TASK-2 has no assignee."}\n```')

    monkeypatch.setattr(critic_module.litellm, "acompletion", fenced)

    assert _invoke() == {"verdict": "revise", "feedback": "TASK-2 has no assignee."}


def test_unparseable_answer_approves(monkeypatch):
    async def prose(**kwargs):
        return _resp("I think the plan looks fine to me.")

    monkeypatch.setattr(critic_module.litellm, "acompletion", prose)

    assert _invoke() == {"verdict": "approve"}


def test_transient_fault_is_retried_once(monkeypatch):
    calls = []

    async def flaky(**kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("Provider returned error")
        return _resp('{"verdict": "revise", "feedback": "no assignee"}')

    monkeypatch.setattr(critic_module.litellm, "acompletion", flaky)

    assert _invoke()["verdict"] == "revise"
    assert len(calls) == 2


def test_retries_are_capped(monkeypatch):
    calls = []

    async def always_transient(**kwargs):
        calls.append(1)
        raise RuntimeError("503 service unavailable")

    monkeypatch.setattr(critic_module.litellm, "acompletion", always_transient)

    assert _invoke() == {"verdict": "approve"}
    assert len(calls) == critic_module._CRITIC_MAX_ATTEMPTS


def test_a_dead_proxy_fails_fast_without_calling_the_model(monkeypatch):
    """A proxy whose VPN is down accepts the connection and answers nothing, so
    the call would otherwise burn the whole deadline on every attempt."""
    calls = []

    async def unreachable():
        raise ConnectionError("Error connecting to proxy server.")

    async def should_not_run(**kwargs):
        calls.append(1)

    monkeypatch.setattr(
        RetryingLiteLlm, "_verify_proxy_reachable", staticmethod(unreachable)
    )
    monkeypatch.setattr(critic_module.litellm, "acompletion", should_not_run)

    assert _invoke() == {"verdict": "approve"}
    assert calls == []


def test_a_permanent_fault_is_not_retried(monkeypatch):
    calls = []

    async def bad_request(**kwargs):
        calls.append(1)
        raise ValueError("model not found")

    monkeypatch.setattr(critic_module.litellm, "acompletion", bad_request)

    assert _invoke() == {"verdict": "approve"}
    assert len(calls) == 1
