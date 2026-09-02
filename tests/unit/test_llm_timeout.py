"""A completion that never returns must fail, not hang forever.

The production symptom: the orchestrator delegated to HypothesesAgent, the
provider accepted the connection and then sent nothing, and the run sat silent
for fourteen minutes with an empty log. No exception was ever raised, so the
retry wrapper never fired. These tests pin the two halves of the fix — every
call carries a deadline, and the resulting Timeout is retried.
"""
import pytest

from CoScientist.agents import common
from CoScientist.config import get_settings


@pytest.fixture
def anyio_backend():
    """The retry backoff sleeps on asyncio; the runtime does too."""
    return "asyncio"


def test_every_model_carries_a_deadline():
    expected = get_settings().llm.request_timeout
    assert expected > 0
    for llm in (common.make_llm(), common.make_coder_llm(), common.make_llm("openrouter/other")):
        assert llm._additional_args.get("timeout") == expected


def test_a_timeout_is_transient():
    """Whatever litellm raises on a deadline must be classified as retryable."""
    assert common._is_transient(TimeoutError("request timed out"))
    assert common._is_transient(RuntimeError("APITimeoutError: connection timeout"))
    assert not common._is_transient(ValueError("malformed tool schema"))


@pytest.mark.anyio
async def test_a_stalled_call_is_retried_then_succeeds(monkeypatch):
    attempts = []

    async def flaky(self, llm_request, stream=False):
        attempts.append(stream)
        if len(attempts) == 1:
            raise TimeoutError("request timed out")
        yield "answer"

    monkeypatch.setattr(common.LiteLlm, "generate_content_async", flaky)
    monkeypatch.setattr(common, "settings", get_settings())

    llm = common.make_llm()
    got = [r async for r in llm.generate_content_async(object())]

    assert got == ["answer"]
    assert len(attempts) == 2


@pytest.mark.anyio
async def test_a_partial_stream_is_never_retried(monkeypatch):
    """Retrying after output has been yielded would duplicate it."""
    attempts = []

    async def stalls_midway(self, llm_request, stream=False):
        attempts.append(stream)
        yield "first half"
        raise TimeoutError("request timed out")

    monkeypatch.setattr(common.LiteLlm, "generate_content_async", stalls_midway)

    llm = common.make_llm()
    with pytest.raises(TimeoutError):
        async for _ in llm.generate_content_async(object()):
            pass

    assert len(attempts) == 1
