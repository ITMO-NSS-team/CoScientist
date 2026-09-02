"""Shared agent initialisation helpers.

Every per-agent module imports from here so settings are resolved once and the
LLM/tooling setup is consistent across agents.
"""
import asyncio
import logging
import os
from typing import Any, AsyncGenerator

import litellm
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse

from CoScientist.config import get_settings
from CoScientist.hitl.handler import ConsoleHITLHandler, DelegatingHITLHandler

settings = get_settings()

_logger = logging.getLogger(__name__)

# Transient upstream failures (provider hiccups, rate limits, 5xx) that are worth
# retrying. OpenRouter wraps a flaky underlying provider as a BadRequestError with
# "Provider returned error", which litellm's own num_retries does NOT retry — so
# we retry around the whole model call ourselves.
_RETRYABLE_SUBSTRINGS = (
    "provider returned error",
    "rate limit",
    "ratelimit",
    "overloaded",
    "service unavailable",
    "temporarily unavailable",
    "timeout",
    "timed out",
    "502",
    "503",
    "504",
    "529",
)
_RETRYABLE_TYPES = (
    "RateLimitError",
    "Timeout",
    "APIConnectionError",
    "ServiceUnavailableError",
    "InternalServerError",
    "APIError",
)
_LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))

# Seconds a single model call may take, and — separately — the longest silence
# tolerated between streamed chunks. A provider that goes quiet raises nothing
# on its own, so without this the agent waits forever.
REQUEST_TIMEOUT = settings.llm.request_timeout


def _is_transient(err: Exception) -> bool:
    if type(err).__name__ in _RETRYABLE_TYPES:
        return True
    msg = str(err).lower()
    return any(s in msg for s in _RETRYABLE_SUBSTRINGS)


class RetryingLiteLlm(LiteLlm):
    """LiteLlm that retries the whole call on transient upstream errors.

    Only retries when nothing has been yielded yet (so a partial stream is never
    duplicated) and only for transient errors; everything else propagates.
    """

    async def _stream(self, llm_request: LlmRequest, stream: bool):
        """Yield the upstream response, bounding the wait between chunks.

        litellm's own `timeout` covers getting the request away, not the silence
        after. A provider that opens the stream and then stops sending leaves
        `async for` waiting on a chunk that never comes: the socket stays
        established, nothing raises, and the agent is parked for good. Giving
        each chunk a deadline turns that silence into an error the caller can
        classify and retry.
        """
        source = super().generate_content_async(llm_request, stream=stream)
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        source.__anext__(), timeout=REQUEST_TIMEOUT
                    )
                except StopAsyncIteration:
                    return
                except asyncio.TimeoutError:
                    # Bare TimeoutError carries no message; say what happened so
                    # `_is_transient` recognises it and the log names the cause.
                    raise TimeoutError(
                        f"model sent nothing for {REQUEST_TIMEOUT}s — request timed out"
                    ) from None
                yield chunk
        finally:
            # Drop the stalled HTTP response rather than leaking the connection.
            await source.aclose()

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        attempt = 0
        while True:
            yielded = False
            try:
                async for resp in self._stream(llm_request, stream=stream):
                    yielded = True
                    yield resp
                return
            except Exception as err:  # noqa: BLE001 — classify then re-raise
                attempt += 1
                max_r = settings.web.max_retries
                if yielded or attempt > max_r or not _is_transient(err):
                    raise
                delay = min(1.5 ** attempt, 8.0)
                _logger.warning(
                    "Transient LLM error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt, max_r, delay, err,
                )
                await asyncio.sleep(delay)

MODEL = settings.llm.main_model
litellm.api_key = settings.llm.openai_api_key
# Silence litellm's "Provider List: https://docs.litellm.ai/docs/providers" spam.
# It fires when litellm can't map a model prefix (e.g. "qwen/...") to a known
# provider during cost/token bookkeeping — harmless, but it floods the console.
litellm.suppress_debug_info = True

hitl_handler = DelegatingHITLHandler(ConsoleHITLHandler())

# The CoderAgent runs on a dedicated (stronger) model — its multi-step tool-use
# benefits from more capability. Falls back to the main model when unset.
#
# Routing mirrors the other agents exactly: the provider prefix in the model
# string (e.g. "openrouter/qwen/...") selects the provider/base-URL, and the
# global `litellm.api_key` (set above) carries the key. We deliberately do NOT
# pass `api_base` here — doing so makes litellm strip the provider prefix, fail
# to re-infer the provider, and spam "Provider List: ..." warnings.
CODER_MODEL = settings.llm.coder_model or settings.llm.main_model


def make_llm(model: str = MODEL) -> LiteLlm:
    """Return a (retry-wrapped) LiteLlm for the main model (or an override)."""
    return RetryingLiteLlm(model=model, timeout=REQUEST_TIMEOUT)


def make_coder_llm() -> LiteLlm:
    """Return a (retry-wrapped) LiteLlm for the dedicated coder model."""
    return RetryingLiteLlm(model=CODER_MODEL, timeout=REQUEST_TIMEOUT)
