"""Shared agent initialisation helpers.

Every per-agent module imports from here so settings are resolved once and the
LLM/tooling setup is consistent across agents.
"""
from typing import Any

import litellm
from google.adk.models.lite_llm import LiteLlm

from CoScientist.config import get_settings
from CoScientist.hitl.handler import ConsoleHITLHandler
from CoScientist.hitl.tool import get_hitl_tools

settings = get_settings()

MODEL = settings.llm.main_model
litellm.api_key = settings.llm.openai_api_key
# Silence litellm's "Provider List: https://docs.litellm.ai/docs/providers" spam.
# It fires when litellm can't map a model prefix (e.g. "qwen/...") to a known
# provider during cost/token bookkeeping — harmless, but it floods the console.
litellm.suppress_debug_info = True

hitl_enabled = settings.hitl.enabled
hitl_handler = ConsoleHITLHandler() if hitl_enabled else None

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
    """Return a LiteLlm wrapper for the main model (or an explicit override)."""
    return LiteLlm(model=model)


def make_coder_llm() -> LiteLlm:
    """Return a LiteLlm wrapper for the dedicated coder model."""
    return LiteLlm(model=CODER_MODEL)


def agent_tools(base_tools: Any = None, *, hitl: bool = False) -> list:
    """Build a tool list, optionally appending HITL tools.

    Args:
        base_tools: a single tool/toolset, a list of tools, or None.
        hitl: whether to append HITL approval/selection tools when enabled.
    """
    if base_tools is None:
        tools: list = []
    elif isinstance(base_tools, list):
        tools = list(base_tools)
    else:
        tools = [base_tools]

    if hitl and hitl_enabled:
        tools.extend(get_hitl_tools())
    return tools
