"""Shared agent initialisation helpers.

All per-agent modules import from here so settings are resolved once.
"""
from typing import Any

import litellm
from google.adk.models.lite_llm import LiteLlm

from CoScientist.config import get_settings
from CoScientist.hitl.tool import get_hitl_tools

settings = get_settings()
MODEL = settings.llm.main_model
litellm.api_key = settings.llm.openai_api_key
hitl_enabled = settings.hitl.enabled


def make_llm() -> LiteLlm:
    """Return a fresh LiteLlm model wrapper (stateless config object)."""
    return LiteLlm(model=MODEL)


def agent_tools(base_tools: Any = None, *, hitl: bool = False) -> list:
    """Build a tool list, optionally appending HITL tools.

    Args:
        base_tools: A single tool/toolset, a list of tools, or None.
        hitl: Whether to append HITL approval/selection tools when enabled.
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
