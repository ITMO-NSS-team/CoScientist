"""Prompt strings and prompt builders for the agents.

Re-exported here so existing `from CoScientist.agents.prompts import <name>`
imports keep working after the split into instructions.py / builder.py.
"""
from CoScientist.agents.prompts.builder import PromptBuilder, render_template
from CoScientist.agents.prompts.instructions import *  # noqa: F401,F403
from CoScientist.agents.prompts.instructions import (
    build_orchestrator_instruction,
    build_research_instruction,
)

__all__ = ["PromptBuilder", "render_template", "build_orchestrator_instruction", "build_research_instruction"]
