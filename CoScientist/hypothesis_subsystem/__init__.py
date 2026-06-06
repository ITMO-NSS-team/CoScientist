"""
Hypothesis Agent Subsystem — public API.

Provides build_hypothesis_subsystem(), which wires HypothesisGenerator,
MooseChemTool, LoopCoordinator (using HypothesisCriticAgent from
CoScientist/agents/hypothesis_critic.py), and AuditLogger into a single
AgentTool that the OrchestratorAgent uses as a drop-in replacement for
the old HypothesesAgent.

Usage in agents.py:
    from CoScientist.hypothesis_subsystem import build_hypothesis_subsystem
    hypothesis_subsystem = build_hypothesis_subsystem(model=MODEL)
"""

from __future__ import annotations

from typing import Optional

from google.adk.tools.agent_tool import AgentTool

from CoScientist.config import get_settings
from CoScientist.hypothesis_subsystem.audit import HypothesisAuditLogger
from CoScientist.hypothesis_subsystem.base_tool import BaseHypothesisTool
from CoScientist.hypothesis_subsystem.generator_agent import (
    add_critic_loop_tool,
    build_hypothesis_generator,
)
from CoScientist.hypothesis_subsystem.loop_coordinator import HypothesisLoopCoordinator
from CoScientist.hypothesis_subsystem.moosechem_tool import MooseChemTool
import logging as _stdlib_logging
from CoScientist.hypothesis_subsystem.tool_registry import HypothesisToolRegistry


def build_hypothesis_subsystem(
    model: Optional[str] = None,
) -> AgentTool:
    """
    Build the complete HypothesisGenerator+Critic subsystem as a single AgentTool.

    The OrchestratorAgent calls this tool without knowing the internal structure:
    it sees a single agent that accepts a research question and returns a
    structured HypothesisList.

    Internal architecture:
        HypothesisGenerator (LlmAgent with tools)
            ├── generate_via_moosechem → MooseChemTool
            └── run_critic_loop → HypothesisLoopCoordinator
                                     └── HypothesisCriticAgent (from agents/hypothesis_critic.py)

    Args:
        model: LLM model identifier. Defaults to settings.llm.main_model.

    Returns:
        An AgentTool wrapping the HypothesisGenerator, ready to be added
        to the OrchestratorAgent's tools list.
    """
    settings = get_settings()
    model = model or settings.llm.main_model

    # ---- Audit logger ---------------------------------------------------
    audit = HypothesisAuditLogger(_stdlib_logging.getLogger("hypothesis_subsystem"))

    # ---- Tool registry --------------------------------------------------
    registry = HypothesisToolRegistry()
    moosechem_tool = MooseChemTool(model=model)
    registry.register(moosechem_tool)

    # ---- Loop coordinator (uses HypothesisCriticAgent) ------------------
    loop_coordinator = HypothesisLoopCoordinator(
        model=model,
        audit=audit,
    )

    # ---- Build Generator agent ------------------------------------------
    generator_agent = build_hypothesis_generator(
        model=model,
        tool_registry=registry,
        audit=audit,
    )

    # Inject the critic loop tool (requires loop_coordinator to exist)
    add_critic_loop_tool(generator_agent)

    # ---- Configure the Generator's state for tool access ---------------
    original_before = generator_agent.before_agent_callback

    async def inject_state(callback_context):
        """Inject registry, loop_coordinator, and audit into agent state."""
        callback_context.state["hypothesis_registry"] = registry
        callback_context.state["loop_coordinator"] = loop_coordinator
        callback_context.state["hypothesis_audit"] = audit
        if original_before:
            return await original_before(callback_context)
        return None

    generator_agent.before_agent_callback = inject_state

    # ---- Wrap as AgentTool ----------------------------------------------
    return AgentTool(agent=generator_agent)


# ---- Exports ------------------------------------------------------------

__all__ = [
    "build_hypothesis_subsystem",
    "HypothesisToolRegistry",
    "HypothesisAuditLogger",
    "BaseHypothesisTool",
    "MooseChemTool",
    "HypothesisLoopCoordinator",
    "build_hypothesis_generator",
]
