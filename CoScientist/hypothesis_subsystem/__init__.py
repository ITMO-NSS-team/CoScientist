"""
Hypothesis Agent Subsystem — public API.

Provides:
  * build_hypothesis_subsystem() — wires Generator + MooseChemMCPTool into an
    AgentTool, for standalone use (tests, direct embedding).
  * HypothesisSubsystemAgent — an ADK LlmAgent subclass that the assembly
    framework instantiates via ``class: custom:hypothesis_subsystem`` in
    system.yaml. The assembler wraps it in an AgentTool and attaches it to
    the OrchestratorAgent as a single black-box entry point.

Internal architecture:
    HypothesisGenerator (LlmAgent with tools)
        ├── retrieve_validation_tools → FedotMAS RAG DB / static fallback
        └── generate_via_moosechem → MooseChemMCPTool (MCP server)
"""

from __future__ import annotations

from typing import Any, Optional

from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

from CoScientist.config import get_settings
from CoScientist.hypothesis_subsystem.audit import HypothesisAuditLogger
from CoScientist.hypothesis_subsystem.base_tool import BaseHypothesisTool
from CoScientist.hypothesis_subsystem.generator_agent import build_hypothesis_generator
from CoScientist.hypothesis_subsystem.moosechem_mcp_tool import MooseChemMCPTool
import logging as _stdlib_logging
from CoScientist.hypothesis_subsystem.tool_registry import HypothesisToolRegistry


# ============================================================================
# Internal helpers
# ============================================================================

def _make_inject_state(
    registry: HypothesisToolRegistry,
    audit: HypothesisAuditLogger,
):
    """Return a before_agent_callback that injects subsystem objects into state."""

    async def inject_state(callback_context):
        callback_context.state["hypothesis_registry"] = registry
        callback_context.state["hypothesis_audit"] = audit
        return None

    return inject_state


def _wire_generator(
    model: str,
    registry: HypothesisToolRegistry,
    audit: HypothesisAuditLogger,
) -> LlmAgent:
    """Build and fully wire the HypothesisGenerator LlmAgent.

    Returns the bare generator (NOT wrapped in AgentTool) so the assembly
    framework can wrap it itself.
    """
    generator_agent = build_hypothesis_generator(
        model=model,
        tool_registry=registry,
        audit=audit,
    )

    # Compose inject_state ON TOP of any existing before_agent callback.
    original_before = generator_agent.before_agent_callback

    async def inject_state(callback_context):
        callback_context.state["hypothesis_registry"] = registry
        callback_context.state["hypothesis_audit"] = audit
        if original_before is None:
            return None
        import inspect as _inspect
        if _inspect.iscoroutinefunction(original_before):
            return await original_before(callback_context)
        else:
            return original_before(callback_context)

    generator_agent.before_agent_callback = inject_state

    return generator_agent


# ============================================================================
# Public API
# ============================================================================

def build_hypothesis_subsystem(
    model: Optional[str] = None,
) -> AgentTool:
    """Build the complete HypothesisGenerator subsystem as a single AgentTool.

    The caller receives a single AgentTool that the OrchestratorAgent can use
    as a drop-in replacement for the old HypothesesAgent. The internal
    architecture (retrieval + MooseChemMCPTool) is hidden behind the AgentTool
    boundary.

    Args:
        model: LLM model identifier. Defaults to settings.llm.main_model.

    Returns:
        An AgentTool wrapping the HypothesisGenerator.
    """
    settings = get_settings()
    model = model or settings.llm.main_model

    audit = HypothesisAuditLogger(_stdlib_logging.getLogger("hypothesis_subsystem"))
    registry = HypothesisToolRegistry()
    registry.register(MooseChemMCPTool())

    generator_agent = _wire_generator(model, registry, audit)
    return AgentTool(agent=generator_agent)


class HypothesisSubsystemAgent(LlmAgent):
    """Assembly-level adapter for ``class: custom:hypothesis_subsystem``.

    The assembly framework instantiates this class, passes it ``name``,
    ``description``, ``output_key``, and any ``options`` from system.yaml,
    then wraps the resulting instance in an :class:`AgentTool` and attaches
    it to the OrchestratorAgent.

    Internally it builds the full hypothesis generation subsystem
    (retrieve_validation_tools + MooseChemMCPTool) and exposes the
    HypothesisGenerator LlmAgent attributes — model, instruction, tools,
    output_schema — so the ADK runtime treats it as a regular LlmAgent.

    The ``before_agent_callback`` from system.yaml (``before_get_task``) is
    composed ON TOP of the internal state-injection callback so both run.

    Constructor args match what :func:`CoScientist.assembly.assembler._build_custom_agent`
    passes for ``issubclass(cls, LlmAgent)`` custom classes.
    """

    def __init__(
        self,
        name: str = "HypothesisGenerator",
        description: str = "",
        model: Any = None,
        output_key: Optional[str] = None,
        **kwargs: Any,
    ):
        settings = get_settings()
        model_str: str = (
            model if isinstance(model, str)
            else (kwargs.pop("model_str", None) or settings.llm.main_model)
        )

        assembler_before_agent = kwargs.pop("before_agent_callback", None)

        audit = HypothesisAuditLogger(
            _stdlib_logging.getLogger("hypothesis_subsystem")
        )
        registry = HypothesisToolRegistry()
        registry.register(MooseChemMCPTool())

        generator = build_hypothesis_generator(
            model=model_str,
            tool_registry=registry,
            audit=audit,
        )

        # Compose before_agent callbacks
        _inject = _make_inject_state(registry, audit)

        async def composed_before_agent(callback_context):
            """Inject subsystem state, then run the assembler-provided callback."""
            await _inject(callback_context)
            if assembler_before_agent is None:
                return None
            import inspect as _inspect
            if _inspect.iscoroutinefunction(assembler_before_agent):
                return await assembler_before_agent(callback_context)
            else:
                return assembler_before_agent(callback_context)

        super().__init__(
            name=name,
            description=description,
            model=generator.model,
            instruction=generator.instruction,
            tools=generator.tools,
            output_schema=generator.output_schema,
            output_key=output_key or generator.output_key,
            before_agent_callback=composed_before_agent,
            **kwargs,
        )


# ---- Exports ------------------------------------------------------------

__all__ = [
    "build_hypothesis_subsystem",
    "HypothesisSubsystemAgent",
    "HypothesisToolRegistry",
    "HypothesisAuditLogger",
    "BaseHypothesisTool",
    "MooseChemMCPTool",
    "build_hypothesis_generator",
]
