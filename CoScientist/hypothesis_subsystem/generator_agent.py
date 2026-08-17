"""
HypothesisGenerator — ADK LlmAgent for hypothesis generation.

This agent owns the tool registry and orchestrates:
1. Tool invocation (e.g., MooseChem) to produce raw hypotheses
2. Critic loop invocation to refine hypotheses
3. Final structured HypothesisList output
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from opik import track

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext

from CoScientist.hypothesis_subsystem.audit import HypothesisAuditLogger
from CoScientist.hypothesis_subsystem.models import (
    Hypothesis,
    HypothesisList,
    HypothesisQuery,
    HypothesisStatus,
    Provenance,
    ProvenanceRecord,
    ToolResult,
)
from CoScientist.hypothesis_subsystem.prompts import GENERATOR_INSTRUCTION
from CoScientist.hypothesis_subsystem.tool_registry import HypothesisToolRegistry

logger = logging.getLogger(__name__)


# ============================================================================
# Function tool: generate_via_moosechem
# ============================================================================

@track(name="generate_via_moosechem")
async def generate_via_moosechem(
    research_question: str,
    background_survey: Optional[str] = None,
    domain_constraints: Optional[str] = None,
    max_hypotheses: int = 5,
    temperature: Optional[float] = None,
    tool_context: Optional[ToolContext] = None,
) -> Dict[str, Any]:
    """
    Generate hypotheses using the MooseChem pipeline (PubMed corpus → LLM generation → scoring).

    Args:
        research_question: The scientific research question to generate hypotheses for.
        background_survey: Optional background context or literature survey.
        domain_constraints: Optional constraints on the applicability domain.
        max_hypotheses: Maximum number of hypotheses to generate (1-20).
        temperature: LLM temperature for generation (0.0-2.0).

    Returns:
        A dict with 'hypotheses' key containing the generated HypothesisList.
    """
    # Retrieve registry from tool_context state
    registry: Optional[HypothesisToolRegistry] = None
    audit: Optional[HypothesisAuditLogger] = None
    if tool_context:
        registry = tool_context.state.get("hypothesis_registry")
        audit = tool_context.state.get("hypothesis_audit")

    # Fallback: create a default tool if registry unavailable
    from CoScientist.hypothesis_subsystem.moosechem_tool import MooseChemTool

    if registry is None:
        tool = MooseChemTool()
    else:
        tool = registry.get("MooseChem")
        if tool is None:
            tool = MooseChemTool()

    query = HypothesisQuery(
        research_question=research_question,
        background_survey=background_survey,
        domain_constraints=domain_constraints,
        max_hypotheses=max_hypotheses,
        temperature=temperature,
    )

    start_ts = audit.log_generation_start(research_question, "MooseChem") if audit else 0
    result: ToolResult = await tool.invoke(query)
    if audit:
        audit.log_tool_invocation(
            strategy_type="MooseChem",
            query=query,
            result=result,
        )

    # Convert to HypothesisList dict for ADK
    hypotheses_dicts = [h.model_dump(mode="json") for h in result.hypotheses]
    return {"hypotheses": hypotheses_dicts, "metadata": result.metadata}


# ============================================================================
# Function tool: run_critic_loop
# ============================================================================

@track(name="run_critic_loop")
async def run_critic_loop(
    hypotheses_json: str,
    research_question: str,
    tool_context: Optional[ToolContext] = None,
) -> Dict[str, Any]:
    """
    Run the Critic review loop on generated hypotheses.

    The Critic evaluates each hypothesis for rigor and falsifiability.
    Approved hypotheses pass through; those needing revision are refined;
    rejected hypotheses are marked as deferred. Max 3 iterations per hypothesis.

    Args:
        hypotheses_json: JSON string of HypothesisList with 'hypotheses' key.
        research_question: The original research question for context.

    Returns:
        Dict with 'hypotheses' key containing the refined HypothesisList.
    """
    loop_coordinator = None
    audit: Optional[HypothesisAuditLogger] = None
    if tool_context:
        loop_coordinator = tool_context.state.get("loop_coordinator")
        audit = tool_context.state.get("hypothesis_audit")

    if loop_coordinator is None:
        # No critic loop available — return as-is
        return json.loads(hypotheses_json)

    # Parse input
    parsed = json.loads(hypotheses_json) if isinstance(hypotheses_json, str) else hypotheses_json
    raw_hypotheses = parsed.get("hypotheses", [])

    # Convert to Hypothesis objects
    hypotheses: List[Hypothesis] = []
    for h in raw_hypotheses:
        if not isinstance(h, dict):
            logger.warning("[run_critic_loop] skipping non-dict hypothesis entry: %r", h)
            continue
        try:
            hypotheses.append(Hypothesis(**h))
        except Exception as exc:
            # A malformed entry must not abort the whole loop — log it and build
            # a minimal wrapper carrying the required provenance so the critic
            # can still judge it (rather than raising ValidationError inside the
            # except handler).
            logger.warning(
                "[run_critic_loop] malformed hypothesis; using minimal wrapper: %s", exc
            )
            hypotheses.append(
                Hypothesis(
                    claim=str(h.get("claim") or "Unknown hypothesis"),
                    domain=str(h.get("domain") or ""),
                    reasoning=str(h.get("reasoning") or ""),
                    strategy_type=str(h.get("strategy_type") or "unknown"),
                    verification_plan=str(h.get("verification_plan") or ""),
                    refutation_conditions=str(h.get("refutation_conditions") or ""),
                    provenance=Provenance(creator="HypothesisGenerator/fallback"),
                )
            )

    hypothesis_list = HypothesisList(hypotheses=hypotheses)

    # Run critic loop
    refined = await loop_coordinator.run_critic_loop(
        hypothesis_list, research_question, tool_context=tool_context
    )

    return {"hypotheses": [h.model_dump(mode="json") for h in refined.hypotheses]}


# ============================================================================
# Builder
# ============================================================================

def build_hypothesis_generator(
    model: str,
    tool_registry: HypothesisToolRegistry,
    audit: HypothesisAuditLogger,
) -> LlmAgent:
    """
    Build the HypothesisGenerator ADK LlmAgent.

    Args:
        model: LLM model identifier (e.g., 'gpt-4').
        tool_registry: Registry of hypothesis-generation strategies.
        audit: Audit logger for observability.

    Returns:
        A configured LlmAgent ready for use in the ADK runtime.
    """
    # Register tools
    generator_tools: List[FunctionTool] = [
        FunctionTool(generate_via_moosechem),
    ]
    # run_critic_loop is added later by build_hypothesis_subsystem
    # after the loop_coordinator is constructed.

    agent = LlmAgent(
        name="HypothesisGenerator",
        model=LiteLlm(model=model),
        instruction=GENERATOR_INSTRUCTION,
        description=(
            "Generates structured, falsifiable scientific hypotheses using "
            "configurable strategy tools (MooseChem pipeline). Iteratively "
            "refines hypotheses via a built-in Critic loop."
        ),
        output_key="generated_hypotheses",
        tools=generator_tools,
    )

    return agent


def add_critic_loop_tool(agent: LlmAgent) -> None:
    """
    Add the run_critic_loop tool to an existing HypothesisGenerator agent.

    Called after the loop_coordinator is constructed, since it requires
    the coordinator to be available.

    Args:
        agent: The HypothesisGenerator LlmAgent to add the tool to.
    """
    agent.tools.append(FunctionTool(run_critic_loop))
