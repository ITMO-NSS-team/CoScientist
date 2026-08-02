"""
HypothesisGenerator — ADK LlmAgent for hypothesis generation.

This agent owns the tool registry and orchestrates:
1. Tool invocation (e.g., MooseChem) to produce raw hypotheses
2. Critic loop invocation to refine hypotheses
3. Final structured HypothesisList output
"""

from __future__ import annotations

import json
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
from CoScientist.hypothesis_subsystem.models import ToolCatalog


# ============================================================================
# Function tool: retrieve_validation_tools
# ============================================================================

@track(name="retrieve_validation_tools")
async def _retrieve_validation_tools(
    research_question: str,
    tool_context: Optional[ToolContext] = None,
) -> Dict[str, Any]:
    """Discover MCP validation tools relevant to the research question.

    Queries the FedotMAS RAG database for tools that can test/validate
    hypotheses for this question. Falls back to a static list (chemical-mcp-server)
    when the RAG DB is unreachable. Stores the result in state['tool_catalog']
    for downstream use by generate_via_moosechem and run_critic_loop.

    Args:
        research_question: The research question to find validation tools for.

    Returns:
        Dict with 'tool_catalog' (serialized ToolCatalog) and 'source'.
    """
    registry: Optional[HypothesisToolRegistry] = None
    if tool_context:
        registry = tool_context.state.get("hypothesis_registry")

    if registry is None:
        from CoScientist.hypothesis_subsystem.tool_registry import HypothesisToolRegistry
        registry = HypothesisToolRegistry()

    catalog = await registry.discover_validation_tools(research_question)

    # Stash in state so generate_via_moosechem picks it up
    if tool_context:
        tool_context.state["tool_catalog"] = catalog

    return {
        "tool_catalog": [t.model_dump(mode="json") for t in catalog.tools],
        "source": catalog.source,
        "tool_count": len(catalog.tools),
        "message": (
            f"Discovered {len(catalog.tools)} validation tools "
            f"(source: {catalog.source}). Use these to prioritize "
            f"testable hypotheses in generate_via_moosechem."
        ),
    }


# ============================================================================
# Function tool: generate_via_moosechem
# ============================================================================

@track(name="generate_via_moosechem")
async def _generate_via_moosechem(
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
    tool_catalog: Optional[ToolCatalog] = None
    if tool_context:
        registry = tool_context.state.get("hypothesis_registry")
        audit = tool_context.state.get("hypothesis_audit")
        tool_catalog = tool_context.state.get("tool_catalog")

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
        tool_catalog=tool_catalog,
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

    # [MooseChem-MCP integration] Stash exact hypotheses via direct state
    # mutation (verified visible to the next tool call in the same step) so the
    # critic loop can read them without depending on the LLM-relayed
    # `hypotheses_json`, which corrupts large (>~15KB) payloads on round-trip.
    if tool_context is not None:
        try:
            tool_context.state["_moosechem_raw_hypotheses"] = {"hypotheses": hypotheses_dicts}
        except Exception:
            pass

    return {"hypotheses": hypotheses_dicts, "metadata": result.metadata}


# ============================================================================
# Function tool: run_critic_loop
# ============================================================================

@track(name="run_critic_loop")
async def _run_critic_loop(
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

    # Parse input. [MooseChem-MCP integration] hypotheses_json is relayed by the
    # LLM and can be corrupted/truncated on large payloads. Try to parse it, but
    # fall back to the exact hypotheses stashed in state by
    # _generate_via_moosechem (verified visible within the same step).
    parsed = None
    if isinstance(hypotheses_json, str):
        try:
            parsed = json.loads(hypotheses_json)
        except Exception:
            parsed = None
    elif isinstance(hypotheses_json, dict):
        parsed = hypotheses_json

    if not parsed or not parsed.get("hypotheses"):
        stashed = tool_context.state.get("_moosechem_raw_hypotheses") if tool_context else None
        if stashed and stashed.get("hypotheses"):
            parsed = stashed

    if not parsed:
        parsed = {"hypotheses": []}

    raw_hypotheses = parsed.get("hypotheses", [])

    # Convert to Hypothesis objects.
    # [MooseChem-MCP integration] The hypotheses JSON is round-tripped through
    # the LLM between tool calls, which can drop required technical fields
    # (provenance, strategy_type, domain, refutation_conditions) while keeping
    # the scientific content. Backfill ONLY missing required fields via
    # setdefault (never overwrites what the LLM kept) so the full hypothesis —
    # variables, tools, evidence_basis — survives instead of collapsing to the
    # minimal wrapper below. Safe to remove if it proves unnecessary.
    hypotheses: List[Hypothesis] = []
    for h in raw_hypotheses:
        if isinstance(h, dict):
            h.setdefault("provenance", {"creator": "MooseChem"})
            h.setdefault("strategy_type", "MooseChem")
            h.setdefault("domain", "chemistry")
            h.setdefault("refutation_conditions", "Refuted if the predicted effect is not observed under the verification plan.")
        try:
            hypotheses.append(Hypothesis(**h))
        except Exception:
            # If parsing fails, create a minimal wrapper
            hypotheses.append(
                Hypothesis(
                    claim=str(h.get("claim", "Unknown")),
                    domain=str(h.get("domain", "")),
                    reasoning=str(h.get("reasoning", "")),
                    strategy_type=str(h.get("strategy_type", "unknown")),
                    verification_plan=str(h.get("verification_plan", "")),
                    refutation_conditions=str(h.get("refutation_conditions", "")),
                    provenance=Provenance(creator="HypothesisGenerator"),
                )
            )

    hypothesis_list = HypothesisList(hypotheses=hypotheses)

    # Run critic loop
    refined = await loop_coordinator.run_critic_loop(hypothesis_list, research_question)

    result = {"hypotheses": [h.model_dump(mode="json") for h in refined.hypotheses]}

    # [MooseChem-MCP integration] Write refined hypotheses straight into state.
    # The agent's output_key only captures the LLM's free-text final answer,
    # which drops/truncates the rich hypothesis JSON on round-trip. Writing the
    # exact critic-loop output here restores what downstream readers expect.
    # Safe to remove if it proves unnecessary.
    if tool_context is not None:
        try:
            tool_context.state["generated_hypotheses"] = result
            # Also stage via actions.state_delta so ADK commits it to session state
            if getattr(tool_context, "actions", None) is not None:
                tool_context.actions.state_delta["generated_hypotheses"] = result
        except Exception:
            pass

    return result


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
    # Register tools — retrieve_validation_tools FIRST so the LLM calls it
    # before generate_via_moosechem (Step 0 → Step 1 of GENERATOR_INSTRUCTION).
    generator_tools: List[FunctionTool] = [
        FunctionTool(_retrieve_validation_tools),
        FunctionTool(_generate_via_moosechem),
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
        # [MooseChem-MCP integration] output_key disabled: it wrote the LLM's
        # free-text final answer (a truncated ~25KB JSON) into
        # state["generated_hypotheses"] on the final event, overwriting the
        # exact hypotheses dict that _run_critic_loop already stores there.
        # _run_critic_loop is now the single writer of this key. Restore this
        # line if the state-write in _run_critic_loop is reverted.
        # output_key="generated_hypotheses",
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
    agent.tools.append(FunctionTool(_run_critic_loop))
