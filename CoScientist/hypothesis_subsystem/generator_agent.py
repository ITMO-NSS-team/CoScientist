"""
HypothesisGenerator — ADK LlmAgent for hypothesis generation.

This agent owns the tool registry and orchestrates:
1. Validation tool discovery (retrieve_validation_tools)
2. Hypothesis generation via MooseChemMCPTool
3. Final structured HypothesisList output with validation_tool_matching
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from opik import track

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext

from CoScientist.hypothesis_subsystem.audit import HypothesisAuditLogger
from CoScientist.hypothesis_subsystem.models import (
    HypothesisQuery,
    ToolCatalog,
    ToolResult,
)
from CoScientist.hypothesis_subsystem.prompts import GENERATOR_INSTRUCTION
from CoScientist.hypothesis_subsystem.tool_registry import HypothesisToolRegistry


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
    for downstream use by generate_via_moosechem.

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
    Generate hypotheses using the MooseChem MCP pipeline.

    Delegates to MooseChemMCPTool which connects to the MOOSE-Chem MCP server
    (Docker). The server runs the original evolutionary-algorithm pipeline:
    corpus building (PubMed + OpenAlex), LLM generation, screening, scoring.

    The tool automatically receives the tool_catalog from state (populated by
    retrieve_validation_tools) and annotates each hypothesis with
    validation_tool_matching.

    Args:
        research_question: The scientific research question to generate hypotheses for.
        background_survey: Optional background context or literature survey.
        domain_constraints: Optional constraints on the applicability domain.
        max_hypotheses: Maximum number of hypotheses to generate (1-20).
        temperature: LLM temperature for generation (0.0-2.0).

    Returns:
        A dict with 'hypotheses' key containing the generated HypothesisList.
    """
    registry: Optional[HypothesisToolRegistry] = None
    audit: Optional[HypothesisAuditLogger] = None
    tool_catalog: Optional[ToolCatalog] = None
    if tool_context:
        registry = tool_context.state.get("hypothesis_registry")
        audit = tool_context.state.get("hypothesis_audit")
        tool_catalog = tool_context.state.get("tool_catalog")

    from CoScientist.hypothesis_subsystem.moosechem_mcp_tool import MooseChemMCPTool

    if registry is None:
        tool = MooseChemMCPTool()
    else:
        tool = registry.get("MooseChem")
        if tool is None:
            tool = MooseChemMCPTool()

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

    hypotheses_dicts = [h.model_dump(mode="json") for h in result.hypotheses]

    # Write hypotheses directly into state so downstream consumers
    # (orchestrator, tests) find them without depending on LLM relay.
    if tool_context is not None:
        try:
            tool_context.state["generated_hypotheses"] = {"hypotheses": hypotheses_dicts}
            if getattr(tool_context, "actions", None) is not None:
                tool_context.actions.state_delta["generated_hypotheses"] = {"hypotheses": hypotheses_dicts}
        except Exception:
            pass

    return {"hypotheses": hypotheses_dicts, "metadata": result.metadata}


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
    generator_tools: List[FunctionTool] = [
        FunctionTool(_retrieve_validation_tools),
        FunctionTool(_generate_via_moosechem),
    ]

    agent = LlmAgent(
        name="HypothesisGenerator",
        model=LiteLlm(model=model),
        instruction=GENERATOR_INSTRUCTION,
        description=(
            "Generates structured, falsifiable scientific hypotheses via the "
            "MooseChem MCP pipeline (PubMed+OpenAlex corpus → LLM generation → "
            "scoring). Discovers available validation tools and annotates each "
            "hypothesis with validation_tool_matching."
        ),
        tools=generator_tools,
    )

    return agent
