"""Inventory indexing + bind-by-name/score (no ask→family regex).

Cover = this-run retrieve returned **compute MCP** tools. Empty fedot/react
binds from exact name in the task text, else highest retrieval score.
Promote coder → MCP/family only when THIS task names that tool. Unnamed
coder stays required coder (or alembic if a repo fits) even if leftover
inventory exists for a different operation.

Research/medical families are declared route-agent capabilities (not RAG MCP).
They must not count as compute coverage for feasibility.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Mapping


FAMILY_MCP = "mcp"
FAMILY_RESEARCH = "research"
FAMILY_MEDICAL = "medical"

RESEARCH_SERVER_ID = "__research__"
MEDICAL_SERVER_ID = "__medical__"
_SYNTHETIC_SERVER_IDS = frozenset({RESEARCH_SERVER_ID, MEDICAL_SERVER_ID})

# Tool names + purposes from CoScientist.assembly.bindings ToolDoc entries
# for ResearchAgent / MedicalAgent. Coverage is still name/score bind, not
# ask-phrase tables.
_DECLARED_FAMILY_TOOLS: tuple[tuple[str, str, str, str], ...] = (
    (FAMILY_RESEARCH, RESEARCH_SERVER_ID, "tavily_search",
     "General web search."),
    (FAMILY_RESEARCH, RESEARCH_SERVER_ID, "tavily_extract",
     "Read the content of specific pages/URLs."),
    (FAMILY_RESEARCH, RESEARCH_SERVER_ID, "tavily_crawl",
     "Crawl a site starting from a URL when one page is not enough."),
    (FAMILY_RESEARCH, RESEARCH_SERVER_ID, "explore_chemistry_database",
     "RAG search over an internal scientific literature database."),
    (FAMILY_RESEARCH, RESEARCH_SERVER_ID, "explore_my_papers",
     "Answers questions using user-uploaded or previously downloaded papers."),
    (FAMILY_RESEARCH, RESEARCH_SERVER_ID, "search_papers",
     "Searches scientific papers in OpenAlex using metadata and search filters."),
    (FAMILY_RESEARCH, RESEARCH_SERVER_ID, "download_papers_from_search",
     "Searches and downloads papers for downstream analysis."),
    (FAMILY_MEDICAL, MEDICAL_SERVER_ID, "search_pubmed",
     "Find peer-reviewed literature on a clinical topic, drug, condition, or intervention."),
    (FAMILY_MEDICAL, MEDICAL_SERVER_ID, "get_pico",
     "Extract Population / Intervention / Comparison / Outcome from a paper abstract."),
    (FAMILY_MEDICAL, MEDICAL_SERVER_ID, "get_study_taxonomy",
     "Classify a paper's study design (observational vs experimental vs literature review)."),
    (FAMILY_MEDICAL, MEDICAL_SERVER_ID, "analyze_medical_image",
     "Interpret an uploaded DICOM or image file; differential diagnosis and ICD-10."),
)


def declared_family_capabilities(*families: str) -> list[dict[str, Any]]:
    """Static research/medical tool descriptors for planner context."""
    want = {str(item).strip() for item in families if str(item).strip()} or {
        FAMILY_RESEARCH, FAMILY_MEDICAL,
    }
    out: list[dict[str, Any]] = []
    for family, server_id, tool, description in _DECLARED_FAMILY_TOOLS:
        if family not in want:
            continue
        out.append({
            "family": family,
            "tool": tool,
            "server_id": server_id,
            "description": description,
            "input_schema": {},
            "score": None,
            "url": None,
        })
    return out


def row_family(item: Mapping[str, Any] | None) -> str:
    if not isinstance(item, Mapping):
        return FAMILY_MCP
    family = str(item.get("family") or "").strip()
    if family in {FAMILY_MCP, FAMILY_RESEARCH, FAMILY_MEDICAL}:
        return family
    server_id = str(item.get("server_id") or "").strip()
    if server_id == RESEARCH_SERVER_ID:
        return FAMILY_RESEARCH
    if server_id == MEDICAL_SERVER_ID:
        return FAMILY_MEDICAL
    return FAMILY_MCP


def index_inventory_tools(
    available_tools: Iterable[dict[str, Any]],
    *,
    families: Iterable[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Map tool name → inventory row (higher retrieval score wins).

    Default families={mcp}: RAG compute tools only. Pass research/medical to
    index declared route-agent capabilities.
    """
    allow = (
        {FAMILY_MCP}
        if families is None
        else {str(item).strip() for item in families if str(item).strip()}
    )
    out: dict[str, dict[str, Any]] = {}
    for item in available_tools:
        if not isinstance(item, dict):
            continue
        tool = str(item.get("tool") or item.get("name") or "").strip()
        server_id = str(item.get("server_id") or "").strip()
        if not tool or not server_id:
            continue
        family = row_family(item)
        if family not in allow:
            continue
        if family == FAMILY_MCP and server_id in _SYNTHETIC_SERVER_IDS:
            continue
        row = {
            "family": family,
            "tool": tool,
            "server_id": server_id,
            "description": item.get("description") or "",
            "input_schema": item.get("input_schema"),
            "score": item.get("score"),
            "url": item.get("url"),
        }
        prior = out.get(tool)
        if prior is None or float(row.get("score") or 0) > float(prior.get("score") or 0):
            out[tool] = row
    return out


def inventory_pairs(inventory: Iterable[dict[str, Any]]) -> set[tuple[str, str]]:
    """Exact (server_id, tool) pairs for registry feasibility checks (MCP only)."""
    return {
        (str(item["server_id"]), str(item.get("tool") or item.get("name")))
        for item in inventory
        if isinstance(item, dict)
        and item.get("server_id")
        and (item.get("tool") or item.get("name"))
        and row_family(item) == FAMILY_MCP
        and str(item.get("server_id") or "") not in _SYNTHETIC_SERVER_IDS
    }


def inventory_nonempty(available_tools: Iterable[dict[str, Any]] | Mapping[str, Any] | None) -> bool:
    """True when this-run retrieve left at least one compute MCP (server_id, tool)."""
    if not available_tools:
        return False
    if isinstance(available_tools, Mapping):
        first = next(iter(available_tools.values()), None)
        if isinstance(first, dict) and first.get("server_id") and row_family(first) == FAMILY_MCP:
            if str(first.get("server_id") or "") not in _SYNTHETIC_SERVER_IDS:
                return True
        if isinstance(first, dict):
            return bool(index_inventory_tools(list(available_tools.values())))
        return False
    return bool(index_inventory_tools(available_tools))


def _named_match(text: str, by_tool: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if not text:
        return None
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", text):
        if token in by_tool:
            return by_tool[token]
    low = text.lower().replace("-", "_")
    for tool, item in by_tool.items():
        if tool.lower() in low:
            return item
    return None


def match_inventory_tool(
    blob: str, by_tool: dict[str, dict[str, Any]], *, source_request: str = "",
) -> dict[str, Any] | None:
    """Pick inventory tool: exact name in task text, else highest retrieval score."""
    if not by_tool:
        return None
    task_text = (blob or "").strip()
    request = (source_request or "").strip()
    combined = f"{task_text}\n{request}".strip()
    for text in (task_text, combined):
        if hit := _named_match(text, by_tool):
            return hit
    ranked = sorted(
        by_tool.values(),
        key=lambda item: float(item.get("score") or 0),
        reverse=True,
    )
    return ranked[0] if ranked else None


def match_named_inventory_tool(
    blob: str, by_tool: dict[str, dict[str, Any]], *, source_request: str = "",
) -> dict[str, Any] | None:
    """Like match_inventory_tool but never fall back to max retrieval score."""
    if not by_tool:
        return None
    task_text = (blob or "").strip()
    request = (source_request or "").strip()
    combined = f"{task_text}\n{request}".strip()
    for text in (task_text, combined):
        if hit := _named_match(text, by_tool):
            return hit
    return None


_SLOT_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9]{3,}")


def _slot_tokens(text: str) -> set[str]:
    return {token.lower() for token in _SLOT_TOKEN_RE.findall(text or "")}


def _tool_name_tokens(tool: str) -> set[str]:
    return {
        part for part in str(tool or "").lower().replace("-", "_").split("_")
        if len(part) >= 4
    }


def slot_tool_score(statement: str, item: Mapping[str, Any]) -> int:
    """Lexical overlap of a frame slot with one inventory row (no domain tables)."""
    stmt = _slot_tokens(statement)
    if not stmt:
        return 0
    name_hits = sum(1 for token in _tool_name_tokens(str(item.get("tool") or "")) if token in stmt)
    desc_hits = sum(
        1 for token in _slot_tokens(str(item.get("description") or ""))
        if len(token) >= 6 and token in stmt
    )
    return name_hits * 2 + desc_hits


def match_slot_inventory_tool(
    blob: str, by_tool: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Named match, else a unique lexical fit of this slot to this-run inventory."""
    if not by_tool:
        return None
    if hit := match_named_inventory_tool(blob, by_tool):
        return hit
    ranked: list[tuple[int, dict[str, Any]]] = []
    for item in by_tool.values():
        score = slot_tool_score(blob, item)
        if score >= 2:
            ranked.append((score, item))
    if not ranked:
        return None
    ranked.sort(key=lambda pair: (pair[0], float(pair[1].get("score") or 0)), reverse=True)
    best_score, best = ranked[0]
    tied = [item for score, item in ranked if score == best_score]
    if len(tied) == 1:
        return best
    tied.sort(key=lambda item: float(item.get("score") or 0), reverse=True)
    return tied[0]


def match_named_family_capability(blob: str) -> dict[str, Any] | None:
    """First research/medical tool name appearing in *task* text.

    Does not consult source_request: mixed asks mention several families, and
    per-task routing must follow this task's text, not the whole brief.
    """
    rows = declared_family_capabilities(FAMILY_RESEARCH, FAMILY_MEDICAL)
    by_tool = index_inventory_tools(
        rows, families={FAMILY_RESEARCH, FAMILY_MEDICAL},
    )
    return match_named_inventory_tool(blob, by_tool)


__all__ = [
    "FAMILY_MEDICAL",
    "FAMILY_RESEARCH",
    "declared_family_capabilities",
    "index_inventory_tools",
    "inventory_nonempty",
    "inventory_pairs",
    "match_inventory_tool",
    "match_named_family_capability",
    "match_named_inventory_tool",
    "match_slot_inventory_tool",
]
