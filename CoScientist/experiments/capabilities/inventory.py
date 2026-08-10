"""Inventory indexing + capability-family helpers (critique advisory, MCP repair bind/demote).

Repair: demote empty MCP → coder only when inventory empty or no tool covers a requested
capability. Prefer the primary operation of the ask.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

# request ↔ inventory tool text. Order docs-only; repair uses PRIMARY_CAP_PRIORITY.
CAPABILITY_SPECS: tuple[tuple[str, re.Pattern[str], re.Pattern[str]], ...] = (
    (
        "docking",
        re.compile(r"\b(dock|docking|vina|affinity\s*score|докинг|аффинност)", re.I),
        re.compile(r"\b(dock|docking|vina|affinity)\b|докинг|аффин", re.I),
    ),
    (
        "toxicity_profile",
        re.compile(
            r"\b(toxic|toxicity|ld50|ld₅₀|dili|hepato|cardio.?toxic|carcinogen|"
            r"токсич|гепато|кардиот|канцер|ld\s*50)\b", re.I),
        re.compile(
            r"\b(toxic|toxicity|ld50|dili|hepato|cardio|carcinogen|"
            r"molecule[_\s]?profile|general[_\s]?toxicity)\b|токсич|гепато|кардиот|канцер",
            re.I),
    ),
    (
        "molecule_generation",
        re.compile(
            r"\b("
            r"generat\w*\s+\w*\s*candidat|generat\w*.{0,40}mol\w*|"
            r"de\s*novo|generate_case_mols|generate_mols|"
            r"drug[_\s-]?like\s+mol\w*|new\s+drug\w*|suggest\s+\w*\s*mol\w*|"
            r"генерац\w*\s+молекул|кандидат\w*"
            r")\b", re.I),
        re.compile(
            r"\b(generate[_\s]?case[_\s]?mols|generate[_\s]?mols|generat\w*\s+mol|"
            r"molecule generation|gan)\b", re.I),
    ),
    (
        "molecular_properties",
        re.compile(
            r"\b(synthesizab|sa\s*score|molecular\s+propert|smiles2prop|дескриптор|"
            r"синтезируем\w*|synspace)\b", re.I),
        re.compile(r"\b(smiles2prop|molecular\s+propert|descriptor|sa\s*score|synthesiz|synspace)\b", re.I),
    ),
    (
        "chemical_clustering",
        re.compile(r"\b(cluster|clustering|chemical.?space|butina|tsne|кластер)", re.I),
        re.compile(r"\b(cluster|clustering|butina|tsne|chemical.?space)\b", re.I),
    ),
    (
        "dataset_curation",
        re.compile(r"\b(dataset.?overview|curat|metabolite.?selection|dedup|обзор\s+датасет)\b", re.I),
        re.compile(r"\b(dataset.?overview|curat|dedup|metabolite.?selection)\b", re.I),
    ),
    (
        "medchem_filter",
        re.compile(r"\b(medchem|apply.?medchem.?filter|фильтр\w*\s+medchem)\b", re.I),
        re.compile(r"\b(medchem|apply.?medchem)\b", re.I),
    ),
)

# Multi-family asks: bind primary op (not first regex hit).
PRIMARY_CAP_PRIORITY: tuple[str, ...] = (
    "molecule_generation", "chemical_clustering", "dataset_curation",
    "toxicity_profile", "medchem_filter", "molecular_properties", "docking",
)


def index_inventory_tools(available_tools: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map tool name → {tool, server_id, description} (first wins)."""
    out: dict[str, dict[str, Any]] = {}
    for item in available_tools:
        if not isinstance(item, dict):
            continue
        tool = str(item.get("tool") or item.get("name") or "").strip()
        server_id = str(item.get("server_id") or "").strip()
        if not tool or not server_id:
            continue
        out.setdefault(tool, {"tool": tool, "server_id": server_id, "description": item.get("description") or ""})
    return out


def inventory_pairs(inventory: Iterable[dict[str, Any]]) -> set[tuple[str, str]]:
    """Exact (server_id, tool) pairs for registry feasibility checks."""
    return {
        (str(item["server_id"]), str(item.get("tool") or item.get("name")))
        for item in inventory
        if isinstance(item, dict) and item.get("server_id") and (item.get("tool") or item.get("name"))
    }


def request_capabilities(request: str) -> set[str]:
    text = request or ""
    return {name for name, req_re, _ in CAPABILITY_SPECS if req_re.search(text)}


def tool_capabilities(tool_name: str, description: str = "") -> set[str]:
    blob = f"{str(tool_name or '').replace('_', ' ')} {description}"
    return {name for name, _, tool_re in CAPABILITY_SPECS if tool_re.search(blob)}


def is_paper_demo_tool(tool_name: str, description: str = "") -> bool:
    blob = f"{tool_name} {description}".lower()
    return bool(re.search(
        r"\b(reproduce_figure|paper\s+fig|characterised molecules|characterized molecules|"
        r"six characterised|demo.?only)\b", blob,
    ))


def inventory_covers_capabilities(by_tool: dict[str, dict[str, Any]], needed: set[str]) -> bool:
    """True when a non-demo inventory tool covers any needed family."""
    if not needed or not by_tool:
        return False
    for tool, item in by_tool.items():
        if is_paper_demo_tool(tool, str(item.get("description") or "")):
            continue
        if tool_capabilities(tool, str(item.get("description") or "")) & needed:
            return True
    return False


def match_inventory_tool(
    blob: str, by_tool: dict[str, dict[str, Any]], *, source_request: str = "",
) -> dict[str, Any] | None:
    """Pick inventory tool for empty MCP bind: exact name, else PRIMARY_CAP_PRIORITY."""
    task_text = (blob or "").strip()
    request = (source_request or "").strip()
    combined = f"{task_text}\n{request}".strip()
    if not combined or not by_tool:
        return None
    for text in (task_text, combined):
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", text):
            if token in by_tool:
                return by_tool[token]
        low = text.lower().replace("-", "_")
        for tool, item in by_tool.items():
            if tool.lower() in low:
                return item
    # Family bind needs a task-local capability signal (not plan-level alone).
    task_caps = request_capabilities(task_text) if task_text else set()
    if not task_caps:
        return None
    needed = request_capabilities(combined)
    if not needed:
        return None
    preferred = [c for c in PRIMARY_CAP_PRIORITY if c in needed and c in task_caps]
    preferred += [c for c in PRIMARY_CAP_PRIORITY if c in needed and c not in task_caps]
    for cap in preferred:
        for tool, item in by_tool.items():
            if is_paper_demo_tool(tool, str(item.get("description") or "")):
                continue
            if cap in tool_capabilities(tool, str(item.get("description") or "")):
                return item
    return None


__all__ = [
    "CAPABILITY_SPECS", "PRIMARY_CAP_PRIORITY", "index_inventory_tools",
    "inventory_covers_capabilities", "inventory_pairs", "is_paper_demo_tool",
    "match_inventory_tool", "request_capabilities", "tool_capabilities",
]
