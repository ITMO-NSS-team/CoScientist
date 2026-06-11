---
id: F003
title: ResearchAgent workflow (literature + web + RAG, paper upload/cleanup)
type: feature
status: done
created: 2026-06-11
updated: 2026-06-11
owners: [ITMO-NSS-team]
derives_from: [F000]
depends_on: [F000, F007, F009, F012]
sources: []
tags: [research, retrieval, literature, rag]
code:
  - CoScientist/agents/research_callbacks.py:cleanup_uploaded_papers
  - CoScientist/tools/research_tools.py:_http_mcp_toolset
  - CoScientist/tools/research_tools.py:paper_analysis_toolset_instance
  - CoScientist/tools/research_tools.py:papers_search_toolset_instance
benchmarks: []
---

## Goal
The ResearchAgent end-to-end workflow: gather reliable knowledge from web, RAG,
and paper search; handle user-uploaded papers; escalate empty results by
reformulating into "find and download papers about <expanded topic>".

## Current state
Shipped in #265 (commit `9d09dfe`). ResearchAgent wires together HTTP MCP toolsets
for paper analysis and paper search (`research_tools.py`) plus callbacks that
resolve a local papers dir and clean up uploaded papers per user/session
(`research_callbacks.py`). Routing/escalation behavior is defined in
`catalog.py` (ResearchAgent entry).

## Attempts
### F003.A1 — ResearchAgent workflow (#265) · 2026-06-09 · outcome: success
- **Method:** compose research as MCP toolsets (paper-analysis + papers-search)
  behind one agent, with per-session upload handling and an empty-result escalation rule.
- **Result:** agent retrieves literature/RAG knowledge and manages uploaded papers.
- **Evidence:** commit `9d09dfe` (#265); symbols in `research_tools.py` / `research_callbacks.py`.
### F003.A2 — Tavily MCP web search: timeout raise then **disable** · 2026-06-10 · outcome: failed
- **Method:** Tavily was wired as a streamable-HTTP MCP toolset
  (`websearch_toolset_instance`). First the timeout was raised 5s→90s (`79cb9c6`)
  to stop bounded hangs.
- **Result:** still broken behind the lab VPN — the external TLS/SSE stream to
  `mcp.tavily.com` negotiates but `list_tools` never returns; ADK raises
  `ConnectionError` and kills the **entire** ResearchAgent run. So Tavily was
  **disabled** (`websearch_toolset_instance = None`).
- **Evidence:** `CoScientist/tools/research_tools.py:27-38` (disable comment +
  commented-out toolset); the comment notes "every benchmark request failed at
  step 1 because of this."
- **Next:** re-enable only once Tavily traffic bypasses the VPN (route
  `mcp.tavily.com` direct); until then ResearchAgent relies on paper_analysis /
  papers_search MCP tools, not live web search.
- **⚠ Root cause UNCONFIRMED:** the "VPN breaks the stream" attribution comes from
  the in-code comment, not an independent diagnosis. It may instead be a
  **local-machine VPN/DNS/proxy/TLS-MITM config** on the box where this was caught
  — not an inherent VPN↔Tavily incompatibility. Treat as a hypothesis, not a
  settled fact, until F003.A3 reproduces it across environments (see TODO).

## ✅ TODO
- [ ] No eval of retrieval quality / escalation success — add a small QA set.
- [ ] Confirm uploaded-paper cleanup is triggered on all exit paths (not just happy path).
- [ ] **Diagnose Tavily failure properly before trusting the "VPN" verdict (F003.A2).**
      Check it's not a *local-machine* VPN issue: reproduce on a different network /
      machine, and off-VPN. Concretely: (a) `curl -v https://mcp.tavily.com/mcp/...`
      on-VPN vs off-VPN; (b) try from a second host on the same VPN; (c) check
      DNS/proxy/TLS-inspection on the local box (split-tunnel, corporate MITM cert).
      If it fails off-VPN too → not VPN. If it works on another machine on the same
      VPN → local config, not the VPN itself. Record the result as F003.A3 with evidence.
- [ ] Re-enable Tavily web search once the real blocker is identified & cleared (see F003.A2).

## ⚠ Pitfalls / Known problems
- **Tavily web search is currently DISABLED** (`websearch_toolset_instance = None`,
  `research_tools.py`). Don't assume ResearchAgent does live web search — it does
  not. Re-enabling it naïvely re-introduces a hang that kills the whole run at
  step 1. The blocker is attributed to the lab VPN (F003.A2) but that cause is
  **unconfirmed** — could be local-machine VPN/DNS/proxy config; diagnose before
  re-enabling.
- `_http_mcp_toolset` returns `None` if the MCP URL is unset → the agent silently
  loses a capability. Check URLs are configured before assuming research tools exist.

## Symbols
- `CoScientist/tools/research_tools.py:_http_mcp_toolset` — builds an HTTP MCP toolset (or None if URL missing).
- `CoScientist/agents/research_callbacks.py:cleanup_uploaded_papers` — per-user/session upload cleanup.
