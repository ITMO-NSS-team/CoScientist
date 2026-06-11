---
id: F005
title: Tool web search — MCP-server discovery via public registries
type: feature
status: done
created: 2026-06-11
updated: 2026-06-11
owners: [ITMO-NSS-team]
derives_from: [F000]
depends_on: [F000]
sources: []
tags: [mcp, tool-discovery, web-registry]
code:
  - CoScientist/tools/tools_web_search/engine.py:MCPSearchTool
  - CoScientist/tools/tools_web_search/adapters.py
  - CoScientist/tools/servers_web_search.py:search_mcp_servers
benchmarks: []
---

## Goal
Let agents **discover MCP servers** by querying public MCP registries on the web
(natural-language → matching servers), as an alternative/complement to the RAG-DB
retrieval in F009. This is "search the web *for tools*", not general web search.

## Current state
Shipped in #260 (commit `fef5c6f`). `tools_web_search/` holds an `MCPSearchTool`
engine with registry adapters (`_McpServersCom`, `_McpServersOrg`);
`servers_web_search.py:search_mcp_servers` is the agent-callable tool that runs a
query and **accumulates** unique results across calls in session state
(`accumulated_web_mcps`). Returns up to 15 LLM-friendly server descriptions.

> Disambiguation: F005 = scrape public **registries** for servers. F009 = retrieve
> from the local **RAG DB**. F003 owns general web search (Tavily) — see its
> pitfall about Tavily being disabled.

## Attempts
### F005.A1 — Tool web search / MCP discovery (#260) · 2026-06-08 · outcome: success
- **Method:** aggregate two registry adapters behind `MCPSearchTool`; expose
  `search_mcp_servers` with cross-call accumulation in `tool_context.state`.
- **Result:** agents can find MCP servers for a capability by web query.
- **Evidence:** commit `fef5c6f` (#260); `MCPSearchTool`, `search_mcp_servers`.

## ✅ TODO
- [ ] Registry scraping is brittle (adapters target specific sites); add a fallback/health check.
- [ ] No eval of discovery precision (are the top-15 servers actually relevant/usable?).

## ⚠ Pitfalls / Known problems
- Adapters (`_McpServersCom`, `_McpServersOrg`) depend on external site structure —
  a registry HTML change silently breaks discovery. Verify adapters before blaming the agent.
- Don't confuse with Tavily/general web search (F003) — different code path, different failure mode.

## Symbols
- `CoScientist/tools/tools_web_search/engine.py:MCPSearchTool` — registry-search engine.
- `CoScientist/tools/tools_web_search/adapters.py` — `_McpServersCom` / `_McpServersOrg` registry adapters.
- `CoScientist/tools/servers_web_search.py:search_mcp_servers` — agent tool, accumulates results in session state.
