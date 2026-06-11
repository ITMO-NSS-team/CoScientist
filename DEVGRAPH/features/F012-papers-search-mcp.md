---
id: F012
title: Papers-search MCP server (OpenAlex)
type: feature
status: done
created: 2026-06-11
updated: 2026-06-11
owners: [ITMO-NSS-team]
derives_from: [F000]
depends_on: [F000]
sources: []
tags: [papers, search, openalex, mcp]
code:
  - mcp-servers/papers-search-mcp-server/papers_search_server.py
  - mcp-servers/papers-search-mcp-server/openalex_client.py
benchmarks: []
---

## Goal
An MCP server to search and download scientific papers via OpenAlex, feeding
ResearchAgent (F003) and dataset collection (F011).

## Current state
Present since #196 (commit `875487b`, which also fixed this server). Lives in
`mcp-servers/papers-search-mcp-server/` with a dedicated `openalex_client.py`.

## Attempts
### F012.A1 — papers-search MCP + fix (#196) · earlier · outcome: success
- **Method:** OpenAlex client wrapped as an MCP server for search/download.
- **Evidence:** commit `875487b` (#196); `papers_search_server.py`, `openalex_client.py`.

## ✅ TODO
- [ ] "Search and download papers from OpenAlex" is unchecked in tools_checklist.md —
      confirm it's wired into the live ResearchAgent path.
- [ ] Needs `SERVICES__OPENALEX_API_KEY`; document rate limits.

## ⚠ Pitfalls / Known problems
- OpenAlex API key / rate limits gate this; without the key, ResearchAgent's
  "find and download papers" escalation can't actually fetch anything.

## Symbols
- `mcp-servers/papers-search-mcp-server/openalex_client.py` — OpenAlex API client.
- `mcp-servers/papers-search-mcp-server/papers_search_server.py` — MCP entrypoint.
