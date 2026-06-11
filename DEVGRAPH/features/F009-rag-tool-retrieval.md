---
id: F009
title: RAG-based tool / MCP-server retrieval
type: feature
status: done
created: 2026-06-11
updated: 2026-06-11
owners: [ITMO-NSS-team]
derives_from: [F000]
depends_on: [F000]
sources: []
tags: [rag, tool-retrieval, mcp, foundational]
code:
  - CoScientist/tools/retrieval_tools.py:RetrievalToolSet
benchmarks: []
---

## Goal
Let the system discover the right tools/MCP servers for a task via RAG (hybrid
dense+sparse retrieval + reranking) over a local DB, instead of hard-wiring a
fixed tool list. Foundational capability the TaskExecutor/ToolRetriever agents
rely on. (Web-registry discovery is a separate path — F005.)

## Current state
Shipped in #212 (commit `b37cb1f`). `retrieval_tools.py:RetrievalToolSet` exposes
RAG retrieval over the `rag_tools` DB. Backed by the external `rag_tools` package
(GitHub) + Postgres (see project card / README). Note: `search_mcp_servers` is
**not** part of F009 — that's the web-registry path in F005.

## Attempts
### F009.A1 — Tool RAG (#212) · earlier · outcome: success
- **Method:** integrate `rag_tools` for hybrid retrieval + rerank; expose as a
  toolset and a `search_mcp_servers` discovery call.
- **Result:** ToolRetriever/TaskExecutor can find appropriate MCP tools by task.
- **Evidence:** commit `b37cb1f` (#212); `RetrievalToolSet`, `search_mcp_servers`.

## ✅ TODO
- [ ] No retrieval@k / precision eval for tool discovery recorded.
- [ ] Document Postgres + `rag_tools` setup as a hard dependency for this to work.

## ⚠ Pitfalls / Known problems
- Hard external deps: needs the `rag_tools` git package **and** a populated Postgres.
  If empty/unreachable, tool discovery returns nothing and downstream agents fail
  quietly — check the DB before debugging the agents.

## Symbols
- `CoScientist/tools/retrieval_tools.py:RetrievalToolSet` — RAG retrieval toolset over the rag_tools DB.
