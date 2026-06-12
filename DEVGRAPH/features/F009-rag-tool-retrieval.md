---
id: F009
title: RAG-based tool / MCP-server retrieval
type: feature
status: done
created: 2026-06-11
updated: 2026-06-12
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
### F009.A2 — Graceful degradation when the tool-RAG DB is down · 2026-06-12 · outcome: success
- **Method:** the pitfall assumed "fail quietly", but reality was worse — with
  Postgres/qdrant unreachable, `list_available_tools` / `retrieve_tools` →
  `rag_tools.create_manager()` → `PostgresClient.initialize()` raised
  `ConnectionRefusedError`/`TimeoutError`, which propagated and **crashed the whole
  orchestrator run** (the run never reached the literature tools). Fix: factor a
  `_create_rag_manager` helper and wrap DB access; on failure return
  `{"status": "unavailable", "tools": []}` so the orchestrator falls back to
  ResearchAgent instead of dying.
- **Result:** literature/knowledge queries no longer depend on the tool-registry DB
  being up; when the DB **is** reachable it still returns tools (observed 5).
- **Evidence:** Opik trace `019eb4fc` (pre-fix `list_available_tools` err=True,
  `OSError` to `10.32.1.36:5432`); post-fix returns `unavailable`; code in
  `retrieval_tools.py` (`_create_rag_manager` + try/except in both entry points).
  See [[opik-tracing-access]].

## ✅ TODO
- [ ] No retrieval@k / precision eval for tool discovery recorded.
- [ ] Document Postgres + `rag_tools` setup as a hard dependency for this to work.

## ⚠ Pitfalls / Known problems
- Hard external deps: needs the `rag_tools` git package **and** a populated Postgres
  + qdrant. **Correction (F009.A2):** an unreachable DB did **not** "fail quietly" —
  it raised from `create_manager()` and **crashed the whole run**. Now it degrades to
  `{"status": "unavailable", "tools": []}` and the orchestrator falls back. Still
  check the DB before debugging agents, but a DB outage no longer kills the run.

## Symbols
- `CoScientist/tools/retrieval_tools.py:RetrievalToolSet` — RAG retrieval toolset over the rag_tools DB.
- `CoScientist/tools/retrieval_tools.py:_create_rag_manager` — builds the rag_tools manager; callers wrap it so a DB outage degrades to "unavailable" instead of crashing (F009.A2).
