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
### F009.A3 — Stop truncating tool descriptions + harden get_server_info · 2026-06-12 · outcome: success
- **Method:** two fixes in `retrieval_tools.py`. (1) **Critical:** `list_available_tools`
  (the orchestrator's only window into MCP tools, `agents.py:314`) truncated each tool's
  `description` to **200 chars** (`[:200]`), so the orchestrator never saw the full
  description — including the concrete cases/datasets/models a parameterized tool supports.
  Removed the cap (now matches `retrieve_tools`, which already returned full descriptions).
  (2) **Similar problem found:** `get_server_info` did raw `PostgresClient.initialize()/get_server()`
  with NO error handling — same crash-on-DB-down bug F009.A2 fixed elsewhere. Wrapped it to
  degrade to `{"status":"unavailable"}` with a `finally` close.
- **Result:** the orchestrator now sees full tool descriptions (can ground a request against
  the real cases a tool lists); a registry/DB outage no longer crashes `get_server_info`.
  This directly attacks the "invented case → false success" failure (F015c/F015h).
- **Evidence:** `retrieval_tools.py:216` (cap removed) + `get_server_info` try/except; py_compile OK.
- **Note (verified):** `MCPServer` has NO tools field, so `get_server_info` returns server
  metadata (url/description), not a tool list — the per-tool **description** is the only
  plan-time source of a tool's cases, which is exactly why the truncation was critical.
  The web-registry path (`to_agent_text`) does NOT truncate descriptions (only caps server count).
### F009.A4 — Add list_server_tools (full toolset + full descriptions) for plan-time grounding · 2026-06-12 · outcome: success
- **Method:** even after F009.A3, the RAG path (`list_available_tools`) returns only top-k tools
  and a **500-char** description — that cap is in rag_tools' embedding text
  (`ingestion/text_builder.py:223`), NOT the stored row. Added `list_server_tools(server_id)` →
  `PostgresClient.get_tools_by_server` → returns a server's COMPLETE toolset with each tool's
  **full** description + `input_schema` (graceful on DB down). Wired into the orchestrator
  (`agents.py` import + `FunctionTool(list_server_tools)`) + the orchestrator prompt.
- **Result (live, Postgres up via VPN, server `d36e3d994404e957`):** full descriptions —
  `generate_case_mols` = **1729 chars** (vs 500 via RAG) and enumerates **all 6 cases**:
  `alzheimer, skleroz, cancer, parkinson, dyslipidemia, drug_resist`. So the orchestrator can now
  read a tool's real cases at PLAN time → the GSK-3β "false success" (F015h) is groundable WITHOUT
  hardcoding (research bridges GSK-3β→Alzheimer → `case='alzheimer'`, which exists).
- **Evidence:** live query (count=5, 6 cases parsed); `retrieval_tools.py:list_server_tools`;
  prompt render shows `list_server_tools`. **⚠ The `agents.py` wiring (import + FunctionTool) is
  applied in the working tree but NOT yet in its own commit** — `agents.py` currently holds a
  parallel session's uncommitted changes (ResilientAgentTool etc.), and partial hunk staging is
  unavailable here, so the 2 lines land with that file's next commit.
- **Finding:** the ~500-char limit is an **embedding-text** cap (for vector search), not the
  stored description — full grounding info IS available at plan time via `get_tools_by_server`;
  RAG snippets are only for ranking.
### F009.A5 — Guard `{accumulated_tools}` in the ToolReranker instruction (Bug D) · 2026-06-12 · outcome: success
- **Method:** the ToolReranker instruction (`prompts.py:121`) interpolates
  `{accumulated_tools}`, which the upstream ToolRetriever's `retrieve_tools` populates.
  When the retriever's LLM didn't accumulate (no tool call, or DB returned
  `unavailable` per F009.A2), the key was absent and ADK's `inject_session_state`
  raised `KeyError: Context variable not found: accumulated_tools`, killing the
  **whole** run (hit constantly on the dataset_S → TaskExecutor path). Fix: add
  `callbacks.py:before_tool_reranker_agent` as the reranker's `before_agent_callback`
  to seed `state['accumulated_tools'] = []` before the prompt renders — a mirror of
  the existing `before_fullset_reranker_agent` guard for `{accumulated_web_mcps}`.
- **Result:** dataset_S / literature runs no longer crash on `accumulated_tools`;
  this is what made the clean F014.A4 model A/B possible (the path had to survive).
- **Evidence:** crash traces (lit `019ebb9c`, `019ebbb2` — span err "Context variable
  not found: accumulated_tools"); fix in `callbacks.py:before_tool_reranker_agent`
  wired at `agents.py` `tool_reranker_agent`; unit tests
  `tests/unit/test_accumulated_tools_guard.py` (2 pass).

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
