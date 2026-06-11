# DEVGRAPH — Index (FAST BOOT)

> Read this first. One-screen map of the whole graph. Full spec: [README.md](./README.md).
> Update the relevant row whenever you finish a task (status, `updated`, "now").

**Project card:** [project_card.md](./project_card.md) — what the system can do today.
**Last graph update:** 2026-06-11 · **Branch when seeded:** `refactoring_full_pipe`

## Features

| ID | Title | Status | Updated | Derives | Now (one line) |
|----|-------|--------|---------|---------|----------------|
| [F000](./features/F000-baseline-orchestrator.md) | Baseline orchestrator + agents | done | 2026-06-11 | — | Pre-DEVGRAPH baseline: ADK orchestrator, RAG retrieval, FEDOT.MAS, paper pipeline. |
| [F001](./features/F001-hitl.md) | Human-in-the-Loop (HITL) | **in_progress** | 2026-06-11 | F000 | Module + types exist (`CoScientist/hitl/`); **live wiring unverified**, eval TODO. |
| [F002](./features/F002-coder-agent.md) | CoderAgent + sandbox exec | done | 2026-06-11 | F000 | Sandbox coder/git engineer shipped (#268); no benchmark yet. |
| [F003](./features/F003-research-agent-workflow.md) | ResearchAgent workflow | done | 2026-06-11 | F000 | Literature/RAG retrieval + upload/cleanup (#265). **Tavily web search DISABLED** (VPN/TLS hang — F003.A2). |
| [F004](./features/F004-medical-agent-frontend.md) | Medical agent + ADK frontend | done | 2026-06-11 | F000 | PubMed/PICO/DICOM + web UI (#262); answers clinically unvalidated. |
| [F005](./features/F005-tool-web-search.md) | Tool web search (MCP-registry discovery) | done | 2026-06-11 | F000 | Discover MCP servers via public registries (#260); NOT Tavily. Adapters brittle. |
| [F006](./features/F006-critic-executor-refinement.md) | Pre/post critic + executor | done | 2026-06-11 | F000 | Self-correction loop over tool calls (#249); no over-block eval. |
| [F007](./features/F007-paper-analysis-pipeline.md) | Paper analysis & parsing | done | 2026-06-11 | F000 | Marker→Chroma→S3 + QA + MCP (#204/#239/#256); PDF extraction gap. |
| [F008](./features/F008-observability-opik.md) | Observability — Opik tracer | done | 2026-06-11 | F000 | Opik tracing + orchestrator prompt fix (#225); enablement undocumented. |
| [F009](./features/F009-rag-tool-retrieval.md) | RAG tool/MCP retrieval (DB) | done | 2026-06-11 | F000 | Hybrid RAG-DB retrieval + rerank (#212); hard dep on rag_tools + Postgres. (≠ F005 web registries.) |
| [F010](./features/F010-fedotmas-integration.md) | FEDOT.MAS integration | done | 2026-06-11 | F000 | Text→ML pipelines (#211, fix #224); needs SSH `fedotmas` install. |
| [F011](./features/F011-dataset-collection-mcp.md) | Dataset-collection MCP | done | 2026-06-11 | F000 | Build datasets from papers (#196, fix #269); coverage unchecked. |
| [F012](./features/F012-papers-search-mcp.md) | Papers-search MCP (OpenAlex) | done | 2026-06-11 | F000 | OpenAlex search/download (#196); needs API key, wiring unconfirmed. |
| [F013](./features/F013-chemical-mcp-docker.md) | Chemical MCP + docker | done | 2026-06-11 | F000 | Dockerized chemistry tools (#187, S3 #197); most verified capabilities. |
| [F014](./features/F014-benchmark-reliability-dataset-s.md) | Benchmark reliability (dataset_S) | **in_progress** | 2026-06-11 | F000 | All 3 errors confirmed via **Opik** (yesterday=qwen3): empties+APIErrors, tool-name hallucination, runaways (≤81 LLM calls). Live test: **pinning routes deterministically**, qwen ≠ flakiness fix. Fix dir = F015 (D1) + re-pin. |
| [F015](./features/F015-experiments-orchestration-module.md) | Experiments orchestration module (АМ) | **proposed** | 2026-06-11 | F010, F006 | Designed in approved ТП (S009): plan→critic→bounded-execute + per-step **tool-sufficiency** + **Alembic** builds missing MCP tools. The orchestration fix for F014's loops & tool-not-found, w/o touching MCP tools. Alembic on branches; not integrated here. |

Legend: `proposed` · `in_progress` (=TODO) · `blocked` · `done` (=CLOSE) · `rejected` (=REJECT) · `superseded`.

## Hot / attention
- **F001 HITL** — the only `in_progress` feature and biggest open thread. Confirm
  where HITL actually fires in the live pipeline before extending. See its `## ⚠ Pitfalls`.
- **Tavily web search is DISABLED** (F003.A2): the TLS/SSE stream to
  `mcp.tavily.com` hangs in `list_tools` and kills the whole ResearchAgent run.
  Attributed to the lab VPN, but **root cause unconfirmed** — may be local-machine
  VPN/DNS/proxy config; diagnose (F003 TODO) before re-enabling. No live web search
  meanwhile.
- **F014 benchmark reliability (dataset_S)** — Opik-backed analysis done (F014.A2/A3).
  Read runs from **Opik** (memory [[opik-tracing-access]]), not stdout. Conclusions:
  (1) all 3 errors are real, mostly on **qwen3** (empties+`finish=None`, hallucinated
  tool names, runaways ≤81 LLM calls / 700s ceiling); (2) **pinning routes
  deterministically** (gpt-oss→DeepInfra when pinned; grab-bag incl. a flaky near-empty
  when off) but is **currently disabled** in prod; (3) **qwen is not a flakiness fix**.
  Next: re-pin gpt-oss-120b, loop guard, constrain tool names; full A/B needs real infra.
- **F015 experiments module (АМ)** is the *structural* fix for F014's loops &
  tool-not-found: plan→critic→bounded-execute + per-step tool-sufficiency + Alembic
  (build missing MCP tools, don't edit existing ones — decision F014.D1, spec S009).
  Building blocks exist (PlannerAgent, CriticAgent, code_exec_server, alembic branches);
  the integrated loop isn't wired on this branch yet. Provider flakiness stays separate.
- **No automated benchmarks/evals exist anywhere** (project card: "Benchmarks: none
  recorded"). Almost every `done` feature carries a "no eval" TODO — first team to
  add an eval harness should record results in that feature's `benchmarks:`.
- **Capability gaps** (from `tools_checklist.md`): molecule/reaction extraction from
  **PDF** (F007/F013), training/AutoML for molecule generation & properties, paper-DB
  Q&A wiring (F007/F012).

## Backfill note
F003–F013 were reconstructed from git history (PRs #187–#268) on 2026-06-11, not
from first-hand session work. Their attempts cite commits; treat finer details as
inferred until confirmed in a live session.

## Sources sub-graph
Registry: [sources/INDEX.md](./sources/INDEX.md). Trust values change as ideas are
tried — check before relying on a cited idea.

## ID counters (next free)
- Features: **F016**
- Sources: **S010**
