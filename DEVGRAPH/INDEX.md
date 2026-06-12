# DEVGRAPH — Index (FAST BOOT)

> Read this first. One-screen map of the whole graph. Full spec: [README.md](./README.md).
> Update the relevant row whenever you finish a task (status, `updated`, "now").

**Project card:** [project_card.md](./project_card.md) — what the system can do today.
**Roadmap:** [ROADMAP.md](./ROADMAP.md) — ordered work + execution state (R-steps). Read for "what to build next".
**Last graph update:** 2026-06-12 · **Branch when seeded:** `refactoring_full_pipe`

## Features

| ID | Title | Status | Updated | Derives | Now (one line) |
|----|-------|--------|---------|---------|----------------|
| [F000](./features/F000-baseline-orchestrator.md) | Baseline orchestrator + agents | done | 2026-06-12 | — | ADK orchestrator/RAG/FEDOT.MAS/paper pipeline; +`ResilientAgentTool` & CLI answer-extraction fixes (F000.A2). |
| [F001](./features/F001-hitl.md) | Human-in-the-Loop (HITL) | **in_progress** | 2026-06-12 | F000 | Live path **confirmed**: callback on CoderAgent outward-facing cmds (F001.A2) + headless auto-reject; edit/tool paths still TODO. |
| [F002](./features/F002-coder-agent.md) | CoderAgent + sandbox exec | done | 2026-06-11 | F000 | Sandbox coder/git engineer shipped (#268); no benchmark yet. |
| [F003](./features/F003-research-agent-workflow.md) | ResearchAgent workflow | done | 2026-06-12 | F000 | Literature pipeline **verified e2e** (F003.A3); `MCP__*` paper URLs added to .env. **Tavily web search DISABLED** (F003.A2). |
| [F004](./features/F004-medical-agent-frontend.md) | Medical agent + ADK frontend | done | 2026-06-11 | F000 | PubMed/PICO/DICOM + web UI (#262); answers clinically unvalidated. |
| [F005](./features/F005-tool-web-search.md) | Tool web search (MCP-registry discovery) | done | 2026-06-11 | F000 | Discover MCP servers via public registries (#260); NOT Tavily. Adapters brittle. |
| [F006](./features/F006-critic-executor-refinement.md) | Pre/post critic + executor | done | 2026-06-12 | F000 | Self-correction loop (#249); fixed delegation-arg clobber → `KeyError: 'request'` (F006.A3); no over-block eval. |
| [F007](./features/F007-paper-analysis-pipeline.md) | Paper analysis & parsing | done | 2026-06-11 | F000 | Marker→Chroma→S3 + QA + MCP (#204/#239/#256); PDF extraction gap. |
| [F008](./features/F008-observability-opik.md) | Observability — Opik tracer | done | 2026-06-11 | F000 | Opik tracing + orchestrator prompt fix (#225); enablement undocumented. |
| [F009](./features/F009-rag-tool-retrieval.md) | RAG tool/MCP retrieval (DB) | done | 2026-06-12 | F000 | Hybrid RAG-DB retrieval + rerank (#212); now **degrades gracefully** when DB down (F009.A2); hard dep on rag_tools + Postgres. (≠ F005.) |
| [F010](./features/F010-fedotmas-integration.md) | FEDOT.MAS integration | done | 2026-06-11 | F000 | Text→ML pipelines (#211, fix #224); needs SSH `fedotmas` install. |
| [F011](./features/F011-dataset-collection-mcp.md) | Dataset-collection MCP | done | 2026-06-11 | F000 | Build datasets from papers (#196, fix #269); coverage unchecked. |
| [F012](./features/F012-papers-search-mcp.md) | Papers-search MCP (OpenAlex) | done | 2026-06-11 | F000 | OpenAlex search/download (#196); needs API key, wiring unconfirmed. |
| [F013](./features/F013-chemical-mcp-docker.md) | Chemical MCP + docker | done | 2026-06-11 | F000 | Dockerized chemistry tools (#187, S3 #197); most verified capabilities. |
| [F014](./features/F014-benchmark-reliability-dataset-s.md) | Benchmark reliability (dataset_S) | **in_progress** | 2026-06-11 | F000 | All 3 errors confirmed via **Opik** (yesterday=qwen3): empties+APIErrors, tool-name hallucination, runaways (≤81 LLM calls). Live test: **pinning routes deterministically**, qwen ≠ flakiness fix. Fix dir = F015 (D1) + re-pin. |
| [F015](./features/F015-experiments-orchestration-module.md) | Experiments orchestration module (АМ) — **epic** | **proposed** | 2026-06-11 | F010, F006 | Designed in approved ТП (S009). Decomposed into F015a–F015h (below). The orchestration fix for F014's loops & tool-not-found, w/o touching MCP tools. |
| [F015a](./features/F015a-experiment-planner.md) | АМ: experiment planner (JSON step-plan DAG) | proposed | 2026-06-11 | F015, F006 | Routine/ReWOO/Data-Interpreter; reverse planner's no-tools rule + inject live inventory. |
| [F015b](./features/F015b-plan-critic-loop.md) | АМ: plan-critic loop (bounded) | proposed | 2026-06-11 | F015, F006 | Deterministic gate first (folds F015c), LLM critic advisory; self-critique unreliable. |
| [F015c](./features/F015c-tool-sufficiency-check.md) | АМ: tool-sufficiency / capability-gap (shared) | proposed | 2026-06-11 | F015 | **Build first.** MCP-Zero/RAG-MCP/AnyTool. Re-scoped: ensures right *servers*, not full fix (see F015g.D1). Fail-closed on backend-down. |
| [F015d](./features/F015d-repo-search-agent.md) | АМ: repo-search (literature-first) | proposed | 2026-06-11 | F015 | AutoSOTA/SUPER; "no repo found" → HITL. |
| [F015e](./features/F015e-alembic-repo-to-mcp.md) | АМ: Alembic repo→MCP pipeline | **in_progress** | 2026-06-11 | F015, F002 | Mostly **built on branches** (5-agent chain + docker). Integrate; packaging/import-path blocker. |
| [F015f](./features/F015f-tool-deploy-registration.md) | АМ: deploy + register/reuse + sandbox | proposed | 2026-06-11 | F015, F015e | ScaleMCP/AutoMCP; the missing registration seam + tool-poisoning security. |
| [F015g](./features/F015g-single-step-fedotmas-dispatch.md) | АМ: single-step FEDOT.MAS dispatch | proposed | 2026-06-12 | F015, F010 | **D1 RESOLVED**: per-step server filter (existing RAG seam) + tool details in task text + `generate_config`→verify→`build_and_run` net. Reuse Alembic guards. |
| [F015h](./features/F015h-am-eval-harness.md) | АМ: eval harness on dataset_S | proposed | 2026-06-11 | F015, F014 | Acceptance gate; proves the fix via Opik (anti-entrenchment). |

Legend: `proposed` · `in_progress` (=TODO) · `blocked` · `done` (=CLOSE) · `rejected` (=REJECT) · `superseded`.

## Hot / attention
- **F001 HITL** — still `in_progress`. Live path now **confirmed** (F001.A2): fires
  via **callback** on the CoderAgent's outward-facing-command approval; the console
  handler now auto-rejects with no TTY (headless-safe). Still open: tool/internal_loop
  paths, edit/select flows, frontend rendering, `timeout_seconds`. See its `## ⚠ Pitfalls`.
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
- **F015 experiments module (АМ)** — epic decomposed into F015a–F015h. **F015g.D1 RESOLVED
  2026-06-12** (per-step server filter via existing RAG seam + tool details in step text +
  re-plan after Alembic + `generate_config`→verify→`build_and_run` net). Build order now
  unblocked: **F015c → F015a → F015b → F015g → F015f → F015d → F015e(integrate)**, F015h alongside.
  Dedup audit done: Alembic already implements the loop-guards / guard-retry / real-call
  validation — **reuse, don't rebuild** (tables in F015e/F015f/F015g). Provider flakiness separate.
- **No automated benchmarks/evals exist anywhere** (project card: "Benchmarks: none
  recorded"). Almost every `done` feature carries a "no eval" TODO — first team to
  add an eval harness should record results in that feature's `benchmarks:`.
  *Note (2026-06-12):* the **first unit tests** landed — `tests/unit/` (12 pass,
  offline) cover the F000.A2 / F001.A2 / F006.A3 robustness fixes. Regression tests,
  not yet an eval harness. The repo's only prior test (`tests/integration/`) needs
  `chromadb` + ITMO VPN and doesn't even collect locally.
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
- Features: **F016** (top-level). F015 sub-features use letter suffixes F015a–F015h; next free **F015i**.
- Sources: **S033**
