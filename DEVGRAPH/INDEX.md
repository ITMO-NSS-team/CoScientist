# DEVGRAPH — Index (FAST BOOT)

> Read this first. One-screen map of the whole graph. Full spec: [README.md](./README.md).
> Update the relevant row whenever you finish a task (status, `updated`, "now").

**Project card:** [project_card.md](./project_card.md) — what the system can do today.
**Roadmap:** [ROADMAP.md](./ROADMAP.md) — ordered work + execution state (R-steps). Read for "what to build next".
**Last graph update:** 2026-06-13 · **Branch when seeded:** `refactoring_full_pipe`

## Features

Grouped by **status**. Feature files live in `features/{done,in_progress,todo}/`.
Changing a feature's status = move its row to the matching section here **and**
`git mv` the file to the matching folder (keep the file's `status:` in sync, fix
inbound links). Within a section, ordered by ID.

### ✅ Done

| ID | Title | Updated | Derives | Now (one line) |
|----|-------|---------|---------|----------------|
| [F000](./features/done/F000-baseline-orchestrator.md) | Baseline orchestrator + agents | 2026-06-13 | — | +`ResilientAgentTool`/CLI fixes (A2), staged prompt (A3), **callable-names + PLAN-step** (A4: tool-not-found 0/9, ground→plan→delegate). Planner/finalization A/B in flight. |
| [F002](./features/done/F002-coder-agent.md) | CoderAgent + sandbox exec | 2026-06-11 | F000 | Sandbox coder/git engineer shipped (#268); no benchmark yet. |
| [F003](./features/done/F003-research-agent-workflow.md) | ResearchAgent workflow | 2026-06-13 | F000 | Pipeline verified e2e (A3); +`MCP__*` URLs; **STOP/anti-thrash** rule (A4: stop re-calling empty `explore_chemistry_database`). **Tavily DISABLED** (A2). |
| [F004](./features/done/F004-medical-agent-frontend.md) | Medical agent + ADK frontend | 2026-06-12 | F000 | PubMed/PICO/DICOM + web UI (#262); upload intake renamed `upload_intake_before_model` (F004.A2); answers clinically unvalidated. |
| [F005](./features/done/F005-tool-web-search.md) | Tool web search (MCP-registry discovery) | 2026-06-11 | F000 | Discover MCP servers via public registries (#260); NOT Tavily. Adapters brittle. |
| [F006](./features/done/F006-critic-executor-refinement.md) | Pre/post critic + executor | 2026-06-12 | F000 | Self-correction loop (#249); fixed delegation-arg clobber → `KeyError: 'request'` (F006.A3); no over-block eval. |
| [F007](./features/done/F007-paper-analysis-pipeline.md) | Paper analysis & parsing | 2026-06-11 | F000 | Marker→Chroma→S3 + QA + MCP (#204/#239/#256); PDF extraction gap. |
| [F008](./features/done/F008-observability-opik.md) | Observability — Opik tracer | 2026-06-13 | F000 | Opik tracing + orchestrator prompt fix (#225); **F008.A2** reliable run→trace correlation (server-side `thread_id` filter + `trace_locator.py` manifest); enablement undocumented. |
| [F009](./features/done/F009-rag-tool-retrieval.md) | RAG tool/MCP retrieval (DB) | 2026-06-12 | F000 | Hybrid RAG-DB retrieval + rerank (#212); now **degrades gracefully** when DB down (F009.A2); hard dep on rag_tools + Postgres. (≠ F005.) |
| [F010](./features/done/F010-fedotmas-integration.md) | FEDOT.MAS integration | 2026-06-13 | F000 | Text→ML pipelines (#211, fix #224); needs SSH `fedotmas` install. **⚠ F010.A3: `molecule_generator` sub-agent returns an LLM paraphrase — drops the tool's S3 link & hallucinates SMILES; real molecules lost in S3. Fix→F015g.** |
| [F011](./features/done/F011-dataset-collection-mcp.md) | Dataset-collection MCP | 2026-06-11 | F000 | Build datasets from papers (#196, fix #269); coverage unchecked. |
| [F012](./features/done/F012-papers-search-mcp.md) | Papers-search MCP (OpenAlex) | 2026-06-11 | F000 | OpenAlex search/download (#196); needs API key, wiring unconfirmed. |
| [F013](./features/done/F013-chemical-mcp-docker.md) | Chemical MCP + docker | 2026-06-11 | F000 | Dockerized chemistry tools (#187, S3 #197); most verified capabilities. |

### 🔧 In progress

| ID | Title | Updated | Derives | Now (one line) |
|----|-------|---------|---------|----------------|
| [F001](./features/in_progress/F001-hitl.md) | Human-in-the-Loop (HITL) | 2026-06-12 | F000 | Live path **confirmed**: callback on CoderAgent outward-facing cmds (F001.A2) + headless auto-reject; edit/tool paths still TODO. |
| [F014](./features/in_progress/F014-benchmark-reliability-dataset-s.md) | Benchmark reliability (dataset_S) | 2026-06-12 | F000 | **F014.A4** full-pipeline A/B (n=5) **contradicts** earlier claims: qwen **0 empties**, gpt-oss 3-4 (`finish=None`), **pinning didn't reduce them** (4→3). Re-pin now wired via `.env` (variant 1). Larger provider-logged A/B still needed. |
| [F015a](./features/in_progress/F015a-experiment-planner.md) | АМ: experiment planner (JSON step-plan DAG) | 2026-06-12 | F015, F006 | **R05 done** (F015a.A1): `CoScientist/experiments/` plan-DAG schema + strict-JSON gen w/ repair; 3/3 dataset_S plans, surfaced capability gaps. Next: live inventory + ADK wiring (R09). |
| [F015e](./features/in_progress/F015e-alembic-repo-to-mcp.md) | АМ: Alembic repo→MCP pipeline | 2026-06-11 | F015, F002 | Mostly **built on branches** (5-agent chain + docker). Integrate; packaging/import-path blocker. |

### 📋 To do (proposed)

| ID | Title | Updated | Derives | Now (one line) |
|----|-------|---------|---------|----------------|
| [F015](./features/todo/F015-experiments-orchestration-module.md) | Experiments orchestration module (АМ) — **epic** | 2026-06-11 | F010, F006 | Designed in approved ТП (S009). Decomposed into F015a–F015h (below). The orchestration fix for F014's loops & tool-not-found, w/o touching MCP tools. |
| [F015b](./features/todo/F015b-plan-critic-loop.md) | АМ: plan-critic loop (bounded) | 2026-06-11 | F015, F006 | Deterministic gate first (folds F015c), LLM critic advisory; self-critique unreliable. |
| [F015c](./features/todo/F015c-tool-sufficiency-check.md) | АМ: tool-sufficiency / capability-gap (shared) | 2026-06-11 | F015 | **Build first.** MCP-Zero/RAG-MCP/AnyTool. Re-scoped: ensures right *servers*, not full fix (see F015g.D1). Fail-closed on backend-down. |
| [F015d](./features/todo/F015d-repo-search-agent.md) | АМ: repo-search (literature-first) | 2026-06-11 | F015 | AutoSOTA/SUPER; "no repo found" → HITL. |
| [F015f](./features/todo/F015f-tool-deploy-registration.md) | АМ: deploy + register/reuse + sandbox | 2026-06-11 | F015, F015e | ScaleMCP/AutoMCP; the missing registration seam + tool-poisoning security. |
| [F015g](./features/todo/F015g-single-step-fedotmas-dispatch.md) | АМ: single-step FEDOT.MAS dispatch | 2026-06-12 | F015, F010 | **D1 RESOLVED**: per-step server filter (existing RAG seam) + tool details in task text + `generate_config`→verify→`build_and_run` net. Reuse Alembic guards. |
| [F015h](./features/todo/F015h-am-eval-harness.md) | АМ: eval harness on dataset_S | 2026-06-11 | F015, F014 | Acceptance gate; proves the fix via Opik (anti-entrenchment). |
| [F016](./features/todo/F016-benchmark-evaluation.md) | Benchmark evaluation (external science-agent benchmarks) | 2026-06-12 | F000 | Maps ~30 survey benchmarks → capability; ✅ MCP-Bench/LitQA2/HypoBench/ChemBench runnable; needs an adapter harness. |
| [F017](./features/todo/F017-scientific-process-metamodel.md) | Scientific-process meta-model + runtime research graph | 2026-06-12 | F000, F015 | Ontology (6 layers) + per-study research graph the orchestrator **queries** (S033 docx). Module I/O contracts + validation. Epic; near-term slice = structured hypothesis(required_tools)+tool-status+"run only if tools available". ≠ DEVGRAPH. |

Legend: `proposed` · `in_progress` (=TODO) · `blocked` · `done` (=CLOSE) · `rejected` (=REJECT) · `superseded`.

## Hot / attention
- **⚠ FEDOT.MAS sub-agent fabricates molecules & drops the S3 result link (F010.A3, 2026-06-13).**
  Empirically confirmed: `generate_mols` (GenerativeMoleculeModels MCP) returns a **presigned S3 URL**
  to a results CSV — no inline SMILES. The `molecule_generator` sub-agent (ADK `LlmAgent`) returns
  the LLM's **paraphrase** (state `output_key` = `part.text`, not the raw `function_response`,
  `llm_agent.py:837-851`), so it **dropped the S3 link and hallucinated 15 SMILES**. Master
  orchestrator then drops even that in final synthesis (defect B, cross-ref F000). The real molecules
  never reach the orchestrator/chat → reactplan-vs-inline planner A/B is **moot** until fixed.
  **Fix lives in F015g** (capture the structured S3 artifact; success = `expected_artifacts` in S3,
  not the LLM summary). Evidence: live MCP call, Opik trace `019ebdab-…`, workflow `fedot-s3-link-trace`.
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
- Features: **F018** (top-level). F015 sub-features use letter suffixes F015a–F015h; next free **F015i**.
- Sources: **S033**
