---
id: F015g
title: Single-step FEDOT.MAS dispatch — one plan step + sufficient server subset (replaces the firehose)
type: feature
status: proposed
created: 2026-06-11
updated: 2026-06-11
owners: [SoloWayG]
derives_from: [F015, F010]
depends_on: [F015a, F015c]
sources: [S030, S020]
tags: [fedotmas, dispatch, loop-guard, scheduler]
code:
  - CoScientist/tools/fedotmas_tools.py:FedotMASToolset    # fedot_tool -> mas.run(task), whole servers
benchmarks: []
---

## Goal
Execute the plan **one step at a time**: a non-LLM scheduler resolves artifact placeholders
to S3 URLs and hands FEDOT.MAS exactly ONE ready step plus only the F015c-sufficient **server
subset** — replacing the current single overloaded query (the F014 runaway root).

## Best practices to adopt
- **LLMCompiler Task-Fetching-Unit [S030]:** a small NON-LLM scheduler resolves
  `{stepN.artifact}`/`$N` → S3 URLs and dispatches one ready step. Sequential topological
  execution first; parallel only for provably-independent branches (FEDOT.MAS runs are heavy,
  contend for docker/S3). Present tools as name+description+I/O schema (ChemCrow).
- **RAG-MCP — never pass the full tool list [S020]:** hand only the selected subset (the
  token/loop win F015 needs).
- **Externally-enforced loop-guard trio — ⚠ REUSE Alembic's guards, don't rebuild:** Alembic
  `main.py` already implements the pattern locally (verified on branch): `MAX_TOOL_REPEATS=3`
  (consecutive same tool+args → break, L106/148-153), `MAX_STEPS=120` (L163), per-stage
  `STAGE_TIMEOUT` via `asyncio.wait_for` (L289), and `_safe_get_tool`/`_UnknownToolStub`
  (L42-61: hallucinated tool name → error message instead of crash). **Lift/generalize these
  into the АМ executor** with two upgrades: (1) NFKC/homoglyph-normalize the dedup key
  (Alembic's is exact-match and consecutive-only; F014's `‐` U+2010 vs `‑` U+2011 duplicate
  slips through) and make it non-consecutive; (2) add the completion check tied to
  `expected_artifacts` (S3 artifact materialized with the right shape — NOT the LLM's success
  summary) + per-step $/call caps wired to Opik (F008). Neither exists in Alembic.

## Decisions
### F015g.D1 — FEDOT.MAS dispatch granularity · 2026-06-11 · **RESOLVED 2026-06-12 (user direction)**
- **Context (verified in the installed package):** `fedotmas_tools.py:fedot_tool` calls
  `mas.run(task_description)` with whole servers as `mcp_servers`. `mas.run` =
  `generate_config(task)` + `build_and_run(config, task)` (`core/base.py:238`). The meta-agent
  routes on **server descriptions only** (`meta/mas_gen.py:44` → `get_server_descriptions` →
  `{name: cfg.description}`, `mcp/registry.py:75`); `create_toolset` binds a whole `McpToolset`
  per server (no per-tool filter). Tool-level descriptions are visible to a worker's LLM only
  at runtime — AFTER the roster is fixed. This is exactly how F014's `molecule_generator` got
  only the 5 generation tools and then called docking/property tools it didn't have.
- **Trace evidence (Opik, 2026-06-12):** dumped a real FEDOT-launching run — trace
  `019eb27d-7950-714f-bb56-d315266eaffc` ("Generate GSK-3beta inhibitors with high activity",
  qwen3, 246 spans) via `scripts/opik_eval/dump_fedot_trace.py`. Confirms: each `fedot_tool`
  span INPUT is **only** `{"task_description": "..."}` (the orchestrator already embeds tool
  names there, e.g. *"use the 'train_ml' tool with case=…, target_column=…"*) — **no servers in
  the call payload**; the server list is injected from `state['filtered_tools']` inside
  `fedot_tool`. FEDOT.MAS then spawns its OWN workers via the meta-agent (visible:
  `routing_meta_agent`, `ml_task_coordinator`/`model_trainer`, `molecule_design_coordinator`/
  `generative_molecule_worker` — names generated per run). So today: one big task in, autonomous
  worker generation inside — exactly the firehose F015 replaces with per-step plans.
- **Resolution (user, 2026-06-12):** the experiments orchestrator solves this at the
  ORCHESTRATION layer, with no fedotmas changes:
  1. **Per-step server filtering** via the existing RAG-over-tools (`retrieve_tools` →
     `state['filtered_tools']` → `fedot_tool`'s `servers_payload` — this seam already exists,
     today driven per-whole-task by the master orchestrator; drive it **per step** instead).
  2. **Detailed per-step `task_description`** that EMBEDS tool-level details (exact tool names,
     which server provides them, expected I/O) — since the meta-agent reads only server
     descriptions, the task text is the lever we fully control for steering its routing.
  3. **Many small FEDOT.MAS calls** instead of one big one: the orchestrator stores the initial
     plan, and after an Alembic build **updates the plan** (re-plan) and may split a stage into
     several simpler, well-specified FEDOT.MAS invocations.
- **Deterministic safety net (found in the public API):** use the documented two-step mode —
  `config = await mas.generate_config(step_task)` → **verify** the generated `MASConfig`
  (`MASAgentConfig` has per-worker `tools: list[str]`, `mas/models.py:19`): does some worker
  carry the server(s) owning this step's `required_tools`? If not, patch the worker's list or
  retry → `mas.build_and_run(config, step_task)`. Catches meta-agent mis-routing without
  touching FEDOT.MAS internals.
- **Consequence:** F015c's job is per-step server selection + gap detection; F015a owns
  plan-storage/re-plan-after-Alembic; F015g composes step text + config-verify + dispatch.

## ⚠ Risks / open questions
- **Plan-execution drift:** FEDOT.MAS may run something other than the step's declared
  `run_params`/`required_tools` — diff actual-vs-declared after each step and flag mismatch.
- **Empty-response false "completed":** back the success signal with deterministic
  `expected_artifacts` checks (F014's empty-response mode).
- **⚠ Worker paraphrase DROPS the S3 artifact link & HALLUCINATES (CONFIRMED, F010.A3):** the
  `molecule_generator` sub-agent returned 15 fabricated SMILES while the real `generate_mols`
  result was a presigned S3 CSV URL (`results_presigned_url`) that ADK never persisted — `output_key`
  holds only the sub-agent's `part.text` (`llm_agent.py:837-851`), not the raw `function_response`.
  Concrete proof that (1) success MUST be `expected_artifacts` materialized in S3, NOT the
  sub-agent's text, and (2) F015g's dispatch must CAPTURE the tool's structured S3 result
  (`skip_summarization` / `after_tool_callback` lifting `results_presigned_url` into state) rather
  than trust the LLM paraphrase. The experiments orchestrator then downloads the CSV (or via `vault`
  MCP) to judge completion.
- **Overlap with F006:** the existing critic already mutates/blocks tool calls — decide whether
  F015g's guards live in / replace the F006 path or sit alongside (avoid two competing critics).

## ✅ TODO
- [x] Resolve F015g.D1 (granularity) — RESOLVED 2026-06-12, see Decisions.
- [ ] Make `fedot_tool`'s `filtered_tools`→`servers_payload` seam **per-step** (it exists per-task today).
- [ ] Step-text composer: embed tool names + owning servers + I/O into `task_description`.
- [ ] Config-verify safety net: `generate_config` → check per-worker `tools` covers the step's
      `required_tools`' servers → patch/retry → `build_and_run`.
- [ ] Non-LLM scheduler (placeholder→S3, one ready step at a time).
- [ ] **Capture the tool's structured S3 result, not the sub-agent paraphrase (F010.A3):** lift
      `results_presigned_url`/`results_s3_key` from the raw `function_response` (via `skip_summarization`
      or an `after_tool_callback`) so the orchestrator gets the real artifact link, not the
      `molecule_generator` sub-agent's (lossy, hallucination-prone) text.
      **HOW — no fedotmas fork needed (verified 2026-06-13):** `MAS(plugins=[...])` is a public kwarg
      threaded to the ADK `Runner` (`core/base.py:174` `App(plugins=…)` → `_adk_runner.py:189`
      `Runner(plugins=…)`). Inject an ADK `BasePlugin.after_tool_callback` from `fedot_tool`
      (`fedotmas_tools.py:71`) that stashes `results_presigned_url` into `tool_context.state['fedot_artifacts']`
      — it runs at the tool-call boundary (inside the run), so the key survives into `mas.run()`'s
      returned state. **Must happen inside the run:** `fedot_tool` only sees the final state, where the
      link is already gone. ⚠ Passing `plugins=` **REPLACES** the defaults (`MAS.__init__`:
      `if plugins is not None: self._plugins = list(plugins)`; default = `[LoggingPlugin(),
      WebSearchLimitPlugin(...)]`) — re-include them or lose web-search limit + finalize-mode
      (`runner.py:_enter_finalize_mode` keys on `WebSearchLimitPlugin`). Prefer `after_tool_callback`
      over `skip_summarization` (the latter drops the worker's final text turn → empties `output_key`,
      breaks multi-step workers).
- [ ] Generalize Alembic's guards into the АМ executor (+NFKC dedup, artifact-completion check,
      Opik-wired caps); actual-vs-declared diff after each step.

## Symbols
- `CoScientist/tools/fedotmas_tools.py:FedotMASToolset.fedot_tool` — current `mas.run(task)` seam to rework.
