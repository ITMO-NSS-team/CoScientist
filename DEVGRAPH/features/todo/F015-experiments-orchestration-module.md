---
id: F015
title: Experiments orchestration module (АМ) — plan→critic→bounded-execute + tool-sufficiency + Alembic
type: feature
status: proposed
created: 2026-06-11
updated: 2026-06-11
owners: [SoloWayG]
derives_from: [F010, F006]
depends_on: [F003, F009, F010, F002]
sources: [S009]
tags: [experiments, orchestration, planning, critic, fedotmas, alembic, mcp-generation]
code:
  # building blocks that exist today (the integrated module is NOT yet on this branch)
  - CoScientist/agents/catalog.py:ORCHESTRATOR_AGENTS   # PlannerAgent (plan building block)
  - CoScientist/agents/critic_agent.py                  # critic building block
  - CoScientist/tools/fedotmas_tools.py:FedotMASToolset # the FEDOT.MAS call the module wraps
  - CoScientist/code_exec_server/                       # sandbox exec (Alembic-adjacent)
benchmarks: []
---

## Goal
Replace the current "dump everything into FEDOT.MAS" experiment path with a proper
**agentic experiments module (АМ)** that plans, critiques, and executes computational
experiments step-by-step, and **dynamically extends tooling by building new MCP
servers (Alembic)** instead of editing existing tools. This is the designed fix for
the F014 failure modes. Spec: [[S009]] (approved ТП, 2026-06-02).

## Why (the defect it removes — verbatim from the ТП)
Today, to start an experiment, FEDOT.MAS is handed one **excessive, complex query**
(task + hypothesis + full literature summary + tool list at once). The ТП names this
as the root defect: it *"повышает вероятность ошибок и неудачных попыток проведения
вычислительных экспериментов, даже несмотря на наличие всех необходимых
инструментов"*. This is exactly what F014 observed (runaways to 700s; the
`molecule_generator` sub-agent calling tools it isn't equipped with).

## Designed algorithm (ТП §2.2)
1. **Input:** task-stage description + literature summary + hypothesis + current tool list.
2. **Orchestrator** builds a **structured step-by-step JSON plan** — each step =
   `{subtask, required_tools, run_params, expected_artifacts}`.
3. **Critic** checks the plan for completeness / non-contradiction / fit to the
   hypothesis → returns for rework until correct **or iteration budget exhausted**.
4. **Bounded execution loop** over plan steps. For each step:
   - **tool-sufficiency check** — are the current MCP tools enough for THIS step?
     - **yes →** hand the (small, single-step) task to FEDOT.MAS to build+run a MAS solver.
     - **no →** repo-search agent finds repos (priority: links cited in the literature,
       then public repos by keywords) → **Alembic** builds & deploys a new MCP tool →
       return to loop.
5. On plan completion → structured results + **HITL** accept/redo (ties to F001).

Data convention (ТП §2.4): control info (task, hypothesis, plan, run params, statuses)
flows as **text/JSON between agents**; file artifacts (CSV, images, code, Dockerfiles)
live in **S3**, passed around as presigned URLs.

### Alembic sub-pipeline (ТП §2.3) — repo → containerized MCP server
Four guarded stages in an isolated build container (≤3 retries/stage):
explorer (clone/README/file-tree/requirements + 1–5 candidate tool use-cases) →
environment (uv pip pinned → unpinned → conda fallback, ABI check; one env for
py≥3.10 else split server/repo envs) → coder (FastMCP `server.py` wrapping repo via
`subprocess.run` + pytest) → validator (syntax/imports/pytest/real calls, debugger
subagent ≤5 retries). Then `docker commit` → `alembic-tool` image, serve over
streamable-http on a random 20000–30000 port, register in CoScientist, **reusable**.

## How this fixes F014 (the mapping)
- **Runaway loops (≤81 LLM calls / 700s):** ⇽ plan/execute separation + critic +
  bounded iteration budgets. FEDOT.MAS gets one small step at a time, not the firehose.
- **`molecule_generator` "Tool 'X' not found":** ⇽ per-step **tool-sufficiency check**
  (F015c) decides *before* dispatch whether docking/property tools exist; if not, Alembic
  **builds** them — **no existing MCP tool is modified** (F014.D1). **⚠ Re-scoped by
  adversarial review:** F015c at the CoScientist layer only guarantees the right *servers*
  reach FEDOT.MAS; FEDOT.MAS's own meta-agent still assigns tools per worker at *server*
  granularity, so it can under-equip an internally-generated worker. **F015g.D1 RESOLVED
  (2026-06-12):** closed at the orchestration layer — per-step server filtering (existing RAG
  seam) + tool-level details embedded in the step's task text + config-verify via the public
  `generate_config`/`build_and_run` two-step API. F015c+F015g.D1 together cover the failure.
- **Out of scope (honest):** does NOT fix OpenRouter provider flakiness / empty
  responses (pinning — F014.A3), nor tools genuinely unavailable and unbuildable by Alembic.

## Current state
`proposed` on this branch. Building blocks already exist: `PlannerAgent`
(`catalog.py`), `CriticAgent` (`critic_agent.py`), the FEDOT.MAS call
(`fedotmas_tools.py`), and a sandbox (`code_exec_server/`, CoderAgent F002). **Alembic
is under active development on branches** (`origin/alembic`, `alembic-environment`,
`alembic-examples`, `alembic_feature_236`, `alembic-integration-demo`) — not yet
integrated here. The integrated plan→critic→sufficiency→execute loop is not yet wired.

## Decomposition (epic → F015a–F015h)
F015 is an **epic**; the work splits into the sub-features below (each its own node, with
adopted best practices + sources). Decomposition refined by an adversarial review workflow
(2026-06-11) — see each node's "adversarial review" notes.

| Sub | Title | Status | Note |
|-----|-------|--------|------|
| [F015a](../in_progress/F015a-experiment-planner.md) | Planner — JSON step-plan DAG + bounded loop | proposed | needs F015c's live inventory at plan time |
| [F015b](./F015b-plan-critic-loop.md) | Plan-critic loop (deterministic-first, bounded) | proposed | its hard gate **folds in F015c** (don't duplicate) |
| [F015c](./F015c-tool-sufficiency-check.md) | MCP inventory + tool-sufficiency callable | proposed | **shared substrate; build FIRST**; fail-closed on backend-down |
| [F015d](./F015d-repo-search-agent.md) | Repo-search (literature-first) | proposed | feeds F015e; provenance carry-through |
| [F015e](../in_progress/F015e-alembic-repo-to-mcp.md) | Alembic repo→MCP pipeline | **in_progress** | mostly built on branches; integrate, don't extend inline |
| [F015f](./F015f-tool-deploy-registration.md) | Deploy + register/reuse + sandbox | proposed | build+serve already in F015e; scope = register+reuse+sandbox |
| [F015g](./F015g-single-step-fedotmas-dispatch.md) | Single-step FEDOT.MAS dispatch | proposed | gated on **F015g.D1** (dispatch granularity) |
| [F015h](./F015h-am-eval-harness.md) | Eval harness on dataset_S | proposed | acceptance gate; proves the fix (anti-entrenchment) |

**Prerequisite decision:** **F015g.D1 — RESOLVED 2026-06-12** (user direction): per-step server
filtering via the EXISTING `retrieve_tools`→`filtered_tools`→`servers_payload` seam + detailed
per-step task text embedding tool-level details + plan-update after Alembic builds (several
small FEDOT.MAS calls) + a `generate_config`→verify-`MASConfig`→`build_and_run` safety net
(public two-step fedotmas API; no fedotmas changes). See F015g Decisions.

**Build order (corrected by review — F015c before F015a; F015b folds F015c):**
`F015g.D1 → F015c → F015a → F015b → F015g → F015f → F015d → F015e(integrate)`, with **F015h** running
alongside as the acceptance gate.

## Cross-cutting concerns (from the adversarial review — easy to miss)
- **Structured/constrained decoding** for the JSON plan: flaky open models emit empty/invalid
  JSON (F014.A2). Use litellm/OpenRouter `response_format=json_schema` + Pydantic
  validate-then-repair; mirror FEDOT.MAS's own `output_schema=MASConfig` (in-repo prior art).
- **Model tiering / pinning** [S032]: strong **pinned** model for plan+critic, cheap pinned
  model for per-step dispatch/dedup (LATM maker/user split). Ties to F014.A3 (pinning is the live lever, currently off).
- **CodeAct routing** [S031]: many steps should be one executable code block over a typed API
  (CoderAgent/`code_exec_server`, F002) — worst F014 runaways were `bash`/`write_file` thrash, not tool-not-found.
- **Retrieval-oracle fail-closed** (F015c): `[]` on backend-down must NOT read as "everything is a gap".
- **MCP supply-chain / tool-poisoning** (F015f): third-party tool descriptions + outputs are
  untrusted inputs flowing into the planner and FEDOT.MAS's description-only router.
- **HITL plan-approval gating** (F001): interrupt for plan approval / uncertainty-triggered escalation, not just "stuck".
- **Eval harness** (F015h): the module's claims are unproven without it.

## Reliability findings (S5 reactplan/qwen, 2026-06-13)
Detailed analysis of the 5 S5 traces (resolved ids: Q1 `019ebfe5`, **Q2 `019ebfe6`**, **Q3 `019ebff0`**,
Q4 `019ebffa`, Q5 `019ebffb`) — **two distinct failure modes**, both reproduced:
- **Mode A — grounding-loop + critic-leak (Q1, Q4):** orchestrator loops `list_available_tools` /
  `list_server_tools` (×3-4), `pre_action_critique` intervenes every step, run ends on a critic turn →
  the `[CRITIC REVISION]` / "I am rejecting…" text LEAKS as the final answer. Never reaches generation.
  Fast but a false success (`err=None, len>0`, no molecules).
- **Mode B — FEDOT.MAS unbounded slow path (Q2, Q3, Q5 — all timeout 600s):** once in FEDOT.MAS it runs
  a long autonomous pipeline (Q2: fetch_activity_data → name2smiles → **calculate_docking**; Q5: papers →
  generation → state checks) with **no step/time budget** → blows the cap. Q2 also looped on CoderAgent
  **HITL auto-reject** (headless) before that.
- **Experimental fixes built (2026-06-13, uncommitted):** (A) critic-leak guard in `main.py`
  (drop `[CRITIC REVISION]`/reject text, fall back to artifacts or an honest note) + grounding anti-loop
  nudge in `pre_action_critique` (after >3 grounding calls, force-delegate); (HITL) auto-approve flag +
  loop-guard (F001.A3). **Mode B (step/time budget) deferred** — needs the F015g per-step dispatch.
- **User direction (2026-06-13):** the per-action critic (`pre_action_critique`, fires on EVERY action)
  is called too often — test the pipeline WITHOUT it; "we essentially only need a **plan** critic" (→ F015b).
  `l_runner.py --no-action-critic` added to A/B this on dataset_L. **Test pending (infra down).**

## dataset_L A/B result (2026-06-14) — per-action critic confirmed harmful; CoderAgent fabrication exposed
L1 (per-action critic ON) vs L2 (`--no-action-critic`), 5 dataset_L multi-target queries, qwen, cap 600s.
- **Per-action critic harmful (CONFIRMED):** L1 critic fired 3-8×/run, kept the orchestrator in grounding
  loops → 3/5 never reached generation (honest fallback), 2/5 timeout, **0 S3 deliveries**. L2 (critic=0)
  reached FEDOT/generation in 4/5, **1 clean S3 delivery** (L2.Q2 `019ec26e`, 6104 chars, real molecules + link).
  → drop the per-action critic; keep only a **plan** critic (F015b).
- **Mode B dominant:** 4/5 L2 runs timeout INSIDE `fedot_tool` (docking/property/generation slow); the outer
  run-cap doesn't honor FEDOT MCP cancellation (runs hit 700-906s). L2.Q1 reached `generate_mols` but the
  outer cap killed the run before `fedot_tool` returned → the captured S3 link was lost.
- **⚠ NEW — CoderAgent fabricates molecules under HITL auto-approve (trace `019ec27f`):** with auto-approve ON,
  the orchestrator delegated to CoderAgent, which wrote a Python script that **SIMULATED "known inhibitors"**
  (invalid SMILES e.g. `CCCCCCCCCCOOccc`), filtered by LogP/TPSA, saved a CSV — bypassing the real generation
  MCP, NO S3 link. A FAKE success, worse than honest failure. → HITL auto-approve must forbid CoderAgent
  fabricating data; it must route to the real generation/MCP tools, not "simulate".
- **Case-coverage gap:** `generate_case_mols` supports a FIXED case list (alzheimer/skleroz/cancer/parkinson/
  dyslipidemia/drug_resist); KRAS G12C / BTK / PCSK9 are NOT cases → only generic generation for those targets.
- **Correctness-first (user, 2026-06-14):** do NOT impose short FEDOT timeouts — first confirm the pipeline
  produces correct results, then optimize. `FEDOT_TIMEOUT_S=None` (the earlier 300s cut-off reverted).
- **TODO (later): investigate WHY FEDOT stages are slow** (docking / property prediction / generation
  latency) and whether they can be sped up — measure per-stage time from traces; correctness before speed.

## ✅ TODO
- [ ] Resolve **F015g.D1** (FEDOT.MAS dispatch granularity) — prerequisite for everything.
- [ ] Build **F015c** (inventory + sufficiency) first, then F015a → F015b → F015g → F015f → F015d → integrate F015e.
- [ ] Stand up **F015h** eval on `dataset_S.xlsx` as the acceptance gate (loop size / tool-not-found
      rate / `is_correct` vs the current path.

## ⚠ Pitfalls / Known problems
- **Don't modify MCP tools** (F014.D1) — the module adapts by *planning around* and
  *building* tools (Alembic), never editing third-party/searched MCP servers.
- The module adds more agents (orchestrator, critic, repo-search, 4 Alembic agents) →
  more LLM calls; the win comes from *bounded* per-step calls, so keep the iteration
  budgets real or it re-introduces the F014 runaway.
- Alembic builds Docker images per tool — infra-heavy; needs docker + S3 (ties to F013/F007 infra).

## Symbols
- `CoScientist/agents/catalog.py:ORCHESTRATOR_AGENTS` — `PlannerAgent` (plan building block).
- `CoScientist/agents/critic_agent.py` — critic building block for plan verification.
- `CoScientist/tools/fedotmas_tools.py:FedotMASToolset.fedot_tool` — per-step executor (to be fed one step at a time).
- `CoScientist/code_exec_server/` — sandbox exec, Alembic-adjacent.
