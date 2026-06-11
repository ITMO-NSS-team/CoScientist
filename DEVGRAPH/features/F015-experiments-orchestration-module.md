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
- **`molecule_generator` "Tool 'X' not found":** ⇽ the per-step **tool-sufficiency
  check** decides *before* dispatch whether docking/property tools exist; if not,
  Alembic **builds** them. The sub-agent is never asked to run a step it lacks tools
  for — and **no existing MCP tool is modified** (see F014.D1).
- **Out of scope (honest):** does NOT fix OpenRouter provider flakiness / empty
  responses (that's pinning — F014.A3), nor tools that are genuinely unavailable and
  cannot be found or built by Alembic.

## Current state
`proposed` on this branch. Building blocks already exist: `PlannerAgent`
(`catalog.py`), `CriticAgent` (`critic_agent.py`), the FEDOT.MAS call
(`fedotmas_tools.py`), and a sandbox (`code_exec_server/`, CoderAgent F002). **Alembic
is under active development on branches** (`origin/alembic`, `alembic-environment`,
`alembic-examples`, `alembic_feature_236`, `alembic-integration-demo`) — not yet
integrated here. The integrated plan→critic→sufficiency→execute loop is not yet wired.

## ✅ TODO
- [ ] Implement the experiment **orchestrator** that emits the structured JSON plan
      (`{subtask, required_tools, run_params, expected_artifacts}` per step).
- [ ] Wire the **plan critic** loop with an explicit iteration budget (reuse F006).
- [ ] Implement the **per-step tool-sufficiency check** against the live MCP tool list
      (reuse F009 retrieval / `search_mcp_servers` F005) — this is the direct F014 fix.
- [ ] Dispatch **one step at a time** to FEDOT.MAS (stop the all-at-once query).
- [ ] Integrate **Alembic** from its branches; register built MCP servers for reuse.
- [ ] Bench against `dataset_S.xlsx` (F014) and compare loop size / tool-not-found
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
