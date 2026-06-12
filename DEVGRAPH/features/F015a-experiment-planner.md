---
id: F015a
title: Experiment planner/orchestrator — structured JSON step-plan (DAG) + bounded execution loop
type: feature
status: in_progress
created: 2026-06-11
updated: 2026-06-12
owners: [SoloWayG]
derives_from: [F015, F006]
depends_on: [F015c, F003, F009]
sources: [S010, S013, S014, S015, S031]
tags: [planning, orchestration, json-plan, dag, experiments]
code:
  - CoScientist/experiments/plan.py:ExperimentPlan       # the step-plan DAG schema (R05)
  - CoScientist/experiments/planner.py:generate_plan     # strict-JSON gen + validate-then-repair (R05)
  - CoScientist/experiments/prompts.py:build_planner_messages
  - CoScientist/agents/prompts.py            # planner_instruction (L801-802 prohibition, L836 flat format) — still to rewrite
  - CoScientist/agents/catalog.py            # PlannerAgent
benchmarks: []
---

## Goal
The АМ entry point: decompose a research-stage task (+ hypothesis + literature summary)
into a **structured JSON step-plan** — each step `{subtask, required_tools, run_params,
expected_artifacts, dep}` — and drive a **bounded** execution loop over it. Replaces the
current free-form roadmap that feeds FEDOT.MAS a firehose (root of F014's runaways).

## Designed approach (ТП §2.2/§3.2)
Orchestrator emits the plan; steps carry required tools + expected artifacts; execution
is a bounded loop with a per-step tool-sufficiency gate (F015c) before dispatch (F015g).

## Best practices to adopt
- **Routine — make `required_tools` a HARD, exact-MCP-name field [S013]:** reverse the
  current planner rule. `prompts.py:801` says *"NEVER specify data sources, tools, or
  methods"* and L802 *"WHAT not HOW"*; the L836 flat `[Agent]|ACTION|INPUT|OUTPUT` format
  must become the ТП JSON schema, and the **live MCP inventory** (from F015c) must be fed
  into the planner context so it names real tools. (Routine reports large planning gains.)
- **ReWOO + HuggingGPT — variable-binding DAG [S014]:** give each step's artifacts a stable
  id; later steps reference `{stepN.artifact}` / `dep` → the plan is a topologically
  executable DAG, so independent steps run **without re-prompting the planner** between
  steps (directly attacks the runaway loop; ReWOO ~5× fewer tokens than ReAct). Resolve
  placeholders to S3 presigned URLs at dispatch (ТП §2.4).
- **Data Interpreter — runtime sub-DAG re-planning [S015]:** on a failed step regenerate
  only the downstream sub-DAG and splice it back (don't freeze the whole plan up front),
  bounded by a max depth. The LLM proposes; a deterministic harness enforces the loop.
  **User direction (2026-06-12, = F015g.D1 resolution):** the canonical re-plan trigger is an
  **Alembic build** — the orchestrator stores the initial plan; when a step hits a capability
  gap and Alembic builds a new tool, the orchestrator UPDATES the plan downstream of that step
  and may split one stage into **several smaller, well-specified FEDOT.MAS calls**. Plan
  storage + versioned re-plan is therefore a first-class F015a responsibility.

## Attempts
### F015a.A1 — Plan schema + strict-JSON generator + dataset_S benchmark (roadmap R05) · 2026-06-12 · outcome: success
- **Method:** new `CoScientist/experiments/` module. `plan.py` = Pydantic
  `ExperimentPlan`/`ExperimentStep`/`Artifact` with structural validators (unique ids,
  deps resolve, **acyclic** via Kahn, no self-dep) + `topological_order()`. `planner.py:generate_plan`
  = litellm `response_format=json_object` + **validate-then-repair** (≤2 repairs feeding the
  Pydantic error back), with gpt-oss provider-pinning (F014 fix) auto-applied. `prompts.py`
  encodes the Routine rule (name EXACT inventory tools) + ReWOO DAG (deps + `{artifact_id}`).
- **Result:** unit tests `tests/unit/test_experiment_plan.py` (8 pass: cycle/dup/dangling-dep/
  empty/roundtrip). Benchmark `scripts/experiments/r05_plan_benchmark.py` on 3 dataset_S tasks:
  **3/3 valid DAG plans, attempts=1 each** (no repair needed — pinning gave clean gpt-oss JSON).
  GSK-3β=7 steps, KRAS-G12C=13 (handles selectivity: dock vs KRAS/HRAS/NRAS→selectivity→filter),
  STAT3=4. The planner correctly surfaced **capability gaps** — `filter_candidates`,
  `compute_selectivity`, `filter_molecules` (named but not in the inventory) — i.e. exactly the
  Type-A gaps F015c must detect and F015d/F015e build. This validates the schema + gap-detection seam.
- **Evidence:** `scripts/experiments/results/r05_plans_2026-06-12.json`; tests pass
  (`PYTHONPATH=. pytest tests/unit/test_experiment_plan.py -q` → 8 passed); run
  `python scripts/experiments/r05_plan_benchmark.py`.
- **⚠ Caveat:** the inventory was a hardcoded **stand-in**, not the live MCP index (that's R07/F015c);
  validity = structural (schema/DAG), NOT that the named real tools exist or that the plan is
  scientifically optimal. The planner is standalone here — not yet wired as the ADK PlannerAgent
  (`agents/prompts.py` rewrite, R09, still pending).

## ⚠ Risks / open questions (incl. adversarial review)
- **Coupling to F015c (build F015c first):** reversing the tool-naming prohibition only helps
  if the planner SEES the live MCP inventory at plan time; otherwise it names plausible-but-
  nonexistent tools (hallucination moved earlier). An out-of-inventory name must NOT be a
  hard error — it's the F015c "insufficient → Alembic" branch.
- **Prompt blast radius (undersold):** `prompts.py:805-834` also encodes an ACTION taxonomy
  (SEARCH/COMPUTE/HYPOTHESIZE) and an agent-routing contract (Experiment vs Coder vs Research
  agent). The new schema must coexist with or replace these — not a clean 3-line delete.
- **CodeAct routing [S031]:** the worst observed runaways were `execute_bash`/`write_file`
  thrash, not tool-not-found. F015a needs a rubric for when a step is emitted as **executable
  code** over a typed API (route to CoderAgent/`code_exec_server`, F002) vs a server-routed
  FEDOT.MAS step vs an Alembic build — fewer steps = fewer loop chances.
- **Structured decoding:** plans must be schema-valid from flaky open models (F014.A2
  empties) → enforce via litellm/OpenRouter `response_format=json_schema` + Pydantic
  validate-then-repair (mirror FEDOT.MAS's own `output_schema=MASConfig`). See F015.

## ✅ TODO
- [x] Define the step JSON schema (+ Pydantic model) incl. a `provenance` field (F015a.A1, R05).
- [x] Strict-JSON generation with validate-then-repair (F015a.A1, R05).
- [ ] Inject the LIVE MCP inventory (F015c/R07) into the planner instead of the stand-in list.
- [ ] Rewrite the ADK `planner_instruction` (reverse L801-802, replace L836) and wire `generate_plan` as PlannerAgent (R09).
- [ ] DAG executor (topological, `{artifact_id}`→S3 resolution) with external (non-prompt) loop bound (R10).
- [ ] CodeAct-vs-FEDOT-vs-Alembic routing rubric (R11).

## Symbols (targets/building blocks)
- `CoScientist/agents/prompts.py` — `planner_instruction` (to rewrite: L801-802, L836).
- `CoScientist/agents/catalog.py` — `PlannerAgent` spec.
