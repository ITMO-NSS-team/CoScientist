---
id: F015a
title: Experiment planner/orchestrator — structured JSON step-plan (DAG) + bounded execution loop
type: feature
status: proposed
created: 2026-06-11
updated: 2026-06-11
owners: [SoloWayG]
derives_from: [F015, F006]
depends_on: [F015c, F003, F009]
sources: [S010, S013, S014, S015, S031]
tags: [planning, orchestration, json-plan, dag, experiments]
code:
  - CoScientist/agents/prompts.py            # planner_instruction (L801-802 prohibition, L836 flat format)
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
- [ ] Define the step JSON schema (+ Pydantic model) incl. a `provenance` field (F015d carry-through).
- [ ] Rewrite `planner_instruction` (reverse L801-802, replace L836) + inject live inventory.
- [ ] DAG executor (topological, placeholder→S3 resolution) with external (non-prompt) loop bound.
- [ ] CodeAct-vs-FEDOT-vs-Alembic routing rubric.

## Symbols (targets/building blocks)
- `CoScientist/agents/prompts.py` — `planner_instruction` (to rewrite: L801-802, L836).
- `CoScientist/agents/catalog.py` — `PlannerAgent` spec.
