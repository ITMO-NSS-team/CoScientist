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
  - CoScientist/experiments/plan.py:ServerTools          # per-step server→tools binding (F015a.A2)
  - CoScientist/experiments/planner.py:generate_plan     # strict-JSON gen + validate-then-repair (R05)
  - CoScientist/experiments/prompts.py:build_planner_messages
  - CoScientist/experiments/gate.py:deterministic_gate   # R12 model-free gate; closes empty-compute-step hole (F015a.A3)
  - CoScientist/experiments/plan.py:ExperimentStep.kind  # compute|research|hypothesize|code_exec (F015a.A3)
  - CoScientist/experiments/submit_plan_tool.py:make_submit_plan_tool  # R09 plan-object tool + fidelity fix (F015a.A4)
  - CoScientist/experiments/bridge.py:check_conformance  # Bridge A "plan-as-contract" (F015a.A4)
  - CoScientist/experiments/executor.py:execute_plan     # Bridge B DAG-executor over real sub-agents (F015a.A4)
  - scripts/experiments/plan_critic_probe.py             # FEDOT-free critic-mode probe harness (F015a.A3)
  - scripts/experiments/full_system_probe.py             # REAL bridge A/B battery on dataset_S/L (F015a.A4)
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

### F015a.A2 — Schema carries server→tools binding (user direction) · 2026-06-12 · outcome: success
- **Method:** per user direction, a step no longer names bare tools — it carries
  `tool_servers: list[ServerTools]` where each `ServerTools = {server, tools, url?}`
  groups the needed tools by the MCP server that provides them. This is the exact unit
  FEDOT.MAS dispatch consumes (`servers_payload`, see F015g.D1). The inventory + prompt
  are now server-grouped; the planner is meant to source this from the **live
  tool-analysis tool** (`retrieve_tools` / F015c, R07) rather than a static list.
- **Result:** 3/3 valid plans, attempts=1. Steps bind correctly, e.g.
  `chemical-mcp-server[calculate_docking]`, `admet-mcp[predict_admet]`. Gap check is now
  at **(server, tool)** granularity and caught two real things: (a) missing filtering
  parked under `UNKNOWN:filter_molecules` (true capability gap → F015d/F015e); (b) a
  **server-name mismatch** — the planner wrote `chemical-mcp` instead of the exact
  `chemical-mcp-server` → flagged `chemical-mcp:get_rdkit_properties`.
- **Evidence:** `scripts/experiments/results/r05_plans_2026-06-12.json`; unit tests 8/8
  (`tests/unit/test_experiment_plan.py` updated for `ServerTools`).
- **Finding:** server-name fidelity matters. With a static stand-in inventory + a flaky
  model the planner can mis-name a server; **F015c must resolve/normalize server names
  against the live inventory** and treat a mis-named server as a correctable gap, not a
  hard failure. The live inventory (R07) makes the names exact.
- **Next:** feed the LIVE inventory (R07); map `step.tool_servers` → FEDOT.MAS
  `servers_payload` in the executor (F015g, R14).

### F015a.A3 — Plan-critic mode experiment (FEDOT-free) + robust-architecture decision + deterministic gate (R12) · 2026-06-14 · outcome: partial
- **Method:** built a FEDOT-free probe harness (`scripts/experiments/plan_critic_probe.py`): the REAL
  OrchestratorAgent + live qwen + PlanReAct + critic, but sub-agents replaced by instant `StubAgentTool`s
  (byte-identical names) and the orchestrator's grounding (`list_available_tools`/`list_server_tools`)
  backed by a FROZEN inventory — so it grounds+plans with NO VPN/Postgres/FEDOT. Metrics read offline from
  `state['critic_pre_history']`. Grid: 4 critic modes {none, per-action, tags(=`plan_critic_only`),
  delegation-gate} × 4 probes {ctrl, fp_alz, gap, wrong} × 2 reps = 32 runs (all traced to Opik,
  `thread_id = session_id = pc_*_195250`).
- **Result (32 runs):** **tags** (current `plan_critic_only`) = **plan-fire 0/8** — NEVER critiqued a
  delegation (fired only on grounding turns) and **missed both bad plans** (gap, wrong: TP 0/4) → the
  current plan-critic is INERT. **delegation-gate** = **plan-fire 8/8, TP 4/4**, firings 2.25 vs
  per-action's 5.12 (≈half the churn). **per-action** TP 4/4 but churny (up to 10 firings). FP=2/4 for both
  per-action & delegation BUT trace-verified CONFOUNDED: part stub-URL artifact (critic flagged the
  `stubsig` fake signature), part soft "confirm the case" revise; the EXPLICIT alzheimer probe was APPROVED
  by all (FP=0) — the alzheimer-FP only appears on an IMPLICIT, unconfirmed case mapping. Latency confounded
  by a network slowdown (not the critic).
- **Decision:** RETIRE tag-mode (inert). delegation-gate = validated **STOPGAP** (detection 0/8→8/8, less
  churn, ~40 lines, no new tool). **ROBUST TARGET** = a `submit_plan(ExperimentPlan)` FunctionTool:
  delegation-gate as the tag-free TRIGGER → deterministic gate → roadmap + HITL approve/EDIT → execute the
  edited plan. The SAME object satisfies the plan→roadmap→HITL requirement. See [[F015b]] for the
  critic-mode comparison + deterministic-gate-first.
- **Verified blockers for the robust target (adversarial, code-grounded):** (1) **no execution bridge** —
  `ExperimentPlan`/`submit_plan` have ZERO refs in `agents/tools/main.py`; a validated/edited plan has
  nobody to execute (roadmap **R10**). (2) **gate hole** — `tool_servers` defaulted to `[]` and
  `_validate_structure` checks only ids/deps/acyclicity → a tool-less compute step passed, so gap/wrong were
  uncatchable unless the model hallucinated a tool name. **FIXED this session (R12):** new
  `experiments/gate.py:deterministic_gate` + `ExperimentStep.kind` — a `compute` step with no resolvable
  `tool_server` now fails `reject:empty_compute_step`; unknown server / unresolvable tool fail too. (3)
  **dead HITL wiring** — `CoScientistManager._hitl_handler` (main.py:87) never reaches the orchestrator, and
  `HITL__HEADLESS_AUTO_APPROVE=true` would silently auto-pass EDIT paths.
- **Evidence:** `scripts/experiments/results/plan_critic_2026-06-14_195250.json` (32 runs);
  `opik_dump/traces_since_2026-06-14/` (33 traces w/ `critic_llm` verdicts — tags-miss `019ec71d…`,
  delegation-catch `019ec731…` "fetch_protein_activities not in inventory"); `CoScientist/tests/unit/test_plan_gate.py`
  (5 passed: empty-compute-step + unresolvable-tool now caught). ADK fact (prior code-read): a `submit_plan`
  function_call injected from `after_model_callback` survives `PlanReActPlanner.process_planning_response`
  and executes; `tool_choice` is NOT forwarded by LiteLlm, so plan-fill cooperation is a bounded gamble the
  gate makes fail-closed.
- **Next:** minimal plan→delegation bridge (`state['experiment_plan']` + delegation conformance) → wave-2
  (clean stub, submit_plan-gate cell + prompt-matched NO-gate control, N≥25, attribute FP to gate vs prompt).

### F015a.A4 — submit_plan + deterministic gate + BOTH bridges built & tested on REAL dataset_S/L · 2026-06-15 · outcome: partial
- **Method:** built the plan-object seam end-to-end and tested it on the LIVE pipeline (NO stubs):
  `experiments/submit_plan_tool.py` (validate `ExperimentPlan` → deterministic gate → HITL approve/EDIT w/
  re-validation → `state['experiment_plan']`; **fidelity-fix toggle** = exact schema error + schema/example
  in the tool description); `experiments/bridge.py` (Bridge **A** "plan-as-contract": `step.kind→agent`
  conformance nudge, PlanReAct keeps driving); `experiments/executor.py` (Bridge **B**: topological execute +
  `{artifact_id}` substitution, dispatching to REAL sub-agents via `AgentTool.run_async`). Harness
  `scripts/experiments/full_system_probe.py`: real grounding (RAG/Postgres) accumulated into
  `state['seen_inventory']` for the gate, advisory delegation-gate critic, **≥30s inter-run delay** (MCP not
  async), per-run JSONL manifest. Battery: (3 S + 2 L) × {bridge_A, bridge_B} + 1 no-fix = 11 runs, cap=1600, qwen.
- **Result (manifest `scripts/experiments/results/full_system_2026-06-15_003152.jsonl`):**
  - **Reached REAL molecules (verified by complex SMILES in-band, NOT has_s3 alone): bridge_B 4/5, bridge_A 3/5.**
    e.g. bridge_A S_kras_sel `Cc1cccc(CC(C)NCC(=O)Nc2ccccc2)c1`; bridge_B L_2 `CN(C)CCSc1ncccc1Cl`; feasibility
    CSV = 50 real molecules + real property cols (QED/SA/PAINS/BBB/IC50).
  - **Fidelity fix VALIDATED:** with fix, submit_plan accepted on the first call (mostly 1.00 vs pre-fix ~0.25);
    the NO-FIX run regressed to the old hallucinated-tool failure (`Tool 'fetch_activity_data' not found`,
    trace 019ec888) — qwen bypassed submit_plan entirely.
  - **Bridge A drifts** (off-plan up to 2/5); the FEDOT-free 42-turn blowup did **NOT** reproduce on real tasks
    (~7-10 turns) — it was a stub artifact (instant stub returns invited improvisation). **Bridge B never drifts**
    (executor follows the plan) but is **slow on big plans** (L_1: 6-step plan ran the full 1600s cap, exec 3/6).
- **4 failures, pinpointed by span (the "where/what"):**
  1. **qwen malformed tool-call JSON → `JSONDecodeError`** (bridge_A S_gsk char 441 / trace 019ec80c; bridge_B
     L_1 char 2122 / trace 019ec85a). Span = the model call. **ROOT: ADK `lite_llm.py:1630`
     `json.loads(tool_call.function.arguments)` is UNGUARDED in the non-streaming path** (the streaming path has
     a guard at :2158-2159); qwen emits broken JSON, worse for big submit_plan payloads. Same task passed in
     feasibility → run-to-run variance (F014 family). **Fix (designed): JSON-repair shim at :1630 (json_repair on
     the args string) + bounded re-prompt.**
  2. **MCP 300s timeout** (bridge_A S_stat3 / trace 019ec814): `search_papers` inside ResearchAgent (reached via
     A's DRIFT) hit the 300s MCP ceiling → McpError propagated → `len=0` (work happened, no answer — F014 pitfall).
  3. **Bridge B big-plan cap** (L_1 / trace 019ec85a): the executor ran a 6-step plan sequentially with real
     FEDOT; a `generate_case_mols` 300s timeout + a JSON error → never finished within 1600s. Needs per-step
     timeout + partial delivery (parallelism deferred per user).
  4. **No-fix hallucinated tool** (trace 019ec888) — see fidelity fix above.
- **Evidence:** manifest above; traces 019ec80c/14/85a/888 + the 7 S3 successes in
  `opik_dump/traces_since_2026-06-14/`; unit tests `tests/unit/test_bridges.py`+`test_plan_gate.py` (11 pass).
- **Verdict:** Bridge B slightly more robust (4/5 vs 3/5) and never drifts, but its sequential executor stalls on
  big plans; Bridge A is the lighter PlanReAct-preserving option but its drift can wander into a slow tool that
  sinks the run. Both viable; the dominant failures are SYSTEMIC (qwen JSON variance + MCP 300s timeouts), not
  bridge-specific.
- **Next:** (1) JSON-repair shim (`RepairingLiteLlm`) → fixes failure #1 (2/4); (2) per-MCP-call timeout +
  partial-result delivery → fixes #2/#3 (`len=0` → partial molecules); (3) Bridge B per-step timeout.
  Parallelism of independent DAG steps DEFERRED (user direction 2026-06-15).

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
- [x] **Deterministic gate (R12)** — tool-resolvability + empty-compute-step → `experiments/gate.py:deterministic_gate` + `ExperimentStep.kind` (F015a.A3; closes the wave-1 gap hole; `test_plan_gate.py` 5/5).
- [x] **`submit_plan(ExperimentPlan)` FunctionTool** (`submit_plan_tool.py`) + **fidelity fix** (exact schema error + schema/example) — built & validated on real runs (F015a.A4; no-fix regressed to hallucinated tool).
- [x] **DAG executor (Bridge B, R10)** — `executor.py:execute_plan` (topological + `{artifact_id}` substitution) dispatching to REAL sub-agents; works, but **stalls on big plans** (per-step timeout TODO).
- [x] **Bridge A "plan-as-contract"** — `bridge.py:check_conformance` (conformance nudge); built, drifts on real tasks.
- [ ] **JSON-repair shim (`RepairingLiteLlm`)** — guard ADK `lite_llm.py:1630` (json_repair + bounded re-prompt); fixes the qwen-malformed-tool-call `JSONDecodeError` (F015a.A4 failure #1, 2/4 of real failures).
- [ ] **Per-MCP-call timeout + partial-result delivery** — a 300s MCP stall (`search_papers`/`generate_case_mols`) must yield partial molecules, not `len=0` (F015a.A4 #2/#3).
- [ ] **Bridge B per-step timeout** so one slow step doesn't blow the whole-run cap (F015a.A4 #3).
- [ ] **Plan-level HITL at the submit_plan handler** — re-validation on EDIT is built (`run_submit_plan`); still TODO: wire a real ChatHITLHandler + fix the dead `CoScientistManager._hitl_handler` (main.py:87).
- [ ] CodeAct-vs-FEDOT-vs-Alembic routing rubric (R11).

## Symbols (targets/building blocks)
- `CoScientist/agents/prompts.py` — `planner_instruction` (to rewrite: L801-802, L836).
- `CoScientist/agents/catalog.py` — `PlannerAgent` spec.
