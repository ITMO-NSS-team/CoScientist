---
id: F000
title: Baseline orchestrator + agents (pre-DEVGRAPH)
type: feature
status: done
created: 2026-06-11
updated: 2026-06-12
owners: [ITMO-NSS-team]
derives_from: []
depends_on: []
sources: [S007]
tags: [baseline, orchestration, adk]
code:
  - CoScientist/agents/agents.py
  - CoScientist/agents/agents.py:ResilientAgentTool
  - CoScientist/agents/catalog.py:ORCHESTRATOR_AGENTS
  - CoScientist/agents/prompt_builder.py:PromptBuilder
  - CoScientist/main.py:CoScientistManager
  - CoScientist/tools/retrieval_tools.py:RetrievalToolSet
  - CoScientist/tools/fedotmas_tools.py:FedotMASToolset
benchmarks: []
---

## Goal
The pre-existing system that DEVGRAPH starts from. Captured as a node so later
features have something concrete to `derives_from`.

## Current state
Working multi-agent pipeline on Google ADK: an Orchestrator delegates to
Planner/Hypotheses/Research/TaskExecutor/Coder/Medical agents (single source of
truth in `catalog.py`), with a pre-action Critic. Tool retrieval is RAG-based
(`retrieval_tools.py`); experiments run via FEDOT.MAS; papers are parsed
(Marker) and stored (S3). Prompts are composed, not hard-coded
(`prompt_builder.py`). See `project_card.md` for the full capability list.

## Baseline (this predates the graph)
Starting symbols this node anchors:
- `CoScientist/agents/agents.py` — `LlmAgent` definitions + `make_llm` (provider
  pinning, retries).
- `CoScientist/agents/catalog.py` — `ORCHESTRATOR_AGENTS` registry that renders
  the orchestrator/critic prompts and attaches agents as tools.
- `CoScientist/tools/` — toolsets exported in `__init__.py`.

## Attempts
### F000.A1 — Pipeline refactoring (provider pinning, retries, format alignment) · 2026-06-10 · outcome: success
- **Method:** pin OpenRouter providers to a known-good set + `num_retries`;
  remove `PlanReActPlanner`; align planner/orchestrator roadmap format; raise
  Tavily MCP timeout 5s→90s.
- **Result:** stops empty/error responses from flaky upstream providers; planner
  output is executed in order without reformat loops.
- **Evidence:** commit `79cb9c6` (`settings.pinned_providers`, `agents.make_llm`).
- **Sources used:** —
### F000.A2 — Sub-agent delegation & CLI output robustness (thinking-model fixes) · 2026-06-12 · outcome: success
- **Method:** two orchestrator-level fixes for gpt-oss "thinking" output.
  (1) **Empty delegation result** — ADK `AgentTool` returns only the *final event's*
  non-thought text; a sub-agent that ends a long tool loop on a thought-only/tool
  event yields `''`, so the orchestrator treats it as failed and escalates (e.g. to
  CoderAgent → HITL crash). New `agents.py:ResilientAgentTool` wraps every delegation
  and falls back to the agent's `output_key` state (ADK saves it on the
  final-response turn) when the direct result is blank — **no callbacks**.
  (2) **CLI printed the reasoning, not the answer** — `main.py` took
  `event.content.parts[0].text` (the `thought` part); now it joins the non-thought
  parts (falls back to any text).
- **Result:** delegations return the real answer; the CLI prints the answer (table +
  Key Points) instead of the chain-of-thought.
- **Evidence:** Opik trace `019eb552` (ResearchAgent span output 0→2100 chars after
  fix; final parts = `[thought, answer]`); CLI final response is the answer in
  `019ebb5d`; unit tests `tests/unit/test_resilient_agent_tool.py` (5 pass) + a
  deterministic check of the `main.py` part-selection. See [[opik-tracing-access]].
- **Next:** `ResilientAgentTool`'s fallback uses `output_key`, which ADK only writes
  on a final-response turn; if a sub-agent never emits one it still returns ''
  (acceptable). All orchestrator sub-agents inherit the wrapper.
### F000.A3 — Staged orchestrator prompt (live A/B/B2 on dataset_S, qwen) · 2026-06-12 · outcome: partial
- **Method:** replaced the flat "tool-first" rule in `prompts.py:ORCHESTRATOR_TEMPLATE` with a
  **staged research pipeline** + module descriptions: scope → ground tools
  (`list_available_tools`/`list_server_tools`) → **STOP exploring** once a fitting tool is named →
  only-if-unclear delegate to Research/Hypotheses → **RUN** via TaskExecutorAgent → finalize.
  Plus an explicit rule: the orchestrator can only LIST tools, it **cannot execute MCP tools**
  (`get_state_from_server`/`generate_*`/docking) itself — must delegate execution. Tested live
  A→B→B2 on 4 dataset_S tasks (GSK-3β/KRAS/BTK/STAT3), qwen-235b, cap 600s; harness
  `scripts/experiments/ab_runner.py` + `scripts/opik_eval/ab_analyze.py` (Opik thread_id `ab_*`).
- **Result:**
  - **A (baseline):** 4/4 capped 600s, **over-explored** (`search_mcp_servers×9`, deep paper
    search), **0/4 reached generation, 0 answers** — over-exploration is the baseline killer.
  - **B (staged, pre-fix):** 3/4 **crashed <80s** on `ValueError: Tool 'get_state_from_server'
    not found` — the orchestrator tried to EXECUTE an MCP tool directly. Decisive but broken.
  - **B2 (staged + can't-execute fix):** **first condition to reach REAL generation** — GSK-3β:
    FEDOT `molecule_generator` → `Pipeline complete 73.5s`; 1/4 produced a final answer (KRAS, 548 chars).
- **Evidence:** commit `5a39485`; `scripts/experiments/results/ab_{A,B,B2}_2026-06-12_*.json`;
  Opik traces by thread_id `ab_A_/ab_B_/ab_B2_*`. See [[opik-tracing-access]].
- **Next (residual levers, biggest first):** (1) orchestrator does NOT **finalize** after a
  substantive result — GSK-3β generated then over-ran the cap → add "got a result → synthesize
  & finish"; (2) sometimes **answers without running** (KRAS: `fedot=0`, text only) → "must run
  the experiment, not a plan"; (3) **per-step pre-action critic ~doubles LLM calls** (latency);
  (4) FEDOT-internal crashes (STAT3 ExceptionGroup). Cross-ref: F014 (dataset_S reliability),
  F017 (the staged flow is the near-term slice of the meta-model), F015a planner (parallel session).
- **Update 2026-06-13 (F014.A5, locus CONFIRMED = orchestrator):** the failing call's error
  lists `Available tools: list_available_tools, list_server_tools, HypothesesAgent,
  ResearchAgent, TaskExecutorAgent, CoderAgent, MedicalAgent` — i.e. the **master
  orchestrator's** roster. So the bug is the orchestrator emitting `get_state_from_server`
  (an MCP tool it can only LIST, not execute). Genuine `Tool 'get_state_from_server' not
  found` = **4 occurrences, ALL on 06-12** (`ab_B_00/01` B-crash + 2× `session_001` gpt-oss).
  - **CORRECTION (anti-entrenchment):** an earlier draft of this note claimed the over-reach
    "recurs on 06-13 in `ab_S5`/`l_Lplan`/`l_Lnone`". **That was a false positive** — in those
    traces `get_state_from_server` only appears inside *another* agent's `Available tools:`
    list, because it is a **legitimate** tool of the FEDOT.MAS `molecule_generator` sub-agent
    (F014.A2). No genuine orchestrator over-reach on 06-13.
  - ⚠ But "none on 06-13" is **weak evidence the fix holds**: the 06-13 sample is small and the
    `l_L1`/`l_L2` batch died on OpenRouter credits before reaching execution — so the B2/staged
    prompt is **not yet proven** to have eliminated it. → see TODO reminder below.
### F000.A4 — Orchestrator prompt: callable-names + explicit PLAN step (folds the tool-not-found fix) · 2026-06-13 · outcome: success
- **Method:** two additions to `prompts.py:ORCHESTRATOR_TEMPLATE` (on top of F000.A3):
  (1) **"CALLABLE NAMES — STRICT"** block — the orchestrator's ONLY callable tools are the
  catalog agents (EXACT CamelCase) + `list_available_tools`/`list_server_tools`; the `name`
  field INSIDE a registry result (`search_papers`, `generate_case_mols`, …) is a LEAF MCP
  tool and is NOT callable → delegate. Targets the two observed not-found crashes: a leaf
  call (`search_papers`) and a casing slip (`research_agent` vs registered `ResearchAgent`;
  ADK lookup is exact & case-sensitive, `functions.py:997-1010`).
  (2) **"PLAN" stage** — after grounding the real tools, build a short ordered plan on the
  tools that ACTUALLY exist, THEN delegate ("ground → plan → delegate" — the F015a direction,
  done inline in the orchestrator).
- **Result:** a runtime 3×3 A/B of the same block (before committing it to the prompt)
  eliminated tool-not-found **0/9**; `none` 0→2/3 (avg n_llm 12.3→5.3); PlanReAct 2→3/3. The
  name-normalizer (`research_agent`→`ResearchAgent`) **never fired** — the prompt alone
  sufficed, so it was NOT committed.
- **Evidence:** `prompts.py:ORCHESTRATOR_TEMPLATE` (CALLABLE NAMES + PLAN); runtime traces
  `019ebd19/1f/21/25/26/28/32/35/37` (0/9 not-found); plan_2 trace `019ebcc5` (orchestrator
  emitted `search_papers` directly). See [[opik-tracing-access]].
- **4-variant A/B done (n=3, in-code fixes, CRISPR literature):** finish / avg n_llm —
  **none (inline PLAN): 3/3, 4.3** · **PlanReAct: 3/3, 5.3** · PlannerAgent: 2/3, 6.0 (avg 219s) ·
  **post_action_critique: 1/3, 9.0** (2 timeouts). **0 tool-not-found in all 12.** Earlier
  no-fix `none` was 0/3 (all timeout) → now 3/3: the combined fixes (callable-names + PLAN +
  ResearchAgent STOP [F003.A4]) fixed convergence. Traces `019ebd63..8e`.
- **Decisions (this A/B):** (1) **Separate free-form PlannerAgent — DROPPED** (worse on every
  axis; `use_planner=False`). (2) **`post_action_critique` — REJECTED as a finalizer** (doubles
  LLM calls → timeouts; left commented at `agents.py:313`). (3) **PlanReAct — committed**
  (`agents.py:306` `planner=PlanReActPlanner()`): equal to inline on reliability, chosen for the
  explicit plan-act-replan structure; *note the data slightly favored plain inline (cheaper)* —
  being validated on the harder **dataset_L** (multi-target molecule gen) before final commit.
- **Next:** dataset_L A/B (PlanReAct vs inline) with per-subtask completion grading; if inline
  wins there too, revert PlanReAct (1 line). Finalization without the LLM post-critic = the
  prompt STOP/PLAN rules (already 3/3) + a future cheap deterministic gate.

## ✅ TODO
- [ ] No eval harness exists for the baseline — add one (shared with project card gap).
- [ ] Orchestrator: **finalize after a substantive result**; **must run** the experiment (not a plan) — top fixes from F000.A3.
      NB `post_action_critique` (SUFFICIENT/INSUFFICIENT/WRONG vs the trajectory) is BUILT but
      **COMMENTED** at `agents.py:313` — enabling it is the obvious finalize signal (cost: +LLM calls);
      4-variant A/B (F000.A4) tests it. Termination is otherwise purely "LLM stops emitting tool calls"
      (no `max_iterations`; ADK default `max_llm_calls=500` ≈ unbounded).
- [ ] **Sub-agents have NO internal loop-guard** — the orchestrator critic governs only the
      orchestrator's delegations, not a sub-agent's own loop. ResearchAgent ran away internally
      (6× empty `explore_chemistry_database`); patched via prompt (F003.A4). Generalize (prompt
      STOP rules and/or `max_iterations`) to other sub-agents.
- [ ] 🔎 **REMINDER — additionally re-check the `get_state_from_server` over-reach (it is in the
      ORCHESTRATOR).** Confirmed locus = master-orchestrator roster (F000.A3 update / F014.A5):
      the orchestrator emits `get_state_from_server` (an MCP tool it may only LIST) → `Tool not
      found` → fatal. Genuine cases: 4×, all 06-12 (`ab_B_00/01`, 2× `session_001`); none on
      06-13 **but unproven** — the 06-13 runs died on credits before reaching execution. **To
      close it:** run a clean, credit-funded batch that actually reaches TaskExecutor/FEDOT and
      confirm the orchestrator never emits `get_state_from_server` (or any `generate_*`/MCP tool)
      directly; if it still does, the durable fix is delegation via the experiments module (F015),
      not prompt wording. NB `get_state_from_server` IS legitimate for the FEDOT `molecule_generator`
      sub-agent — only an *orchestrator-level* call is the bug.

## ⚠ Pitfalls / Known problems
- Upstream OpenRouter providers are flaky → **do not** remove provider pinning /
  retries without a replacement; that was the actual bug fixed in `79cb9c6`.
- **Config drift (found in F014.A1) — FIXED 2026-06-12 (F014.A4, variant 1):**
  `pinned_providers` was `[]` (pinning OFF) while `.env` ran `gpt-oss-120b`. Now `.env`
  sets `LLM__PINNED_PROVIDERS=["deepinfra","groq","together","fireworks"]` →
  `provider_routing()` engages for gpt-oss. **Two gotchas remain:** (1) `provider_routing()`
  reads `pinned_providers`, NOT `allowed_providers` (the latter, `LLM__ALLOWED_PROVIDERS`,
  is a dead/legacy field). (2) the pin set is gpt-oss-specific — if `LLM__MAIN_MODEL`
  switches to qwen, set `LLM__PINNED_PROVIDERS=[]` (else 429/500). NB pinning did NOT
  actually reduce full-pipeline empties (F014.A4) — it's engaged but not the empties fix.
- Integration tests need ITMO VPN + hosted services; they won't run locally.

## Symbols
- `CoScientist/agents/catalog.py:ORCHESTRATOR_AGENTS` — the one place to add/remove a delegatable agent.
- `CoScientist/agents/prompt_builder.py:PromptBuilder` — composable prompt assembly (`<<NAME>>` sentinels).
- `CoScientist/agents/agents.py:make_llm` — LLM factory with provider pinning + retries.
- `CoScientist/agents/agents.py:ResilientAgentTool` — AgentTool wrapper that falls back to the agent's `output_key` when the stock result is blank (F000.A2). All sub-agents are wrapped in it.
- `CoScientist/main.py:CoScientistManager.run` — extracts the non-thought answer part from the final event (F000.A2).
