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

## ✅ TODO
- [ ] No eval harness exists for the baseline — add one (shared with project card gap).
- [ ] Orchestrator: **finalize after a substantive result**; **must run** the experiment (not a plan) — top fixes from F000.A3.

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
