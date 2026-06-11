---
id: F000
title: Baseline orchestrator + agents (pre-DEVGRAPH)
type: feature
status: done
created: 2026-06-11
updated: 2026-06-11
owners: [ITMO-NSS-team]
derives_from: []
depends_on: []
sources: [S007]
tags: [baseline, orchestration, adk]
code:
  - CoScientist/agents/agents.py
  - CoScientist/agents/catalog.py:ORCHESTRATOR_AGENTS
  - CoScientist/agents/prompt_builder.py:PromptBuilder
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

## ✅ TODO
- [ ] No eval harness exists for the baseline — add one (shared with project card gap).

## ⚠ Pitfalls / Known problems
- Upstream OpenRouter providers are flaky → **do not** remove provider pinning /
  retries without a replacement; that was the actual bug fixed in `79cb9c6`.
- **Config drift (found in F014.A1):** `settings.pinned_providers` is currently `[]`
  (pinning OFF) with a comment assuming the active model is qwen3-235b, but `.env`
  runs `gpt-oss-120b` — the exact model the comment says needs the 4-provider set.
  So pinning is *not engaged for the model in use*. Verify `(main_model,
  pinned_providers)` agree before trusting that the empty-response fix is active.
- Integration tests need ITMO VPN + hosted services; they won't run locally.

## Symbols
- `CoScientist/agents/catalog.py:ORCHESTRATOR_AGENTS` — the one place to add/remove a delegatable agent.
- `CoScientist/agents/prompt_builder.py:PromptBuilder` — composable prompt assembly (`<<NAME>>` sentinels).
- `CoScientist/agents/agents.py:make_llm` — LLM factory with provider pinning + retries.
