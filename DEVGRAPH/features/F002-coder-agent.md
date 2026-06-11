---
id: F002
title: CoderAgent + sandbox execution
type: feature
status: done
created: 2026-06-11
updated: 2026-06-11
owners: [ITMO-NSS-team]
derives_from: [F000]
depends_on: [F000]
sources: [S002]
tags: [coder, sandbox, tools, registry]
code:
  - CoScientist/tools/coder_tools.py:CoderToolset
  - CoScientist/agents/catalog.py:ORCHESTRATOR_AGENTS   # CoderAgent entry
benchmarks: []
---

## Goal
Give the orchestrator a general-purpose engineering agent that can *do* work —
write & run code, shell/git (clone/commit/push), manage files, install deps, run
long jobs — in an isolated sandbox, for tasks no existing MCP tool covers.

## Current state
Shipped in #268 (commit `f863802`), together with a dynamic prompt/agent registry.
`CoderAgent` is registered in `catalog.py` and backed by
`coder_tools.py:CoderToolset` (`coder_toolset_instance`). Routing distinguishes it
from `TaskExecutorAgent`: Coder *writes/runs* new code; TaskExecutor only *runs
existing* MCP tools.

## Attempts
### F002.A1 — CoderAgent + sandbox + dynamic registry (#268) · 2026-06-10 · outcome: success
- **Method:** add a sandboxed coder toolset; register the agent via the single
  `ORCHESTRATOR_AGENTS` source of truth so prompts/critic/tool-attachment stay in sync.
- **Result:** orchestrator can delegate engineering work to a sandbox; agent
  roster is data-driven (no prompt duplication).
- **Evidence:** commit `f863802`; `tools/__init__.py` exports `coder_toolset_instance`;
  `catalog.py` has the `CoderAgent` spec with routing.
- **Sources used:** [S002] Voyager (skill-library-as-executable-code) — `partial`:
  the "skills are runnable code, retrieved/composed by the agent" idea applies;
  Voyager's automatic curriculum + skill-DB retrieval are **not** implemented.

## ✅ TODO
- [ ] No benchmark/eval for coder success rate — add a small task suite and record in `benchmarks:`.
- [ ] Persist useful sandbox-built scripts as reusable skills (Voyager-style) instead of one-shot.

## ⚠ Pitfalls / Known problems
- Overlap with `TaskExecutorAgent`: if routing blurs, the orchestrator may write
  code for something an existing MCP tool already does. Keep the "write new" vs
  "run existing" boundary explicit in routing (`catalog.py`).

## Symbols
- `CoScientist/tools/coder_tools.py:CoderToolset` — sandbox code/shell/git toolset.
- `CoScientist/agents/catalog.py` — `CoderAgent` AgentSpec (description + routing).
