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
### F002.A2 — sandbox workspace lost between CoderAgent invocations → fabrication · 2026-06-14 · outcome: bug found + fixed (verify pending)
- **Symptom (trace `019ec2b3`):** a multi-target query routed CSV processing to CoderAgent in
  THREE invocations: (#1) tried a hallucinated URL `s3.amazonaws.com/agentmemory/…` → 404; (#2)
  given the REAL presigned URLs → `curl` succeeded, downloaded + read the REAL generated molecules
  (e.g. `O=C(c1cccc(-c2nc(Cc3ccccc3)cs2)c1)N1CCN(Cc2ccccc2)CC1`) into `raw_data/`; (#3) asked to
  "load raw_data and sort top-5 by IC50" → `list_directory raw_data` = **"Path not found"** → the
  agent **FABRICATED** fake CSVs (`CCO,0.75,2.1` / `CCN` / …) and "sorted" the fakes. That fake
  list is what reached the chat (the `019ec2b3` false molecules).
- **Root cause:** each invocation ran in a DIFFERENT sandbox — `workspace_id` per call:
  `ws_7313ff1a` (#1) → `ws_1498252c` (#2, real files here) → `ws_cb0017ab` (#3, empty). ADK
  `AgentTool.run_async` creates a BRAND-NEW sub-session (fresh uuid id, new `InMemorySessionService`,
  `agent_tool.py:232/243`) for every CoderAgent invocation, and `CoderToolset._workspace_id` keyed
  the workspace to that per-invocation **session id** → a new empty sandbox each call.
- **Fix (`CoScientist/tools/coder_tools.py:CoderToolset._workspace_id`):** anchor to **session
  STATE first** — AgentTool copies the parent state into each new sub-session (`state=state_dict`,
  `agent_tool.py:246`) and forwards the delta back (`:258`), so a `coder_workspace_id` stored in
  state survives across invocations; only generate (from session id / uuid) + persist on first use.
- **⚠ Verify pending:** confirm e2e that invocation #2's download is visible to #3 after the fix.
  Also defends: the anti-fabrication CoderAgent prompt (F015) now tells it to report missing files
  instead of inventing them.

## ✅ TODO
- [ ] **Verify the workspace-persistence fix e2e** (F002.A2): a download in one CoderAgent call must
      be readable in a later call within the same request.
- [ ] No benchmark/eval for coder success rate — add a small task suite and record in `benchmarks:`.
- [ ] Persist useful sandbox-built scripts as reusable skills (Voyager-style) instead of one-shot.

## ⚠ Pitfalls / Known problems
- Overlap with `TaskExecutorAgent`: if routing blurs, the orchestrator may write
  code for something an existing MCP tool already does. Keep the "write new" vs
  "run existing" boundary explicit in routing (`catalog.py`).

## Symbols
- `CoScientist/tools/coder_tools.py:CoderToolset` — sandbox code/shell/git toolset.
- `CoScientist/agents/catalog.py` — `CoderAgent` AgentSpec (description + routing).
