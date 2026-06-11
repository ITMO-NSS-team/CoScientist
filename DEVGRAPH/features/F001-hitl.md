---
id: F001
title: Human-in-the-Loop (HITL)
type: feature
status: in_progress
created: 2026-06-11
updated: 2026-06-11
owners: [SoloWayG]
derives_from: [F000]
depends_on: [F000]
sources: [S008]
tags: [hitl, orchestration, ux, control]
code:
  - CoScientist/hitl/models.py:HITLRequest
  - CoScientist/hitl/models.py:HITLResponse
  - CoScientist/hitl/handler.py
  - CoScientist/hitl/callbacks.py
  - CoScientist/hitl/session_agent.py
  - CoScientist/hitl/tool.py
benchmarks: []
---

## Goal
Let a human intervene in the agent loop — approve/reject a planned action, pick
among options, edit an agent's proposed output, or supply free-form input —
instead of the system running fully autonomously.

## Current state
A dedicated `CoScientist/hitl/` module exists (added in #246, commit `8118d01`).
The data model is in place: `HITLAction` = {approve, reject, select, edit,
provide_input}; `HITLRequest` (agent_name, action_type, message, options,
context, `invoked_via` ∈ {callback, tool, internal_loop}, timeout) and
`HITLResponse` (action, selected_option, instructions, free_input, approved).
There is a handler, ADK callbacks, a session agent, and a tool wrapper.

> ⚠ **Not yet verified by this graph:** whether/where HITL is actually invoked in
> the *live* orchestrator pipeline (callback vs tool vs internal loop), and how
> the human response is surfaced in the CLI/frontend. Confirm before extending.

## Attempts
### F001.A1 — Initial HITL module (#246) · prior work · outcome: partial
- **Method:** model HITL as typed request/response objects + handler + ADK
  callbacks + a tool, so any agent can pause for a human via `invoked_via`.
- **Result:** module and types exist and import; covers the 5 interaction types.
- **Evidence:** commit `8118d01`; files present under `CoScientist/hitl/`
  (`models.py`, `handler.py`, `callbacks.py`, `session_agent.py`, `tool.py`).
- **Sources used:** [S008] (HITL agent-control patterns) — `unverified`.
- **Next:** trace one concrete invocation end-to-end (orchestrator → HITLRequest →
  human → HITLResponse → resumed action) and record it as F001.A2 with evidence.

## ✅ TODO
- [ ] Verify and document where HITL fires in the live pipeline (which callback / which agents).
- [ ] End-to-end demo: approve + reject + edit paths, with CLI/frontend rendering.
- [ ] Timeout behavior: what happens when `timeout_seconds` elapses with no human?
- [ ] Decide default policy: which agent actions require approval out of the box?

## ⚠ Pitfalls / Known problems
- The module's *existence* ≠ it being *wired in*. Don't assume HITL gates actions
  until F001.A2 proves the path with evidence.
- Three invocation sources (`callback`/`tool`/`internal_loop`) can diverge — keep
  them consistent or one path will silently bypass the human.

## Symbols
- `CoScientist/hitl/models.py:HITLAction` — enum of the 5 interaction types.
- `CoScientist/hitl/models.py:HITLRequest` / `:HITLResponse` — the wire contract.
- `CoScientist/hitl/handler.py` — request/response orchestration (verify entrypoints).
- `CoScientist/hitl/tool.py` — exposes HITL as an agent-callable tool.
