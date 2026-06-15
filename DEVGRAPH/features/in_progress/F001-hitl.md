---
id: F001
title: Human-in-the-Loop (HITL)
type: feature
status: in_progress
created: 2026-06-11
updated: 2026-06-12
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

> ✅ **Live wiring confirmed (F001.A2, 2026-06-12):** HITL fires via the **callback**
> path on the **CoderAgent's outward-facing/hard-to-reverse command** approval
> (`coder_tools.py:_maybe_request_approval` → `ConsoleHITLHandler.handle_request`,
> `invoked_via="callback"`, `action_type=APPROVE`). The default handler when the
> manager gets `hitl_handler=None` is `ConsoleHITLHandler`. The **tool** and
> **internal_loop** paths, edit/select flows, and frontend rendering remain unverified.

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
### F001.A2 — Live wiring confirmed + headless-safety fix · 2026-06-12 · outcome: partial
- **Method:** ran the orchestrator headlessly and traced (Opik) where HITL fires.
- **Result:** HITL **does** fire live — the CoderAgent's outward-facing command path
  calls `coder_tools.py:_maybe_request_approval` → `ConsoleHITLHandler.handle_request`
  (`invoked_via="callback"`, `APPROVE`). But `ConsoleHITLHandler` blocks on `input()`
  via `asyncio.to_thread`; in a non-TTY/headless run that raises `EOFError` and
  **kills the entire orchestrator run**. Fix: when `not sys.stdin.isatty()` (or
  `input()` EOFs) auto-**REJECT** — safe default, never run outward-facing actions
  unattended — instead of crashing, so the orchestrator continues and finalizes.
- **Evidence:** Opik traces `019ebb49` / `019ebb58` (`CoderAgent` span err=True,
  `EOFError: EOF when reading a line`, frame `hitl/handler.py:52`); fix in
  `hitl/handler.py:ConsoleHITLHandler.handle_request`; unit tests
  `tests/unit/test_hitl_handler_headless.py` (3 pass). See [[opik-tracing-access]].
- **Next:** still unverified — edit/reject/select paths end-to-end, frontend
  rendering, and `timeout_seconds` behavior. Make the headless default (reject vs
  approve) configurable.
### F001.A3 — headless auto-approve flag + loop-guard (EXPERIMENTAL, uncommitted) · 2026-06-13 · outcome: built, test pending (infra down)
- **Why:** HITL is NOT disabled — `settings.hitl.enabled=True` and the headless handler
  AUTO-REJECTS (`handler.py`). On reject the before-callback returns "blocked by human"
  → the orchestrator **re-delegates to CoderAgent → auto-reject → loop → timeout** (S5 Q2,
  7+ HITL requests). User direction (2026-06-13): auto-approve under a flag.
- **Change:** `HITLSettings.headless_auto_approve: bool=False` + `loop_guard_repeats: int=3`
  (`config/settings.py`); `ConsoleHITLHandler` now: in headless, APPROVE if
  `headless_auto_approve`, else count identical requests and AUTO-APPROVE after
  `loop_guard_repeats` to break an infinite reject→retry loop, else REJECT. Per-instance
  `_seen` signature counter. `l_runner.py` sets `HITL__HEADLESS_AUTO_APPROVE=true` for tests.
- **Status:** built + imports OK; **e2e test on dataset_L pending** (ITMO infra/VPN down at the moment).

## ✅ TODO
- [x] Verify and document where HITL fires in the live pipeline (F001.A2: **callback** on CoderAgent outward-facing commands).
- [x] Headless default policy: reject-vs-approve now configurable (`HITL__HEADLESS_AUTO_APPROVE`) + loop-guard (F001.A3).
- [ ] End-to-end demo: approve + reject + edit paths, with CLI/frontend rendering.
- [ ] Timeout behavior: what happens when `timeout_seconds` elapses with no human?
- [ ] Decide default policy: which agent actions require approval out of the box?

## ⚠ Pitfalls / Known problems
- The module's *existence* ≠ it being *wired in*. F001.A2 proved **one** path
  (callback on CoderAgent commands); tool/internal_loop paths still unproven.
- Three invocation sources (`callback`/`tool`/`internal_loop`) can diverge — keep
  them consistent or one path will silently bypass the human.
- `ConsoleHITLHandler.handle_request` blocks on `input()`. In a headless/server run
  (no TTY) that raises `EOFError` and kills the whole run; it now auto-rejects when
  there is no TTY (F001.A2). Any non-console handler must implement the same
  non-interactive fallback, or headless runs will crash on the first approval gate.

## Symbols
- `CoScientist/hitl/models.py:HITLAction` — enum of the 5 interaction types.
- `CoScientist/hitl/models.py:HITLRequest` / `:HITLResponse` — the wire contract.
- `CoScientist/hitl/handler.py` — request/response orchestration (verify entrypoints).
- `CoScientist/hitl/tool.py` — exposes HITL as an agent-callable tool.
