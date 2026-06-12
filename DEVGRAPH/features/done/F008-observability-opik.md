---
id: F008
title: Observability — Opik tracer
type: feature
status: done
created: 2026-06-11
updated: 2026-06-13
owners: [ITMO-NSS-team]
derives_from: [F000]
depends_on: [F000]
sources: []
tags: [observability, tracing, opik]
code:
  - CoScientist/logging/opik_tracer.py
  - CoScientist/config/settings.py
  - scripts/opik_eval/trace_locator.py
  - scripts/opik_eval/ab_analyze.py
benchmarks: []
---

## Goal
Trace agent/LLM activity (Opik) so runs are observable and debuggable, and fix the
orchestrator prompt issues surfaced while tracing.

## Current state
Shipped in #225 (commit `5ddbe1e`). `logging/opik_tracer.py` provides the tracer;
it is wired into `agents.py` and `critic_agent.py` and configured via
`config/settings.py`. The same PR fixed orchestrator prompt issues.

## Attempts
### F008.A1 — Opik tracer + orchestrator prompt fix (#225) · earlier · outcome: success
- **Method:** add an Opik tracer integration and correct the orchestrator prompt.
- **Result:** runs are traceable; orchestrator prompt issues resolved.
- **Evidence:** commit `5ddbe1e` (#225); `logging/opik_tracer.py`, settings entries.
### F008.A2 — Reliable run→trace correlation (server-side filter + manifest) · 2026-06-13 · outcome: success
- **Method:** the eval harness was finding traces by `search_traces(max_results=N)` then
  filtering `thread_id` **on the client** — fragile: a run's trace silently drops off once
  >N other traces arrive. Replaced with two reliable mechanisms in
  `scripts/opik_eval/trace_locator.py`: (1) **server-side OQL** — `thread_id` is a
  filterable string column, so `search_traces(filter_string='thread_id = "<session_id>"')`
  matches regardless of volume (ops `=`/`starts_with`/`contains`; `wait_for_at_least=1`
  blocks until a just-finished run is indexed); (2) a **manifest**
  (`results/trace_manifest.jsonl`) — `ab_runner.py` calls `record_run(session_id, query,
  condition, model)` after each run, persisting `session_id→trace_id` (+query/model/start)
  for later direct lookup with NO Opik call. The join key is `session_id`
  (`main.py:CoScientistManager`) == trace `thread_id` (ADK OpikTracer,
  `opik/integrations/adk/opik_tracer.py:173`). `ab_analyze.py` now uses the server-side
  resolver. CLI: `trace_locator.py <session>|--prefix|--contains|--query`.
- **Result:** any past run's trace is retrievable at any time by session_id (exact),
  run-family prefix, or query text. Verified live against opik 2.0.21 / project
  `adk-coscientist`: server-side `thread_id` filter returned the real trace UUIDs;
  backfilled the manifest with all 13 existing `ab_*` runs; lookup-by-query →
  trace_id → spans round-trips correctly.
- **Evidence:** `scripts/opik_eval/trace_locator.py` (+ `ab_analyze.py`/`ab_runner.py`
  wiring); manifest at `scripts/experiments/results/trace_manifest.jsonl` (13 rows, e.g.
  `ab_B2_00_225459 → 019ebcf9-440d-7f91-88c9-ef42cb1e787a`). See [[opik-tracing-access]].
  Cross-ref F014 / F015h (the eval harness that consumes this).
- **Next:** ad-hoc `python -m CoScientist.main` runs use the static default
  `session_001` (not individually resolvable) — pass a unique `session_id`, or
  optionally make `CoScientistManager` default to a unique id + auto-`record_run`.

## ✅ TODO
- [ ] Document how to enable/point Opik (env vars) in the project card / docs.
- [ ] Confirm tracing covers the newer agents (Coder F002, Medical F004).
- [ ] (Optional) Auto-record the manifest from `CoScientistManager` so EVERY run (not
  just the A/B harness) is correlatable; needs a unique default `session_id`.

## ⚠ Pitfalls / Known problems
- Tracing depends on settings/env being present; if misconfigured it can no-op
  silently — verify traces actually arrive before relying on them for debugging.
- **Do NOT find a run's trace by `search_traces(max_results=N)` + client-side
  `thread_id` filter** — it drops your trace once >N other traces exist. Use the
  server-side `filter_string` / the manifest (F008.A2).

## Symbols
- `CoScientist/logging/opik_tracer.py` — Opik tracing integration; ADK `OpikTracer`
  records the run's `session_id` as the trace `thread_id` (the run↔trace join key).
- `scripts/opik_eval/trace_locator.py` — reliable run→trace resolution: server-side
  `thread_id` OQL filter (`resolve_traces`/`find_trace`) + a durable `record_run`/
  `lookup` manifest (`results/trace_manifest.jsonl`). CLI by session/prefix/query.
