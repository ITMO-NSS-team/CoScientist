---
id: F008
title: Observability — Opik tracer
type: feature
status: done
created: 2026-06-11
updated: 2026-06-11
owners: [ITMO-NSS-team]
derives_from: [F000]
depends_on: [F000]
sources: []
tags: [observability, tracing, opik]
code:
  - CoScientist/logging/opik_tracer.py
  - CoScientist/config/settings.py
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

## ✅ TODO
- [ ] Document how to enable/point Opik (env vars) in the project card / docs.
- [ ] Confirm tracing covers the newer agents (Coder F002, Medical F004).

## ⚠ Pitfalls / Known problems
- Tracing depends on settings/env being present; if misconfigured it can no-op
  silently — verify traces actually arrive before relying on them for debugging.

## Symbols
- `CoScientist/logging/opik_tracer.py` — Opik tracing integration.
