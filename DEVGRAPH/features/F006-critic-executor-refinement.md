---
id: F006
title: Pre/post critic + executor refinement
type: feature
status: done
created: 2026-06-11
updated: 2026-06-12
owners: [ITMO-NSS-team]
derives_from: [F000]
depends_on: [F000]
sources: []
tags: [critic, self-correction, orchestration, reliability]
code:
  - CoScientist/agents/critic_agent.py:PreVerdict
  - CoScientist/agents/critic_agent.py:PostVerdict
  - CoScientist/agents/critic_agent.py:_extract_completed_trajectory
  - CoScientist/agents/critic_agent.py:_apply_revisions
benchmarks: []
---

## Goal
A critic that vets agent actions before and after execution (pre-action approval
of pending tool calls; post-action verdict over the completed trajectory), with
the executor refined to act on critic revisions.

## Current state
Shipped in #249 (commit `d0cf370`); later given provider routing + retries in the
pipeline refactor (`79cb9c6`). `critic_agent.py` defines `PreVerdict`/`PostVerdict`
enums, extracts the completed trajectory and pending calls, and applies revisions.
The critic roster is rendered from `catalog.py:render_critic_roster()`.

## Attempts
### F006.A1 — Critic & executor refinement (#249) · 2026-06-07 · outcome: success
- **Method:** add pre-action and post-action critics; feed verdicts back into the
  executor to revise/approve tool calls.
- **Result:** orchestrator gets a self-correction loop around tool calls.
- **Evidence:** commit `d0cf370` (#249); enums + `_apply_revisions` in `critic_agent.py`.
### F006.A2 — Critic provider routing + retries · 2026-06-10 · outcome: success
- **Method:** route the critic to pinned providers with retries (same flakiness fix as F000.A1).
- **Evidence:** commit `79cb9c6`.
### F006.A3 — Critic clobbered sub-agent delegation args → `KeyError: 'request'` · 2026-06-12 · outcome: success
- **Method:** on verdict REVISE, `_apply_revisions` overwrote each function_call's
  args wholesale with the critic's `revised_calls[i].args`. For a sub-agent
  delegation (ADK `AgentTool` requires a single `{"request": <str>}`) the critic
  "revised" the call into a domain shape (`{keywords, year, limit}`), dropping
  `request`; `AgentTool.run_async` then did `args['request']` → `KeyError` and
  killed the whole run. Fix: if the original call had a `request` key and the
  revision drops it, keep the original `request` string.
- **Result:** delegation survives critic revision; the "Find 3 papers…" query
  completes end-to-end instead of crashing at the TaskExecutorAgent call.
- **Evidence:** Opik trace `019ebb58` (delegation args became `{keywords, year, limit}`
  → `KeyError: 'request'`) vs fixed run `019ebb5d` (clean final answer);
  `critic_agent.py:_apply_revisions` request-preserve guard; unit tests
  `tests/unit/test_critic_revisions.py` (4 pass). See memory [[opik-tracing-access]].
- **Next:** ideally the critic shouldn't rewrite delegation args at all — consider a
  prompt rule to leave agent-tool `request` calls untouched.

## ✅ TODO
- [ ] No measurement of whether the critic improves task success vs. its latency/cost.
- [ ] Risk of over-blocking: confirm the critic doesn't reject valid actions (needs eval).
- [ ] **Orchestrator loop guard (F014.A1 + A2):** real runs reached **28–81 LLM calls**
      for a single query, several hitting a 700s ceiling; offenders are `write_file` /
      `execute_bash` / `fedot_tool` thrash (worst on qwen3), plus 9 near-duplicate
      ResearchAgent delegations (two differing only by a U+2010 vs U+2011 hyphen).
      The critic/executor should NFKC-normalize + dedup near-identical calls and cap
      repeated same-agent/same-tool calls that add no new info.

## ⚠ Pitfalls / Known problems
- The critic is itself an LLM call → adds latency and a failure point; that's why
  it needed provider pinning + retries (F006.A2). Don't remove those.
- Trajectory/critic prompts truncate values (`_truncate`, 1500 chars) — very large
  tool outputs may be judged on a clipped view.
- `_apply_revisions` rewrites function-call args **by index, wholesale**. It must
  not strip the `request` key of sub-agent (`AgentTool`) delegations — that crashes
  the run with `KeyError: 'request'` (fixed F006.A3). Keep the request-preserve guard.

## Symbols
- `CoScientist/agents/critic_agent.py:PreVerdict` / `:PostVerdict` — verdict enums.
- `CoScientist/agents/critic_agent.py:_extract_completed_trajectory` / `:_apply_revisions`.
