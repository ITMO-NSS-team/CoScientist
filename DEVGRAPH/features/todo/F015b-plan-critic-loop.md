---
id: F015b
title: Plan-critic loop — deterministic-first, externally-grounded, hard-bounded plan verification
type: feature
status: proposed
created: 2026-06-11
updated: 2026-06-11
owners: [SoloWayG]
derives_from: [F015, F006]
depends_on: [F015a, F015c]
sources: [S016, S017, S018, S011, S010]
tags: [critic, plan-verification, bounded-iteration, reliability]
code:
  - CoScientist/agents/critic_agent.py        # existing pre/post critic (_CRITIC_MODEL = main_model)
benchmarks: []
---

## Goal
Verify the F015a plan **before execution** for completeness / consistency / fit, looping
back to the planner with a **hard iteration budget**. The structural antidote to F014's
≤81-call runaways: catch a bad plan on paper, not after 700s of thrash.

## Best practices to adopt
- **SOPStruct — deterministic gate first, LLM critic second [S016]:** the HARD gate is
  code-only (no LLM): JSON-schema validity, DAG connectivity (start→end, acyclic), every
  step input is external or a declared upstream artifact, every `required_tool` resolvable
  against the live inventory — **this last check IS F015c; fold it in, don't make it a
  separate downstream stage** (see F015 gap). Checklist taxonomy: {missing_prereq, ordering,
  redundancy, infeasible}; emit per-step `{error, category, justification, corrective_guidance}`.
- **Self-critique is unreliable / can DEGRADE [S017]:** do NOT build F015b as the current
  `critic_agent.py` pure-LLM APPROVE/REVISE/REJECT. Intrinsic self-critique hurt GPT-4 on
  GSM8K. Keep subjective hypothesis-fit judgment **advisory-only**; gate only on deterministic
  checks; prefer a critic model **instance separate** from the planner (dodge self-preference).
- **Self-Refine + Reflexion bounding [S018]:** three **code-enforced** stop rules: (1) max
  2–3 critic→rework rounds; (2) invoke the LLM critic only after a deterministic check flags
  something; (3) **no-progress detection** — hash plan JSON + fired-error set each round; if
  either repeats, STOP → escalate to HITL (F001).
- **⚠ REUSE, don't rebuild:** the bounded "deterministic gate → nudge re-prompt → give up"
  mechanic already exists in Alembic — `main.py:196-222` (`MAX_GUARD_RETRIES=3`: guard fires →
  agent re-invoked with a nudge message → max retries → give up). F015b is this same pattern
  with the plan validator as the guard; lift it rather than writing a new loop.

## ⚠ Risks / open questions (incl. adversarial review)
- **Reflection entrenchment** (DEVGRAPH §6): an LLM critic can entrench a false
  "incomplete"/"fine" verdict and oscillate. Require it to cite the specific failing
  field/step as evidence; max-iters alone is insufficient without no-progress detection.
- **Critic shares the flaky main model:** `critic_agent.py` sets `_CRITIC_MODEL =
  settings.llm.main_model` — the very model F014 shows is flaky. "Use a separate instance"
  is asserted, not designed: which **pinned** model does the critic use? (cost/pinning, F015.)
- **Schema-valid JSON prerequisite:** without constrained decoding the gate burns its 2–3
  iterations on malformed JSON, not semantic errors (F014.A2 empties). See F015.

## ✅ TODO
- [ ] Deterministic plan validator (schema + DAG + input-provenance + tool-resolvability via F015c).
- [ ] Bounded critic loop (≤3, failure-triggered, no-progress hash → HITL).
- [ ] Decide critic model/instance + pinned provider; bench vs self-consistency (k=3–5 plans).

## Symbols
- `CoScientist/agents/critic_agent.py` — existing critic to refactor (verdict enums, `_CRITIC_MODEL`).
