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
  - CoScientist/experiments/gate.py:deterministic_gate   # deterministic-gate-first (S016), built F015a.A3
  - scripts/experiments/plan_critic_probe.py             # FEDOT-free critic-mode comparison harness (F015a.A3)
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

## Findings (F015a.A3, 2026-06-14 — FEDOT-free critic-mode experiment, 32 runs)
- The current tag-based plan-critic (`critic_agent.py:plan_critic_only`, "tags" mode) is **INERT**:
  **plan-fire 0/8**, fired only on grounding turns, **missed both bad plans** (TP 0/4). Confirms the
  tag-detection thesis empirically — RETIRE it.
- A **delegation-gate** critic (fire on the first roster `function_call.name`, tag-free) restored
  detection to **8/8** and caught all bad plans (**TP 4/4**) at ~half the per-action churn (2.25 vs 5.12
  firings). This is the validated TRIGGER half of the F015b design.
- The S016 **deterministic-gate-first** lever is now real: `experiments/gate.py:deterministic_gate`
  catches the gap/wrong hazards (empty-compute-step + unresolvable tool) with ZERO LLM. The LLM critic is
  demoted to advisory (REVISE / thought-note only — a bare-text REJECT TERMINATES the ADK invocation).
- Evidence: `scripts/experiments/results/plan_critic_2026-06-14_195250.json`, `opik_dump/traces_since_2026-06-14/`.

## ✅ TODO
- [~] Deterministic plan validator — schema+DAG done (`plan.py`); **tool-resolvability + empty-compute-step
      gate DONE** (`experiments/gate.py`, F015a.A3); input-provenance + live-inventory (F015c) pending.
- [ ] Bounded critic loop (≤3, failure-triggered, no-progress hash → HITL) — note: `loop_guard_repeats` is
      HITL-only (settings.py:170), invisible to the callback; needs a NEW callback-side bound.
- [ ] Wire delegation-gate as the tag-free TRIGGER in prod (`agents.py:320`), retire `plan_critic_only`.
- [ ] Decide critic model/instance + pinned provider; bench vs self-consistency (k=3–5 plans).

## Symbols
- `CoScientist/agents/critic_agent.py` — existing critic to refactor (verdict enums, `_CRITIC_MODEL`).
