---
id: F015h
title: АМ eval harness on dataset_S — prove the module fixes F014 (don't entrench "it works")
type: feature
status: in_progress
created: 2026-06-11
updated: 2026-06-12
owners: [SoloWayG]
derives_from: [F015, F014]
depends_on: [F008]
sources: [S019]
tags: [eval, benchmark, opik, reliability, anti-entrenchment]
code:
  - CoScientist/dataset_S.xlsx
  - scripts/opik_eval/run_baseline.py
  - scripts/opik_eval/metrics.py
  - scripts/opik_eval/opik_client.py
benchmarks: []
---

## Goal
A named, owned **eval harness** (flagged as missing by both synthesis and the adversarial
review). The whole АМ justification — lowers tool-not-found rate, kills runaways — is
**unproven without measurement**, and the project records zero benchmarks. This is the only
way to prove F015 fixes F014 instead of entrenching a false "it works" (DEVGRAPH §6).

## Designed approach
- Reuse **`dataset_S.xlsx`** (`case` / `content` / `decomposers_tasks` / `is_correct`) and read
  run internals from **Opik** (the approach validated in F014.A2; see `scripts/opik_eval/*` —
  to be committed) — trace name `multi-agent-orchestrator`, `metadata.main_model`, span types.
- **Metrics** (current firehose path vs the АМ path): loop size (#LLM calls), tool-not-found
  count, runaway/700s-ceiling incidence, `is_correct` per case, calls-per-task, $/latency.
- **MetaTool-style labeled sufficiency set** [S019]: ground-truth sufficient servers per task +
  a **held-out genuinely-unsupported class** → measure F015c gap-detection precision/recall
  (catch false Type-A gaps that would trigger needless Alembic builds).

## Attempts
### F015h.A1 — Opik baseline harness + first firehose snapshot (roadmap R04) · 2026-06-12 · outcome: partial
- **Method:** built `scripts/opik_eval/` (standalone, no CoScientist import; creds from
  `~/.opik.config`/env). `metrics.py` computes per-trace reliability (empty-LLM, tool-not-found
  via regex on span `error_info`, runaway = ≥25 LLM calls or ≥690s, repeated-toolcall, errors,
  tokens) with 429-backoff; `run_baseline.py` aggregates by `main_model` → markdown + JSON.
- **Result (last 24 traces, `results/baseline_2026-06-12.md`):** qwen3 (16): median **8** LLM
  calls, **18.8%** errored, **6.2%** empty-resp, 6 tool-not-found. gpt-oss-120b (5): median 11,
  **60%** errored, **40%** empty-resp. ALL: median 7.5, 25% errored, 20.8% empty, 6 tool-not-found.
  Hallucinated tools: `predict_ml`×4 (FEDOT `molecule_generator`, F015c), `request_approval`×2
  (HITL, F001). Error types: KeyError×2, ValueError×2, ExceptionGroup×1, EOFError×1.
- **Evidence:** `scripts/opik_eval/results/baseline_2026-06-12.md` (+ `.json`); run
  `.venv/bin/python scripts/opik_eval/run_baseline.py --limit 24`.
- **⚠ Caveat:** the last-24 window mixes ad-hoc debugging runs (CRISPR-papers, CoderAgent probes)
  with benchmark tasks — NOT a clean dataset_S pass. The KeyError/EOFError here are exactly the
  bugs just fixed (F006.A3 / F001.A2), i.e. pre-fix traces. The extreme runaways (≤81 LLM calls,
  F014.A1) are OLDER than this window — raise `--limit` or add a date filter to capture them.
  This is a *snapshot baseline*, not the controlled acceptance run (that's roadmap R19).
- **Next:** clean controlled dataset_S baseline (fixed task set, one model, file-logged) for R19.

## ⚠ Risks / open questions
- **Local infra caps live runs here** (Postgres `:5432` down, ITMO MCP needs VPN — F014.A1);
  full multi-step runs need real infra. Read existing traces from Opik where live runs aren't possible.
- Cost: 9-condition × N-query matrices are real money/time — scope and pin a cheap model for runs (LATM).
- Cross-cuts F015a/F015b/F015c/F015g — keep it ONE shared harness, not folded silently into each.

## ✅ TODO
- [x] Commit the Opik-reading eval scripts under `scripts/opik_eval/` (F015h.A1: client+metrics+run_baseline).
- [ ] Clean controlled dataset_S baseline (fixed task set, single model, file-logged) — not a mixed window.
- [ ] Alembic axis: **extend** the existing `run_benchmark.py` (branch: parallel repo→MCP build
      harness, parses `validation.md` per image) — don't write a second harness for build success.
- [ ] dataset_S runner (firehose vs АМ) → metrics table; read from Opik.
- [ ] Labeled sufficiency set (ground-truth servers + held-out unsupported) for F015c precision/recall.
- [ ] Wire as the acceptance gate for F015 (ТП KPI: ≤20% time vs firehose; fewer failed runs).

## Symbols
- `CoScientist/dataset_S.xlsx` — the benchmark (drug-design molecule generation).
