---
id: F010
title: FEDOT.MAS integration (experiment execution)
type: feature
status: done
created: 2026-06-11
updated: 2026-06-11
owners: [ITMO-NSS-team]
derives_from: [F000]
depends_on: [F000]
sources: []
tags: [fedotmas, experiments, automl, pipelines]
code:
  - CoScientist/tools/fedotmas_tools.py:FedotMASToolset
benchmarks: []
---

## Goal
Run computational experiments by building multi-agent pipelines from text
descriptions via FEDOT.MAS — the ExperimentAgent's execution backend.

## Current state
Shipped in #211 (commit `bfb4fe3`) with a follow-up bug fix in #224 (`c842f2d`).
`fedotmas_tools.py:FedotMASToolset` wraps FEDOT.MAS. Depends on the external
`fedotmas` package (ITMO, SSH install — see README).

## Attempts
### F010.A1 — FEDOT.MAS integration (#211) · earlier · outcome: success
- **Method:** wrap FEDOT.MAS as an ADK toolset that turns text task descriptions
  into runnable multi-agent ML pipelines.
- **Evidence:** commit `bfb4fe3` (#211); `FedotMASToolset`.
### F010.A2 — FEDOT bug fix (#224) · earlier · outcome: success
- **Method:** fix a FEDOT execution bug.
- **Evidence:** commit `c842f2d` (#224).

## ✅ TODO
- [ ] No recorded benchmark of pipeline-build success / experiment validity.
- [ ] Document the SSH-based `fedotmas` install as a hard prerequisite.
- [ ] **"Non-existent tools" re-rooted (F014.A2):** Opik traces show the symptom is an
      LLM in the FEDOT.MAS `molecule_generator` sub-agent calling tools it isn't
      equipped with (`Tool 'smiles2props'/'predict_ml' not found`) — NOT an unvalidated
      server payload. Fix is the experiments module **F015** (per-step tool-sufficiency
      + Alembic), per decision **F014.D1** — do not edit MCP tools. Still worth logging
      the `servers_payload` sent to FEDOT.MAS.

## ⚠ Pitfalls / Known problems
- `fedotmas` installs from a private ITMO repo over SSH — without access this whole
  capability is unavailable; the ExperimentAgent then can't run experiments.

## Symbols
- `CoScientist/tools/fedotmas_tools.py:FedotMASToolset` — FEDOT.MAS experiment toolset.
