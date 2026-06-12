---
id: F004
title: Medical agent + ADK frontend
type: feature
status: done
created: 2026-06-11
updated: 2026-06-11
owners: [ITMO-NSS-team]
derives_from: [F000]
depends_on: [F000]
sources: []
tags: [medical, dicom, pubmed, frontend, adk]
code:
  - CoScientist/tools/med_tools.py:MedToolset
  - CoScientist/agents/med_callbacks.py:_make_artifact_id
benchmarks: []
---

## Goal
A MedicalAgent for clinical questions (PubMed search, PICO extraction, study
taxonomy, DICOM image analysis) plus the ADK web frontend for interacting with
the system (sessions/artifacts).

## Current state
Shipped in #262 (commit `9976bc8`). `MedToolset` (`med_tools.py`) provides the
medical tools; `med_callbacks.py` turns uploaded medical images into artifacts
(`_make_artifact_id`) passed via `artifact_id`. The ADK frontend persists sessions
under `.adk/artifacts/users/<user>/sessions/` (present in-repo).

## Attempts
### F004.A1 — Medical agent + ADK frontend (#262) · 2026-06-09 · outcome: success
- **Method:** add a domain agent with a dedicated toolset + image-artifact callbacks;
  expose the ADK frontend for chat/sessions.
- **Result:** clinical Q&A + DICOM analysis available; web UI for sessions.
- **Evidence:** commit `9976bc8` (#262); `med_tools.py:MedToolset`; `.adk/artifacts/.../sessions/` exists.

## ✅ TODO
- [ ] No clinical-accuracy eval — medical answers are unverified for correctness.
- [ ] Document the frontend run/serve command in the project card.

## ⚠ Pitfalls / Known problems
- Medical output is **not** clinically validated — must not be presented as advice.
- DICOM handling depends on the artifact_id being passed through from the orchestrator;
  if routing drops it, image analysis silently degrades to text-only.

## Symbols
- `CoScientist/tools/med_tools.py:MedToolset` — PubMed/PICO/taxonomy/DICOM toolset.
- `CoScientist/agents/med_callbacks.py:_make_artifact_id` — registers uploaded medical images as artifacts.
