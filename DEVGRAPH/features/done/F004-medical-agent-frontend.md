---
id: F004
title: Medical agent + ADK frontend
type: feature
status: done
created: 2026-06-11
updated: 2026-06-12
owners: [ITMO-NSS-team]
derives_from: [F000]
depends_on: [F000]
sources: []
tags: [medical, dicom, pubmed, frontend, adk]
code:
  - CoScientist/tools/med_tools.py:MedToolset
  - CoScientist/agents/med_callbacks.py:_make_artifact_id
  - CoScientist/agents/med_callbacks.py:upload_intake_before_model
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
### F004.A2 — Rename the upload-intake callback for clarity · 2026-06-12 · outcome: success
- **Method:** `med_callbacks.py:before_model_modifier` — the orchestrator's
  `before_model_callback`, imported under the misleading alias `med_before_model` —
  is actually generic upload intake (intercepts ANY `inline_data` part, saves it to
  the artifact store, registers the id in state, strips raw bytes because LiteLLM
  can't take binary MIME). Renamed to `upload_intake_before_model`; alias dropped.
- **Result:** call site (`agents.py:311`) now reads honestly; no behavior change.
  The state key `uploaded_medical_artifacts` was deliberately left as-is —
  renaming it would orphan registered uploads in persisted `.adk/` sessions.
- **Evidence:** `agents.py:21,311`, `med_callbacks.py:13`; `tests/unit/` pass (20).

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
- `CoScientist/agents/med_callbacks.py:upload_intake_before_model` — orchestrator-level
  upload intake (any file type, not just medical); ex-`before_model_modifier` (F004.A2).
- `CoScientist/agents/med_callbacks.py:med_agent_before_model` — MedicalAgent-side
  reminder injecting available `artifact_id`s from session state.
