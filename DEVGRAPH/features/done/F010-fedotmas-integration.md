---
id: F010
title: FEDOT.MAS integration (experiment execution)
type: feature
status: done
created: 2026-06-11
updated: 2026-06-13
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
### F010.A3 — `molecule_generator` paraphrases away the S3 result link & HALLUCINATES SMILES · 2026-06-13 · outcome: defect confirmed (fix → F015g)
- **Question chased:** does FEDOT's result reach the orchestrator/chat, and "the MCP should return
  an S3 link with the molecules" (user). Traced where the mol-gen result dies.
- **Method:** (1) live-called the real generative MCP `generate_mols` (GenerativeMoleculeModels,
  `http://10.32.2.2:8764/mcp`); (2) read Opik trace `019ebdab-9c96-7da4-ba48-9ef4bde6dbb2`
  (dataset_L row1); (3) read the FEDOT.MAS/ADK `output_key` wiring (3-reader workflow `fedot-s3-link-trace`).
- **Finding 1 — the tool returns an S3 link, NOT inline molecules.** Raw `generate_mols(num=3)`:
  `{status:"ok", columns:["Smiles","QED","LogP",…], bucket_name:"molecule-generative-mcp",
  results_s3_key:"generated/gan_default/<hash>.csv",
  results_presigned_url:"http://10.32.1.114:9000/…csv?X-Amz-Signature=…", expires_in:3600}`, with
  `upload_results_to_s3` default **true**. The real molecules live in the CSV behind the presigned
  URL; the payload carries **no SMILES**.
- **Finding 2 — output_key = the sub-agent's LLM paraphrase, not the raw tool result.** The
  `molecule_generator` sub-agent is a vanilla ADK `LlmAgent`; ADK feeds the tool `function_response`
  back to the LLM and persists to state **only the LLM's next text turn**
  (`google/adk/agents/llm_agent.py:837-851` writes `part.text`; the raw `function_response` with the
  presigned URL is never stored). The worker **technically CAN** echo the link in its text — there's
  no hard block — but it's the LLM's *choice*, not a passthrough. In this run it (a) **dropped** the
  presigned URL and (b) **fabricated** 15 SMILES that were never in the tool output (trace shows no
  CSV-fetch between `generate_mols` and the answer; fedotmas has no S3/CSV download). → hallucinated
  molecules; the real ones lost in S3.
- **Finding 3 — defect B (orchestrator finalization).** Even the paraphrase doesn't reach chat: the
  master orchestrator's final synthesis replaces it with meta-commentary ("generated 15 generic
  molecules, however the request needs target-specific…"); `SMILES-like in final: NONE`. (cross-ref F000.)
- **Intended design (user, 2026-06-13):** the worker should return its result (incl. the S3 link),
  the experiments orchestrator / master-orchestrator then **downloads the CSV** and decides whether
  the task is done; the `vault` MCP (`http://10.32.11.45:8000/mcp`, "S3 vault tool for persisting
  agent's artifacts") is the helper agents use to pull the file link. The channel is fine
  (`fedot_tool`→`mas.run()`→state), but today the state holds the lossy paraphrase, so the real
  artifact never reaches the orchestrator to judge completion.
- **Evidence:** live MCP-call output above; Opik trace 019ebdab (`fedot_tool` out =
  `{status:success, result:{user_query…, molecule_generator_output:"1. COc1cccc(…)"}}`, no S3 key;
  final trace output has no SMILES); workflow `fedot-s3-link-trace` (3 readers, file:line).
- **Outcome:** correctness defect confirmed — a FEDOT.MAS/ADK data-flow issue, NOT a planner/critic
  one (makes the dataset_L reactplan-vs-inline A/B moot). Fix belongs to **F015g**: capture/forward
  the tool's structured S3 artifact (`skip_summarization` or an `after_tool_callback` that lifts
  `results_presigned_url` into state), and define success as `expected_artifacts` materialized in S3
  — NOT the worker's LLM summary. Do NOT edit MCP tools (F014.D1).

## ✅ TODO
- [ ] **Forward the tool's S3 artifact, don't trust the paraphrase (F010.A3):** capture
      `results_presigned_url`/`results_s3_key` from `generate_*` results (raw `function_response`,
      not the worker's text) and surface it — download CSV → real molecules, or hand the link to the
      orchestrator/`vault`. Tracked under F015g.
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
- **⚠ The sub-agent returns an LLM PARAPHRASE, not the raw tool result (F010.A3).** A FEDOT.MAS
  sub-agent's state value (`{name}_output`) is whatever its LLM *wrote* after the tool ran, NOT the
  tool's `function_response`. So structured payloads leak: the mol-gen MCP tools
  (`generate_mols`/`generate_case_mols`) return a **presigned S3 URL to a results CSV** (no inline
  SMILES), but the `molecule_generator` sub-agent paraphrased it — **dropping the S3 link and
  hallucinating 15 SMILES** that weren't in the tool output. Never trust `molecule_generator_output`
  SMILES as real; the real molecules are in the S3 CSV. The sub-agent *can* echo the link, but it's
  the LLM's choice, not a passthrough — capture the raw result deterministically instead. Fix → F015g.

## Symbols
- `CoScientist/tools/fedotmas_tools.py:FedotMASToolset` — FEDOT.MAS experiment toolset; `fedot_tool`
  returns `mas.run(task)` (= FEDOT.MAS session state) verbatim — no S3 handling of its own.
- `google/adk/agents/llm_agent.py:837-851` (installed ADK) — writes only the sub-agent's `part.text`
  to `output_key`; the raw tool `function_response` (carrying `results_presigned_url`) is never
  persisted. **The drop point.**
- GenerativeMoleculeModels MCP `http://10.32.2.2:8764/mcp` — `generate_mols`/`generate_case_mols`
  return `{results_presigned_url, results_s3_key, bucket_name, columns, …}`; `upload_results_to_s3`
  default `true`. Molecules are in the CSV behind the URL, not in the payload.
- `vault` MCP `http://10.32.11.45:8000/mcp` — "S3 vault tool for persisting agent's artifacts"; the
  intended helper for pulling/holding the file link (per intended design, F010.A3).
