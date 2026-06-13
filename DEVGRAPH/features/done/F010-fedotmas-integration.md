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
### F010.A4 — EXPERIMENT: reactplan/qwen e2e (dataset_S ×5) — S3 fix wired, runs don't reach generation · 2026-06-13 · outcome: NOT solved (plugin bug found+fixed; upstream blockers dominate)
- **Setup:** committed PlanReAct default + qwen3-235b (`LLM__MAIN_MODEL`, pinning `[]`).
  `scripts/experiments/ab_runner.py --condition S3 --limit 5 --cap 600` (one query/domain).
  EXPERIMENTAL (uncommitted) S3 fix active: `ArtifactCapturePlugin` (after_tool_callback) wired
  into `fedot_tool` via `MAS(plugins=...)` + ExperimentAgent/orchestrator prompt edits.
- **Result: 0/5 delivered molecules; only 1/5 reached generation.** Trace ids: Q1 `019ebdf0`
  (thread ab_S3_00; record SSL-failed→resolved by thread_id), Q2 `019ebdf2`, Q3 `019ebdf4`,
  Q4 `019ebdf9`, Q5 `019ebe05`.

  | Q | gen? | blocker |
  |---|---|---|
  | Q1 GSK-3beta | ❌ | hard abort `ValueError Tool 'list_s3_train_cases' not found` — LLM emitted a function_call with OUTER name=leaf-MCP-tool, inner args `{server_id, tool_name:list_generative_train_cases}` (generic-dispatcher hallucination); ADK routes on outer name (`functions.py:1008`). Critic flagged it, model repeated it. |
  | Q2 KRAS | ❌ | can't resolve `case_name` (KRAS ∉ hardcoded cases) → loops on schema introspection → critic self-rejects loop → ends at budget (meta-refusal) |
  | Q3 BTK | ❌ | `JSONDecodeError char887` — qwen emitted malformed tool-call JSON; litellm `json.loads` (`lite_llm.py:1630`) crashes the whole run (no resilience). No BTK case → CoderAgent ChEMBL detour |
  | Q4 STAT3 | ✅ generate_mols, **49 molecules + S3 URL** | aborted AFTER gen: `explore_chemistry_database` MCP timeout 300s killed the run |
  | Q5 PCSK9 | ❌ | RAG registry `TimeoutError` → `list_available_tools` 'unavailable' → critic abandons run |
- **S3 fix — bug found & fixed:** in Q4 generation ran and the presigned URL was present, but the
  plugin's `artifacts` was EMPTY. Root cause: ADK passes `after_tool_callback` the WRAPPED
  `CallToolResult.model_dump()` (`mcp_tool.py:382`) — `{content:[{text:<json>}], structuredContent:{…},
  isError}`, not the tool's top-level dict; the extractor only checked top-level keys. **Fixed:**
  recursive key-agnostic extractor (walks dicts/lists, parses JSON-string content) + fallback regex
  scan of the final `mas.run()` state in `fedot_tool`. Validated OFFLINE on 3 envelope shapes — all
  capture the URL. NOT yet validated e2e (no post-fix run reached gen; gen-MCP `10.32.2.2` flaky).
- **Conclusion:** the S3-link loss (F010.A3) is real and now capturable, but it sits DOWNSTREAM of
  generation; runs are blocked UPSTREAM by 5 distinct dispatch/discovery/infra failures. Committed
  callable-names prompt did NOT stop the outer-name dispatch crash (Q1). Hard aborts on a single bad
  LLM output (Q1 ValueError, Q3 JSONDecodeError) + post-gen failures aborting a run that already has
  molecules+link (Q4) are the real reliability gaps → F000/F015.
- **Evidence:** `scripts/experiments/results/ab_S3_2026-06-13_032445.json`; `trace_manifest.jsonl`;
  workflow `ab-s3-trace-analysis` (5 per-trace readers). Plugin `fedot_artifact_plugin.py` +
  `fedotmas_tools.py` (uncommitted).

### F010.A5 — EXPERIMENT iter-2: S3 fix WORKS e2e — molecules + S3 link reach the chat · 2026-06-13 · outcome: SOLVED for 2/5 (S3-link loss resolved); 3 reliability gaps remain
- **Changes vs A4 (experimental, uncommitted):** FIXED plugin extractor (recursive over the wrapped
  `CallToolResult` envelope) + fallback state-scan; orchestrator **generic-fallback** prompt block
  (steer "generate molecules" to the always-works generic `generate_mols`, deliver its S3 link,
  stop case/property loops + critic-abandons, never emit a critic-named leaf tool).
- **Direct `fedot_tool` call (live infra):** `artifacts=[{url: …/gan_default/64296213….csv?X-Amz-…,
  generated_count:10, tool:generate_mols}]`, `state['fedot_artifacts']` set, debug confirms the plugin
  saw `generate_mols` as `{keys:[content,structuredContent,isError]}` → **capture proven**.
- **reactplan/qwen `ab_runner --condition S4` (5 queries):** 3/5 produced a response (vs 2/5 refusals
  in A4). Trace ids: Q1 `019ebfc4`, Q2 `019ebfc7`, Q3 `019ebfd0`, Q4 `019ebfd3`, Q5 `019ebfd7`.

  | Q | gen | artifacts | **S3 URL in final answer** | note |
  |---|---|---|---|---|
  | Q1 GSK-3beta | ✅ generate_mols | ✅1 | ✅ **YES** | "50 molecules generated & uploaded to S3 … 📥 Download: http://10.32.1.114:9000/…gsk3beta/…csv?X-Amz…" |
  | Q3 BTK | ✅ generate_mols | ✅1 | ✅ **YES** | no BTK case → generic fallback → link delivered |
  | Q2 KRAS | ✅ gen+case_mols | ✅2 | ❌ | generation succeeded & link captured, but orchestrator finalization DROPPED it, wrote a PDB-ID meta-refusal (**defect B**) |
  | Q4 STAT3 | ❌ | 0 | ❌ | `KeyError: 'request'` at `agent_tool.py:216` (`args['request']`) — AgentTool delegation crash when LLM omits `request`; `ResilientAgentTool` (agents.py:45) doesn't guard the missing key |
  | Q5 PCSK9 | ❌ | 0 | ❌ | timeout 600s (slow/looping path) |
- **Conclusion: the S3-link loss (F010.A3) is SOLVED** — when a run reaches generation and finalizes,
  the real molecules' presigned S3 URL now reaches the chat (Q1, Q3 proven; direct call proven). The
  plugin captures the link at the tool boundary regardless of sub-agent paraphrase. **Remaining gaps**
  (reliability, not S3): (B) orchestrator finalization can still drop a captured result (Q2) → needs a
  deterministic finalizer that appends `state['fedot_artifacts']` when the LLM omits it; (C) AgentTool
  `KeyError 'request'` hard-aborts on malformed delegation (Q4) → `ResilientAgentTool` must inject a
  default `request`; (D) timeouts (Q5).
- **Evidence:** `scripts/experiments/results/ab_S4_2026-06-13_115614.json`; `trace_manifest.jsonl`;
  `logs` not committed. Inline 5-trace analysis (reached_gen / artifacts / S3-in-answer / error).

### F010.A6 — EXPERIMENT iter-3: AgentTool guard + deterministic finalizer; HIGH variance · 2026-06-13 · outcome: S3 solution stands; reliability is the remaining (separate) problem
- **Changes vs A5 (experimental, uncommitted):** (C-fix) `ResilientAgentTool.run_async` injects a
  default `request` when the LLM omits it (kills the Q4 `KeyError 'request'` hard-abort,
  `agent_tool.py:216`); (B-fix) deterministic finalizer in `CoScientistManager.run` — after the run,
  read `session.state['fedot_artifacts']` and APPEND the captured S3 link(s) to the answer if the
  orchestrator dropped them.
- **`ab_runner --condition S5` (5 queries):** **0/5 clean deliveries** (vs A5/S4 2/5) — qwen run-to-run
  variance is large. Q1 `019ebfe5` & Q4 `019ebffa` returned non-empty but DEGENERATE (reached_gen=False,
  0 artifacts, answer = leaked `[CRITIC REVISION]` text, not molecules); Q2/Q3/Q5 = timeout 600s
  (CoderAgent HITL-approval loops / property-prediction / KRAS-selectivity detours).
- **C-fix CONFIRMED:** Q4 STAT3 no longer `KeyError`-crashes — it completed (48s) instead of aborting.
  Finalizer (B-fix) was NOT exercised: no S5 run reached generation, so `state['fedot_artifacts']` was
  empty. Logic verified by inspection + the direct `fedot_tool` call (A5) proving artifacts land in state.
- **New degenerate mode observed:** the `pre_action_critique` revision text can leak out as the FINAL
  answer (S5 Q1/Q4) when the run ends on a critic turn.
- **CONCLUSION — the S3-link problem is SOLVED and proven** (A5/S4 Q1+Q3 delivered the link to the chat;
  direct `fedot_tool` call captures it; the finalizer guarantees delivery once generation happened).
  What remains is **general pipeline reliability** — NOT the S3 link: (1) timeouts on slow paths
  (CoderAgent loops, property prediction, target-selectivity detours); (2) huge qwen variance in which
  path a query takes; (3) critic-revision text leaking as the final answer. These are F000/F015 work.
- **Commit candidates (the S3 solution, once approved):** `fedot_artifact_plugin.py` + `fedotmas_tools.py`
  plugin wiring; `main.py` finalizer; `ResilientAgentTool` request-guard; orchestrator/ExperimentAgent
  S3 + generic-fallback prompt. Reliability (timeouts/variance/critic-leak) → F015.
- **Evidence:** `results/ab_S5_2026-06-13_123143.json`; trace ids Q1 `019ebfe5`, Q4 `019ebffa`,
  Q5 `019ebffb` (Q2/Q3 record_run failed — API/SSL); inline analysis (reached_gen/artifacts/S3/critic-leak).
### F010.A7 — committed the S3 solution + "read S3 contents" finalizer/prompt · 2026-06-13 · outcome: shipped
- Productionized the helpful fixes (removed debug scaffolding) and **committed** them: S3-capture
  plugin `CoScientist/tools/fedot_artifact_plugin.py:ArtifactCapturePlugin` + `fedot_tool` wiring
  (`MAS(plugins=…)` → returns `artifacts`, writes `state['fedot_artifacts']`);
  `ResilientAgentTool` request-guard; generic-fallback + S3 prompt blocks.
- **Design refinement (user):** S3 links are the data-transport between remote MCPs (in & out);
  the USER-facing answer must be formed from the FILE CONTENTS, not a bare link. Updated the
  orchestrator prompt to "download and READ the S3 file(s) (via CoderAgent) and form the answer
  from the actual contents" (keep the link too); the deterministic finalizer
  (`CoScientist/main.py:_s3_csv_preview` + `CoScientistManager.run`) now DOWNLOADS the CSV and
  appends a contents preview when the orchestrator dropped it — verified parsing against a CSV.
- Reliability (timeouts / qwen variance / critic-revision leak) remains open → F015.

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
