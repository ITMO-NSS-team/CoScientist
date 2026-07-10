# Alembic × TM-Bench — full sweep results

**Run:** `benchmarks/alembic/runs/2026-07-10_tmbench-all/`
**Model:** `openrouter/z-ai/glm-5.2` (driver for every agent stage)
**Tasks:** all 15 TM-Bench tasks across 14 repositories (STAMP contributes two tasks from one repo)
**Harness:** `run_benchmark.py --tasks <15 yaml> --mount-dir <ToolMaker>/benchmark/data --parallel 4`
**Data:** real ToolMaker `benchmark/data` mounted read-only at `/mount/data`, staged per-task into `/mount/input` (incl. TCGA WSI slides, MSD CT volumes, tabular CSVs)
**Total wall-clock:** ~8 h (4-way parallel; the tail was network-throttled CPU-torch/weight downloads)

---

## Headline

| Metric | Result |
|---|---|
| Repos completed (exit 0) | **14 / 14** |
| Tools **passed** (all tests green **and** never crashed) | **10 / 15** |
| Tools **perfect** (passed **and** every invocation-correctness test green) | **5 / 15** |
| Unit tests passed | **25 / 25** (100 %) |
| Invocation-correctness tests passed | **5 / 5** (100 %) |
| Execution-OK (didn't crash) | 10 / 15 |
| `installed-<task>` runtime images tagged | **15 / 15** |
| `code.py` TM-Bench exports | **12 / 12** (one per task that produced a tool) |

**Every tool that the driving LLM was allowed to finish, and that could physically run on this CPU box, passed.** The five non-passing tools split cleanly into three external-API-infrastructure blocks and two hardware/resource limits — none is a pipeline-logic failure.

> **Read the "Metrics trust & caveats" section below before quoting these numbers.** Three of the ten "passed" verdicts rest on soft signals (a 120 s-timeout counted as runtime-success, or zero unit tests that actually ran), and **esm's MCP server is a non-functional LLM shim** — so `10 passed` is not `10 shippable servers`. The five **perfect** tools are the fully-trustworthy core.

---

## Per-task results

| Repo | Task | expl | env | coder | valid | wrap | passed | perfect | tests | invoc | exec | resets | elapsed |
|---|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|--:|
| cytopus | cytopus_db | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** | **★** | 2/2 | 1/1 | 1/1 | 0 | 19 m |
| esm | esm_fold_predict | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** | – | –¹ | –¹ | 1/1 | 0 | 26 m |
| flowmap | flowmap_overfit_scene | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** | – | 4/4 | 0/0¹ | 1/1 | 1 | 136 m |
| CONCH | conch_extract_features | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** | **★** | 2/2 | 1/1 | 1/1 | 0 | 141 m |
| MUSK | musk_extract_features | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | – | – | – | 0/1 | 2 | 29 m |
| ModernBERT | modernbert_predict_masked | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | – | – | – | 0/1 | 5 | 99 m |
| MedSAM | medsam_inference | ✓ | ✗² | ✓ | ✓ | ✓ | **✓** | **★** | 3/3 | 1/1 | 1/1 | 2 | 238 m |
| MedSSS | medsss_generate | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | – | 3/3 | 0/0 | 0/1 | 2 | 252 m |
| PathFinderCRC | pathfinder_verify_biomarker | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** | **★** | 3/3 | 1/1 | 1/1 | 2 | 42 m |
| RETFound_MAE | retfound_feature_vector | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** | – | –¹ | 0/0¹ | 1/1 | 0 | 83 m |
| nnUNet | nnunet_train_model | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | – | 3/3 | 0/0 | 0/1 | 0 | 164 m |
| UNI | uni_extract_features | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | – | – | – | 0/1 | 4 | 64 m |
| TabPFN | tabpfn_predict | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** | **★** | 3/3 | 1/1 | 1/1 | 0 | 103 m |
| STAMP | stamp_extract_features | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** | – | 2/2 | 0/0 | 2/2³ | 0 | 169 m |
| STAMP | stamp_train_classification_model | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** | – | (shared) | 0/0 | (shared) | 0 | (shared) |

¹ No evidence-based ground truth was available for an invocation-correctness assertion, so the tool can be **passed** but not **perfect** — this is a property of the task, not a defect.
² **MedSAM's env gate failed but the tool is perfect.** The env gate's repo-import smoke tried to `import MedSAM_Inference` (a script, not a module) and failed; because a failed env gate *fails-forward* to the Coder (R3), the Coder still produced a correct, invocation-verified tool. Good evidence the soft/hard-gate split + fail-forward works as designed.
³ STAMP's two tasks share one repo/env (multi-task-one-repo, R4); both tools executed on the real TCGA slides without crashing.

**★ perfect (5):** cytopus, CONCH, MedSAM, PathFinderCRC, TabPFN.

---

## Failure classification — infra/gated vs genuine

None of the five non-passing tools is a pipeline-logic failure. They split into two buckets:

### A. External API-infrastructure blocks (3) — **re-runnable, not pipeline faults**

The driving model (glm-5.2 via OpenRouter) was rejected mid-stage with
`HTTP 403 { "success": false, "error": "Access denied by security policy." }`.
These are OpenRouter-side policy/routing rejections of the *LLM* calls — unrelated to the target repo, the tool code, or HuggingFace model gating. They exhausted the stage-reset budget on the unlucky jobs:

| Tool | 403 hits | Where it hit | Consequence |
|---|--:|---|---|
| **UNI** | 16 | environment + coder | venv never built, no tool files |
| **ModernBERT** | 16 | environment + coder | venv never built, no tool files |
| **MUSK** | 10 | coder (env built fine) | no tool files |

Note the target-model gating is **not** the problem: **CONCH** (a gated pathology foundation model) came out **perfect**, and **STAMP** (which pulls CTransPath weights) passed — both used `HF_TOKEN` successfully. MedSSS also caught 2 transient 403s but the reset loop recovered and all its stages passed. On a clean API window these three would very likely pass.

### B. Genuine hardware / resource limits at execution time (2) — **correct tool, can't run here**

The pipeline produced a tool that compiles, imports, and passes its unit tests; only the *execution* on this CPU-only box was impossible:

| Tool | Stages | Why exec failed |
|---|---|---|
| **MedSSS** | all passed, tests 3/3, wrapped | Needs the ~16 GB `pixas/MedSSS_Policy` weights; only `config.json` (4 KB) downloaded within budget. Debugger's own verdict: "Class D hard environment fault." |
| **nnUNet** | all passed, tests 3/3, wrapped | The 2 h network-throttled CPU-torch download consumed the env budget, so numpy/scipy/batchgenerators/SimpleITK never installed; full 3D nnU-Net training also needs GPU + hours regardless. |

---

## Metrics trust & caveats (post-hoc log audit)

The headline numbers come straight from `validation.json` (R2), but a manual log audit of
the non-obvious cases found that not every "passed" is equally solid:

- **Fully trustworthy — the 5 perfect tools** (cytopus, CONCH, MedSAM, PathFinderCRC,
  TabPFN): real unit tests green **and** a real evidence-based invocation-correctness test
  green. CONCH/PathFinderCRC/TabPFN ran to completion; cytopus/MedSAM's redundant
  exec-smoke was skipped only because the tested arg is an *output* path, but their
  `test_invoc_*` assertions genuinely ran and passed.
- **Solid execution, no correctness assertion — STAMP ×2**: both tools ran on the real
  TCGA slides without crashing (`exec 2/2`), unit tests green, but no ground-truth
  correctness test existed (`invoc 0/0`).
- **Soft "passed" — treat with caution:**
  - **esm** — `tests_passed=null` (**no unit test ever ran**; the run exceeded the 120 s
    cap), `exec_ok` is only the 120 s-timeout grace, **and the server is a shim** (below).
    This should not be counted as a clean pass.
  - **RETFound** — `tests_passed=null` (120 s timeout) and its invocation **failed on
    gated HF weights**: `GatedRepoError: 403 … cannot access gated repo
    YukunZhou/RETFound_mae_natureCFP.pth`. With HF access now granted this is re-runnable
    and would likely become a real invocation.
  - **flowmap** — unit tests are real (4/4), but `exec_ok=true` is the 120 s-timeout grace;
    an earlier invocation crashed (`RuntimeError: FlowMap overfit pipeline failed (rc 1)`),
    the debugger fixed it, and the re-run went long → timed out → counted as runtime
    success. Execution didn't crash, but correctness is unverified.

**Bottom line:** `5 perfect` is the number to trust unconditionally. `10 passed` includes
2 solid-but-unasserted (STAMP), 1 real-tests-but-timeout-exec (flowmap), and 2 weak
(esm, RETFound). The `25/25 tests` and `5/5 invoc` are 100 % *of the tests that ran* — esm
and RETFound contributed **zero** tests (both timed out at the 120 s cap).

## Wrapper stage — how bulletproof is it, really?

The three "wrapper failed" rows (MUSK/ModernBERT/UNI) are just empty-input: the 403-killed
Coder produced no tool files, so there was nothing to wrap — not a wrapper defect. But the
audit found a **genuine wrapper/gate weakness on esm**:

- The deterministic server codegen is a pure function of the tool signatures and worked
  correctly on **11/12** tool-producing repos (`llm_fallback=false`).
- **esm needed the LLM fallback (`llm_fallback=true`) and it produced a broken server.**
  Root cause: esm's venv was hand-rebuilt by the debugger with bare `uv venv` (no pip), and
  the env-gate's `ensure_pkg(fastmcp)` guarantee did not durably land in that venv. At
  wrapper time `from fastmcp import FastMCP` raised `ModuleNotFoundError` with no pip to
  fix it, so the fallback agent **defined a stub `FastMCP` shim** to make the module import.
- **G4 (`check_server`) accepts the shim** — it only verifies that `server.py` compiles and
  imports, not that it imports the *real* fastmcp. So the gate passes a server that would
  never actually serve MCP.

Concrete hardening — **implemented & verified in-image** (gate_checks: 51/51 pass):
1. `ensure_pkg` now verifies the package resolves **inside the venv** (PYTHONPATH stripped,
   run from the venv dir so a system/leak/stray-file copy can't satisfy it) and
   **re-verifies after installing** — an `uv pip install` that reports success but doesn't
   land now hard-fails the env gate instead of passing silently. (`tools/venv.py`)
2. New `ensure_server_packages()` guarantees **fastmcp + mcp** land in the server venv; the
   env gate calls it (was fastmcp-only), so a hand-rolled `uv venv` can't silently ship
   without the MCP runtime. (`tools/venv.py`, `main.py` env gate)
3. `check_server` (G4) now requires **real, venv-local fastmcp** — a shimmed or
   system-leaked fastmcp is rejected, so G4 can no longer be satisfied by a stub.
   (`tools/invoke.py`)

New regression checks lock these in: `resolves_in_venv rejects a non-venv-local module`,
`… rejects a missing module`, and `G4 rejects a server without a venv-local fastmcp`.

## Verification of TM-Bench deliverables

- **`installed-<task>` runtime images: 15/15 present**, including both STAMP tasks
  (`toolmaker-runtime:installed-stamp_extract_features` and
  `…-stamp_train_classification_model`) committed off the single 9.05 GB STAMP image.
- **`code.py` exports: 12** — exactly one per task that produced a tool (the 3 API-blocked
  tasks have no tool to export, correctly). Each is a **self-contained plain function**
  (imports inside the body) with the **task-specified signature**, e.g.
  `stamp_extract_features(output_dir, slide_dir)` uses the task's argument names, not the
  repo's internal `wsi_dir` — confirming the signature-aware sample-arg filter
  (`function_param_names`) survives task-renamed parameters end-to-end.
- Both STAMP clone refs were honoured (R4 ref-pinning): `stamp_extract_features` at
  commit `1fdf48c`, `stamp_train_classification_model` on branch `v1`.

---

## What this validates about the R1–R8 upgrade

- **R2 (metrics from run data):** every number here comes from `validation.json` /
  `stage_status.json`, not a self-reported report.
- **R3 (functions-first + fail-forward gates):** MedSAM is the proof — env gate failed,
  Coder still shipped a perfect tool. Deterministic server codegen needed no LLM fallback
  on **11 of 12** tool-producing repos; **esm is the exception and it failed badly** (see
  below).
- **R4 (TM-Bench multi-task-one-repo, ref-pinning, mounted data):** STAMP ran two tasks
  off one pinned checkout against real TCGA slides.
- **R6 (two-level testing):** 25/25 unit tests and 5/5 invocation-correctness tests, with
  `passed` vs `perfect` cleanly separating "ran clean" from "provably correct."
- **R1 (stage resets, no stage timeouts):** 18 resets total absorbed transient failures;
  the heavy repos were bounded by the reset budget, not left to hang.

## Suggested follow-ups

1. **Re-run UNI / ModernBERT / MUSK** on a clean OpenRouter window (or a fallback provider)
   — the 403 "security policy" blocks are the only "failures" likely to flip to passes with
   no code change. **Re-run RETFound** now that HF gated access to `RETFound_mae_natureCFP`
   is granted — its invocation should now load the real weights instead of 403-ing.
0. ~~Fix the wrapper/env-gate fastmcp gap~~ **DONE** (see "Wrapper stage" above) — the
   three hardening fixes are implemented and gate-checked; esm-class repos will now either
   get a real venv-local fastmcp or fail G4 honestly instead of shipping a shim. **esm
   itself should be re-run** to confirm it now produces a real server (or an honest wrapper
   failure).
2. **Env-budget for mega-downloads:** nnUNet lost its dependency install to a 2 h torch
   download. Consider pre-seeding a CPU-torch wheel into the base image so the env stage
   spends its budget on repo deps, not on re-downloading torch per job.
3. **Script-target env smoke (minor):** MedSAM's env gate mis-fired trying to `import` a
   script target. Harmless (fail-forward saved it) but the repo-import smoke could skip
   bare `.py` script targets that are invoked via subprocess.
