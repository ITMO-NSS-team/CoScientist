# Alembic — Improvement Spec

Consolidated, deduplicated spec of features alembic could adopt, drawn from the ToolRosella and ToolMaker comparisons ([TOOLROSELLA_COMPARISON.md](./TOOLROSELLA_COMPARISON.md), [TOOLMAKER_COMPARISON.md](./TOOLMAKER_COMPARISON.md)) and alembic's current design ([DESIGN.md](./DESIGN.md)). Goal: maximize **stability/robustness** and produce the metrics a paper needs, **without giving up** alembic's edge (autonomous multi-tool, served MCP, two-venv, real in-loop invocation).

## Priority summary

| # | Feature | Robustness ↑ | Effort | Source |
|---|---|---|---|---|
| F1 | Static import/symbol AST gate | High | Low | ToolRosella |
| F2 | Semantic output-correctness gate | High | Low–Med | ToolMaker |
| F3 | Held-out validation invocation | High | Med | ToolMaker |
| F4 | Bounded, AST-verified tool selection | Med–High | Med | ToolRosella |
| F5 | First-class conda / `environment.yml` | Med | Med | ToolRosella |
| F6 | External resource acquisition (weights/datasets) | High | Med–High | ToolMaker |
| F7 | Declared, allowlisted repo-secret injection | Med | Low–Med | ToolMaker |
| F8 | Reproducible image from recorded `install.sh` | High | Med | ToolMaker |
| F9 | Fresh-checkpoint isolation per attempt | High | Med | ToolMaker |
| F10 | Persistent failure-memory across debug | Med | Low | ToolMaker |
| F11 | Tiered model routing | Med | Low | ToolMaker |
| F12 | Structured run metrics + error taxonomy | High (paper) | Low | both |
| F13 | Success-memoized, resumable runner | Med | Low | ToolRosella |

Suggested first wave (cheap, high-leverage): **F1, F2, F12, F10**. Second wave (the robustness core): **F9, F8, F6, F4, F3**.

**Update (2026-07-06), after the 12-repo bench run** (see `benchmarks/alembic/runs/2026-06-30_baseline/summary.md`, logs in `benchmarks/alembic/runs/2026-06-30_baseline/logs/`): five more bugs were caught red-handed rather than inferred from comparison, and they are cheaper and higher-confidence than F1–F13 because we have the exact traceback. **F14–F18 below should run before F1–F13** — several of them are the actual reason today's pass rate looks worse than the pipeline's real capability.

| # | Feature | Robustness ↑ | Effort | Source |
|---|---|---|---|---|
| F14 | Validator must not blanket-skip all tools on one test failure | High | Low | bench run (auto-sklearn, biotite) |
| F15 | Debugger request must always carry `repo_url` | Med | Low | bench run (biotite) |
| F16 | Per-debugger-call timeout, distinct from stage timeout | High | Low | bench run (biopython) |
| F17 | Retry once on transient LLM/API error before counting a real failure | Med | Low | bench run (BioSPPy) |
| F18 | Explorer must propose realistic, domain-sized sample inputs | High | Low | bench run (BioSPPy) |

---

## Validation & correctness

### F1 — Static import/symbol AST gate
**Problem.** The coder can hallucinate function/class names; alembic only catches this at `validate_syntax` (module import) or later via the debugger — expensive.
**Spec.** New deterministic tool `check_symbols(repo_url)` run between coder and validator: AST-scan the cloned repo for public exported symbols per module; AST-parse `server.py` + `helpers/*.py`; flag every `from <mod> import <sym>` (and wrapped call) where `<mod>` is a repo module but `<sym>` isn't exported. Return structured issues; on issues, route back to coder (or debugger Class C) **before** any venv execution.
**Integration.** Add as a tool + a thin gate stage in `main.py` after Coder; reuse `tools/scripts/compat_check.py`-style AST walking.
**Effort.** Low.

### F2 — Semantic output-correctness gate
**Problem.** `invoke_mcp_tool` only reports `{ok: True/False}` (no exception). A tool can run cleanly and return garbage — common for scientific tools.
**Spec.** After a successful `invoke_mcp_tool`, add an LLM judge that checks the **returned value** against the explorer's documented "Returns"/example: is it plausible, and does it have the expected keys/types/shapes? Emit `{successful, reasoning}`. A `False` verdict triggers the debugger just like an exception.
**Integration.** New step in the Validator instruction + a small `assess_output` helper; feed it the sample's expected-output description from the coder's `samples:` block.
**Effort.** Low–Medium.

### F3 — Held-out validation invocation
**Problem.** Alembic validates with the same samples the coder declared, so a tool that hard-codes the demo path/file passes.
**Spec.** Require the explorer/coder to emit, per tool, **two** invocations with *different* real inputs (e.g. a second sample file, different params). Validator must pass **both**; a tool that only passes its own demo args is marked FAILED (overfit).
**Integration.** Extend the coder `samples:` schema to `examples: [primary, holdout]`; Validator Step 4 loops over both.
**Effort.** Medium (depends on repos shipping ≥2 inputs; fall back to param variation).

---

## Tool selection

### F4 — Bounded, confidence-ranked, AST-verified tool selection
**Problem.** The explorer proposes "1–5 scenarios" via prompt only — uncapped, unverified, signatures guessed.
**Spec.** Deterministic post-processing of explorer output: (a) hard cap on total tools (e.g. ≤12); (b) per-tool confidence tiers with per-tier caps (high→5/med→3/low→1); (c) **intersect** proposed function/class names with the repo's actual AST symbols, dropping `test`/`example`/private; (d) capture each kept symbol's **real parameter names** and pass them into the coder so wrappers use correct argv (pre-empts wrong-kwarg `TypeError`s the debugger currently fixes reactively).
**Integration.** Run between Explorer and Coder; reuse F1's AST symbol scan; thread `function_signatures` into the coder prompt.
**Effort.** Medium.

---

## Environment & resources

### F5 — First-class conda / `environment.yml` path
**Problem.** Alembic reaches conda only as "Attempt 3"; scientific repos (rdkit, openbabel, pinned CUDA) often need it.
**Spec.** Promote conda + `environment.yml` to a primary environment strategy alongside uv: detect `environment.yml`, honor channels, use `--solver=libmamba`, install conda deps then the pip section. Keep two-venv semantics (conda env = repo venv; server venv stays Py≥3.10 with fastmcp).
**Integration.** Extend `tools/venv.py` (`setup_venv`) + the Environment instruction's decision tree.
**Effort.** Medium.

### F6 — External resource acquisition (weights download, dataset mounts)
**Problem.** Neither alembic nor ToolRosella fetches pretrained weights/models or README-only-mentioned downloads — a dominant failure cause for ML repos.
**Spec.** Add a resource step in the Environment/build phase: (a) the agent may **download model weights** (README-driven, `huggingface-cli`/`wget`/`gdown`, gated by `HF_TOKEN` via F7), recording fetches so they're baked into the image; (b) **datasets/large inputs are NOT baked** — declare them as mount points / sample paths supplied at invocation time. Explorer notes required artifacts in `exploration.md`; Environment acts on "download weights" but defers "needs dataset" to a mount.
**Integration.** New Environment instruction section + allow `bash_env` to run download CLIs; document a `mounts:`/inputs convention for the served container.
**Effort.** Medium–High.

### F7 — Declared, allowlisted repo-secret injection
**Problem.** Alembic injects only its own LLM keys (and scrubs them); a repo-required runtime secret (HF token, inference API key) has no channel, so token-gated tools can't be built or served.
**Spec.** Support a per-repo `env:` declaration (config/task-level) of required secret names; resolve from host env via an **allowlist** substitution (e.g. `${env:HF_TOKEN}`); inject into build + serve containers; keep them out of the committed image (bake-then-remove, combined with alembic's existing secret scrub). Surface available var names to the agents in prompts.
**Integration.** Extend `start_chain.py` env passthrough + add a substitution util; mention available vars in Environment/Coder instructions.
**Effort.** Low–Medium.

---

## Reproducibility & isolation

### F8 — Reproducible image from a recorded `install.sh`
**Problem.** Alembic `docker commit`s the mutated build container — opaque, large, hard to audit/reproduce.
**Spec.** Record the Environment agent's **successful** setup commands (venv creation, installs, apt-get, downloads) into an ordered `install.sh` / generated Dockerfile; build the served image with a clean `docker build` from that script instead of committing the dirty container. Keeps the image minimal, reviewable, and deterministically rebuildable.
**Integration.** Log env-agent commands (already partly in `pipeline.log`) into a structured script; swap the commit step in `start_chain.py` for a build step.
**Effort.** Medium.

### F9 — Fresh-checkpoint isolation per validate/debug attempt
**Problem.** Validator/debugger mutate one shared workdir; accumulated side-effects can mask or fabricate failures across iterations.
**Spec.** Freeze the post-install state as a checkpoint (image from F8, or a snapshot of `output/`+venv); before **each** validation/debug attempt, reset the runtime to that checkpoint so every attempt runs against the same clean installed environment. Debug exploration is allowed but discarded; only the code edit persists.
**Integration.** Wrap the Validator loop with a `reset_runtime()` step; pairs naturally with F8.
**Effort.** Medium.

---

## Agent-loop quality

### F10 — Persistent failure-memory across debug iterations
**Problem.** The debugger is largely stateless per call; alembic only guards with "stop on identical error," allowing oscillating fixes.
**Spec.** Maintain a running list of compact `problem_summaries` (diagnosis + attempted fix) and feed it into each subsequent debug/diagnose prompt ("avoid repeating these"). Use it to detect near-duplicate diagnoses, not just byte-identical errors.
**Integration.** Thread a summaries list through the Validator↔Debugger calls in `main.py`/instructions.
**Effort.** Low.

### F11 — Tiered model routing
**Problem.** One `MODEL` drives every agent; planning/first-implementation benefit from stronger reasoning, the repair loop does not.
**Spec.** Allow per-role model selection: a reasoning model for Explorer planning + initial Coder implementation, a cheaper/faster model for Validator/Debugger iterations. Configurable via env (`MODEL_REASONING`, `MODEL`).
**Integration.** `agents.py` already centralizes `MODEL`; add a second constant and assign per agent.
**Effort.** Low.

---

## Observability & benchmarking

### F12 — Structured run metrics + error taxonomy
**Problem.** Alembic logs human-readable events but emits no structured per-run metrics or failure classification — exactly the data a paper needs.
**Spec.** Per run, emit a JSON record: per-stage durations, step/tool-call counts, LLM call/token/retry stats, guard/timeout firings, and **every failure classified into a taxonomy** (`ModuleNotFound`/`Import`/`Environment`/`Build`/`Syntax`/`Runtime`/`AttributeError`/`FileNotFound`/…). Aggregate across a benchmark into pass-rate-by-stage and error-distribution tables.
**Integration.** Add a metrics sink alongside the loguru file sink in `main.py`; classify in the Validator/Debugger; aggregate in `run_benchmark.py`.
**Effort.** Low.

### F13 — Success-memoized, resumable runner
**Problem.** Alembic wipes the workdir each run; re-benchmarking reprocesses already-solved repos.
**Spec.** Persist a `processed_repos.json` keyed by repo URL + commit; skip repos already converted successfully; combine with the existing `--resume <stage>` for partial reruns. Record per-repo status/artifact path/message.
**Integration.** Wrap `run_benchmark.py` (and optionally `start_chain.py`) with a memo check; reuse the existing resume machinery.
**Effort.** Low.

---

## Bugs caught in the 2026-06-30 bench run (F14–F18)

### F14 — Validator must not blanket-skip all tools on one test failure
**Problem.** `auto-sklearn` (1 of 17 pytest cases failed on an unrelated kwarg mismatch) and `biotite` (a one-character `SyntaxError` in the generated test file) both ended with **all 4 declared tools marked SKIPPED** — `invoke_mcp_tool` was never called for any of them. The Validator instruction's Step 3 wording ("record the error and proceed to Step 5") is evidently being read as "skip Step 4 entirely" rather than "skip only the affected tool." This directly produced 2 of the 5 FAILED rows in the baseline run's summary and understates the true pass rate — several of those 4×2=8 SKIPped tools likely work fine.
**Spec.** Make per-tool invocation independent of overall `run_tests` pass/fail: only the specific tool(s) implicated by a failing test should be withheld from `invoke_mcp_tool`; every other declared sample must still be invoked and scored PASS/FAIL on its own merits.
**Effort.** Low — tighten the Validator instruction's Step 3→Step 4 transition logic.

### F15 — Debugger request must always carry `repo_url`
**Problem.** In `biotite`, the Validator's one debugger call omitted the repo URL prefix that every other observed call includes; the Debugger replied "cannot locate test_server.py... verify repository URL" and gave up on try 1 of 3 — turning a trivial missing-brace fix into a total stage failure (all 4 tools SKIPPED). The `debugger` `AgentTool` takes a free-text `request: str` with nothing enforcing that `repo_url` is present.
**Spec.** Either make `repo_url` a required structured field on the debugger call (not embedded in free text), or add a pre-flight check that rejects/re-prompts a debugger request missing it.
**Effort.** Low.

### F16 — Per-debugger-call timeout, distinct from the stage timeout
**Problem.** `biopython`'s Validator stage hit its full 30-minute wall-clock timeout while a single debugger call was stuck chasing an `ImportError` (helper scripts `sys.path.insert`-ing into the raw cloned source instead of an installed package, so compiled C-extensions were missing) — the whole budget was consumed with **zero** `validation.md` written, losing all signal for that repo.
**Spec.** Bound each individual debugger round-trip (e.g. 3–5 min) independent of the stage-level timeout, so a stuck call fails fast and the stage can still write a partial report instead of losing everything.
**Effort.** Low — wrap the debugger `AgentTool` invocation with its own timeout in `main.py`.

### F17 — Retry once on transient LLM/API error
**Problem.** In `BioSPPy`, the first debugger call for `process_ecg_signal` never returned: OpenRouter/LiteLLM returned an empty body mid-call → `JSONDecodeError`/`OpenAIError`, caught by `main.py`'s blanket `except Exception: logger.exception(...)`, which silently burned one of the tool's ≤2 debugger attempts with zero diagnostic value.
**Spec.** Classify transient provider errors (empty response, 5xx, rate limit) separately from real failures and retry once (with backoff) before charging it against the attempt budget.
**Effort.** Low.

### F18 — Explorer must propose realistic, domain-sized sample inputs
**Problem.** `BioSPPy` failed 4 of 5 tools because the Explorer's example inputs were toy arrays (a 5-element list for an ECG filter requiring `padlen=4503`, a 5 ms segment where `assess_ecg_quality` requires ≥5 s) — the tools are correct, the *samples* violate the domain's basic preconditions. This is a distinct, cheaper problem than F3's held-out-invocation generalization concern: today's single sample isn't even large enough to exercise the tool once.
**Spec.** Explorer instruction should require sample data sized to the domain (signal duration/length, image dimensions, sequence length) inferred from the repo's own docs/tests/example data, not an arbitrary placeholder shape.
**Effort.** Low — instruction change plus, where the repo ships its own example/test fixtures, prefer sampling from those over synthesizing new toy data.

**2026-07-06 rerun finding — first attempt at F18 (Explorer-only) did not work.** Re-running `BioSPPy` after patching only `instructions/explorer.py` still produced `ecg_processing(signal=[0.1, 0.2, 0.3])` and other 3-5-element arrays, failing with the exact same "signal too short" class of error. Root cause: the concrete `samples:` block that `invoke_mcp_tool` actually runs is written by the **Coder**, not the Explorer, and `instructions/coder.py` had its own, directly conflicting instruction: *"Use the most minimal args you can."* The Coder was never reading the Explorer's sizing guidance for this — it was independently told to minimize. Fixed by rewording `coder.py`'s sample-writing rules to distinguish "cheap to run" (small batch size, CPU device) from "smaller than the function's own precondition," and to check the wrapped function's docstring/call-sites for a minimum size before synthesizing a value.

**2026-07-06 second rerun (rerun2, 11 repos, `coder.py` patch in place) — confirmed fixed at scale.** Cross-checked every `invoke_mcp_tool` call across all 11 executed logs: no repo passed a tiny/all-placeholder numeric array to a precondition-sensitive function anymore. `BioSPPy` now passes `signal: 'examples/ecg.txt'` (a real bundled ECG recording); `astropy` passes real repo-bundled `.fits` test fixtures; `auto-sklearn` copies the repo's own real `load_breast_cancer`/`load_diabetes` example datasets; `ase`/`astronomy`/`aizynthfinder`/`backtrader` all use real physical/chemical/financial parameters. F18 is closed.

One adjacent miss surfaced by the same cross-check (not F18 itself — sizing was never the issue here): **`biotite`** invented plausible-but-nonexistent filenames (`example.fasta`, `example.pdb`, `fixed.pdb`/`mobile.pdb`) instead of the real files it had already seen via `read_file`/`ls` (`doc/examples/download/lysozyme_md.pdb`, `dppc_n128.pdb`, `waterbox_md.pdb`), failing 4/5 tools on "file cannot be read." `coder.py` already had a "Do NOT invent paths" rule, so this is an instruction-following miss, not a missing instruction — strengthened the wording with a concrete counter-example and an explicit "verify the exact path via a directory listing or read_file result first" requirement. Not yet re-benchmarked after this wording change — a stronger, deterministic fix (reject file-path-shaped args that don't resolve inside the container before invocation) is F1/F4's territory, tracked there rather than duplicated here.

### F19 — `_UnknownToolStub` missing attributes
**Problem.** Found in the 2026-07-06 rerun (`backtrader.log`): when the LLM hallucinates a tool name, `main.py`'s `_safe_get_tool` patch returns an `_UnknownToolStub` with only `.name` and `.run_async`; something in ADK's flow also reads `.description` on a tool object, producing `AttributeError: '_UnknownToolStub' object has no attribute 'description'`. This burned a debugger attempt (masked by F17's retry-once, which happened to succeed on the next try — without F17 this would have been a silent wasted attempt like the original BioSPPy case).
**First fix (v1, incomplete).** Added a `description` attribute to `_UnknownToolStub.__init__`.
**2026-07-06 rerun2 finding — v1 was incomplete.** Re-running with the `.description` fix in place, the exact same class of error recurred in `AgML.log` and `BioSPPy.log`, this time for a *different* attribute: `AttributeError: '_UnknownToolStub' object has no attribute 'is_long_running'`. In `AgML` the retry hit the identical error and the debugger call was abandoned as an unresolved failure (graceful, but a genuinely lost attempt). In `BioSPPy` the retry never returned at all — it hung until the Validator's 1800s stage timeout fired, truncating validation after only 1 of 5 tools had been checked, and (see F20) the pipeline log then misleadingly claimed the run had completed. Root cause: `_UnknownToolStub` is duck-typed against ADK's real `BaseTool`, whose full public attribute surface is `name`, `description`, `is_long_running`, `custom_metadata` — the v1 fix patched only the one attribute that had been observed failing, not the class the stub is standing in for.
**Fix (v2).** `_UnknownToolStub.__init__` now sets `is_long_running = False` and `custom_metadata = None` alongside `name`/`description`, matching `BaseTool`'s complete surface so no further attribute is missing regardless of which ADK code path introspects it (`main.py`).
**2026-07-06 targeted rerun (rerun3) confirmed fixed.** Re-ran all three previously-affected/adjacent repos (`AgML`, `BioSPPy`, `biotite`) with v2 in place: zero `_UnknownToolStub`/`AttributeError` occurrences across all three logs. F19 closed.
**Effort.** Trivial (fixed twice; v2 is the complete fix, confirmed).

### F20 — Stage-timeout logging falsely claims the stage completed
**Problem.** Found via the `BioSPPy` rerun2 case above: when `_run_stage()` hits the per-stage `asyncio.wait_for` timeout (F16's mechanism), it logs `STAGE TIMEOUT ... aborting stage, pipeline continues` and returns `""` — but every call site in `main.py`'s `run_pipeline()` unconditionally logged `"[<Stage> done] report → .../reports/<stage>.md"` afterward regardless of that return value, and for the Validator stage additionally logged a `logger.success("Pipeline complete: ...")` banner. For `BioSPPy` this meant the pipeline log claimed success and a written `validation.md` when the file never existed (confirmed via `docker run --rm --entrypoint cat alembic-tool:BioSPPy .../validation.md` → "No such file or directory") — directly causing the benchmark harness's "validation.md not readable" row, which without this explanation reads as an unrelated harness bug rather than an honest reflection of an aborted stage.
**Spec/fix.** Capture each stage's return value; a normal completion always returns at least the literal fallback string `"Agent did not produce a final response."` (never `""`), so `bool(final)` reliably distinguishes "timed out" from "finished" without new state. Gated all four stages' "done" log lines on this, and replaced the Validator's unconditional success banner with an explicit "Pipeline incomplete — Validator stage timed out, no report was written" error banner on the timeout path (`main.py`). Also generalized `run_benchmark.py`'s `write_summary()` so the Overall column shows `ERROR — <reason>` whenever validation extraction failed for *any* reason (not just the repo-unreachable/`not_run` case it already handled) — previously this rendered as a bare `—`, the same class of "reason isn't visible" bug already fixed once for unreachable repos.
**2026-07-06 targeted rerun (rerun3) confirmed fixed.** `AgML`'s Validator stage genuinely timed out again in this rerun (1800s, a real slow/hard repo — same as rerun2), and this time the log honestly printed `"Pipeline incomplete: AgML"` instead of a false `"Pipeline complete"`; the benchmark summary row correctly reads `ERROR — validation.md not readable` instead of a bare `—`. `BioSPPy`'s Coder stage also timed out (1500s, unrelated LLM-latency cause — see F22) and produced the equally honest `"[coder] STAGE TIMEOUT..."` + `"required artefacts missing... skipping validator stage"` pair, again with no false success claim. F20 closed.
**Effort.** Trivial (fixed, confirmed).

### F21 — Coder invents plausible-but-nonexistent sample file paths
**Problem.** See the F18 rerun2 note above — `biotite` passed `example.fasta`/`example.pdb`/`fixed.pdb`/`mobile.pdb` to `invoke_mcp_tool`, none of which exist in the repo, despite `coder.py` already instructing "Do NOT invent paths" and despite the Coder having already seen the real bundled files via `read_file`/`ls` earlier in the same run.
**Spec/fix.** Strengthened the existing rule with a concrete counter-example matching what was actually observed, and an explicit requirement to verify the exact path via a directory listing or `read_file` result before using it (`instructions/coder.py`). This is an instruction-following miss, not a missing rule, so a wording fix is a partial mitigation at best — a deterministic gate (reject file-path-shaped args that don't resolve inside the container before invocation) would close this properly and is F1/F4's territory.
**Effort.** Low (wording fix applied).
**2026-07-06 targeted rerun (rerun3) confirmed fixed.** Re-ran `biotite` alone after the wording change: all `invoke_mcp_tool` calls now use real repo test-fixture paths (`tests/structure/data/pdb/4gxy.pdb`, `1o1z.pdb`, etc.). Its remaining 2/4 tool failures are genuine (`zip() argument 2 is longer than argument 1` on a real structural mismatch; `Cannot import name 'read' from 'biotite.structure'` because the repo requires Python ≥3.12 but the container venv is 3.11.15 — an unrelated environment-setup gap, not a path-invention issue). F21 closed for the observed case; the deterministic gate (F1/F4) remains the durable fix for the general class.

### F22 — Silent provider fault: unmapped `finish_reason: "error"` burns ~4 min per occurrence, uncaught by F17
**Problem.** Found in the rerun3 targeted-verification run (`BioSPPy.log`, 2026-07-06): during the Coder stage, two separate LLM turns ~4 minutes apart each ended with `LiteLLM:WARNING core_helpers.py:107 - Unmapped finish_reason 'error', defaulting to 'stop'`, followed immediately by a stub/non-actionable "final" response (e.g. *"Now I'll implement the MCP server file..."* with no tool call), which then only recovered via `main.py`'s unrelated `write_report`-guard nudge (`MAX_GUARD_RETRIES=3`, i.e. `[guard] Retry 1/3`, `Retry 2/3`). The stage's `server.py` write finally succeeded on the 3rd attempt, one second before the 1500s Coder `STAGE_TIMEOUT` fired — i.e. the fault very nearly cost the whole stage outright, and did cost ~8 of the 25 available minutes plus 2 of 3 guard-retry slots.
**Root cause (confirmed by reading source inside the built image).** `litellm/litellm_core_utils/core_helpers.py`'s `map_finish_reason()` holds an explicit table mapping every provider's finish-reason string to the OpenAI-standard set (`stop`/`length`/`tool_calls`/`content_filter`/`function_call`) — it even has a deliberate `"ERROR": "stop"` entry for a different provider's uppercase variant. The lowercase `"error"` seen here is *not* in that table, so it falls through to the `else` branch: log a warning and silently default to `"stop"` — **no exception is raised**. Whatever partial/garbage `message.content` came back with it is then treated by ADK as a normal completed turn. The fault itself originates upstream — OpenRouter (or whichever backend it proxied to for that call) surfaced an internal failure as `finish_reason: "error"` instead of a clean HTTP error — but LiteLLM compounds it by making the fault invisible to any caller instead of raising.
**Why F17 doesn't already cover this.** F17's retry-once lives entirely inside `_DebuggerAgentTool.run_async` (`agents.py`), wrapping only calls *to* the Debugger sub-agent — the main per-stage loop (`run_agent`/`_run_agent_once` in `main.py`) has no equivalent protection at all. More importantly, F17 catches Python exceptions (it was built for a raised `JSONDecodeError` on an empty response body); this fault **never raises**, so even copy-pasting F17's `except Exception: retry` pattern onto the main stage loop would not catch it — there is nothing to catch.
**Frequency (checked across all 4 benchmark rounds so far).** `grep -rc "Unmapped finish_reason" benchmarks/alembic/runs/*/logs/*.log` → exactly 2 occurrences, both in the same one `BioSPPy` run, 0 in the baseline/rerun/rerun2/rerun3-biotite/rerun3-AgML logs. Rare (roughly 1 affected run out of ~15 so far), but each occurrence is expensive (~4 min dead time) and, per the near-miss above, is not reliably survivable — a 3rd occurrence in the same stage would have exhausted `MAX_GUARD_RETRIES` and produced `"Agent did not produce a final response."` with no recovery.
**Spec — what a real fix needs (deferred, not yet implemented).**
1. **Detect it at the source, not downstream.** Since ADK never sees the raw `finish_reason` (LiteLLM already renamed it to `"stop"` by the time ADK's event is built), the only place capable of detecting this is a shim around the model call itself — either (a) monkeypatch `litellm.litellm_core_utils.core_helpers.map_finish_reason` (same monkeypatch style already used for `_safe_get_tool` in `main.py`) to raise a distinguishable exception (e.g. `TransientProviderError`) instead of silently defaulting when the *raw, unmapped* input is exactly `"error"` (leave every other provider's existing mapped/defaulted reasons alone — don't touch legitimately-handled cases), or (b) find/set a LiteLLM `drop_params`/callback hook that surfaces the raw finish_reason to application code without needing to monkeypatch a private helper.
2. **Generalize the retry.** Once (1) makes the fault raise-able, extend F17's retry-once semantics from "Debugger-only" to the main per-stage loop (`run_agent`/`_run_agent_once`), so a caught `TransientProviderError` triggers an immediate retry of just that turn — not a full extra guard-retry nudge cycle, and not dependent on borrowing budget from `MAX_GUARD_RETRIES` (which exists for a different purpose: nudging the agent toward a missed tool call, not recovering from provider flakiness).
3. **Regression-test before trusting it**, since this patches a choke point every one of the 4 pipeline stages runs through — a mistake here risks destabilizing all of them at once, unlike the narrow, single-call-site fixes in F14–F21. Re-run the same 3-repo targeted set (or a wider sample) afterward and confirm no new stalls/regressions in stages that previously worked cleanly.
**Effort.** Medium (shared-plumbing change, needs careful testing) — **explicitly deferred**: rare enough (1/15 runs) not to block the 2026-07-10 demo submission, but documented here with full root cause and a concrete implementation path so it doesn't need to be re-diagnosed later.

### F23 — F16's per-debugger-call timeout can be defeated by a chain of individually-bounded subprocess calls
**Problem.** Found in the rerun3 targeted-verification run (`AgML.log`, 2026-07-06), while investigating why `AgML`'s Validator stage produced *no* report at all this run (vs. rerun2, where the same repo's Validator finished cleanly with real 0/2/3 tool verdicts — see the F14 section for context on why per-tool detail matters here). Exact timeline:
- `09:34:47` — validator sends a 6th debugger call, asking it to re-investigate a `ModuleNotFoundError: No module named 'ensemble_boxes'` (a real, ordinary environment-fixing task — install a missing pip package and re-verify).
- **Nothing is logged for the next ~20 minutes** — not even `_DebuggerAgentTool`'s own `"[debugger] call timed out after 600s"` message (`agents.py`), which should have fired at `09:44:47` per `DEBUGGER_CALL_TIMEOUT=600`.
- `09:54:36` — the *outer* Validator `STAGE_TIMEOUT` (1800s, `main.py`) finally fires — itself ~2 minutes late (stage started `09:21:14`; 1800s later is `09:51:14`, not `09:54:36`).
**Root cause (inferred from the evidence, not yet confirmed by reading ADK's scheduler internals).** Every individual subprocess call in `tools/shell.py`/`tools/venv.py`/`tools/invoke.py` already has its own bound (`bash`=15s, `bash_env`=300s "for slow installs", venv setup=240s, tool invocation=900s) — so no single call can hang forever. But `asyncio.wait_for(timeout=600)` (F16's mechanism, wrapping the *entire* debugger sub-agent turn in `_DebuggerAgentTool.run_async`) can only actually cancel at a point where the wrapped coroutine yields control back to the event loop. If the debugger sub-agent's one "turn" internally chains several of those individually-bounded calls together (e.g. a few sequential `bash_env` install attempts while chasing the `ensemble-boxes` fix, each capped at 300s but cumulatively unbounded) before its own reasoning loop naturally yields, the 600s ceiling is real in theory but can't take effect until that chain finishes — which is consistent with both symptoms above: the missing 600s log line, and the outer 1800s timeout itself firing a couple minutes late (it was queued behind the same non-yielding block).
**Why this matters.** F16 was written and confirmed (2026-07-06, first rerun) specifically to stop one stuck debugger call from burning the *entire* Validator budget (the original `biopython` case). This finding shows that guarantee has a real, evidenced gap under at least one condition (a multi-step install-and-verify sub-task), even though F16's fix is otherwise working correctly (it did fire in `BioSPPy`/`auto-sklearn`/`ase` during rerun2, per that section).
**Spec — what a real fix needs (deferred, not yet implemented).**
1. Confirm the mechanism directly (this write-up is inference from timing evidence, not a source-level confirmation) — instrument or trace whether ADK's tool-execution loop genuinely withholds control from the awaiting `asyncio.wait_for` across multiple chained tool calls within one sub-agent turn.
2. If confirmed, the fix is a *cap on the debugger's own turn*, not just the outer wrapper: either (a) enforce a hard ceiling on the number of tool calls a single debugger turn may chain before being forced to yield/return, or (b) lower the per-tool-call timeouts specifically inside debugger-invoked installs, or (c) have `_DebuggerAgentTool` poll/cancel via a mechanism that doesn't rely on the wrapped coroutine's own cooperative yielding (e.g. running the debugger turn in a separate task that can be forcibly cancelled, or in a subprocess/thread that can be killed outright rather than just have its awaiting task cancelled).
3. Regression-test against `AgML` and any other repo whose Validator needed genuine multi-step package-install debugging, to confirm the fix doesn't just move the bottleneck.
**Effort.** Medium-High (requires confirming the actual ADK scheduling behavior before a fix can be designed with confidence) — **explicitly deferred**, same reasoning as F22: not blocking the 2026-07-10 submission, documented in full so it doesn't need re-diagnosing later.

---

## Out of scope / explicitly not recommended
- **gitingest / DeepWiki external analysis (ToolRosella).** Fragile (rate limits, "Loading…" states, `verify=False`). Alembic's direct file-reading explorer is more robust; treat external knowledge as optional augmentation only.
- **Single-function / task-spec model (ToolMaker).** Do **not** drop autonomous multi-tool discovery — it is alembic's core advantage; borrow ToolMaker's *validation/isolation* mechanisms, not its task-scoped generation model.
