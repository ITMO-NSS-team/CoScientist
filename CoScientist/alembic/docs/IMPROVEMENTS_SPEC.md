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

**Update (2026-07-06), after the 12-repo bench run** (see `benchmarks/alembic/base_results.md`, logs in `alembic_bench_logs/`): five more bugs were caught red-handed rather than inferred from comparison, and they are cheaper and higher-confidence than F1–F13 because we have the exact traceback. **F14–F18 below should run before F1–F13** — several of them are the actual reason today's pass rate looks worse than the pipeline's real capability.

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
**Problem.** `auto-sklearn` (1 of 17 pytest cases failed on an unrelated kwarg mismatch) and `biotite` (a one-character `SyntaxError` in the generated test file) both ended with **all 4 declared tools marked SKIPPED** — `invoke_mcp_tool` was never called for any of them. The Validator instruction's Step 3 wording ("record the error and proceed to Step 5") is evidently being read as "skip Step 4 entirely" rather than "skip only the affected tool." This directly produced 2 of the 5 FAILED rows in `base_results.md` and understates the true pass rate — several of those 4×2=8 SKIPped tools likely work fine.
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

**2026-07-06 rerun finding — first attempt at F18 (Explorer-only) did not work.** Re-running `BioSPPy` after patching only `instructions/explorer.py` still produced `ecg_processing(signal=[0.1, 0.2, 0.3])` and other 3-5-element arrays, failing with the exact same "signal too short" class of error. Root cause: the concrete `samples:` block that `invoke_mcp_tool` actually runs is written by the **Coder**, not the Explorer, and `instructions/coder.py` had its own, directly conflicting instruction: *"Use the most minimal args you can."* The Coder was never reading the Explorer's sizing guidance for this — it was independently told to minimize. Fixed by rewording `coder.py`'s sample-writing rules to distinguish "cheap to run" (small batch size, CPU device) from "smaller than the function's own precondition," and to check the wrapped function's docstring/call-sites for a minimum size before synthesizing a value. Not yet re-benchmarked after this second patch — do that before relying on F18 being closed.

### F19 — `_UnknownToolStub` missing `.description` attribute
**Problem.** Found in the 2026-07-06 rerun (`backtrader.log`): when the LLM hallucinates a tool name, `main.py`'s `_safe_get_tool` patch returns an `_UnknownToolStub` with only `.name` and `.run_async`; something in ADK's flow also reads `.description` on a tool object, producing `AttributeError: '_UnknownToolStub' object has no attribute 'description'`. This burned a debugger attempt (masked by F17's retry-once, which happened to succeed on the next try — without F17 this would have been a silent wasted attempt like the original BioSPPy case).
**Spec/fix.** Add a `description` attribute to `_UnknownToolStub.__init__`. Trivial, applied directly (`main.py`).
**Effort.** Trivial (already fixed).

---

## Out of scope / explicitly not recommended
- **gitingest / DeepWiki external analysis (ToolRosella).** Fragile (rate limits, "Loading…" states, `verify=False`). Alembic's direct file-reading explorer is more robust; treat external knowledge as optional augmentation only.
- **Single-function / task-spec model (ToolMaker).** Do **not** drop autonomous multi-tool discovery — it is alembic's core advantage; borrow ToolMaker's *validation/isolation* mechanisms, not its task-scoped generation model.
