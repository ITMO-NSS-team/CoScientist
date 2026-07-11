# Alembic Remaster — Design Choices

Concise architectural decisions for redesigning `CoScientist/alembic`. Goal:
a **bulletproof, model-agnostic** repo→served-MCP pipeline that beats the
current `toolmaker_subset.txt` numbers, supports a **target-tool list** (for
TM-Bench), and is **concise and easy to manage** — while staying 100%
compatible with `benchmarks/alembic/run_benchmark.py` and `docker/alembic/`.

Sources: current [DESIGN.md](../alembic/docs/DESIGN.md), the F1–F49 backlog
([IMPROVEMENTS_SPEC.md](../alembic/docs/IMPROVEMENTS_SPEC.md)), the independent
audit ([audit/](../alembic/docs/audit/)), and fresh reads of ToolMaker
(arXiv:2502.11705, 80% on TM-Bench) and ToolRosella/code2mcp
(arXiv:2603.09290, 61.5%).

---

## 1. What's wrong today (diagnosis)

The current pipeline works but is **fragile and bloated for three structural
reasons**, all confirmed by the audit and the benchmark logs:

1. **Machine-critical data lives in LLM-authored markdown.** The tool list,
   signatures, sample args, SKIP set and venv layout are all free-text the
   *next* LLM stage must re-read correctly every run. This is the root cause
   behind F25, F1/F4 staying "optional," the validator inventing `/dev/null`
   args, and `validation.md` mis-formatting dropping data from the harness
   aggregate (audit 2.5 / N8 / N15).
2. **Stability was bolted on reactively.** F14–F49 are 30+ point-fixes each
   patching one observed failure — argv-bug catalogs, device-default rules,
   path-join reminders, per-shape import checks — mostly as *more prompt text*
   the model must obey. Prompts ballooned to ~1600 lines; the coder prompt
   alone is ~495. Every fix assumes the model follows instructions.
3. **Everything is tuned around a non-frontier model (qwen3-235b).** Much of
   the hand-holding exists because qwen mis-follows or loops. A frontier model
   (glm-5) needs far less of it — but the *deterministic* guarantees should
   not depend on model strength at all.

Net effect on the target subset: mostly `ERROR — validation.md not readable`
and syntax-FAILED rows. The signal is lost to timeouts, formatting drift, and
hallucinated symbols — not to genuinely impossible repos.

---

## 2. The core architectural shift: **the LLM proposes, code disposes**

Every mechanical decision moves out of the prompt into **deterministic code**;
the LLM is left with the genuinely creative work (understand the repo, choose
what to wrap, write helper logic, diagnose a bug). This single principle
subsumes most of the F-backlog and both competitors' strongest mechanism (AST
verification), and it makes the system model-agnostic — a weaker model can't
break a code-enforced gate, and a stronger model isn't slowed by prompt
hand-holding it doesn't need.

**Structured contract, not markdown handoff.** Each stage emits a small typed
**`plan.json`-style contract** (parsed + schema-checked by code, with precise
parse-error feedback on retry) carrying the machine fields; markdown reports
are kept only as human/LLM-readable context, and `validation.md` is
**rendered by code** from structured results — never hand-formatted by the
LLM. This kills report-drift (the biggest source of lost harness signal) and
unlocks the deterministic gates below as *code*, not *prompting*.

---

## 3. Redesigned workflow

Five phases; **4 LLM agents interleaved with 2 deterministic gates**. Same
four stage names the harness/`--resume`/`--until` expect
(`explorer/environment/coder/validator`) — the gates run *inside* the
orchestrator between stages, invisibly to the contract.

```
 clone → EXPLORE ──▶ [GATE: plan] ──▶ ENVIRONMENT ──▶ CODER ──▶ [GATE: static] ──▶ VALIDATE(+repair loop)
          (LLM,        (code:            (LLM, given      (LLM,        (code: AST        (LLM validator + debugger
        read-only,    AST-verify         computed        given         import/symbol/    with memory; code renders
        propose)      symbols +          layout)         verified      undefined-name    validation.md from results)
                      layout)                            signatures)   check)
```

### Explorer (LLM, read-only) — *understand + propose*
Clone (with submodules), read README + **at least one real test/example**,
identify env requirements + external weights, and propose tool **candidates**
as a structured block: `{name, target ("module.path:symbol" or "script:path"),
purpose, sample_args, holdout_args, returns_schema}`. If a `target_task` was
supplied (§5), it must include a candidate satisfying it. No more "1–5
scenarios in prose."

### Gate: Plan (deterministic, no LLM) — *verify + decide*
The highest-leverage new piece, validated by **both** competitors:
- **AST symbol verification (F1/F4/ToolRosella `code_check`).** Scan the clone,
  build a `module → {functions(with real params), classes}` table. For each
  candidate `target`: confirm the symbol exists (or the script file exists),
  drop hallucinated/`test`/`example`/private ones, and **inject the real
  parameter names** into the contract so the Coder wraps correct argv.
- **Tool-count cap** with simple confidence ordering (ToolRosella caps at 12).
- **Venv layout decision (N15).** Parse `requires-python` /
  `python_requires` with `packaging.SpecifierSet` → `one-venv` (≥3.10) vs
  `two-venv` (older), server python + repo python. This is a pure function of
  the declared constraint — no LLM needed, no more prose decision tree.

Output: the authoritative `plan.json`.

### Environment (LLM, given the computed layout) — *build*
Told the layout + verified deps; builds venv(s) via `uv`, apt-gets system
libs, downloads weights (HF via `HF_TOKEN`, gdown for Drive). Records the
successful commands into an `install.sh` artifact (cheap; supports the paper's
reproducibility story without changing the commit flow — see §6).

### Coder (LLM, given verified signatures) — *write*Check the latest run of the same repo - it's running in circles again. I think we should split the repos for
  double-env path for good. First, let's upgrade to python 3.12 in our base image. Second, the server-env does not
  even bother us, we'll only need it when the wrapper has done its job and we launch mcp-server into production. No
  more need to install any extras in the working tools env - is always static-install, python always gets only
  pytest and fastmcp
Writes `server.py` + `helpers/*.py` + `tests/test_server.py`. Prompt is short
because signatures are pre-verified and the gates catch the rest. Keeps the
proven patterns: subprocess-through-`PYTHON`, helper-per-tool, JSON on stdout,
no editable installs. Result contract is **sentinel-delimited** (N5:
`print("<<<ALEMBIC_RESULT>>>"); print(json.dumps(...))`) so library banners /
progress bars on stdout no longer break the JSON parse.

### Gate: Static check (deterministic) — *catch before running*
AST import/symbol + whole-file undefined-name check on `server.py` and every
helper, in the correct venv (F1/F28/F45/F39, already built — kept and made a
first-class pre-execution gate). Failures route back to the Coder (bounded),
not to an expensive live-invoke + debugger round.

### Validate + repair loop (LLM validator + debugger with memory) — *prove*
- **Deterministic sample gate (N3):** resolve path-shaped args against
  `REPO_PATH`/cwd and existence-check *before* invoking — a bad sample returns
  a distinct `bad_sample` outcome (ask for a correction, don't burn a
  debugger round).
- **Real invocation** of each non-SKIP tool (alembic's headline edge over
  ToolRosella's import-only smoke test — kept). SKIP is code-enforced (F25).
- **Semantic correctness gate (F2, ToolMaker's ASSESS):** check the returned
  value against the declared `returns_schema` (keys/types/shape); optionally an
  LLM judge. `{ok:True}` alone ("ran without raising") is not "correct."
- **Held-out invocation (F3, ToolMaker's example-vs-test split):** also invoke
  `holdout_args` (different inputs) so a tool that hard-codes its demo path
  fails honestly.
- **Debugger with problem-summary memory (F10, both competitors, still
  missing in alembic).** A running compact `problem_summaries` list is fed
  into each debug round so fixes don't oscillate (the live MUSK
  torch-version thrash, F49). The validator **independently re-invokes** after
  every debugger claim (F24 — kept).
- Code renders `validation.md` + `validation.json` from the structured
  per-tool verdicts. No LLM formatting.

**Debugger stays a sub-agent of the validator** (kept), with F29 step-logging.
Flattening it into a top-level stage was considered and rejected as churn for
little gain now that the validator already re-verifies independently.

---

## 4. Robustness carried over vs. dropped

**Kept (fold into the clean design, they are real):** honest stage-timeout
logging (F20), async tool offload so timeouts fire (F23), per-debugger-call
timeout (F16) + process-group kill (F37), transient-provider-fault retry
(F17/F22), unknown-tool stub (F19), submodule clone (F34), `IsADirectoryError`
guard (F33), soft-deadline "write what you have" nudge (F32), the F35 fallback
reporter, F12 metrics, F6 weight download, F25 SKIP gate, F28/F39/F45 static
checks, F30 result-size cap, F36 cycle-exempt tools.

**Dropped / simplified (were reactive prompt patches now covered by gates or a
frontier model):** the argv-bug catalog, device-default rule, path-join
reminders, mega-tool warnings, per-shape import special-casing, and most of
the coder/validator prose — replaced by the Plan gate (correct signatures up
front), the static gate (catches the rest), and the sample gate. Prompts
target ~60–70% smaller.

**Deferred with rationale (documented, not silently skipped):**
- **F8 build-from-`install.sh` image / F9 reset-per-iteration checkpoint**
  (ToolMaker's reproducibility wins). Heavy Docker-in-Docker engineering; the
  current `docker commit` is harness-compatible and works. We *record*
  `install.sh` (cheap) for the paper but keep committing. Revisit post-benchmark.
- **F11 tiered models.** One `MODEL` knob (env-configurable) is what the
  glm-5 final run needs; per-role routing is a cost optimization, not a
  correctness one.
- **F5 first-class conda.** Audit measured conda needed 0/14 times on this
  subset; `uv` + apt-get covers it. Keep conda as a fallback only.

---

## 5. Target-tool / TM-Bench support (explicit new capability)

The user asked for "setting a list of target tools." Add an **optional
`--target-task` / `ALEMBIC_TARGET_TASK` input** (a JSON/YAML task spec:
`description`, typed `arguments`, typed `returns`, one `example`) threaded into:
- **Explorer:** "ensure one proposed tool implements this capability."
- **Coder:** "expose a tool with exactly these params returning these keys."
- **Validator:** the semantic gate checks against the task's `returns`; the
  held-out invocation uses the task's `test_cases`.

This maps 1:1 onto a TM-Bench `task.yaml` (typed args/returns + example +
held-out test_cases) and gives the two paper metrics the audit recommends:
**coverage/discovery rate** (URL-only mode — did autonomous exploration even
propose a matching tool?) and **strict task pass rate** (targeted mode + a
per-task MCP→`def(...)->dict` shim). The shim + license/HF-gating remain future
work for a strict head-to-head; the *input plumbing* is built now so the
pipeline can run either mode. Native alembic mode (no target task) is unchanged.

---

## 6. Model-agnostic & determinism decisions

- **No pinned temperature by default.** Both ToolMaker (gpt-4o, provider
  default) and ToolRosella (0.1) reach their numbers without temp-0, and
  pinning temp-0 reliably looped qwen (audit N1). Determinism comes from the
  **gates**, not the sampling. Keep `MODEL_TEMPERATURE`/`MODEL_TOP_P` env
  overrides; leave unset by default.
- **Report as mean ± std / pass@k over k≥3** — the honest metric for a
  stochastic system (audit N1 methodology), not a single run.
- **One `MODEL` env knob** (`openrouter/qwen/qwen3-235b-a22b-2507` default →
  `z-ai/glm-5` for the final run).
- **Structured output via fenced-`json`-block + code parse + guard-retry** —
  the proven F25 pattern generalized; no new ADK plumbing, degrades gracefully.

---

## 7. Module layout (concise, one concern per file)

```
alembic/
  common.py        # get_repo_name, ensure_base_image (UNCHANGED — harness imports these)
  config.py        # MODEL, all timeouts, caps — single source of truth
  contract.py      # dataclasses + JSON parse/validate + validation.md renderer
  runtime.py       # ADK driver: guarded run loop, patches, failure taxonomy (was agent_runtime.py)
  pipeline.py      # orchestration + the two deterministic gates (was main.py's body)
  agents.py        # slim agent defs + debugger memory
  main.py          # thin `python -m alembic.main` entrypoint shim
  start_chain.py   # docker build/commit/serve (UNCHANGED behaviour)
  tools/
    paths.py shell.py fs.py venv.py invoke.py
    analysis.py    # NEW — deterministic AST symbol table + signature extraction (Plan gate)
    scripts/{invoke_tool.py, compat_check.py}
  instructions/    # 5 slim prompts (explorer/environment/coder/validator/debugger); reporter kept
```

Target: prompts ~1600→~550 lines, `invoke.py` ~510→~260, total module
meaningfully smaller, with the growth concentrated in *deterministic,
testable* code (gates) rather than prose.

---

## 8. Success criteria

1. `run_benchmark.py` runs unchanged; every repo produces a parseable
   `validation.md` + `metrics.json` (no more `ERROR — validation.md not
   readable` from formatting/timeout).
2. Higher tool-invocation pass rate on `toolmaker_subset.txt` than the current
   ~0–3 passing tools/run, measured qwen-vs-qwen against recorded baselines,
   then a final `z-ai/glm-5` run.
3. Fewer, deterministic failure modes; hallucinated-symbol and
   report-drift classes eliminated by construction.
4. Module is smaller and each file has one clear job.

---

# Part II — Upgrade (R1–R8, 2026-07-10)

Second architectural pass per [upgrade.md](./upgrade.md). Supersedes parts of
§3–§6 above where stated. Maintainer choices locked in Q&A: harness moves to
JSON extraction; per-tool metrics with passed/perfect semantics; **full
TM-Bench export** (score in their harness, no debugger on their tests);
deterministic MCP wrapper + LLM fallback; in-container fs checkpoints;
code-recorded `setup.sh`.

## II.1 Target-focused retries, not graceful fails (R1)

The qwen run's esm failure (Explorer never produced a report) showed the old
posture — "run once, salvage what you can, move on" — optimizes cost, not
completion. Inverted:

- **Stage-reset loops.** Every LLM stage runs inside a reset loop. If its exit
  gate fails, the stage's owned files are rolled back to the checkpoint taken
  at stage start, a fresh agent session is created, and the stage reruns with
  a one-paragraph note about what failed last time. `STAGE_RESET` env, default
  2 extra loops. Stochastics happens; a reroll usually lands.
- **Stage timeouts optional, default OFF.** Loop breakers (step ceiling,
  repeat/cycle breakers, per-command subprocess timeouts) still bound runaway
  behaviour, but no wall-clock guillotine kills an honest slow install.
  `ALEMBIC_TIMEOUT_<STAGE>` re-enables per stage when needed.
- **Checkpoints are in-container filesystem snapshots** (Q&A): each stage owns
  a known set of paths (reports, plan.json, venvs, output/tools, server.py);
  reset = delete/restore those + fresh session. ToolMaker-style per-stage
  `docker commit` was rejected: the pipeline runs inside the build container
  (no docker socket), and host-orchestrated multi-commit runs are a heavy
  rewrite for the same effect. The final `docker commit` flow is unchanged.

## II.2 Metrics from run data, not the final report (R2)

`validation.md` as the harness contract was the last report-shaped dependency
— and it still zeroed two repos in the qwen run. Now:

- The pipeline writes **`reports/stage_status.json` incrementally** — after
  every stage attempt and every gate: status, resets used, gate details.
  `validation.json` (rich per-tool metrics) and `metrics.json` are likewise
  written by code during the run. A crash at any point leaves valid JSON.
- **`run_benchmark.py` extracts only JSON** (stage_status/validation/metrics)
  from the committed image. `validation.md` is still rendered — for humans —
  but nothing parses it anymore. Report-not-readable as a failure class is
  eliminated by construction.
- Reports are now **only written by the Explorer** (its output feeds the plan
  gate and downstream prompts). Environment/Coder reports are replaced by
  gates + stage_status; downstream agents get the exploration report appended
  directly to their opening prompt (no read_report round-trip).

## II.3 Functions first, server last (R3) — the structural inversion

The old pipeline made the Coder write `server.py` + argparse helpers, then
validated tools *through* that plumbing. Testing therefore always exercised
subprocess/argv wiring — the layer where most generated-code bugs lived.
Inverted, ToolMaker-style:

- **The Coder writes each tool as a plain Python function** —
  `output/tools/<name>.py`, one `def <name>(...) -> dict` per file, thorough
  docstring (purpose, per-arg description, usage examples), **imports inside
  the function body** (keeps import cheap for pytest collection, and makes the
  function self-contained so the TM-Bench `code.py` export is a verbatim copy).
- **Tests import the function directly** (`from tools.<name> import <name>`) —
  no argparse, no subprocess in the test path. Two kinds, split by naming
  convention in `tests/test_<name>.py`:
  - `test_smoke_*` — quick sanity, always present;
  - `test_invoc_*` — evidence-based correctness (reference values, shapes,
    output files), written **only where the Explorer documented grounds**
    (README numbers, repo tests, example outputs). The Explorer's plan now
    carries per-tool `sample_args` + `evidence` for exactly this.
- **New pipeline order** (stage names for --resume/--until in parens):

```
 clone → EXPLORER ─▶ [G1 plan gate] ─▶ ENVIRONMENT ─▶ [G2 env gate] ─▶ CODER
 (explorer)          AST verify +      (environment)  venv compat +   (coder)
                     layout            build venvs    repo-import
                                                      smoke
   ─▶ [G3 artefact gate] ─▶ VALIDATION ─▶ WRAPPER ─▶ [G4 server gate] ─▶ EXPORT
       per-tool: compiles,  (validator:    (wrapper: deterministic      setup.sh,
       imports, undefined   code loop +    codegen; LLM agent only      code.py,
       names, tests exist   batch debug)   as G4-failure fallback)      artefacts
```

- **G2 env gate (deterministic, was missing).** `check_venv_compat` runs on
  `.venv` (and `.venv-repo` in two-venv mode) with no LLM discretion, plus a
  repo-import smoke test (import the top-level modules of `plan.tools`
  targets in the tools venv). On failure: one bounded debugger round →
  re-check → still broken = stage reset with the failure note. The Coder
  never writes against a broken venv.
- **Validation is a deterministic code loop** over per-tool checks: direct
  invocation (execution status) + the pytest suite (smoke/invoc split). All
  failures across tools are **batched into one debugger call** ("fix all at
  once") so shared root causes (a missing dep hitting 5 tools) cost one round,
  and the fix propagates to neighboring tools. Debugger rounds capped by
  `DEBUGGING_ROUNDS` (default 10, on by default).
- **The MCP server is generated last, deterministically** (Q&A). `server.py`
  is rendered by code from the verified function signatures (AST) — every tool
  call shells through the tools venv via a generic `helpers/run_function.py`
  runner (the same script validation used, so serving and validating share one
  execution path; two-venv works unchanged). The wrapper **LLM agent exists
  only as a fallback** when the G4 compile/import gate fails. Tests passing on
  the functions + a compiling generated wrapper ⇒ the server is presumed
  valid — no live re-validation through MCP.
- **Exported artefacts** per run: `server.py`, `tools/`, `tests/`,
  `helpers/`, `setup.sh` — copied by the harness into
  `runs/<ts>/output/<repo>/`.

## II.4 Two-level tool verdicts (R6)

TM-Bench's split adopted verbatim: **execution status** (shallow — didn't
crash) vs **invocation correctness** (evidence-based tests). Per-tool metrics
(Q&A):

| metric | source | notes |
|---|---|---|
| tests passed / total | `test_smoke_*` results | `-` if none |
| exec ok | direct invocation w/ sample_args, 120 s cap | timeout or missing input files ⇒ **runtime success** (not a failure); crash ⇒ failed |
| invoc passed / total | `test_invoc_*` results | `-` if no evidence basis |

- **Tool `passed`** = all its tests passed **and** it never crashed.
- **Tool `perfect`** = passed **and** all its invocation-correctness tests
  passed (requires at least one).
- **Repo overall** = the counts: tests passed, invocations exec-ok,
  invocations correct, № passed tools, № perfect tools. No single binary
  verdict is load-bearing anymore.

GUI / no-visible-output tools get execution-status testing only (documented as
such by the Explorer), never invocation tests.

## II.5 TM-Bench compatibility (R4, R7)

- **Task-targeted mode** accepts one **or a list of** task specs
  (`ALEMBIC_TASKS`, JSON/YAML text or paths; old `ALEMBIC_TARGET_TASK` still
  works). The Explorer gets each task's name/description/typed
  arguments/returns/example verbatim and MUST plan a tool per task with
  exactly that name and signature; the plan gate enforces presence.
- **Multi-task, one repo** (the STAMP case): both tasks ride one pipeline run
  — the Explorer scopes for both, the Coder implements both functions. If the
  dual run fails, fall back to separate runs.
- **Mount data**: `start_chain --mount-dir <dir>` bind-mounts ToolMaker's
  `benchmark/data` read-only; the pipeline copies each task's
  `example.mount`/`test_cases` files to `/mount/input/<dst>` before
  validation. Missing files ⇒ exec-only testing (runtime success), never a
  crash.
- **Full export, scored in their harness** (Q&A): after the pipeline, render
  `code.py` per task (the tool function copied verbatim — self-contained by
  construction) and tag the committed image
  `toolmaker-runtime:installed-<task>`. No debugger is wired to their tests;
  we export what we have and score there.
- **R7 dataset firewall**: in task mode the Environment agent's prompt
  prohibits dataset downloads (weights/configs/deps explicitly allowed) even
  if the exploration report requests them.

## II.6 Reproducibility artefacts (R5)

`setup.sh` is **recorded, not authored** (Q&A): the pipeline logs every
*successful* `setup_venv`/`bash_env` command the Environment agent runs
(ToolMaker's `installed_state.bash()` approach) and renders `setup.sh`
deterministically at env-gate pass. Always matches what actually ran; zero
LLM effort; exported with the run artefacts.

## II.7 Token hygiene (R8)

`repo_url` is gone from every tool signature except `clone_repo`. The pipeline
(and `clone_repo`) set a process-wide current-repo; path helpers resolve
against it. One pipeline = one repo; multi-repo tooling was pure token tax.

## II.8 Updated success criteria

1. Zero repos lost to report formatting/absence — all benchmark data comes
   from JSON written during the run.
2. A stage that fails its gate retries from checkpoint (visible as
   `resets > 0` in stage_status), instead of cascading a zero downstream.
3. Per-tool passed/perfect counts on `toolmaker_subset.txt` beat the qwen
   baseline (9 tools passing / 4 repos); final numbers from
   `openrouter/z-ai/glm-5.2`.
4. STAMP dual-task run produces both tools from one pipeline; `code.py` +
   `installed-<task>` tags score in ToolMaker's harness.
