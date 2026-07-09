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

### Coder (LLM, given verified signatures) — *write*
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
