# 03 — Benchmarking: TM-Bench / ToolArena integration

You are set on TM-Bench (the ToolMaker benchmark) as the external gold set, and
believe it is the only fit. This report: (1) corrects three points in the
current mental model, verified against the source; (2) evaluates your proposed
"give alembic the repo + target task" merge; (3) gives the concrete integration
design and the two metrics to report; (4) states feasibility for 2026-07-10.

Facts below are verified from `github.com/KatherLab/ToolMaker` (branch
`original`), the ACL-2025 paper (arXiv:2502.11705 / 2025.acl-long.1266), and the
ToolArena repo. Directional items are flagged.

## 1. Three corrections to the current plan

### Correction A — it is the **`original` branch**, and the task file has NO gold output

- The paper's **15 tasks** live on branch **`original`**, in
  `benchmark/tasks/*.yaml` (exactly 15 files). Branch `main` is rewired to the
  **ToolArena** superset (more tasks; DEMO_READINESS saw "26"). So "the right
  branch with 15 tasks" = **`original`**, not a count you filter `main` down to.
  (Your earlier note about commit `bc3b56f`/`dd6d770` is consistent — that's the
  history of the same 15-task set; branch `original` is the stable pointer.)
- **The `task.yaml` does *not* contain gold outputs.** Each task is:
  `repo{url, commit, env}` + one-sentence `description` + typed `arguments`
  (name→{description,type}) + typed `returns` (name→{description,type}, shape
  only) + **one** `example` invocation **with no expected value** + `test_cases`
  (held-out invocations with *different* inputs). The gold lives **separately**
  as pytest unit tests in `benchmark/tests/test_<name>.py` plus reference
  tensors (`.npy`) in `benchmark/tests/data/`.

So "gold tool-call examples + taxonomy + env setup" maps to TM-Bench as: the
**example/test_cases** are the tool-calls, the **`category` field** is the
taxonomy (Pathology 6 / Radiology 2 / Omics 2 / Other 5), and **env setup is
declared but not scripted** — `repo.commit` + `repo.env` (e.g.
`HF_TOKEN: ${env:HF_TOKEN}`) are given; the tool-maker must produce the install
itself. **Alembic already does its own env setup**, so you do not consume
TM-Bench's — you consume its tasks + gold tests.

### Correction B — the required artifact is a **Python function returning a dict**, not MCP

TM-Bench scores *"any tool-maker that produces (1) an environment definition and
(2) a tool implementation"*, where the tool implementation is a **`def
<name>(...) -> dict`** with the signature generated from the `arguments`/`returns`
schema, returning a dict whose keys match `returns`. It is **not** an MCP server.
Therefore comparing alembic ⇒ TM-Bench **requires a per-task adapter** ("shim")
that presents the exact expected function signature and internally calls
alembic's running MCP tool, reshaping args in and the returned dict out to the
keys `tests.py` asserts on. DEMO_READINESS already flagged this as "real
engineering per task" — confirmed; it is unavoidable for a strict number.

### Correction C — the harness is **built to score alternative tool-makers**

The test harness supports a `TOOLMAKER_BENCHMARK_PREFIX ∈ {None, "toolmaker",
"openhands"}` that namespaces the produced tools so the **same unit tests** grade
different makers. That is your clean plug-in point: **alembic becomes a third
prefix.** You are not fighting the harness; it was designed for exactly this
kind of comparison (the paper's only external baseline, **OpenHands**, is scored
this way).

## 2. Evaluating your proposed merge

> "the alembic working on the repo → mcp exploration as usual PLUS the target
> tasks from the bench — this way we can measure with both our current method
> and theirs."

**This is the right shape, with two required pieces you'd need to build:**

1. **An optional `target_task` input threaded into Explorer + Coder.** Today
   alembic takes a URL only. To guarantee a tool matching the bench task exists
   (and has a compatible signature), pass the task's `description` + typed
   `arguments`/`returns` into the Explorer ("ensure one scenario implements
   this") and the Coder ("expose a tool with these params returning these
   keys"). Modest change — an extra optional field in the stage messages, not a
   new architecture.
2. **The per-task shim** from Correction B (bridge MCP ⇒ the expected
   `def(...) -> dict`), registered under an `alembic` prefix (Correction C).

With both, you get exactly your "measure both methods" goal — **and it is
actually two different, both-worth-reporting metrics**, not one:

## 3. The two metrics (report both — they tell different stories)

### Metric 1 — Coverage / discovery rate (alembic's *native* mode, favourable)

Give alembic **only the repo URL** (no target task). Ask: did its autonomous
exploration produce a tool whose capability matches the bench task? Scored by
mapping alembic's emitted tools → the target task (LLM-judge or human).

This is the metric that plays to alembic's thesis: **ToolMaker is told what to
build; alembic is not.** ToolMaker's paper has no coverage concept to compare
against, so a coverage-rate column is a claim only alembic can make. Needs no
shim — just the mapping judgment. Report it as "fraction of the 15 tasks where
autonomous discovery even proposed a matching tool."

### Metric 2 — Strict task pass rate (apples-to-apples with ToolMaker's 80%)

Targeted mode (`target_task` fed in) + shim + run TM-Bench's `tests.py`. A tool
is *correct* iff **all** unit tests across **all** its held-out `test_cases`
pass. This is directly comparable to the published baselines:

| Method | Tasks correct | Notes |
|--------|:-------------:|-------|
| **ToolMaker** (gpt-4o) | **12/15 (80%)** | told the exact signature |
| OpenHands (gpt-4o) | 3/15 (20%) | only external baseline; ~half its installs crashed |
| OpenHands (Claude 3.5 Sonnet) | 2/15 | |
| **Alembic** | *to measure* | autonomous discovery + shim |

Framing for the paper: even a **lower** strict number than ToolMaker is a strong
result *if paired with Metric 1*, because alembic solves the harder,
un-specified problem and ships a served MCP endpoint rather than an
in-container function.

## 4. Cheaper intermediate: adopt TM-Bench's *shape* now, its *harness* later

You do not need the full shim to get most of the scientific value. TM-Bench's
real contribution to your stability problem is **typed, held-out, gold-graded
invocations**. Adopt that shape inside alembic's own validator immediately:

- Have the Coder emit a typed `returns` schema per tool (keys + types), like
  TM-Bench's `returns`.
- Require **two** invocations per tool with different inputs (this is F3,
  held-out) — TM-Bench's `example` vs `test_cases` split is the exact pattern.
- Grade with **property assertions** (structure/values/files/execution), not a
  bare `{ok}` — TM-Bench deliberately chose unit tests over equality checks for
  this reason.

This gives your **internal** benchmark a correctness signal (closing the F2/F24
gap and much of the [02](./02-stability.md) instability) without waiting on
license/shim work — and it makes the eventual TM-Bench shim trivial, because
alembic already speaks typed-returns.

## 5. Feasibility for 2026-07-10 and the blockers

**Recommendation: do not attempt full TM-Bench parity for this submission.**
Consistent with DEMO_READINESS §4. Blockers, all real:

- **No LICENSE on ToolMaker/ToolArena** → redistributing tasks/tests needs the
  authors' permission. Email now if you want the stretch.
- **Per-task shim is genuine engineering** (one adapter per matched task).
- **HF-gated weights** on several tasks (UNI, CONCH, MUSK, RETFound…) need
  `HF_TOKEN` handling — that is F7, not yet built.
- **Task drift** — use branch `original` for the paper-exact 15; `main` is the
  moving ToolArena superset.

**For submission:** ship the internal 12-repo benchmark (with N1's k-seed
reporting) as the Evaluation; mark TM-Bench head-to-head as future work in
`limitations.tex`, naming the shim + permission + coverage-metric plan so it
reads as scoped, not hand-waved.

**Stretch (only if §1–§4 of DEMO_READINESS land early):** shim 2–3
**non-gated** tasks — good candidates by inspection: `tabpfn_predict`
(PriorLabs/TabPFN), `modernbert_predict_masked`, `cytopus_db` (no HF-gated model
weights). Report both metrics on that small slice as an illustrative row.

## 6. On "TM-Bench is the only option" — refined, not confirmed

TM-Bench is the **best** fit for the full combination (env-setup-from-arbitrary-
repo + gold held-out outputs + taxonomy + a callable typed tool), but it is not
unique:

- **ToolArena** (KatherLab) — same `task.yaml` schema, **maintained superset**,
  and the direction the ToolMaker authors themselves moved to. If you're
  building the shim anyway, target ToolArena's harness so you inherit >15 tasks
  and future updates. This is the strongest complement to TM-Bench, not a rival.
- **GitTaskBench** (arXiv 2508.18993, AAAI) — 54 tasks, repo + env setup +
  curated automated eval + a 7-modality × 7-domain taxonomy. Closest independent
  match; differs in grading **end-task delivery** rather than a reusable typed
  tool, and lacks TM-Bench-style typed gold tool-call I/O. Worth a citation and
  possibly a second external comparison.
- **SUPER** (EMNLP 2024) and **CORE-Bench** — env-setup-from-repo + gold
  *answers*, but they yield an *answer*, not a callable tool, and have no typed
  tool-call schema. Not a fit for alembic's output, but cite as the neighbouring
  "set up and run a repo" line of work.
- **ScienceAgentBench** — gold reference *programs* + outputs, but ships a
  dataset per task rather than env-setup-from-arbitrary-repo. Partial fit.
- **ResearchEnvBench** (arXiv 2603.06739, 2026) — "environment synthesis for
  research code"; very recent, **unverified** — flag for a look.

**Bottom line:** keep TM-Bench as the headline external target, but (a) build the
shim against **ToolArena** so it doesn't rot, and (b) cite GitTaskBench/SUPER/
CORE-Bench to show you surveyed the space — a reviewer will know these exist.
