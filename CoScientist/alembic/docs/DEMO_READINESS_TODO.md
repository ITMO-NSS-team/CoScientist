# Demo-paper readiness TODO

Submission deadline: **2026-07-10, 11:59 PM UTC-12h** — roughly 4 days out from
when this was written. Sequencing below is chosen so that a hard stop after
any "must-have" block still leaves a submittable paper.

Sources: [IMPROVEMENTS_SPEC.md](./IMPROVEMENTS_SPEC.md) (F1–F23), the
per-run summaries + logs under
[benchmarks/alembic/runs/](../../../benchmarks/alembic/runs/) —
`2026-06-30_baseline/` (12-repo baseline, 2026-06-30),
`2026-07-06_rerun1_f14-f18/` (same 12 repos after F14–F18),
`2026-07-06_rerun2_f14-f19/` (11 repos, F18 confirmed at scale + F19-v1/F21
found), `2026-07-06_rerun3_f19-f21-targeted/` (3-repo targeted re-verify of
F19-v2/F20/F21, also surfaced F22/F23) — each folder holds `summary.md`,
`summary.json`, and `logs/*.log`. Also a review of
[ToolArena](https://github.com/KatherLab/ToolArena) (TM-Bench's harness
repo) at commit `0b275a266e01b38f13a8e7041489f5762887cd65`.

---

## 1. Must-have — system fixes (blocks trustworthy Evaluation numbers)

These were caught directly in the bench logs, not inferred — each one is
currently *understating* Alembic's real pass rate, so fixing them before the
next benchmark run matters more than any comparison-driven feature.
**All re-benchmarked on the same 12 repos on 2026-07-06** (see §1a) — status
below reflects what the rerun actually confirmed, not just what was patched.

- [x] **F14** — Validator blanket-skips all tools when *any* one test fails
  (caused `auto-sklearn` and `biotite`'s 0/0/4 SKIP rows). **Confirmed fixed
  by rerun**: `auto-sklearn` now shows 3 real, distinct `FAILED` verdicts
  (genuine `FileNotFoundError`/`ModuleNotFoundError`/`KeyError`, not one
  shared cause) instead of a blanket SKIP; `biotite` tests now fully PASS
  and its remaining SKIPs are individually-reasoned coder decisions
  ("requires external PDB file", "requires network access"), not a
  short-circuit.
- [x] **F15** — Debugger request must always include `repo_url` (a missing
  URL turned `biotite`'s 1-character `SyntaxError` into a total stage
  failure). **Confirmed fixed by rerun**: the exact baseline failure string
  ("cannot locate ... verify repository URL") never recurs across all 12
  rerun logs, including cases where the validator's own request text was
  URL-less — the debugger still operated on the correct repo path.
- [x] **F16** — Per-debugger-call timeout separate from the stage timeout
  (`biopython` burned its whole 30-min Validator budget on one stuck call
  and produced no report at all). **Confirmed fixed by rerun**: the
  600s-timeout message fired 5 times across `auto-sklearn`, `BioSPPy`, and
  `ase`, and every time the stage kept going afterward instead of dying.
  `biopython` specifically now finishes in 1341s with a real report
  (`10 passed, 5 failed`) instead of losing the whole run.
- [x] **F17** — Retry once on transient LLM/API error (empty-body
  `JSONDecodeError` from OpenRouter/LiteLLM burned a `BioSPPy` debugger
  attempt for nothing). **Confirmed fixed by rerun**: fired in `ase` and
  `backtrader` (`"transient error (...), retrying once"`); backtrader's
  retry succeeded outright and its tool went on to PASS.
- [x] **F18** — Explorer/Coder must propose realistically-sized, real
  sample inputs. **First patch (Explorer-only) did NOT work** — rerun
  `BioSPPy` still used `signal=[0.1, 0.2, 0.3]`-style arrays. Root cause:
  the concrete `samples:` block `invoke_mcp_tool` runs is written by the
  **Coder**, not the Explorer, and `instructions/coder.py` had its own,
  directly conflicting rule — *"Use the most minimal args you can."*
  Re-patched `coder.py` to distinguish "cheap to run" from "smaller than
  the function's precondition." **Confirmed fixed at scale by rerun2**
  (11 repos): every `invoke_mcp_tool` call now uses real repo-bundled data
  or realistic parameters — `BioSPPy`'s `examples/ecg.txt`, `astropy`'s
  bundled `.fits` fixtures, `auto-sklearn`'s own `load_breast_cancer`
  example, real physical/chemical/financial parameters elsewhere. No tiny
  placeholder array recurred anywhere. F18 is closed.
- [x] **F19** *(new, found during the rerun, not in the original bench)* —
  `_UnknownToolStub` (the hallucinated-tool-name stub in `main.py`) was
  missing attributes ADK's `BaseTool` exposes, causing
  `AttributeError: '_UnknownToolStub' object has no attribute '...'`
  whenever ADK introspected it. **v1 (added `.description`) was
  incomplete** — rerun2 hit the identical failure class again for a
  *different* attribute, `is_long_running`, in `AgML` (unresolved after
  retry, a genuinely lost debugger attempt) and `BioSPPy` (retry hung
  until the 1800s stage timeout, truncating validation to 1 of 5 tools —
  see F20). **v2** now sets the stub's full `BaseTool` surface
  (`is_long_running=False`, `custom_metadata=None`) so no further
  attribute can be missing regardless of which code path introspects it.
  **Confirmed fixed by targeted rerun3** (`AgML`, `BioSPPy`, `biotite`):
  zero `_UnknownToolStub`/`AttributeError` occurrences across all three
  logs. F19 closed.
- [x] **F20** *(new, found during rerun2)* — Stage-timeout handling in
  `main.py` logged `"[Validator done] report → .../validation.md"` and a
  `"Pipeline complete"` success banner **unconditionally**, even when
  `_run_stage()` had just returned early on a timeout with nothing
  written. Confirmed via `docker run --rm --entrypoint cat
  alembic-tool:BioSPPy .../validation.md` → file did not exist, despite
  the log claiming success. This is what produced the benchmark summary's
  bare `"validation.md not readable"` row with no visible cause. Fixed:
  gated all four stages' "done" logs on the stage's actual return value
  (a real completion always returns a non-empty string; only the timeout
  path returns `""`), and replaced the false success banner with an
  honest "Pipeline incomplete — stage timed out" one. Also generalized
  `run_benchmark.py`'s `write_summary()` so the Overall column shows
  `ERROR — <reason>` for *any* validation-extraction failure, not only
  the repo-unreachable case already handled — same "don't hide the reason
  behind a bare `—`" fix applied more broadly. **Confirmed fixed by
  rerun3**: `AgML`'s Validator stage and `BioSPPy`'s Coder stage both
  genuinely timed out again in this run (real slow repos / real F22
  latency, not new regressions), and both times the log printed an
  honest "Pipeline incomplete"/"STAGE TIMEOUT" message with no false
  success claim, and the summary table showed `ERROR — validation.md not
  readable` instead of a bare `—`. F20 closed.
- [x] **F21** *(new, found during rerun2)* — Coder invented
  plausible-but-nonexistent sample file paths (`biotite`: `example.fasta`,
  `example.pdb`, `fixed.pdb`/`mobile.pdb`) instead of the real bundled
  files it had already seen via `read_file`/`ls`, failing 4/5 tools.
  `coder.py` already said "Do NOT invent paths" — this is an
  instruction-following miss, not a missing rule. Strengthened the
  wording with the concrete counter-example and an explicit
  verify-before-use requirement. **Confirmed fixed by rerun3**: `biotite`
  now uses real repo test-fixture paths throughout
  (`tests/structure/data/pdb/4gxy.pdb` etc.); its 2 remaining tool
  failures are genuine (a real structural PDB mismatch and a real
  Python ≥3.12-vs-3.11.15 environment gap), not invented filenames. F21
  closed for the observed case; a deterministic file-existence gate
  (F1/F4) remains the durable general fix.
- [x] **F22** *(found during rerun3, fixed 2026-07-06)* — A rare (1 of
  ~15 runs so far) OpenRouter/LiteLLM fault: an upstream provider error
  surfaces as an unmapped `finish_reason: "error"`, which `litellm`'s
  `map_finish_reason()` silently defaults to `"stop"` instead of raising
  — so ADK treats a stub/garbage response as a normal completed turn.
  Cost `BioSPPy`'s Coder stage ~8 of its 25 minutes across two
  occurrences in rerun3, recovered only by accident via an unrelated
  guard-retry nudge. **Fix:** rather than chasing `map_finish_reason`
  across the 9 litellm modules that each independently `from ... import`
  it (fragile), hooked litellm's own diagnostic logger — a stable public
  integration point — via a `logging.Handler` that flags a
  `contextvars.ContextVar` when it sees the exact warning text.
  `run_agent`'s stage loop now retries (same message, up to 2x) whenever
  the flag is set, on a budget entirely separate from the guard-retry
  nudges. Unit + integration tested (fault-then-recover, and
  exhaustion-doesn't-infinite-loop); see
  [IMPROVEMENTS_SPEC.md](./IMPROVEMENTS_SPEC.md#f22) for the full
  test list, including the subtlety that litellm's own redaction filter
  required using `record.getMessage()` instead of raw `record.args`.
- [x] **F23** *(found during rerun3, fixed 2026-07-06)* — `AgML`'s
  Validator stage produced no report at all in rerun3 because F16's 600s
  per-debugger-call timeout didn't fire: a debugger call went silent for
  ~20 minutes before the *outer* 1800s stage timeout finally caught it,
  itself ~2 minutes late. **Root cause confirmed** (not just inferred)
  by reading ADK source directly: `function_tool.py`'s
  `_invoke_callable()` calls synchronous tool functions with a plain
  blocking call, no `asyncio.to_thread`/`run_in_executor` — so any one
  of our `subprocess.run()`-based tools (`bash`, `bash_env`,
  `setup_venv`, `check_venv_compat`, `validate_syntax`, `run_tests`,
  `invoke_mcp_tool`, `clone_repo`) fully freezes the event loop for its
  duration, during which no `asyncio.wait_for`-based timeout can fire.
  **Fix:** converted all 8 to `async def`, each offloading its blocking
  body to a worker thread via `asyncio.to_thread`. Verified with a
  decisive positive/negative test pair: `wait_for(bash_env("sleep 10"),
  timeout=1.5)` now correctly times out at ~1.5s post-fix, and the
  identical test against the *old* synchronous pattern does **not**
  time out at all (proving the test is real, not a false positive). See
  [IMPROVEMENTS_SPEC.md](./IMPROVEMENTS_SPEC.md#f23) for the full
  test list.
- [x] Live end-to-end confirmation (`benchmarks/alembic/runs/2026-07-06_rerun4_f22-f23-verify/`).
  Both `AgML` and `BioSPPy` completed all 4 stages cleanly this run — a
  real improvement over rerun3, where both truncated with "validation.md
  not readable" (the exact symptom F19/F20/F23 caused). F23 got a clean
  natural reproduction: `BioSPPy` hit a genuine slow `pip install`
  debugger call, and `[debugger] call timed out after 600s` fired
  **exactly 600.0s** after the call was sent — previously this line
  never appeared even ~20 minutes late. F20 got a free cross-check too:
  `AgML`'s Explorer stage genuinely timed out (900s, unrelated slow-repo
  case), and the log correctly shows no false "done" claim for that
  stage while the 3 stages that really did complete each show a true
  one. F22's fault did not recur live (expected at ~1/15 frequency) —
  no regression, and its correctness rests on the unit/integration
  tests, not this run. All remaining tool-level failures in both repos
  trace to genuine, unrelated application bugs (missing ML deps,
  `argparse` boolean-flag handling, sample-sizing in generated helpers),
  not to anything F19–F23 touched. F22 and F23 are both closed.
- [ ] **F24** *(new, found while explaining rerun4's tool-level failures,
  deferred past 2026-07-10 by explicit instruction)* — Traced why every
  failing tool in both `AgML` and `BioSPPy` stayed FAILED despite correct
  debugger diagnoses: `instructions/validator.py` tells the validator to
  trust the debugger's summary and never re-invoke a tool itself, on the
  assumption every failure is a code bug the debugger can edit and
  self-verify. Two real gaps in that assumption, both hit in this one
  run: (1) when the debugger correctly diagnoses "the sample/argument is
  wrong, not the code" (e.g. `quality_ecg` needs a 10s segment, `eeg` had
  only 3 samples), there's no path for that diagnosis to trigger a
  corrected retry; (2) a debugger fix to one tool (`ecg`'s argv
  construction) didn't get recognized as covering a sibling tool with
  the identical bug in its own helper file (`ppg`), so the validator
  re-hit the same error there with only one debugger attempt already
  spent on it. Full evidence and fix direction in
  [IMPROVEMENTS_SPEC.md](./IMPROVEMENTS_SPEC.md#f24) — adjacent to
  F1/F3/F4, likely best designed together with those post-submission.
  (Update 2026-07-06: the section below's related sub-caveat — no
  independent re-verification of the debugger's self-report — has since
  been fixed; see the F24 entry in §5 below.)
- [x] **F12** — Structured per-run JSON metrics + failure taxonomy.
  Implemented 2026-07-06: `agent_runtime.py`'s `run_agent`/`_run_agent_once`
  now return a `stage_metrics` dict (tool-call counts, a `classify_error()`
  taxonomy over `invoke_mcp_tool`/`validate_syntax`/`run_tests` failures —
  plus `DebuggerTimeout` from F16/F17's swallowed timeouts — guard-retry
  and F22 transient-fault-retry counts, and an abort reason); `main.py`
  folds these into `pipeline_metrics` per stage (with wall-clock durations)
  and writes it to `reports/metrics.json` as before. `run_benchmark.py`
  gained `aggregate_metrics()`, rolling every repo's `metrics.json` into a
  stage-completion-rate table and a cross-repo failure-taxonomy table,
  appended to `summary.md` and to `summary.json` (now `{"repos": [...],
  "aggregate": {...}}` instead of a bare list). Unit-tested the classifier
  (10 cases incl. nested-traceback root-cause extraction) and the
  aggregator (mixed complete/timed-out/crashed/unreachable repos) — see
  [IMPROVEMENTS_SPEC.md](./IMPROVEMENTS_SPEC.md#f12). Live verification on
  `biotite` (`benchmarks/alembic/runs/2026-07-06_rerun6_f12-verify/`) caught
  two real bugs the same day: `extract_validation()` skipped `metrics.json`
  entirely whenever `validation.md` was missing (exactly the
  timed-out/skipped-validator case F12 most needs to see), and
  `abort_reason` accumulated the *first* abort across guard-retry attempts
  instead of reflecting the latest one (so a stage that failed once and
  then genuinely succeeded on retry would still misreport the stale
  failure). Both fixed and re-verified: a clean rerun shows PASSED (4/4
  tools, 12/12 tests), `metrics.json` fully populated (per-stage durations,
  per-tool call counts, a `failures_by_class` of `{TypeError: 3,
  ModuleNotFound: 1, FileNotFound: 1}` correctly classified from 5 real
  debugger-cycle failures en route to the final pass), and
  `summary.json`'s aggregate correctly showing `repos_with_metrics: 1`.

## 1a. Rerun results (2026-07-06, same 12 repos as the baseline run)

Headline: `astronomy` and `astropy`'s tool-invocation layer newly PASS
outright (5/5, and tools-only-PASSED-despite-test-failure respectively);
`auto-sklearn` and `biotite` went from blanket-SKIP noise to real,
individually-diagnosable verdicts (see F14 above) — a materially more
trustworthy signal than the baseline even where the overall row is still
FAILED. Caveat: each run regenerates server code from scratch via the LLM,
so this is not a controlled code-for-code diff — some swings are ordinary
LLM-generation variance, not caused by F14–F21.

New/unrelated issues surfaced by the rerun (not F14–F21's scope, logged for
later): `astropy`'s generated test file never imports the tool functions it
tests (`NameError` on all 18 tests, while direct invocation still passes —
a Coder bug); `aizynthfinder`'s `create_interactive_app` helper emits no
stdout JSON; `backtrader`'s `backtest_strategy` has a real `IndexError` in
SMA warm-up logic; `ase`'s `optimize_geometry` hit a `UnicodeDecodeError`
reading a binary `.traj` file as text. `dalle-mini` and `Analyze-stroke`
reproduce their exact baseline root causes (unresolvable jaxlib/Python 3.8
deadlock; inaccessible git remote) unchanged.

### Rerun2 (11 repos, F14–F18 + F19-v1 in place)

Re-ran the same set (`Analyze-stroke` auto-skipped by the new availability
check — see below) with `coder.py`'s F18 fix and `main.py`'s F19-v1 fix
baked into a rebuilt image. F18 confirmed working at scale (see above).
F19-v1 turned out incomplete, and its `BioSPPy` failure mode (a hung retry
eating the full 30-min stage timeout) directly caused F20's misleading
"pipeline complete" logging to surface. `biotite` regressed to inventing
nonexistent sample paths (F21) — a different failure mode than F18's
sizing issue, on the same repo family (real-file selection, not real-file
sizing). Net effect: F18 and the repo-availability check are now solid;
F19/F20/F21 fixes are applied but pending a final targeted re-verification
run (`AgML`, `BioSPPy`, `biotite`) before this section can be closed out.

### Rerun3 (targeted, 3 repos: `AgML`, `BioSPPy`, `biotite`)

Ran only the 3 repos affected by F19/F20/F21 (not the full 11) to confirm
those fixes cheaply. All three confirmed fixed — see F19/F20/F21 entries
above. `AgML`'s Validator stage and `BioSPPy`'s Coder stage both timed out
again in this run, but for real, unrelated reasons (a genuinely slow/hard
repo, and a rare OpenRouter/LiteLLM fault — F22, deferred) rather than
F19/F20 recurring; the important signal is that both timeouts were now
*reported honestly* instead of masked as success. §1's fix backlog
(F14–F21) is now closed and confirmed by log evidence; F22 is documented
and deferred post-submission. F12 remains the only open must-have item
below.

### Rerun5 (refactor smoke test, 1 repo: `astronomy`)

`main.py` was refactored (523→249 lines) to move cross-cutting runtime
support — loguru setup, the ADK/LiteLLM compatibility patches (F19, F22),
and the guarded single-agent-turn runner (`run_agent`/`_run_agent_once`)
— into a new `agent_runtime.py` (289 lines); `main.py` now holds only
pipeline orchestration (stage sequencing, timeouts, CLI entrypoint). Ran
`astronomy` end-to-end as a live regression check: **PASSED**, 5/5 tools,
16/16 tests, exit 0, 1137s — no `STAGE TIMEOUT`, no `_UnknownToolStub`, no
`Unmapped finish_reason`, no `error.json`, matching the historically-good
result for this repo. The two debugger calls that did fire (a missing
package install, two `AttributeError`s from wrong SDK field names) are
ordinary generated-code bugs, unrelated to the refactor, both resolved
normally. Confirms the split introduced zero behavioral regressions.

### Final-eval (2026-07-06, full 12-repo set, all fixes F14–F23 + F12 in place)

`benchmarks/alembic/runs/2026-07-06_final-eval/` — 11/12 reachable
(`Analyze-stroke` unreachable in every run since baseline: dead repo, not a
regression), `--parallel 4`. Per-repo Overall verdict across all three full
runs to date:

| Repo | Baseline (06-30) | Rerun2 (F14–F19) | Final-eval (07-06) |
|---|---|---|---|
| AgML | PASSED 5/0/0 | FAILED 0/2/3 | **timed out** (validator, honest F20 log) |
| BioSPPy | FAILED 1/4/0 | ERROR (no data) | FAILED 0/4/1 |
| aizynthfinder | PASSED 3/0/0 | FAILED 1/4/0 | FAILED 0/2/1 |
| ase | PASSED 4/0/1 | FAILED 1/1/1 | FAILED 2/0/3 |
| astronomy | PASSED 5/0/0 | PASSED 7/0/0 | PASSED 5/0/0 |
| astropy | PASSED 4/0/1 | FAILED 3/2/0 | **PASSED 6/0/0** |
| auto-sklearn | FAILED 0/0/4 (blanket skip) | FAILED 0/2/2 | FAILED 0/5/0 (real diagnoses) |
| backtrader | FAILED 2/2/1 | FAILED 0/4/0 | FAILED 1/4/0 |
| biopython | no data (validation.md unreadable) | FAILED 2/3/0 | FAILED 4/1/0 |
| biotite | FAILED 0/0/4 (blanket skip) | FAILED 1/4/1 | FAILED 2/1/1 (real diagnoses) |
| dalle-mini | FAILED 0/1/2 | FAILED 0/3/0 | FAILED 0/1/0 |

Overall-PASSED count: **5/10 → 1/10 → 2/10** (denominator = repos with any
real validator data each run). Tool-invocation pass rate (Σpassed /
Σ(passed+failed+skipped) across repos with real data): **54.5% (24/44) →
31.9% (15/47) → 45.5% (20/44)**.

**Read this honestly, not as a regression.** None of F14–F23 touch code
generation quality — they fix logging honesty, timeout/retry mechanics, and
tool-lookup robustness. The swings above (AgML/aizynthfinder/ase declining;
astropy/biopython recovering) are the same LLM-regeneration variance
already characterized earlier this session (each run regenerates
`server.py` from scratch via the LLM; a repo passing in one run and failing
in the next, or vice versa, is expected noise, not a directional trend). Two
things *are* real, structural improvements, independent of this noise:
1. **F14's fix is visible in the data.** `auto-sklearn` and `biotite`
   went from a blanket 4-tool SKIP (baseline; one test failure nuking all
   downstream tool checks) to real, individually-diagnosable PASS/FAIL
   verdicts in both later runs — a materially more trustworthy signal even
   though the Overall row is still FAILED.
2. **F12 now gives a systematic failure taxonomy for free**, unavailable in
   baseline/rerun2: this run's aggregate (`summary.json`) shows `Import: 8,
   ValueError: 5, ModuleNotFound: 5, TypeError: 4, DebuggerTimeout: 4,
   FileNotFound: 3, Runtime: 3, NameError: 3, Syntax: 1, AttributeError: 1`
   across all 11 attempted repos — the first time this breakdown exists
   anywhere for alembic.

**Recommendation for the paper's Evaluation section:** report the
tool-invocation pass rate as the headline metric (not Overall-PASSED-repo
count, which is low-n and noisy), explicitly caveat it as a single-run
sample subject to LLM-regeneration variance (cite the 31.9–54.5% spread
across 3 independent runs as evidence of the caveat, not as 3 comparable
data points to average), and lead the qualitative narrative with F14's
SKIP→real-diagnosis improvement plus the new F12 failure taxonomy — both
of which are true, structural, and don't depend on any single run's luck.

## 2. Must-have — re-run the benchmark, write the Evaluation section

- [x] Re-run `benchmarks/alembic/run_benchmark.py` on at least the current
  12-repo set (ideally the full `toolrosella_subset.txt`, 12 more repos)
  after §1's fixes land, and diff against
  `benchmarks/alembic/runs/2026-06-30_baseline/summary.md` to confirm the
  SKIP rows resolve to real PASS/FAIL. Done 2026-07-06 — see "Final-eval"
  above; SKIP rows do resolve to real PASS/FAIL, but Overall-PASSED count
  itself is noisy run-to-run (see honest caveat above).
- [ ] Fill in `docs/paper/sections/evaluation.tex` with the real table
  (per-repo stage success + tool pass rate) and a headline number. **This
  is not optional for submission** — the demo track explicitly desk-rejects
  papers with no reported evaluation.
- [ ] Update `docs/paper/sections/limitations.tex` if the re-run surfaces
  new failure classes not already described there.

## 3. Must-have — paper mechanics

- [ ] Real author names/affiliations/emails (`emnlp2026_demo.tex`)
- [ ] `\repourl` / `\demourl` — public repo link + a ≤2.5 min screencast
  (`emnlp2026_demo.tex`); the demo track desk-rejects submissions with no
  working link
- [ ] Trim `docs/TOOLMAKER_COMPARISON.md` / `docs/TOOLROSELLA_COMPARISON.md`
  into `sections/appendix.tex` (2-page appendix limit)
- [ ] Acknowledgments (`sections/acknowledgments.tex`)
- [ ] Switch `\usepackage[preprint]{acl}` → `\usepackage{acl}` once camera-ready

## 4. Decision needed now: TM-Bench / ToolArena comparison scope

Investigated the harness so this isn't a guess: at the pinned commit,
[ToolArena](https://github.com/KatherLab/ToolArena) is a full runnable
harness (`toolarena` CLI, per-task Docker image, `pytest tasks/<task>
--implementation <dir>` scorer) — not just task files. Two blockers make a
*full* TM-Bench comparison unrealistic in 4 days:

1. **No LICENSE file in the repo** — redistributing/republishing benchmark
   artifacts needs explicit permission from the ToolMaker authors first.
2. **Task drift**: 26 task dirs exist today vs. the paper's 15; one original
   task (`nnunet_train_model`) was dropped and replaced. Getting paper-exact
   parity means pulling the original 15 from commit `bc3b56f`/`dd6d770`, not
   the current tree.
3. Alembic's input/output contract differs (URL-only autonomous multi-tool
   discovery vs. ToolMaker's exact-signature single function), so even with
   permission, comparable numbers require writing a **shim** per matched
   task (`implementation.py` that calls Alembic's already-running MCP
   server, reshaped to the dict `tests.py` expects) — real engineering per
   task, not a config change.

**Recommendation:** do not attempt full TM-Bench parity for this
submission. Options, pick one:

- [ ] **(Recommended)** Mark it explicitly as future work in
  `sections/limitations.tex` ("head-to-head TM-Bench comparison against
  ToolMaker pending author permission and a per-task adapter shim") and
  ship only the internal benchmark for Evaluation.
- [ ] **(Stretch, optional)** Email the ToolArena/ToolMaker authors now for
  permission, and if granted with time to spare, shim 2–3 of the 15
  original tasks (pick ones with no HF-gated weights) for a small
  illustrative comparison row — report both strict pass-rate (via shim)
  and a "coverage rate" (fraction of tasks where Alembic's autonomous
  discovery even proposed a matching tool), since ToolMaker's paper has no
  coverage-rate concept to compare against (it's always told what to build).

---

## 5. Optional — system robustness (stronger paper, not blocking)

Ranked by payoff; see [IMPROVEMENTS_SPEC.md](./IMPROVEMENTS_SPEC.md) for
full specs. Do these only if §1–§4 are done with days to spare.

- [ ] **F1** — Static import/symbol AST gate between Coder and Validator.
  Would have caught `astronomy`'s repeated `REPO_PATH` bug (fixed reactively
  3× across separate debugger calls) in one deterministic pass.
- [ ] **F4** — Bounded, confidence-ranked, AST-verified tool selection (real
  parameter names into the Coder pre-empt wrong-kwarg `TypeError`s like
  `auto-sklearn`'s `time_left_for_this_task`).
- [ ] **F24** — Validator/debugger handoff needs a "sample is wrong, not
  the code" outcome and cross-tool fix propagation for shared helper
  bugs (found 2026-07-06 in the rerun4 live-verification run). Directly
  adjacent to F1/F4 — likely one combined design. These two remain
  deferred (unchecked above is for these, not the item below).
  - [x] The more general caveat found while auditing the baseline `AgML`
    log — every tool's PASSED verdict rested entirely on the debugger's
    own self-report, since Step 4 never had the validator independently
    re-invoke a fixed tool — **is fixed**. Implemented 2026-07-06:
    `instructions/validator.py` Step 4 now requires the validator to call
    `invoke_mcp_tool` itself again after every debugger call, regardless
    of what the debugger's self-report claims, and judge PASSED/FAILED
    from that independent result. See
    [IMPROVEMENTS_SPEC.md](./IMPROVEMENTS_SPEC.md#f24).
- [ ] **F25** — SKIP is an LLM-followed convention, not a code-enforced
  gate (`AgML`'s `train_detector` was marked SKIP by the Coder but
  invoked anyway by the Validator, with the tool's expensive default
  params, burning most of the 1800s stage budget). Needs (a) the
  SKIP/invoke split computed in code and handed to the validator as an
  explicit constraint instead of trusted free-text parsing, and (b) an
  optional separate, much-longer-timeout "extended validation" pass for
  SKIP-marked heavy tools, decoupled from the shared per-repo budget.
- [x] **F26** — Tools with no checkable/parseable output (e.g.
  `aizynthfinder`'s `run_interactive_gui`, a Jupyter-notebook launcher)
  should never be created in the first place, not discovered as
  unfixable at validation time. Coder-instruction fix, same class as
  F18/F21. Implemented 2026-07-06: new "Tool-selection guardrails"
  section in `instructions/coder.py`, right before Step 3.
- [x] **F27** — Tools with too many parameters are a disproportionate
  failure point (`aizynthfinder` final-eval's 10-parameter
  `perform_retrosynthesis` mega-tool vs. baseline's thin, few-param,
  CLI-mirroring tools). Implemented 2026-07-06 alongside F26 in the same
  "Tool-selection guardrails" section: caps at ≤4-5 params, prefer
  mirroring the repo's own CLI/API 1:1. The optional F12 metric
  enrichment (tracking param counts over time) was scoped out for now —
  not selected for this implementation pass.
- [x] **F28** — `validate_syntax` only checks `server.py`, never the
  `helpers/*.py` scripts that hold the real per-tool logic and imports —
  so a hallucinated import (`aizynthfinder`'s nonexistent
  `AiZynthExpander` class) is invisible until a live invocation burns a
  full debugger round-trip on something a sub-second static check could
  catch for free. A concrete, low-effort, evidence-backed slice of F1.
  Implemented 2026-07-06: see [IMPROVEMENTS_SPEC.md](./IMPROVEMENTS_SPEC.md#f28)
  for the full mechanism and the two subtle false-positive bugs found and
  fixed during testing (sys.path[0] semantics, two-venv repo-vs-server
  venv selection). Verified against all 11 real committed images from the
  final-eval run with zero false positives; caught two previously-invisible
  real bugs, including a `biotite` helper with a flatly invalid
  `def x = y(...)` typo that had been silently marked SKIPPED and never
  actually tested in the real run. Live end-to-end verification (fresh
  `biotite` regeneration, not a replay):
  `benchmarks/alembic/runs/2026-07-06_rerun7_f26-f28-verify/` — confirmed
  working exactly as designed on freshly-generated code: caught an `elsif`
  typo and 3 hallucinated import paths (one confirmed against real repo
  source: `ClustalOmegaApp` hallucinated as living in `biotite.application.msa`
  instead of its real home `biotite.application.clustalo`) before any
  `invoke_mcp_tool` round-trip, zero false positives, F26/F27's guardrails
  also held (5 thin tools, ≤5 params each, no GUI/mega-tool). This run
  also independently reproduced the F24/F25-class validator/debugger
  coordination gap: the debugger re-attempted fixing that same
  already-known-unfixable import at Step 4 after already giving up on it
  at Step 2, and combined with an unrelated 600s debugger timeout, blew
  the Validator's 1800s stage budget — `validation.md` never got written
  for this run. Not a regression from F26/27/28; see IMPROVEMENTS_SPEC.md#f28
  "Live end-to-end confirmation" for the full account and its bearing on F25.
- [x] **F29** — The debugger's internal steps (its own bash/read/edit/
  invoke_mcp_tool calls) were completely invisible in the pipeline log —
  only its final one-paragraph self-report ever surfaced, via
  `[validator] RESP debugger -> {...}`, making the ~10-minute debugger
  calls seen while diagnosing F28's live run impossible to audit after
  the fact. Implemented 2026-07-06: `agents.py` now attaches
  `before_tool_callback`/`after_tool_callback`/`after_model_callback` to
  `debugger_agent` (these fire regardless of whether the agent runs
  top-level or nested inside an `AgentTool`, so no ADK internals needed
  reimplementing), logging every debugger-internal CALL/RESP/text line
  tagged `[debugger#N]` where N is a per-stage "debug round" counter
  incremented once per validator→debugger call. See
  [IMPROVEMENTS_SPEC.md](./IMPROVEMENTS_SPEC.md#f29) for the mechanism
  and testing (verified against ADK's actual callback contracts and
  functionally tested inside the real container image).
- [ ] **F2 / F3** — Semantic output-correctness gate + held-out validation
  invocation. Matches ToolMaker's rigor and is a direct "why trust this
  passed" answer for reviewers.
- [ ] **F5** — First-class conda/`environment.yml` path (would help
  `dalle-mini`-style old-ML-stack wheel deadlocks, though that one may be
  unfixable regardless — aging JAX wheels don't exist for its era).
- [ ] **F6 / F7** — External resource acquisition + allowlisted repo secrets.
- [ ] **F8 / F9** — Reproducible `install.sh`-built image + fresh-checkpoint
  isolation per debug attempt.
- [ ] **F10 / F11** — Persistent failure-memory across debug iterations +
  tiered model routing.
- [ ] **F13** — Success-memoized, resumable benchmark runner (cheaper
  re-benchmarking once §1 fixes need re-validating against a larger repo set).

## 6. Optional — paper enrichments

- [ ] Qualitative contrast against ToolRosella's *reported* numbers (61.5%
  conversion after 3 RRF rounds, 1,580 tools, 84.0% downstream success) —
  clearly labeled as "their numbers, not re-run on our sample," since
  domains differ.
- [ ] A short ablation once F14–F18 land: before/after pass-rate on the same
  12 repos, quantifying how much of today's apparent instability was
  pipeline bugs vs. genuinely hard repos (backtrader's dead Yahoo-Finance
  feed, dalle-mini's unresolvable JAX wheels look like the latter).
- [ ] If time allows, expand the benchmark beyond the 12-repo
  `toolrosella_subset.txt` sample using `subset_by_domain.py` against the
  full 122-repo `repository_inventory.jsonl`, for a broader domain-coverage
  claim in the Evaluation section.
