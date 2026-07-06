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
- [ ] **F22** *(new, found during rerun3, deferred — not blocking the
  2026-07-10 submission)* — A rare (1 of ~15 runs so far) OpenRouter/
  LiteLLM fault: an upstream provider error surfaces as an unmapped
  `finish_reason: "error"`, which `litellm`'s `map_finish_reason()`
  silently defaults to `"stop"` instead of raising — so ADK treats a
  stub/garbage response as a normal completed turn. Cost `BioSPPy`'s
  Coder stage ~8 of its 25 minutes across two occurrences, recovered
  only by accident via an unrelated guard-retry nudge; a 3rd occurrence
  in the same stage would have exhausted the guard-retry budget and
  killed the stage outright. Not caught by F17 (whose retry-once is
  scoped only to Debugger calls, and only catches *raised* exceptions —
  this fault never raises). Full root cause, exact code location
  (`litellm/litellm_core_utils/core_helpers.py:107`), and a concrete
  3-step implementation path are written up in
  [IMPROVEMENTS_SPEC.md](./IMPROVEMENTS_SPEC.md#f22) for whenever this
  gets picked up post-submission.
- [ ] **F23** *(new, found during rerun3, deferred)* — `AgML`'s Validator
  stage produced no report at all this run (vs. rerun2's clean 0/2/3
  verdicts on the same repo) because F16's 600s per-debugger-call timeout
  appears **not to have fired**: a debugger call sent at `09:34:47` (to
  re-investigate a missing `ensemble-boxes` package) produced zero log
  output for ~20 minutes — not even F16's own "call timed out after 600s"
  message — before the *outer* 1800s stage timeout finally fired, itself
  ~2 minutes late. Root cause (inferred from timing, not yet confirmed
  against ADK internals): every individual subprocess call already has
  its own bound (15s–900s depending on the tool), but if the debugger
  sub-agent chains several of those within one turn (e.g. a few
  sequential install attempts) before yielding control, `asyncio.wait_for`
  can't actually cancel until that chain finishes — so the 600s ceiling
  is real on paper but defeatable in practice. This is a genuine gap in
  F16, not caused by F19/F20/F21 (which don't touch timeouts/retries) and
  not the same fault as F22 (which is an LLM/provider issue, not a
  subprocess-chaining one). Full evidence and fix options in
  [IMPROVEMENTS_SPEC.md](./IMPROVEMENTS_SPEC.md#f23), deferred
  post-submission like F22.
- [ ] **F12** — Structured per-run JSON metrics + failure taxonomy. Needed
  so the Evaluation section reports a clean table instead of hand-parsed
  logs (which is how this TODO's own failure analysis had to be done).

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

## 2. Must-have — re-run the benchmark, write the Evaluation section

- [ ] Re-run `benchmarks/alembic/run_benchmark.py` on at least the current
  12-repo set (ideally the full `toolrosella_subset.txt`, 12 more repos)
  after §1's fixes land, and diff against
  `benchmarks/alembic/runs/2026-06-30_baseline/summary.md` to confirm the
  SKIP rows resolve to real PASS/FAIL.
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
