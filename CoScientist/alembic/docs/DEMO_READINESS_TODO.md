# Demo-paper readiness TODO

Submission deadline: **2026-07-10, 11:59 PM UTC-12h** — roughly 4 days out from
when this was written. Sequencing below is chosen so that a hard stop after
any "must-have" block still leaves a submittable paper.

Sources: [IMPROVEMENTS_SPEC.md](./IMPROVEMENTS_SPEC.md) (F1–F19),
[benchmarks/alembic/base_results.md](../../../benchmarks/alembic/base_results.md)
+ `alembic_bench_logs/*.log` (12-repo baseline run, 2026-06-30),
[benchmarks/alembic/rerun_f14_f18.md](../../../benchmarks/alembic/rerun_f14_f18.md)
+ `alembic_bench_logs_rerun/*.log` (same 12 repos, re-run 2026-07-06 after
F14–F18), and a review of [ToolArena](https://github.com/KatherLab/ToolArena)
(TM-Bench's harness repo) at commit `0b275a266e01b38f13a8e7041489f5762887cd65`.

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
- [x] **F18** — Explorer must propose realistically-sized sample inputs.
  **First patch (Explorer-only) did NOT work** — rerun `BioSPPy` still used
  `signal=[0.1, 0.2, 0.3]`-style arrays and failed the same way. Root cause:
  the concrete `samples:` block `invoke_mcp_tool` runs is written by the
  **Coder**, not the Explorer, and `instructions/coder.py` had its own,
  directly conflicting rule — *"Use the most minimal args you can."*
  Re-patched `coder.py` to distinguish "cheap to run" from "smaller than
  the function's precondition." **Not yet re-benchmarked after this second
  patch** — do that before checking this box for real.
- [x] **F19** *(new, found during the rerun, not in the original bench)* —
  `_UnknownToolStub` (the hallucinated-tool-name stub in `main.py`) was
  missing a `.description` attribute, causing `AttributeError:
  '_UnknownToolStub' object has no attribute 'description'` when ADK
  introspected it — observed in `backtrader`, masked there by F17's retry
  succeeding on the next attempt, but a wasted attempt otherwise. One-line
  fix applied (`main.py`).
- [ ] **F12** — Structured per-run JSON metrics + failure taxonomy. Needed
  so the Evaluation section reports a clean table instead of hand-parsed
  logs (which is how this TODO's own failure analysis had to be done).

## 1a. Rerun results (2026-07-06, same 12 repos as `base_results.md`)

Headline: `astronomy` and `astropy`'s tool-invocation layer newly PASS
outright (5/5, and tools-only-PASSED-despite-test-failure respectively);
`auto-sklearn` and `biotite` went from blanket-SKIP noise to real,
individually-diagnosable verdicts (see F14 above) — a materially more
trustworthy signal than the baseline even where the overall row is still
FAILED. Caveat: each run regenerates server code from scratch via the LLM,
so this is not a controlled code-for-code diff — some swings are ordinary
LLM-generation variance, not caused by F14–F19.

New/unrelated issues surfaced by the rerun (not F14–F19's scope, logged for
later): `astropy`'s generated test file never imports the tool functions it
tests (`NameError` on all 18 tests, while direct invocation still passes —
a Coder bug); `aizynthfinder`'s `create_interactive_app` helper emits no
stdout JSON; `backtrader`'s `backtest_strategy` has a real `IndexError` in
SMA warm-up logic; `ase`'s `optimize_geometry` hit a `UnicodeDecodeError`
reading a binary `.traj` file as text. `dalle-mini` and `Analyze-stroke`
reproduce their exact baseline root causes (unresolvable jaxlib/Python 3.8
deadlock; inaccessible git remote) unchanged.

## 2. Must-have — re-run the benchmark, write the Evaluation section

- [ ] Re-run `benchmarks/alembic/run_benchmark.py` on at least the current
  12-repo set (ideally the full `toolrosella_subset.txt`, 12 more repos)
  after §1's fixes land, and diff against `base_results.md` to confirm the
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
