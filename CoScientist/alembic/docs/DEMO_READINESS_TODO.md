# Demo-paper readiness TODO

Submission deadline: **2026-07-10, 11:59 PM UTC-12h** — roughly 4 days out from
when this was written. Sequencing below is chosen so that a hard stop after
any "must-have" block still leaves a submittable paper.

Sources: [IMPROVEMENTS_SPEC.md](./IMPROVEMENTS_SPEC.md) (F1–F18),
[benchmarks/alembic/base_results.md](../../../benchmarks/alembic/base_results.md)
+ `alembic_bench_logs/*.log` (12-repo run, 2026-06-30), and a review of
[ToolArena](https://github.com/KatherLab/ToolArena) (TM-Bench's harness repo)
at commit `0b275a266e01b38f13a8e7041489f5762887cd65`.

---

## 1. Must-have — system fixes (blocks trustworthy Evaluation numbers)

These were caught directly in the bench logs, not inferred — each one is
currently *understating* Alembic's real pass rate, so fixing them before the
next benchmark run matters more than any comparison-driven feature.

- [ ] **F14** — Validator blanket-skips all tools when *any* one test fails
  (caused `auto-sklearn` and `biotite`'s 0/0/4 SKIP rows). Make per-tool
  invocation independent of overall `run_tests` pass/fail.
- [ ] **F15** — Debugger request must always include `repo_url` (a missing
  URL turned `biotite`'s 1-character `SyntaxError` into a total stage
  failure).
- [ ] **F16** — Per-debugger-call timeout separate from the stage timeout
  (`biopython` burned its whole 30-min Validator budget on one stuck call
  and produced no report at all).
- [ ] **F17** — Retry once on transient LLM/API error (empty-body
  `JSONDecodeError` from OpenRouter/LiteLLM burned a `BioSPPy` debugger
  attempt for nothing).
- [ ] **F18** — Explorer must propose realistically-sized sample inputs
  (toy 5-element arrays caused 4 of 5 `BioSPPy` tool failures against
  filters requiring `padlen≈4500`/`≥5s` segments — the tools are fine, the
  samples aren't).
- [ ] **F12** — Structured per-run JSON metrics + failure taxonomy. Needed
  so the Evaluation section reports a clean table instead of hand-parsed
  logs (which is how this TODO's own failure analysis had to be done).

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
