# Case 2 — checkpoint plan evaluation

This directory contains two plan-only evaluations of the molecular-generation
case:

- `question_only.txt` — the original research question;
- `question_with_refinements.txt` — the same question with the detailed setup,
  target metric, and H1–H5.

The comparison is in `comparison.md`. Durable run state is stored in the
`pipeline_evaluation_runs` table of `evaluation.sqlite3`.

## Final runs

| Variant | Run ID | Status | T0 bundle |
|---|---|---|---|
| Question only | `case2_question_only_checkpoint_20260812T155017Z_619b2707` | `awaiting_review` | `ckpt_20260812T155017_T0_before_hitl_619b2707.zip` |
| With refinements | `case2_with_refinements_checkpoint_20260812T195528Z_16c16683` | `awaiting_review` | `ckpt_20260812T195528_T0_before_hitl_16c16683.zip` |

Both runs used `plan_checkpoint_only` mode and stopped at the fail-closed
`T0_before_hitl` boundary. No experimental task, model training, GOLEM process,
or docking job was executed. Earlier failed attempts remain in SQLite as an
audit trail, but only the two final T0 bundles are versioned.

## Result

Adding the refinements materially changed the generated plan: the original
question produced 3 hypotheses and one end-to-end coder task (estimated 240
minutes), while the refined prompt produced 5 hypotheses and a five-task
pipeline (estimated 600 minutes). See `comparison.md` for the omissions,
routing results, and hypothesis-label inconsistency found during review.

At evaluation time the tool registry contained no usable molecular tools, so
all planned tasks fell back to the `coder` route. These records assess plan
generation only and are not scientific evidence for or against H1–H5.

## Reproduction

The runner requires an integration checkout that contains both the experiment
module/profile from PR #305 (`CoScientist/agents/experiments.yaml`) and the
checkpoint implementation from this branch. It reads credentials with
`python-dotenv`; do not commit the env file.

```bash
uv run python scripts/run_case2_evaluation.py \
  --variant question_only \
  --prompt evaluation/case2/question_only.txt \
  --db evaluation/case2/evaluation.sqlite3 \
  --trace evaluation/case2/traces/question_only_checkpoint.log \
  --checkpoint-dir evaluation/case2/checkpoints \
  --env-file .env \
  --postgres-port 5433

uv run python scripts/run_case2_evaluation.py \
  --variant with_refinements \
  --prompt evaluation/case2/question_with_refinements.txt \
  --db evaluation/case2/evaluation.sqlite3 \
  --trace evaluation/case2/traces/with_refinements_checkpoint.log \
  --checkpoint-dir evaluation/case2/checkpoints \
  --env-file .env \
  --postgres-port 5433
```

Each invocation appends a uniquely identified row. Agent trace logs are kept
out of version control; the SQLite record contains the trace captured for each
run.

## Integrity checks

```bash
sqlite3 -readonly evaluation/case2/evaluation.sqlite3 'PRAGMA integrity_check;'
unzip -t evaluation/case2/checkpoints/<run-directory>/<bundle>.zip
```
