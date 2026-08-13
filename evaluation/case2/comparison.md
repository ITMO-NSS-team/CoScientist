# Case 2 — checkpoint plan comparison

Evaluation mode: `plan_checkpoint_only`. Both runs stopped at the fail-closed
`T0_before_hitl` review boundary. No experiment task, CoderAgent, model training,
GOLEM process, or docking job was started.

| Item | Question only | With refinements |
|---|---:|---:|
| DB status | `awaiting_review` | `awaiting_review` |
| Plan revision | 1 | 1 |
| Hypotheses in plan | 3 | 5 |
| Tasks | 1 | 5 |
| Estimated duration | 240 min | 600 min |
| Routes | 1 × `coder` | 5 × `coder` |
| Tool inventory | empty/unavailable | empty/unavailable |

## Question-only plan

The planner collapsed the study into one end-to-end task and split the claim
into three endpoint hypotheses: Docking, SA, and validity. The single task tests
all three against a static Transformer baseline. It proposes 1,000 samples per
model, 50 epochs, and a placeholder `protein_target.pdb`.

Important omissions relative to the refined statement: GOLEM, GSK-3β, MCPhub,
CVAE/latent-space design, 10,000 trajectories, novelty, trajectory
informativeness/length ablations, and KL ablation.

## Refined plan

The planner expanded the call into five tasks:

1. Generate at least 10,000 GOLEM trajectories and a static control set.
2. Train trajectory/static CVAE variants.
3. Evaluate the joint threshold metric and novelty.
4. Analyze step-size informativeness and trajectory length.
5. Run the KL-regularization ablation.

It explicitly names GOLEM, MCPhub, CVAE, GSK-3β docking, novelty, and the KL
ablation. Thus the refinements materially changed both the content and topology
of the plan: 1 task became a 5-stage pipeline.

## Quality findings

- The refined tasks cover the user's intended H1–H5 substantially better.
- The generated *hypothesis statements* drifted from the supplied H1–H5
  (e.g. H5 became “diverse successful pathways”), while EXP-5 still uses H5 for
  the KL ablation. This is an internal label/semantics inconsistency.
- Every task fell back to `coder`, because local retrieval failed with
  `Embedder not initialized` and public MCP discovery returned zero candidates.
- The plans are planning artifacts only. They do not constitute scientific
  evidence for or against the hypotheses.

## Durable records

- SQLite table: `pipeline_evaluation_runs`
- Base run: `case2_question_only_checkpoint_20260812T155017Z_619b2707`
- Refined run: `case2_with_refinements_checkpoint_20260812T195528Z_16c16683`
- Both T0 ZIP bundles passed `unzip -t`; SQLite passed `PRAGMA integrity_check`.
