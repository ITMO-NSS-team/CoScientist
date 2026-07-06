# Alembic benchmark — 2026-07-06 12:07

Repos processed: 14

| Repo | Time | Exit | Syntax | Tests | Tools (P/F/S) | Overall |
|---|---:|---:|---|---|---|---|
| CONCH | 2107s | 0 | FAILED | PASSED — 20 passed, 0 failed | 0/4/1 | FAILED — All attempted tool invocations failed due to missing model checkpoint file. The model `pytorch_model.bin` is gated on Hugging Face and requires authentication, which cannot be automated. This is a data/environment issue outside the scope of code fixes. |
| MUSK | 2367s | 0 | PASSED | FAILED — 11 passed, 1 failed | 1/3/1 | FAILED |
| MedSAM | 1975s | 0 | PASSED | PASSED — 13 passed, 0 failed | 0/1/3 | FAILED — Syntax and tests passed, but `run_inference` invocation failed due to an unresolvable environment mounting issue preventing dependency installation. |
| MedSSS | 1776s | 0 | PASSED | PASSED — 14 passed, 0 failed | 1/1/3 | FAILED (tool invocation: compare_performance) |
| ModernBERT | 2303s | 0 | PASSED | PASSED — 22 passed, 0 failed | 0/3/2 | FAILED (invocation stage — 3 tools failed with hard environment or design issues) |
| PathFinderCRC | 983s | 0 | PASSED | PASSED — 11 passed, 0 failed | 3/0/0 | PASSED |
| RETFound_MAE | 2166s | 0 | PASSED | PASSED — 10 passed, 0 failed | 0/3/0 | FAILED — Test suite passed, but all tool invocations failed due to environment and access restrictions. |
| STAMP | 1549s | 0 | PASSED | PASSED — 10 passed, 0 failed | 0/0/5 | PASSED |
| TabPFN | 2019s | 0 | PASSED | FAILED — 28 passed, 1 failed | 0/3/5 | FAILED |
| UNI | 3275s | 0 | — | — | 0/0/0 | — |
| cytopus | 3193s | 0 | PASSED | PASSED — 24 passed, 0 failed | 3/1/1 | FAILED — Test stage invocation for `construct_kb` failed. All other stages passed. |
| esm | 3449s | 0 | — | — | 0/0/0 | — |
| flowmap | 2955s | 0 | PASSED | PASSED — 17 passed, 0 failed | 0/4/0 | FAILED (All tool invocations failed) |
| nnUNet | 2646s | 0 | PASSED | FAILED — 6 passed, 4 failed | 1/4/0 | FAILED |

## Per-repo details

### CONCH
- URL: https://github.com/mahmoodlab/CONCH
- Duration: 2106.7s
- Exit code: 0
- Log: /home/server2/denis/CoScientist/alembic_bench_logs/CONCH.log
  - load_model: FAILED
  - encode_image: FAILED
  - zero_shot_classification: FAILED
  - encode_text: FAILED
  - mi_zero_shot_classification: SKIPPED

### MUSK
- URL: https://github.com/lilab-stanford/MUSK
- Duration: 2366.8s
- Exit code: 0
- Log: /home/server2/denis/CoScientist/alembic_bench_logs/MUSK.log
  - extract_image_embeddings: FAILED
  - extract_text_embeddings: FAILED
  - run_zeroshot_classification: PASSED
  - evaluate_benchmark: SKIPPED
  - compute_similarity: FAILED

### MedSAM
- URL: https://github.com/bowang-lab/MedSAM
- Duration: 1975.2s
- Exit code: 0
- Log: /home/server2/denis/CoScientist/alembic_bench_logs/MedSAM.log
  - run_inference: FAILED
  - train_model: SKIPPED
  - convert_checkpoint: SKIPPED
  - launch_gui: SKIPPED

### MedSSS
- URL: https://github.com/pixas/MedSSS
- Duration: 1775.6s
- Exit code: 0
- Log: /home/server2/denis/CoScientist/alembic_bench_logs/MedSSS.log
  - generate_reasoning_trajectories: SKIPPED
  - train_policy_model: SKIPPED
  - train_prm_model: SKIPPED
  - evaluate_model: PASSED
  - compare_performance: FAILED

### ModernBERT
- URL: https://github.com/AnswerDotAI/ModernBERT
- Duration: 2303.2s
- Exit code: 0
- Log: /home/server2/denis/CoScientist/alembic_bench_logs/ModernBERT.log
  - train_modernbert: FAILED
  - finetune_glue: FAILED
  - evaluate_superglue: SKIPPED
  - train_embedding_dpr: FAILED
  - evaluate_embedding: SKIPPED

### PathFinderCRC
- URL: https://github.com/LiangJunhao-THU/PathFinderCRC
- Duration: 983.2s
- Exit code: 0
- Log: /home/server2/denis/CoScientist/alembic_bench_logs/PathFinderCRC.log
  - calculate_tissue_fraction: PASSED
  - generate_combined_dataset: PASSED
  - run_survival_analysis: PASSED

### RETFound_MAE
- URL: https://github.com/rmaphoh/RETFound_MAE
- Duration: 2165.7s
- Exit code: 0
- Log: /home/server2/denis/CoScientist/alembic_bench_logs/RETFound_MAE.log
  - fine_tune_retfound_model: FAILED
  - extract_latent_features: FAILED
  - evaluate_trained_model: FAILED

### STAMP
- URL: https://github.com/KatherLab/STAMP
- Duration: 1548.9s
- Exit code: 0
- Log: /home/server2/denis/CoScientist/alembic_bench_logs/STAMP.log
  - preprocess_stamp: SKIPPED
  - train_stamp: SKIPPED
  - crossval_stamp: SKIPPED
  - deploy_stamp: SKIPPED
  - statistics_stamp: SKIPPED

### TabPFN
- URL: https://github.com/PriorLabs/TabPFN
- Duration: 2019.2s
- Exit code: 0
- Log: /home/server2/denis/CoScientist/alembic_bench_logs/TabPFN.log
  - train_classifier: FAILED
  - create_classifier_version: FAILED
  - create_regressor_version: FAILED
  - train_regressor: SKIPPED
  - predict_classifier: SKIPPED
  - predict_regressor: SKIPPED
  - save_fitted_model: SKIPPED
  - load_fitted_model: SKIPPED

### UNI
- URL: https://github.com/mahmoodlab/UNI
- Duration: 3274.6s
- Exit code: 0
- Log: /home/server2/denis/CoScientist/alembic_bench_logs/UNI.log
- validation.md not readable

### cytopus
- URL: https://github.com/wallet-maker/cytopus
- Duration: 3192.6s
- Exit code: 0
- Log: /home/server2/denis/CoScientist/alembic_bench_logs/cytopus.log
  - query_genesets: PASSED
  - label_genes: PASSED
  - export_gmt: PASSED
  - construct_kb: FAILED
  - annotate_hierarchy: SKIPPED

### esm
- URL: https://github.com/facebookresearch/esm
- Duration: 3449.4s
- Exit code: 0
- Log: /home/server2/denis/CoScientist/alembic_bench_logs/esm.log
- validation.md not readable

### flowmap
- URL: https://github.com/dcharatan/flowmap
- Duration: 2955.2s
- Exit code: 0
- Log: /home/server2/denis/CoScientist/alembic_bench_logs/flowmap.log
  - run_reconstruction: FAILED
  - run_ablation_study: FAILED
  - export_reconstruction_to_colmap: FAILED
  - pretrain_flow_model: FAILED

### nnUNet
- URL: https://github.com/MIC-DKFZ/nnUNet
- Duration: 2646.4s
- Exit code: 0
- Log: /home/server2/denis/CoScientist/alembic_bench_logs/nnUNet.log
  - run_inference: FAILED
  - run_training: FAILED
  - ensemble_predictions: FAILED
  - apply_postprocessing: FAILED
  - plan_and_preprocess: PASSED
