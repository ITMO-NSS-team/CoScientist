# Alembic benchmark — 2026-07-06 14:27

Repos processed: 2

| Repo | Time | Exit | Syntax | Tests | Tools (P/F/S) | Overall |
|---|---:|---:|---|---|---|---|
| AgML | 2978s | 0 | PASSED | FAILED — 8 passed, 4 failed | 1/3/0 | FAILED |
| BioSPPy | 2652s | 0 | PASSED | FAILED — 15 passed, 5 failed | 0/5/0 | FAILED |

## Per-repo details

### AgML
- URL: https://github.com/Project-AgML/AgML
- Duration: 2978.1s
- Exit code: 0
- Log: benchmarks/alembic/runs/2026-07-06_rerun4_f22-f23-verify/logs/AgML.log
  - download_dataset: PASSED
  - train_model_on_dataset: FAILED
  - run_inference_on_dataset: FAILED
  - evaluate_model_on_dataset: FAILED

### BioSPPy
- URL: https://github.com/scientisst/BioSPPy
- Duration: 2652.0s
- Exit code: 0
- Log: benchmarks/alembic/runs/2026-07-06_rerun4_f22-f23-verify/logs/BioSPPy.log
  - quality_ecg: FAILED
  - eda: FAILED
  - ecg: FAILED
  - ppg: FAILED
  - eeg: FAILED
