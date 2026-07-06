# Alembic benchmark — 2026-07-06 18:27

Repos processed: 12

| Repo | Time | Exit | Syntax | Tests | Tools (P/F/S) | Overall |
|---|---:|---:|---|---|---|---|
| AgML | 3164s | 0 | — | — | 0/0/0 | ERROR — validation.md not readable |
| Analyze-stroke | 0s | — | — | — | 0/0/0 | N/A — repo unreachable: ERROR: Repository not found. |
| BioSPPy | 2326s | 0 | PASSED | PASSED — 20 passed, 0 failed | 0/4/1 | FAILED |
| aizynthfinder | 2171s | 0 | PASSED | PASSED — 13 passed, 0 failed | 0/2/1 | FAILED (2 tools failed at invocation stage; debugger could not resolve issues within time limit) |
| ase | 2114s | 0 | PASSED | FAILED — 16 passed, 4 failed | 2/0/3 | FAILED (Tests stage failed; debugger could not resolve dependency installation within time limits) |
| astronomy | 1762s | 0 | PASSED | PASSED — 15 passed, 0 failed | 5/0/0 | PASSED |
| astropy | 1886s | 0 | PASSED | PASSED — 18 passed, 0 failed | 6/0/0 | PASSED |
| auto-sklearn | 2064s | 0 | PASSED | FAILED — 0 passed, 15 failed | 0/5/0 | FAILED (tests failed, multiple tool invocations failed) |
| backtrader | 1282s | 0 | PASSED | FAILED — 1 passed, 22 failed | 1/4/0 | FAILED |
| biopython | 1586s | 0 | PASSED | PASSED — 21 passed, 0 failed | 4/1/0 | FAILED — Tool invocations failed: fetch_genbank_record |
| biotite | 1485s | 0 | PASSED | PASSED — 12 passed, 0 failed | 2/1/1 | FAILED (1 tool invocation failed) |
| dalle-mini | 1792s | 0 | PASSED | PASSED — 6 passed, 0 failed | 0/1/0 | FAILED |

## Per-repo details

### AgML
- URL: https://github.com/Project-AgML/AgML
- Duration: 3163.8s
- Exit code: 0
- Log: benchmarks/alembic/runs/2026-07-06_final-eval/logs/AgML.log
- validation.md not readable

### Analyze-stroke
- URL: https://github.com/ghh1125/Analyze-stroke
- Duration: 0s
- Exit code: N/A — pipeline not run
- Log: —
- repo unreachable: ERROR: Repository not found.

### BioSPPy
- URL: https://github.com/scientisst/BioSPPy
- Duration: 2325.8s
- Exit code: 0
- Log: benchmarks/alembic/runs/2026-07-06_final-eval/logs/BioSPPy.log
  - ecg_processing: FAILED
  - eda_processing: FAILED
  - emg_onset_detection: FAILED
  - hrv_feature_extraction: FAILED
  - ppg_pulse_detection: SKIPPED

### aizynthfinder
- URL: https://github.com/MolecularAI/aizynthfinder
- Duration: 2170.7s
- Exit code: 0
- Log: benchmarks/alembic/runs/2026-07-06_final-eval/logs/aizynthfinder.log
  - perform_retrosynthesis: FAILED
  - expand_molecule: FAILED
  - download_data: SKIPPED

### ase
- URL: https://github.com/yfyh2013/ase
- Duration: 2114.2s
- Exit code: 0
- Log: benchmarks/alembic/runs/2026-07-06_final-eval/logs/ase.log
  - build_structure: PASSED
  - optimize_geometry: SKIPPED
  - interpolate_neb_path: SKIPPED
  - run_molecular_dynamics: SKIPPED
  - query_database: PASSED

### astronomy
- URL: https://github.com/cosinekitty/astronomy
- Duration: 1761.5s
- Exit code: 0
- Log: benchmarks/alembic/runs/2026-07-06_final-eval/logs/astronomy.log
  - moon_phase: PASSED
  - find_rise_set: PASSED
  - predict_lunar_eclipses: PASSED
  - calculate_jupiter_moons: PASSED
  - get_seasons: PASSED

### astropy
- URL: https://github.com/astropy/astropy
- Duration: 1885.6s
- Exit code: 0
- Log: benchmarks/alembic/runs/2026-07-06_final-eval/logs/astropy.log
  - fits2bitmap: PASSED
  - fitsheader: PASSED
  - query_simbad: PASSED
  - convert_coordinates: PASSED
  - compute_cosmology: PASSED
  - fit_table: PASSED

### auto-sklearn
- URL: https://github.com/automl/auto-sklearn
- Duration: 2063.6s
- Exit code: 0
- Log: benchmarks/alembic/runs/2026-07-06_final-eval/logs/auto-sklearn.log
  - run_classification: FAILED
  - run_regression: FAILED
  - get_ensemble_info: FAILED
  - inspect_pipeline_components: FAILED
  - implement_custom_optimization: FAILED

### backtrader
- URL: https://github.com/mementum/backtrader
- Duration: 1282.5s
- Exit code: 0
- Log: benchmarks/alembic/runs/2026-07-06_final-eval/logs/backtrader.log
  - backtest_sma_crossover: FAILED
  - optimize_sma_strategy: FAILED
  - run_pairs_trade: FAILED
  - plot_strategy_results: FAILED
  - calculate_indicator: PASSED

### biopython
- URL: https://github.com/biopython/biopython
- Duration: 1586.3s
- Exit code: 0
- Log: benchmarks/alembic/runs/2026-07-06_final-eval/logs/biopython.log
  - bio_search: PASSED
  - read_fasta_sequences: PASSED
  - find_restriction_sites: PASSED
  - fetch_genbank_record: FAILED
  - parse_blast_results: PASSED

### biotite
- URL: https://github.com/biotite-dev/biotite
- Duration: 1485.1s
- Exit code: 0
- Log: benchmarks/alembic/runs/2026-07-06_final-eval/logs/biotite.log
  - align_sequences: PASSED
  - fetch_structure: PASSED
  - to_protein_blocks: FAILED
  - run_blast: SKIPPED

### dalle-mini
- URL: https://github.com/borisdayma/dalle-mini
- Duration: 1792.3s
- Exit code: 0
- Log: benchmarks/alembic/runs/2026-07-06_final-eval/logs/dalle-mini.log
  - generate_images: FAILED

## Aggregate metrics (F12)

Repos with metrics.json: 11/12

**Stage completion (completed/attempted):**

- explorer: 11/11
- environment: 11/11
- coder: 11/11
- validator: 10/11

**Failure taxonomy (tool-invocation failures across all repos):**

- Import: 8
- ValueError: 5
- ModuleNotFound: 5
- TypeError: 4
- DebuggerTimeout: 4
- FileNotFound: 3
- Runtime: 3
- NameError: 3
- Syntax: 1
- AttributeError: 1

- Guard retries (write_report/venv nudges) total: 15
- Transient provider-fault retries (F22) total: 0
