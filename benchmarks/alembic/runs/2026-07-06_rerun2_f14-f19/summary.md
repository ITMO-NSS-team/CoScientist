# Alembic benchmark — 2026-07-06 12:00

Repos processed: 12

| Repo | Time | Exit | Syntax | Tests | Tools (P/F/S) | Overall |
|---|---:|---:|---|---|---|---|
| AgML | 2359s | 0 | PASSED | PASSED — 16 passed, 0 failed | 0/2/3 | FAILED |
| Analyze-stroke | 0s | — | — | — | 0/0/0 | N/A — repo unreachable: ERROR: Repository not found. |
| BioSPPy | 2599s | 0 | — | — | 0/0/0 | ERROR — validation.md not readable |
| aizynthfinder | 1719s | 0 | PASSED | PASSED — 21 passed, 0 failed | 1/4/0 | FAILED — Syntax and tests passed, but 4/5 tools failed on end-to-end invocation. |
| ase | 1313s | 0 | PASSED | PASSED — 15 passed, 0 failed | 1/1/1 | FAILED |
| astronomy | 2322s | 0 | PASSED | PASSED — 25 passed, 0 failed | 7/0/0 | PASSED |
| astropy | 1970s | 0 | PASSED | PASSED — 20 passed, 0 failed | 3/2/0 | FAILED (fitscheck and fits2bitmap tools not fully functional) |
| auto-sklearn | 1969s | 0 | PASSED | FAILED — 8 passed, 4 failed | 0/2/2 | FAILED |
| backtrader | 1656s | 0 | PASSED | PASSED — 11 passed, 0 failed | 0/4/0 | FAILED (all 4 tools failed at runtime due to HTTP compression issue) |
| biopython | 2025s | 0 | PASSED | PASSED — 19 passed, 0 failed | 2/3/0 | FAILED (run_blast, parse_genbank, and calculate_protein_features tools failed during invocation) |
| biotite | 1301s | 0 | PASSED | PASSED — 12 passed, 0 failed | 1/4/1 | FAILED — Syntax & Imports: PASSED, Tests: PASSED, but 4 out of 5 tools failed during invocation due to missing input files. |
| dalle-mini | 1200s | 0 | PASSED | FAILED — 14 passed, 1 failed | 0/3/0 | FAILED |

## Per-repo details

### AgML
- URL: https://github.com/Project-AgML/AgML
- Duration: 2359.1s
- Exit code: 0
- Log: logs/AgML.log
  - load_and_preprocess_dataset: FAILED
  - train_detection_model: FAILED
  - evaluate_detection_model: SKIPPED
  - visualize_data: SKIPPED
  - export_to_pytorch_loader: SKIPPED

### Analyze-stroke
- URL: https://github.com/ghh1125/Analyze-stroke
- Duration: 0s
- Exit code: N/A — pipeline not run
- Log: —
- repo unreachable: ERROR: Repository not found.

### BioSPPy
- URL: https://github.com/scientisst/BioSPPy
- Duration: 2598.9s
- Exit code: 0
- Log: logs/BioSPPy.log
- validation.md not readable

### aizynthfinder
- URL: https://github.com/MolecularAI/aizynthfinder
- Duration: 1718.8s
- Exit code: 0
- Log: logs/aizynthfinder.log
  - predict_single_route: FAILED
  - predict_batch_routes: FAILED
  - run_interactive_gui: FAILED
  - expand_molecule: FAILED
  - download_public_data: PASSED

### ase
- URL: https://github.com/yfyh2013/ase
- Duration: 1312.7s
- Exit code: 0
- Log: logs/ase.log
  - build_structure: PASSED
  - run_calculation: FAILED
  - get_system_info: SKIPPED

### astronomy
- URL: https://github.com/cosinekitty/astronomy
- Duration: 2321.6s
- Exit code: 0
- Log: logs/astronomy.log
  - equator: PASSED
  - illumination: PASSED
  - seasons: PASSED
  - moon_phase: PASSED
  - lunar_eclipse: PASSED
  - rise_set: PASSED
  - geo_vector: PASSED

### astropy
- URL: https://github.com/astropy/astropy
- Duration: 1969.9s
- Exit code: 0
- Log: logs/astropy.log
  - fitsinfo: PASSED
  - fitscheck: FAILED
  - fitsdiff: PASSED
  - fitsheader: PASSED
  - fits2bitmap: FAILED

### auto-sklearn
- URL: https://github.com/automl/auto-sklearn
- Duration: 1969.0s
- Exit code: 0
- Log: logs/auto-sklearn.log
  - fit_classifier: FAILED
  - fit_regressor: FAILED
  - show_models: SKIPPED
  - performance_over_time: SKIPPED

### backtrader
- URL: https://github.com/mementum/backtrader
- Duration: 1656.5s
- Exit code: 0
- Log: logs/backtrader.log
  - backtest_sma_crossover: FAILED
  - optimize_sma_params: FAILED
  - generate_pyfolio_report: FAILED
  - backtest_with_commission: FAILED

### biopython
- URL: https://github.com/biopython/biopython
- Duration: 2025.1s
- Exit code: 0
- Log: logs/biopython.log
  - fetch_entrez: PASSED
  - run_blast: FAILED
  - parse_genbank: FAILED
  - calculate_protein_features: FAILED
  - generate_alignment: PASSED

### biotite
- URL: https://github.com/biotite-dev/biotite
- Duration: 1301.3s
- Exit code: 0
- Log: logs/biotite.log
  - fetch_sequences_from_entrez: PASSED
  - perform_sequence_alignment: FAILED
  - read_structure_and_extract_assembly: FAILED
  - detect_disulfide_bonds: SKIPPED
  - calculate_secondary_structure: FAILED
  - superimpose_homologous_structures: FAILED

### dalle-mini
- URL: https://github.com/borisdayma/dalle-mini
- Duration: 1200.5s
- Exit code: 0
- Log: logs/dalle-mini.log
  - generate_images: FAILED
  - process_text: FAILED
  - rank_images: FAILED
