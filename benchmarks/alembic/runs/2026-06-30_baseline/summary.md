cat /var/essdata/CoScientist/alembic_bench.md
# Alembic benchmark — 2026-06-30 15:32

Repos processed: 12

| Repo | Time | Exit | Syntax | Tests | Tools (P/F/S) | Overall |
|---|---:|---:|---|---|---|---|
| AgML | 1970s | 0 | PASSED | PASSED — 15 passed, 0 failed | 5/0/0 | PASSED |
| Analyze-stroke | 135s | 0 | — | — | 0/0/0 | — |
| BioSPPy | 824s | 0 | PASSED | FAILED — 15 passed, 5 failed | 1/4/0 | FAILED |
| aizynthfinder | 1864s | 0 | PASSED | PASSED — 12 passed, 0 failed | 3/0/0 | PASSED |
| ase | 714s | 0 | PASSED | PASSED — 29 passed, 0 failed | 4/0/1 | PASSED |
| astronomy | 1287s | 0 | PASSED | PASSED — 22 passed, 0 failed | 5/0/0 | PASSED |
| astropy | 1302s | 0 | PASSED | PASSED — 20 passed, 0 failed | 4/0/1 | PASSED |
| auto-sklearn | 1435s | 0 | PASSED | FAILED — 17 passed, 1 failed | 0/0/4 | FAILED |
| backtrader | 797s | 0 | PASSED | PASSED — 12 passed, 0 failed | 2/2/1 | FAILED (Tool invocations: run_sma_crossover, optimize_sma_range) |
| biopython | 2313s | 0 | — | — | 0/0/0 | — |
| biotite | 689s | 0 | PASSED | FAILED — 0 passed, 1 error | 0/0/4 | FAILED (tests stage failed, debugger could not resolve the issue) |
| dalle-mini | 538s | 0 | PASSED | PASSED — 12 passed, 0 failed | 0/1/2 | FAILED — Tool `generate_images` failed due to unsatisfiable dependency (Python version mismatch). |

## Per-repo details

### AgML
- URL: https://github.com/Project-AgML/AgML
- Duration: 1970.5s
- Exit code: 0
- Log: logs/AgML.log
  - load_dataset: PASSED
  - load_detection_model: PASSED
  - evaluate_detection_model: PASSED
  - train_detection_model: PASSED
  - visualize_sample: PASSED

### Analyze-stroke
- URL: https://github.com/ghh1125/Analyze-stroke
- Duration: 134.7s
- Exit code: 0
- Log: logs/Analyze-stroke.log
- validation.md not readable

### BioSPPy
- URL: https://github.com/scientisst/BioSPPy
- Duration: 824.1s
- Exit code: 0
- Log: logs/BioSPPy.log
  - process_ecg_signal: FAILED
  - extract_eda_scrs: PASSED
  - compute_eeg_features: FAILED
  - assess_ecg_quality: FAILED
  - plot_biosignal_summary: FAILED

### aizynthfinder
- URL: https://github.com/MolecularAI/aizynthfinder
- Duration: 1863.6s
- Exit code: 0
- Log: logs/aizynthfinder.log
  - aizynthcli: PASSED
  - download_public_data: PASSED
  - do_expansion: PASSED

### ase
- URL: https://github.com/yfyh2013/ase
- Duration: 714.0s
- Exit code: 0
- Log: logs/ase.log
  - convert_structure: PASSED
  - get_structure_info: PASSED
  - build_surface: PASSED
  - fit_equation_of_state: PASSED
  - search_database: SKIPPED

### astronomy
- URL: https://github.com/cosinekitty/astronomy
- Duration: 1287.3s
- Exit code: 0
- Log: logs/astronomy.log
  - calculate_moon_phase: PASSED
  - find_lunar_eclipses: PASSED
  - calculate_celestial_positions: PASSED
  - find_rise_set_times: PASSED
  - calculate_seasons: PASSED

### astropy
- URL: https://github.com/astropy/astropy
- Duration: 1301.7s
- Exit code: 0
- Log: logs/astropy.log
  - parse_and_transform_sky_coord: PASSED
  - create_and_convert_quantity: PASSED
  - access_constant: PASSED
  - calculate_angular_separation: PASSED
  - summarize_fits_file: SKIPPED

### auto-sklearn
- URL: https://github.com/automl/auto-sklearn
- Duration: 1435.3s
- Exit code: 0
- Log: logs/auto-sklearn.log
  - fit_classification_model: SKIPPED
  - predict_classification_model: SKIPPED
  - fit_regression_model: SKIPPED
  - predict_regression_model: SKIPPED

### backtrader
- URL: https://github.com/mementum/backtrader
- Duration: 796.7s
- Exit code: 0
- Log: logs/backtrader.log
  - run_sma_crossover: FAILED
  - optimize_sma_range: FAILED
  - load_csv_data: PASSED
  - analyze_performance: PASSED
  - generate_crossover_signals: SKIPPED

### biopython
- URL: https://github.com/biopython/biopython
- Duration: 2312.9s
- Exit code: 0
- Log: logs/biopython.log
- validation.md not readable

### biotite
- URL: https://github.com/biotite-dev/biotite
- Duration: 688.9s
- Exit code: 0
- Log: logs/biotite.log
  - fetch_sequence: SKIPPED
  - pairwise_alignment: SKIPPED
  - blast_search: SKIPPED
  - multiple_sequence_alignment: SKIPPED

### dalle-mini
- URL: https://github.com/borisdayma/dalle-mini
- Duration: 538.1s
- Exit code: 0
- Log: logs/dalle-mini.log
  - generate_images: FAILED
  - rank_images_by_clip: SKIPPED
  - query_backend: SKIPPED