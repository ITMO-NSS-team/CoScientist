# Alembic benchmark — 2026-07-06 03:00

Repos processed: 12

| Repo | Time | Exit | Syntax | Tests | Tools (P/F/S) | Overall |
|---|---:|---:|---|---|---|---|
| AgML | 334s | 0 | PASSED | PASSED — 14 passed, 0 failed | 2/0/3 | PASSED |
| Analyze-stroke | 134s | 0 | — | — | 0/0/0 | — |
| BioSPPy | 1644s | 0 | PASSED | FAILED — 17 passed, 3 failed | 0/5/0 | FAILED (Syntax passed, but tests and all tool invocations failed) |
| aizynthfinder | 1034s | 0 | PASSED | PASSED — 11 passed, 0 failed | 3/1/0 | FAILED — Tool invocations: `create_interactive_app` failed with JSONDecodeError after fix attempt |
| ase | 1314s | 0 | PASSED | PASSED — 18 passed, 0 failed | 4/1/0 | FAILED (optimize_geometry tool failed after two debugger attempts) |
| astronomy | 1003s | 0 | PASSED | PASSED — 11 passed, 0 failed | 7/0/0 | PASSED |
| astropy | 1028s | 0 | PASSED | FAILED — 0 passed, 18 failed | 5/0/0 | PASSED |
| auto-sklearn | 2485s | 0 | PASSED | FAILED — 11 passed, 1 failed | 0/3/0 | FAILED |
| backtrader | 1500s | 0 | PASSED | FAILED — 15 passed, 1 failed | 2/1/0 | FAILED |
| biopython | 1342s | 0 | PASSED | FAILED — 10 passed, 5 failed | 0/1/4 | FAILED |
| biotite | 600s | 0 | PASSED | PASSED — 13 passed, 0 failed | 1/0/4 | PASSED |
| dalle-mini | 511s | 0 | PASSED | PASSED — 21 passed, 0 failed | 0/4/1 | FAILED — All 4 invoked tools failed due to unresolvable environment dependency (Python 3.8 incompatibility with jaxlib). |

## Per-repo details

### AgML
- URL: https://github.com/Project-AgML/AgML
- Duration: 334.1s
- Exit code: 0
- Log: logs/AgML.log
  - load_and_preprocess_dataset: PASSED
  - train_detection_model: SKIPPED
  - run_inference: SKIPPED
  - evaluate_model: SKIPPED
  - visualize_dataset_samples: PASSED

### Analyze-stroke
- URL: https://github.com/ghh1125/Analyze-stroke
- Duration: 133.5s
- Exit code: 0
- Log: logs/Analyze-stroke.log
- validation.md not readable

### BioSPPy
- URL: https://github.com/scientisst/BioSPPy
- Duration: 1643.5s
- Exit code: 0
- Log: logs/BioSPPy.log
  - ecg_processing: FAILED
  - hrv_analysis: FAILED
  - quality_assessment: FAILED
  - af_detection: FAILED
  - eda_processing: FAILED

### aizynthfinder
- URL: https://github.com/MolecularAI/aizynthfinder
- Duration: 1034.5s
- Exit code: 0
- Log: logs/aizynthfinder.log
  - single_retrosynthesis: PASSED
  - batch_retrosynthesis: PASSED
  - download_pretrained_models: PASSED
  - create_interactive_app: FAILED

### ase
- URL: https://github.com/yfyh2013/ase
- Duration: 1314.3s
- Exit code: 0
- Log: logs/ase.log
  - build_crystal: PASSED
  - build_molecule: PASSED
  - calculate_eos: PASSED
  - optimize_geometry: FAILED
  - launch_gui: PASSED

### astronomy
- URL: https://github.com/cosinekitty/astronomy
- Duration: 1003.1s
- Exit code: 0
- Log: logs/astronomy.log
  - calculate_lunar_phase: PASSED
  - calculate_rise_set: PASSED
  - calculate_solstices_equinoxes: PASSED
  - calculate_celestial_positions: PASSED
  - predict_lunar_eclipses: PASSED
  - predict_solar_eclipses: PASSED
  - calculate_jupiter_moons: PASSED

### astropy
- URL: https://github.com/astropy/astropy
- Duration: 1028.1s
- Exit code: 0
- Log: logs/astropy.log
  - convert_coordinate: PASSED
  - calculate_separation: PASSED
  - convert_quantity: PASSED
  - calculate_distance: PASSED
  - photon_energy_to_wavelength: PASSED

### auto-sklearn
- URL: https://github.com/automl/auto-sklearn
- Duration: 2485.2s
- Exit code: 0
- Log: logs/auto-sklearn.log
  - run_classification: FAILED
  - run_regression: FAILED
  - evaluate_model: FAILED

### backtrader
- URL: https://github.com/mementum/backtrader
- Duration: 1499.5s
- Exit code: 0
- Log: logs/backtrader.log
  - backtest_strategy: FAILED
  - optimize_strategy: PASSED
  - add_indicator: PASSED

### biopython
- URL: https://github.com/biopython/biopython
- Duration: 1341.7s
- Exit code: 0
- Log: logs/biopython.log
  - read_fasta: FAILED
  - perform_blast: SKIPPED
  - fetch_entrez: SKIPPED
  - parse_blast_xml: SKIPPED
  - parse_genbank: SKIPPED

### biotite
- URL: https://github.com/biotite-dev/biotite
- Duration: 599.7s
- Exit code: 0
- Log: logs/biotite.log
  - create_multiple_sequence_alignment: PASSED
  - calculate_distances: SKIPPED
  - perform_homology_search: SKIPPED
  - analyze_hydrogen_bonds: SKIPPED
  - superimpose_structures: SKIPPED

### dalle-mini
- URL: https://github.com/borisdayma/dalle-mini
- Duration: 510.6s
- Exit code: 0
- Log: logs/dalle-mini.log
  - dalle_mini_generate_image: FAILED
  - dalle_mini_generate_images: FAILED
  - dalle_mini_load_model: FAILED
  - dalle_mini_tokenize: FAILED
  - dalle_mini_decode_image: SKIPPED
