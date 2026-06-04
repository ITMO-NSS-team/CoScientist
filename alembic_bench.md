# Alembic benchmark — 2026-06-04 18:47

Repos processed: 3

| Repo | Time | Exit | Syntax | Tests | Tools (P/F/S) | Overall |
|---|---:|---:|---|---|---|---|
| Img2Mol | 1175s | 0 | PASSED | FAILED — 6 passed, 1 failed | 0/0/2 | FAILED (tests stage) |
| OpenChemIE | 1732s | 0 | PASSED | PASSED — 13 passed, 0 failed | 0/5/0 | FAILED |
| synspace | 324s | 0 | PASSED | PASSED — 7 passed, 0 failed | 1/1/0 | FAILED (Tool Invocations: visualize_reaction FAILED) |

## Per-repo details

### Img2Mol
- URL: https://github.com/bayer-science-for-a-better-life/Img2Mol
- Duration: 1174.9s
- Exit code: 0
- Log: /Users/deni/Desktop/PythonAllProjects/CoScientist/alembic_bench_logs/Img2Mol.log
  - infer_molecule_from_image: SKIPPED
  - infer_molecules_from_images: SKIPPED

### OpenChemIE
- URL: https://github.com/CrystalEye42/OpenChemIE
- Duration: 1732.3s
- Exit code: 0
- Log: /Users/deni/Desktop/PythonAllProjects/CoScientist/alembic_bench_logs/OpenChemIE.log
  - extract_molecules_from_figures_in_pdf: FAILED
  - extract_reactions_from_text_in_pdf: FAILED
  - extract_reactions_from_pdf: FAILED
  - extract_figures_from_pdf: FAILED
  - extract_molecule_corefs_from_figures_in_pdf: FAILED

### synspace
- URL: https://github.com/whitead/synspace
- Duration: 324.1s
- Exit code: 0
- Log: /Users/deni/Desktop/PythonAllProjects/CoScientist/alembic_bench_logs/synspace.log
  - generate_chemical_space: PASSED
  - visualize_reaction: FAILED
