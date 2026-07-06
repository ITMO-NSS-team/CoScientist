# Alembic benchmark — 2026-07-06 12:55

Repos processed: 3

| Repo | Time | Exit | Syntax | Tests | Tools (P/F/S) | Overall |
|---|---:|---:|---|---|---|---|
| AgML | 3060s | 0 | — | — | 0/0/0 | ERROR — validation.md not readable |
| BioSPPy | 1839s | 0 | — | — | 0/0/0 | ERROR — validation.md not readable |
| biotite | 2256s | 0 | PASSED | PASSED — 11 passed, 0 failed | 2/2/0 | FAILED — superimpose_structures and visualize_structure failed end-to-end execution due to input compatibility issue and missing Python version support. |

## Per-repo details

### AgML
- URL: https://github.com/Project-AgML/AgML
- Duration: 3060.4s
- Exit code: 0
- Log: logs/AgML.log
- validation.md not readable

### BioSPPy
- URL: https://github.com/scientisst/BioSPPy
- Duration: 1839.2s
- Exit code: 0
- Log: logs/BioSPPy.log
- validation.md not readable

### biotite
- URL: https://github.com/biotite-dev/biotite
- Duration: 2255.5s
- Exit code: 0
- Log: logs/biotite.log
  - fetch_structure: PASSED
  - align_sequences: PASSED
  - superimpose_structures: FAILED
  - visualize_structure: FAILED
