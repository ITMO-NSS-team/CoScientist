# Alembic benchmark — 2026-07-06 16:48

Repos processed: 1

| Repo | Time | Exit | Syntax | Tests | Tools (P/F/S) | Overall |
|---|---:|---:|---|---|---|---|
| biotite | 1142s | 0 | PASSED | PASSED — 12 passed, 0 failed | 4/0/0 | PASSED |

## Per-repo details

### biotite
- URL: https://github.com/biotite-dev/biotite
- Duration: 1142.5s
- Exit code: 0
- Log: benchmarks/alembic/runs/2026-07-06_rerun6_f12-verify/logs/biotite.log
  - fetch_sequence: PASSED
  - align_sequences: PASSED
  - download_structure: PASSED
  - mafft_alignment: PASSED

## Aggregate metrics (F12)

Repos with metrics.json: 1/1

**Stage completion (completed/attempted):**

- explorer: 1/1
- environment: 1/1
- coder: 1/1
- validator: 1/1

**Failure taxonomy (tool-invocation failures across all repos):**

- TypeError: 3
- ModuleNotFound: 1
- FileNotFound: 1

- Guard retries (write_report/venv nudges) total: 0
- Transient provider-fault retries (F22) total: 0
