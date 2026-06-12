---
id: F015e
title: Alembic — repo→runnable MCP server pipeline (explorer→environment→coder→validator+debugger)
type: feature
status: in_progress
created: 2026-06-11
updated: 2026-06-11
owners: [ITMO-NSS-team]
derives_from: [F015, F002]
depends_on: [F015d]
sources: [S024, S025, S026]
tags: [alembic, tool-creation, mcp-generation, docker, sandbox]
code:
  # lives on branch origin/alembic-integration-demo (src/ layout), NOT on refactoring_full_pipe
  - CoScientist/src/alembic/main.py
  - CoScientist/src/alembic/agents.py
  - CoScientist/src/alembic/tools.py
  - CoScientist/src/alembic/instructions.py
  - CoScientist/src/alembic/start_chain.py
  - docker/alembic/Dockerfile
benchmarks: []
---

## Goal
Turn a code repository into a **validated, containerized, runnable MCP server** — the
"build a missing tool" backend for the АМ (honors F014.D1: only CREATES new servers, never
edits existing tools). Largely **already built** on branches; F015 wraps it, doesn't extend it inline.

## Current state (grounded in origin/alembic-integration-demo)
A complete standalone repo→MCP factory exists: a 5-agent ADK chain
(`agents.py`: explorer/environment/coder/validator + debugger as an AgentTool), staged
sequentially in `main.py:run_pipeline()` with `--resume`. Stages: explorer (clone, README/tree,
1–5 candidate tool scenarios + env reqs), environment (1-venv vs 2-venv; `setup_venv` uv→pip;
`check_venv_compat` AST-replays imports for ABI conflicts), coder (FastMCP `server.py` +
pytest + `samples:` block), validator (`validate_syntax`/`run_tests`/`invoke_mcp_tool` runs
each tool in the server venv; on failure calls the debugger which triages A/B/C/D and
re-verifies). Guards in `main.py`: `_safe_get_tool`/`_UnknownToolStub` (returns an error
instead of crashing on a hallucinated tool name), `MAX_TOOL_REPEATS=3`, `MAX_STEPS=120`,
per-stage `STAGE_TIMEOUT`. Docker: `docker/alembic/` (base image, entrypoint build/serve/shell,
`serve.py` streamable-http on :8000); `start_chain.py` build→`docker commit alembic-tool:<repo>`
→serve on a random 20000–30000 port; `run_benchmark.py` parallel harness over many repos.

## Attempts
### F015e.A1 — Alembic pipeline built on branches · prior work · outcome: success (standalone)
- **Method:** 5-agent ADK chain + docker sandbox + guards + benchmark harness.
- **Evidence:** `origin/alembic-integration-demo` (#259 sandbox, #248 venv rework, #266 examples);
  files under `CoScientist/src/alembic/` + `docker/alembic/`.
- **Outcome:** works standalone (manual `repo_url` in), NOT integrated into the CoScientist runtime.

## Best practices vs what's ALREADY BUILT (dedup audit 2026-06-12 — don't reproduce)
Verified against `origin/alembic-integration-demo` line-by-line:

| Recommended practice | Status in Alembic | Action |
|---|---|---|
| Real-call validation (≈ readiness L1) [S025] | ✅ EXISTS — `tools.py:invoke_mcp_tool` (L612) runs each tool in the server venv | reuse as-is |
| Syntax/import gate (≈ L0) [S025] | 🟡 PARTIAL — `validate_syntax` = `py_compile` + module load (L501-514); pyright `reportMissingImports` is stricter | optional upgrade, low prio |
| uv-first install [S026] | 🟡 PARTIAL — `setup_venv` (L274) prefers uv, pip fallback; but it's called BY the agent, not deterministically BEFORE the agentic cascade | reorder: try plain install pre-agent |
| Hallucinated-tool guard | ✅ EXISTS — `main.py:_safe_get_tool`/`_UnknownToolStub` (L42-61) | lift into АМ executor (F015g) |
| Duplicate-call / step / time caps | ✅ EXISTS — `MAX_TOOL_REPEATS=3`, `MAX_STEPS=120`, `STAGE_TIMEOUT` (L106-113) | generalize (F015g), +NFKC |
| Bounded guard-retry w/ nudges (≈ F015b's loop mechanic) | ✅ EXISTS — `MAX_GUARD_RETRIES=3` (L196-222) | reuse the pattern in F015b |
| ABI/compat check [S026] | ✅ EXISTS — `check_venv_compat` (L376) AST-replays imports | reuse |
| Env-RESET on debug retry [S024] | ❌ ABSENT — debugger patches in place; half-broken state can poison retries | net-new, adopt |
| Per-command commit/rollback [S026] | ❌ ABSENT — single `docker commit` at pipeline end (`start_chain.py`) | adopt selectively (I/O-heavy) |
| Dockerfile-from-history [S026] | ❌ ABSENT — **maybe skip**: the docker-commit image flow already gives reuse; adopt only if rebuild-from-source reproducibility is required | decide, don't auto-adopt |
| GPU/CUDA probe (L2) [S025] | ❌ ABSENT (`--gpus` flag exists, no probe) | adopt if tool advertises GPU |
| `diff_report.md` provenance [S024] | ❌ ABSENT | adopt (cheap, feeds DEVGRAPH evidence) |
| Conservative probe-then-assert self-report [S025] | 🟡 PARTIAL — validator writes per-tool PASS/FAIL from real calls, but no readiness-level recorded in served-tool metadata for F015c | extend (record level at F015f registration) |

## ⚠ Risks / open questions
- **Packaging/import-path blocker (prerequisite for ALL integration):** Alembic is `CoScientist/src/alembic`
  (src/ layout) on a separate branch, invoked as a CLI — distinct from the package layout on
  `refactoring_full_pipe`. Decide import path + expose `run_pipeline`/`start_chain` as an
  **in-process callable** from the F015g/F015c "insufficient" branch. This is an explicit first step.
- **"Validator passed" ≠ scientific correctness** (ToolMaker) — F015g must not treat green tests as hypothesis-grade.
- Per-command `docker commit` is I/O/disk heavy; keep the safe-command bypass list. Low build-success on
  hard research repos → F015c must degrade to a reported gap, not loop.

## ✅ TODO
- [ ] Packaging/import-path decision; expose Alembic as an in-process callable.
- [ ] Net-new only (see dedup table): env-reset-on-debug-retry; `diff_report.md` provenance;
      record readiness level (existing `invoke_mcp_tool` result) into served-tool metadata for F015c.
- [ ] Reorder deterministic install BEFORE the agentic env cascade (uv-first already exists in `setup_venv`).
- [ ] Decide: per-command commit/rollback (I/O cost) and Dockerfile-from-history (likely skip — commit-flow suffices).

## Symbols (on branch origin/alembic-integration-demo)
- `CoScientist/src/alembic/main.py` — `run_pipeline`, guards (`_safe_get_tool`, `MAX_TOOL_REPEATS`, `STAGE_TIMEOUT`).
- `CoScientist/src/alembic/tools.py` — `clone_repo`, `setup_venv`, `check_venv_compat`, `invoke_mcp_tool`, …
- `CoScientist/src/alembic/start_chain.py` — `build_image`/`serve_image` (docker commit → serve).
