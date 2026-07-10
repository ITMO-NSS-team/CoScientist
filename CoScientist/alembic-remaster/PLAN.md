# Alembic Remaster — Work Plan

Ordered implementation plan for [DESIGN_CHOICES.md](./DESIGN_CHOICES.md).
Implementation replaces `CoScientist/alembic` **in place** (Docker COPY +
run_benchmark paths are hardcoded there); the old module is preserved in git.
Each item lists its acceptance check. Compatibility contract to protect is in
DESIGN_CHOICES §8 / the `alembic-remaster` memory.

## Guiding rules
- Reuse proven low-level code (docker infra, shell/venv/fs/invoke bodies,
  ADK patches, F14–F49 fixes worth keeping per DESIGN §4). Rewrite the
  orchestration, contracts, gates, and prompts.
- One concern per file; keep it small and testable.
- Verify each phase in the container (`alembic-base:latest`) before the next.

---

### W1 — Skeleton + config + contract (foundation)
- `config.py`: `MODEL`, all timeouts (from current `main.py`), stage list,
  caps (`MAX_STEPS`, guard/tool-repeat/tool-cycle, tool-count cap, etc.).
- `contract.py`: dataclasses `ToolSpec{name,target,params,sample_args,
  holdout_args,returns,skip,skip_reason}`, `EnvSpec{layout,server_python,
  repo_python,requirements,weights}`, `Plan{env,tools}`, and
  `ToolResult`/`Validation`. Functions: `parse_json_block(md)->dict` (fenced
  ```json extractor, generalizes `parse_samples_block`), `load_plan`/`save_plan`
  (to `reports/plan.json`), and `render_validation_md(Validation)->str` in the
  EXACT harness format (`## Syntax & Imports`/`## Tests`/`## Tool Invocations`/
  `## Overall`).
- **Accept:** `python -c "import alembic.contract"` clean; a unit test asserts
  `render_validation_md` output re-parses correctly through
  `run_benchmark.parse_validation` (import it directly and round-trip).

### W2 — Deterministic AST analysis (`tools/analysis.py`) — the Plan gate core
- `symbol_table(repo_dir) -> {module: {"functions": {name: [params]}, "classes": [names]}}`
  via AST walk (model on `scripts/compat_check.py`), skipping test/private.
- `verify_target(target, table, repo_dir) -> {ok, params|None, reason}`:
  `"mod.path:sym"` → symbol exists + real params; `"script:path"` → file exists.
- `decide_layout(repo_dir) -> EnvSpec-ish` via `packaging.SpecifierSet` on
  `requires-python`/`python_requires` (pyproject/setup.py/setup.cfg).
- **Accept:** unit test on 2–3 real subset clones (e.g. TabPFN, cytopus) —
  known real symbols verify, a fabricated symbol is rejected, layout matches
  the declared python.

### W3 — Runtime (`runtime.py`, was `agent_runtime.py`)
- Port the guarded run loop, ADK/LiteLLM patches (unknown-tool stub F19, fault
  detector F22), failure taxonomy (F12), async offload assumptions, guard/
  cycle/step breakers (F36 exemptions), soft-deadline (F32), `progress` dict
  (F35). Trim dead paths and comments; keep behaviour.
- Add `problem_summaries` threading hook for the debugger (F10).
- **Accept:** imports clean; the guard-retry + soft-deadline unit behaviour
  from the old tests still holds (re-run equivalents).

### W4 — Tools layer (trim + sentinel contract)
- Keep `paths.py`; keep `shell.py`/`venv.py`/`fs.py` bodies, prune obsolete
  special-cases. `invoke.py`: keep static-check (F28/F39/F45), SKIP gate (F25),
  process-group kill (F37), result cap (F30); switch result parsing to the
  **sentinel-delimited** contract (N5) and add the **path-arg existence gate**
  (N3, `bad_sample` outcome). `scripts/invoke_tool.py`: emit the sentinel.
- **Accept:** `invoke_mcp_tool` round-trips a sentinel result with noisy stdout
  before it; a nonexistent path arg returns `bad_sample`; SKIP still refuses.

### W5 — Agents + slim prompts
- `agents.py`: explorer/environment/coder/validator/debugger/reporter defs;
  debugger keeps F29 logging + gets `problem_summaries`.
- `instructions/*`: rewrite all 5 prompts short and model-agnostic — drop the
  reactive catalogs (argv bugs, device default, path reminders, mega-tool
  warnings) now covered by gates. Each prompt emits/consumes the structured
  contract block. Add optional `target_task` sections (§5).
- **Accept:** every prompt imports; each is materially shorter; a dry read
  confirms it references the contract block + its gate outputs.

### W6 — Pipeline orchestration (`pipeline.py`) + `main.py` shim
- Stage loop: Explore → **Plan gate** (analysis + layout, write plan.json) →
  Environment (record install.sh) → Coder → **Static gate** (route back on
  fail, bounded) → Validate(+repair loop) → render validation.md + metrics.json.
- Preserve `--resume`/`--until`, per-stage timeouts, F35 fallback, the
  `finally` metrics/error.json write, honest complete/incomplete banner.
- `--target-task` / `ALEMBIC_TARGET_TASK` plumbing.
- `main.py`: thin `python -m alembic.main <url> [--resume][--until][--target-task]`.
- Verify `start_chain.py` still forwards flags (add `--target-task` passthrough
  + `ALEMBIC_TARGET_TASK` to `PASSTHROUGH_ENV`).
- **Accept:** `python -m alembic.main <url> --until explorer` produces
  `exploration.md` + `plan.json`; full run produces a parseable `validation.md`.

### W7 — Docker rebuild + wiring check
- `docker/alembic/requirements.txt`: add `packaging` if not transitively
  present. Rebuild `alembic-base:latest`. Confirm entrypoint `build` path and
  `serve.py` still load a committed image.
- **Accept:** base image builds; `import alembic.main` inside the container is
  clean; a trivial repo builds end-to-end and commits `alembic-tool:<repo>`.

### W8 — Smoke test (task #5)
- Run 1–2 easy subset repos (e.g. `cytopus`, `TabPFN`) single-threaded with
  qwen. Fix crashes/contract-parse issues. `docker rmi -f` after each.
- **Accept:** both produce `validation.md` with real per-tool verdicts (no
  `ERROR — validation.md not readable`).

### W9 — Full qwen benchmark (task #6)
- `run_benchmark.py --repos-file toolmaker_subset.txt --parallel 4`; `docker
  rmi -f alembic-tool:<repo>` after each `[bench] ↓ done`. Check OpenRouter key
  first. Compare aggregate vs recorded baselines. Iterate on systemic failures.
- **Accept:** measurable improvement over baseline; every repo has metrics.

### W10 — Final glm-5.2 run (task #7)
- Set `MODEL=z-ai/glm-5.2` (per maintainer; ~$0.42/$1.32 per M tokens), rerun
  full subset `--parallel 4`, clean images. Record final numbers into
  `benchmarks/alembic/toolmaker_results.md`.
- **Accept:** completed run + summary table; note glm-5.2 vs qwen delta.

> **Status 2026-07-10:** W1–W9 done (qwen baseline: 12/14 syntax, 9 passing
> tools, 4 repos with passing tools). W10 superseded by Phase 2 below — the
> final glm-5.2 run happens on the upgraded architecture.

---

# Phase 2 — Upgrade (R1–R8)

Implements [upgrade.md](./upgrade.md) per DESIGN_CHOICES Part II. Locked
choices: JSON-only harness extraction; passed/perfect per-tool semantics; full
TM-Bench export; deterministic wrapper + LLM fallback; in-container fs
checkpoints; code-recorded setup.sh; fast run = 2-repo glm-5.2 smoke then
STAMP dual-task.

### U1 — Core plumbing (config / paths / contract)
- `config.py`: `STAGE_RESET` (default 2), `DEBUGGING_ROUNDS` (default 10),
  stage timeouts default **None** with `ALEMBIC_TIMEOUT_<STAGE>` overrides,
  `TEST_TIMEOUT=120`, `ALEMBIC_TASKS` (+ back-compat `ALEMBIC_TARGET_TASK`),
  `STAGES` gains `wrapper`.
- `tools/paths.py`: `set_current_repo(url)` + 0-arg path helpers (R8);
  `tools_python()` (repo venv in two-venv mode, else server venv).
- `contract.py`: `ToolSpec` + `sample_args`/`evidence`; `ToolReport`
  {tests_passed/total, exec_ok/note, invoc_passed/total, passed/perfect};
  `Validation` v2 with repo-level counts; `stage_status.json` writer;
  `validation.md` renderer becomes human-only (new format).
- **Accept:** container-import clean; unit check on ToolReport semantics
  (timeout ⇒ runtime success; passed/perfect boundaries).

### U2 — Tools layer v2
- All agent tools lose `repo_url` (R8); `clone_repo` sets current repo.
- `shell.py`/`venv.py`: record successful env-stage commands → rendered
  `setup.sh` (R5). `setup_venv` also readies pytest in the tools venv.
- `invoke.py` v2: per-tool static gate checks (compile/imports/undefined names
  on `tools/<name>.py` + test files); per-tool pytest runner (120 s cap,
  smoke/invoc split parsed from `-v` output); `invoke_tool_function` running
  `scripts/run_function.py` (sentinel protocol; missing-input/timeout ⇒
  runtime success).
- `codegen.py` (new): render `server.py` from AST signatures + docstrings
  (subprocess through `helpers/run_function.py`); render `setup.sh`; render
  TM-Bench `code.py` (verbatim function copy).
- **Accept:** gate_checks cover: run_function round-trip, pytest split parse,
  codegen output compiles, code.py self-contained.

### U3 — Instructions + agents
- Explorer: plan JSON gains per-tool `sample_args` + `evidence`; documents
  test basis (R6); task section when `ALEMBIC_TASKS` set (name/signature
  MUST match).
- Environment: no write_report; venvs only + weights; dataset prohibition in
  task mode (R7).
- Coder: plain function files (imports in body) + `tests/test_<name>.py`
  (`test_smoke_*`/`test_invoc_*`); no argparse, no server.py.
- Debugger: batch-failure mode (all failures in one message, fix shared root
  causes first).
- New `wrapper` fallback instruction (fix generated server.py only).
- **Accept:** prompts import; each references its gate contract.

### U4 — Orchestration rewrite (main.py)
- Stage-reset loops (fs checkpoints per stage-owned paths, fresh session +
  failure note, ≤ STAGE_RESET) around every LLM stage.
- Gates: G1 plan (≥1 verified tool, task tools present); G2 env
  (check_venv_compat + repo-import smoke; 1 debugger round then reset); G3
  artefacts (per-tool file/def/compile/imports/undefined + tests exist &
  import); G4 server (compile+import in .venv; wrapper LLM fallback).
- Deterministic validation: mount-file staging (tasks), per-tool exec + pytest,
  batched debugger ≤ DEBUGGING_ROUNDS, incremental validation.json.
- Export step: setup.sh, code.py per task; stage_status.json throughout;
  exploration report appended to env/coder prompts; timeouts only when set.
- **Accept:** `--until explorer` yields exploration.md + plan.json +
  stage_status.json; forced gate failure triggers visible reset.

### U5 — Harness v2 (run_benchmark.py + start_chain.py)
- start_chain: `--mount-dir` (ro bind mount), `ALEMBIC_TASKS` passthrough,
  stage choices + `wrapper`.
- run_benchmark: extraction = stage_status.json + validation.json +
  metrics.json (no md parsing); summary columns = stage reached / tests /
  exec-ok / invoc / passed / perfect; artefact export via `docker cp` →
  `runs/<ts>/output/<repo>/`; `--tasks` (files/dir) grouping by repo URL
  (STAMP dual-task); `toolmaker-runtime:installed-<task>` tagging in task mode.
- **Accept:** bench of a stub image produces the new summary from JSON only.

### U6 — Rebuild + in-container regression checks
- Update `tests/gate_checks.py` for v2 contracts; rebuild `alembic-base`;
  all checks pass in-container.

### U7 — Fast run 1: cytopus + TabPFN, `--parallel 2`, `MODEL=openrouter/z-ai/glm-5.2`
- Validates the restructure end-to-end; iterate on failures; `docker rmi`
  per repo.

### U8 — Fast run 2: STAMP dual-task
- Fetch both stamp task yamls (ToolMaker branch `original`); one pipeline run
  with both tasks; verify dual tools + code.py export + tags. Fallback:
  separate runs.

---

## Risks & mitigations
- **Contract-block parse failures** on weaker models → precise parse-error
  guard-retry; fall back to the old free-text path if the block never parses
  (never worse than today).
- **AST target verification false-negatives** (dynamic `__getattr__` exports,
  re-exports) → treat "not found" as a *soft* warning that demotes a candidate,
  not a hard drop, when the module clearly re-exports; keep a `--no-symbol-gate`
  escape hatch.
- **Rewrite regressions** on the kept F-fixes → port behaviour with the
  existing evidence in mind; keep the old module in git for diff/rollback.
- **Disk/DNS/spend** operational traps → per `alembic-remaster` memory
  (parallel 4, rmi per repo, check key).
