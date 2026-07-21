# Alembic — System Design

Alembic turns **any scientific GitHub repo → a deployable FastMCP server**, fully autonomously. A 4-agent pipeline (built on Google ADK + LiteLLM) clones the repo, builds a venv, writes a FastMCP server + pytest tests, validates every tool end-to-end, then `docker commit`s the result into a runnable image.

All agents share one `MODEL` (default `openrouter/qwen/qwen3-235b-a22b-2507`) and run **sequentially inside an ephemeral Docker container** — the container is the security boundary, so agents may run arbitrary shell.

---

## Architecture

```
start_chain.py  ──build base img──>  docker run (build mode)  ──>  main.run_pipeline()
                                                                          │
                  Explorer ─> Environment ─> Coder ─> Validator(─>Debugger)
                                                                          │
                <──── docker commit alembic-tool:<repo> ────  +  docker run (serve mode)
                                                              FastMCP server on random host port
```

### Workdir layout (`.alembic/<repo-name>/`)
| Path | Contents |
|---|---|
| `repos/` | shallow-cloned source |
| `output/server.py`, `output/tests/`, `output/helpers/` | generated artefacts |
| `output/.venv` | **server venv** (Py ≥3.10: fastmcp, pytest, mcp) — always exists |
| `output/.venv-repo` | **repo venv** (repo's own Python+deps) — only when repo needs Py <3.10 or has hard conflicts |
| `reports/{exploration,environment,server,validation}.md` | inter-agent handoff |
| `pipeline.log` | per-run log (scrubbed before commit) |

Stages communicate **only via the `reports/*.md` files** — no shared chat memory across agents.

---

## The 4 Agents (+ Debugger)

### 1. Explorer — *understand the repo*
**Tools:** `clone_repo`, `read_file`, `bash`, `search`, `write_report`
1. Clone repo (shallow) → get file list.
2. Read README, get `ls -R` tree.
3. Read ≤7 key files (setup/pyproject, `*.sh`, `run_*/train_*/predict_*`, configs). **Budget: ~20 tool calls.**
4. Identify env requirements (requirements.txt / pyproject / setup.py / env.yml, Python version, system libs, exact git URLs).
5. `write_report("exploration")` — description, key files, workflows, **Environment Setup**, and **1–5 MCP usage scenarios** with concrete real-data examples.

### 2. Environment — *make it installable*
**Tools:** `read_report`, `setup_venv`, `bash_env`, `check_venv_compat`, `write_report`
1. Read exploration report's Env section.
2. **Decide layout:** one-venv (repo Py ≥3.10) vs two-venv (older Py / hard conflicts).
3. Build venv(s): `setup_venv` first, then fall back to `uv pip` w/o version pins, then conda. `apt-get` for missing C libs. **Stop after 3 failed strategies per venv.**
4. `check_venv_compat` replays the repo's own imports to catch ABI/API conflicts; apply targeted fixes (numpy<2, transformers<4.38, torch extra-index, etc.).
5. `write_report("environment")`. **Guard:** `output/.venv/bin/python` must exist or the stage is retried.

### 3. Coder — *write the server* (runs against the explorer report; env builds in parallel conceptually)
**Tools:** `read_report`, `bash`, `read_file`, `write_file`, `write_report`
1. Read exploration report (tools + examples).
2. Verify real API signatures via `grep`/`read_file` before coding.
3. Write **static helper scripts** (`helpers/*.py`) — argv in, JSON to stdout — for tools that call repo Python.
4. Write `server.py`: each `@mcp.tool()` shells out via `subprocess.run([str(PYTHON), ...])` where `PYTHON` auto-selects `.venv-repo` if present. **Strict rules:** no editable installs, no runtime-built source strings, no defensive path checks, every `check=True` wrapped → `RuntimeError`, no `None` defaults for paths.
5. Write `tests/test_server.py` (mock only `subprocess.run`; success + failure + one example test per explorer example).
6. `write_report("server")` incl. a `samples:` YAML block telling the validator how to invoke each tool.

### 4. Validator — *prove it works* (owns the Debugger as a sub-agent tool)
**Tools:** `read_report`, `validate_syntax`, `run_tests`, `invoke_mcp_tool`, `write_report`, `AgentTool(debugger)`
1. Read server report.
2. `validate_syntax` (py_compile + real import).
3. `run_tests` (pytest).
4. `invoke_mcp_tool` for every non-`SKIP` sample → executes the real tool in the server venv, catching runtime faults pytest's mocks can't (missing OS binary, missing dep, bad argv).
5. On any failure → call **Debugger**, then re-check. Budgets: ≤3 tries/stage, ≤2 debugger calls/tool, and **stop on identical repeated error**.
6. `write_report("validation")` — per-stage PASS/FAIL + overall verdict.

### Debugger (sub-agent, invoked by Validator)
**Tools:** `read_output_file`, `update_file`, `bash`, `bash_env`, `invoke_mcp_tool`
Triages the error into one class and fixes only that: **(A)** missing OS binary → `apt-get`; **(B)** missing module → `uv pip` into the correct venv; **(C)** code bug → `update_file` full server/helper (common argv bugs, defensive-check removal); **(D)** hard env fault → stop & report. Always re-runs `invoke_mcp_tool` to verify before returning a structured summary.

---

## Tools (`tools/`)

| Tool | Purpose |
|---|---|
| `clone_repo` | shallow `git clone`; returns filtered file list |
| `read_file` / `search` | read repo text file (≤40KB) / glob repo |
| `bash` / `bash_env` | shell, 15s / 300s timeout (installs); `glob` shortcut |
| `setup_venv` | create server venv (uv→venv fallback) + install fastmcp/pytest/mcp + deps |
| `check_venv_compat` | run `compat_check.py` — AST-collect repo imports, replay in venv, report conflicts |
| `write_file` / `read_output_file` / `update_file` | manage `output/` artefacts |
| `read_report` / `write_report` | inter-agent `reports/*.md` |
| `validate_syntax` | py_compile + module-load of server.py in venv |
| `run_tests` | pytest the generated suite (120s) |
| `invoke_mcp_tool` | run one `@mcp.tool()` live via `invoke_tool.py` (monkey-patches `mcp.run` to no-op, unwraps FastMCP FunctionTool); returns `{ok, result}` or `{ok:False, error, traceback, stderr}` |

**Helper scripts** (run inside the *target* venv, not the pipeline's): `scripts/compat_check.py` (import-conflict detector) and `scripts/invoke_tool.py` (single-tool invoker).

---

## Orchestration & Robustness (`main.py`)

- **Per-stage wall-clock timeouts:** explorer 15m / environment 40m / coder 25m / validator 30m. A timed-out stage aborts but the pipeline continues.
- **Guard retries (≤3):** re-invoke a stage if its required report wasn't written or the venv guard path is missing.
- **Loop breakers:** abort an agent on the same tool+args called 3× (`MAX_TOOL_REPEATS`) or 120 total events (`MAX_STEPS`).
- **Unknown-tool patch:** ADK's `_get_tool` is monkey-patched so a hallucinated tool name returns an error stub to the LLM instead of crashing the run.
- **Validator skip-guard:** skips if `server.py`/tests/`server.md` are missing.
- `--resume <stage>` preserves the workdir and restarts from any stage.

## Deployment (`start_chain.py` + `docker/alembic/`)
1. `ensure_base_image` builds `alembic-base:latest` once (python:3.11 + git/build tools/C libs + uv + alembic code).
2. **build mode** (`entrypoint.py build`) runs `alembic.main` inside the container; API keys passed via `-e`/`--env-file`.
3. On success: scrub `pipeline.log`, blank secret env vars via `--change`, `docker commit` → `alembic-tool:<repo>`.
4. **serve mode** (`entrypoint.py serve` → `serve.py`) loads `server.py` in the server venv and runs FastMCP over `streamable-http` on `$MCP_PORT` (8000), mapped to a random host port.

`benchmarks/alembic/run_benchmark.py` runs the whole flow over many repos in parallel (after a `git ls-remote` reachability pre-check) and writes a summary + per-repo logs under `benchmarks/alembic/runs/<timestamp>/`.
