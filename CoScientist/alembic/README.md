# Alembic

## Overview

Alembic is a multi-agent pipeline that automatically generates a deployable [FastMCP](https://github.com/jlowin/fastmcp) server from any scientific GitHub repository. Given a repo URL, it clones the code, sets up a reproducible Python environment, writes a working MCP server with pytest tests, validates every tool, and packages the result as a Docker image — all without human intervention.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Getting Started](#getting-started)
- [Live Dashboard (Web UI)](#live-dashboard-web-ui)
- [Running a Benchmark](#running-a-benchmark)
- [Project Structure](#project-structure)
- [Configuration](#configuration)

---

## How It Works

The pipeline runs four sequential stages inside a Docker container:

| Stage | Agent | What it does |
|---|---|---|
| 1 | **Explorer** | Clones the repo, reads the code, writes `reports/exploration.md` |
| 2 | **Environment** | Creates an isolated `.venv`, installs all dependencies, writes `reports/environment.md` |
| 3 | **Coder** | Generates `output/server.py` (FastMCP) and `output/tests/test_server.py` |
| 4 | **Validator** | Runs syntax checks, pytest, and live tool invocations; calls a **Debugger** sub-agent on failures; writes `reports/validation.md` |

On success the build container is committed to `alembic-tool:<repo-name>` and launched with the MCP server listening on a random host port.

---

## Getting Started

**Prerequisites:** Docker, Python 3.11+, and a `.env` file at the project root with at least one LLM API key.

```env
OPENROUTER_API_KEY=sk-or-...
MODEL=openrouter/qwen/qwen3-235b-a22b-2507   # optional override
```

**Build and serve a repo in one command:**

```bash
python CoScientist/alembic/start_chain.py https://github.com/Roestlab/massformer
```

The script:
1. Builds `alembic-base:latest` once (skipped on subsequent runs).
2. Runs the pipeline inside a container.
3. Commits the result and starts the MCP server.

```
[start-chain] MCP server up.
  url       : http://localhost:24371/mcp
  logs      : docker logs -f alembic-serve-massformer-a1b2c3
  stop      : docker stop alembic-serve-massformer-a1b2c3 && docker rm alembic-serve-massformer-a1b2c3
```

**Useful flags:**

```bash
# Resume from a specific stage (workdir is preserved)
python CoScientist/alembic/start_chain.py <repo_url> --resume coder

# Build image without starting the server
python CoScientist/alembic/start_chain.py <repo_url> --no-serve

# Force rebuild of the base image
python CoScientist/alembic/start_chain.py <repo_url> --rebuild-base

# GPU access inside the container
python CoScientist/alembic/start_chain.py <repo_url> --gpus all
```

---

## Live Dashboard (Web UI)

A local, browser-based dashboard that runs the pipeline **without Docker** and
streams every stage and agent action in real time:

- **top** — the five-stage rail (explorer → environment → coder → validator →
  wrapper), lit as each stage runs / passes / fails;
- **left** — an accumulating column: the exploration report, the generated
  output files, how to run them (the recorded `setup.sh`), and per-tool
  invocation examples;
- **right** — the generated tools as cards with live pass/fail validation
  badges (hover a failure for the error) and a **Call** form that invokes each
  tool-function on demand — the same `invoke_tool_function` path the validator
  uses, so you can exercise a tool the instant the Coder writes it (wrapping to
  a real MCP server is only the final stage);
- **bottom** — a live activity feed of every agent tool call.

**Prerequisites:** the same `.env` as [Getting Started](#getting-started), plus
`fastapi` and `uvicorn` (already in the project `requirements.txt`).

**Start it (from the repo root):**

```bash
python CoScientist/alembic/web/server.py
```

Then open **http://127.0.0.1:8100**, paste a repo URL (e.g.
`https://github.com/whitead/synspace`) and press **Run**.

Notes:
- Run it from the project root — the pipeline writes its workdir to `./.alembic/`,
  the same location as the CLI.
- **Stop** invalidates the run; the pipeline unwinds at its next stage/tool
  boundary (a subprocess already in flight finishes first).
- The pipeline itself is unchanged for the CLI/benchmark: it emits events
  through the optional `alembic.events` bus, a no-op unless the dashboard
  installs a sink. All UI enrichment (reading `plan.json` / `validation.json` /
  the output files off disk) lives in `web/app.py`, not in the pipeline.

---

## Running a Benchmark

`benchmarks/alembic/run_benchmark.py` processes multiple repos in parallel
(with a `git ls-remote` reachability pre-check) and writes a Markdown
summary.

```bash
# Explicit list, 4 parallel workers (default)
python benchmarks/alembic/run_benchmark.py \
    --repos https://github.com/Roestlab/massformer \
            https://github.com/whitead/synspace \
            https://github.com/CrystalEye42/OpenChemIE

# From a file (one URL per line, '#' = comment), 8 workers
python benchmarks/alembic/run_benchmark.py \
    --repos-file repos.txt \
    --parallel 8
```

Results default to `benchmarks/alembic/runs/<timestamp>/` (`summary.md`,
`summary.json`, `logs/*.log`) — pass `--output`/`--json-output`/`--log-dir`
to override.

---

## Project Structure

```
CoScientist/alembic/
├── agents.py          # Agent definitions (explorer, environment, coder, debugger, validator)
├── main.py            # Pipeline orchestrator (run_pipeline)
├── start_chain.py     # CLI: build base image → run pipeline → commit → serve
├── run_benchmark.py   # Parallel benchmark runner
├── common.py          # Shared Docker helpers (ensure_base_image, get_repo_name)
├── events.py          # Optional live-event bus (no-op for CLI; feeds the web UI)
├── instructions/      # System prompts for each agent
├── tools/             # Tool implementations (fs, shell, venv, invoke, codegen, paths)
└── web/               # Live dashboard: FastAPI + WebSocket (app.py, server.py, templates/)

docker/alembic/
├── Dockerfile         # Base image (python:3.11 + build deps + alembic code)
├── entrypoint.py      # Container entrypoint (build / serve / help)
├── serve.py           # FastMCP server launcher inside container
└── requirements.txt   # Pipeline dependencies
```

---

## Configuration

All settings are passed through environment variables (`.env` or shell):

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `openrouter/qwen/qwen3-235b-a22b-2507` | LiteLLM model string for all agents |
| `OPENROUTER_API_KEY` | — | Required when using OpenRouter |
| `OPENAI_API_KEY` | — | Required when using OpenAI models |
| `TAVILY_API_KEY` | — | Optional; enables web search inside the Explorer |
| `MCP_PORT` | `8000` | Port the FastMCP server listens on inside the container |
| `ALEMBIC_WORKDIR` | `/work/.alembic` | In-container working directory for repos and reports |
