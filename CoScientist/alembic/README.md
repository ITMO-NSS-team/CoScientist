# Alembic

## Overview

Alembic is a multi-agent pipeline that automatically generates a deployable [FastMCP](https://github.com/jlowin/fastmcp) server from any scientific GitHub repository. Given a repo URL, it clones the code, sets up a reproducible Python environment, writes a working MCP server with pytest tests, validates every tool, and packages the result as a Docker image — all without human intervention.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Getting Started](#getting-started)
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

### Inside CoScientist (A2A)

The same pipeline is available to the CoScientist orchestrator as the
**AlembicAgent** — an A2A service (`:8007`) whose `build_mcp_server(repo_url)`
tool runs this whole flow and returns the live MCP server URL and its tools.
See [docs/A2A.md](docs/A2A.md).

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

## Running a Benchmark

`run_benchmark.py` processes multiple repos in parallel and writes a Markdown summary.

```bash
# Explicit list, 4 parallel workers (default)
python CoScientist/alembic/run_benchmark.py \
    --repos https://github.com/Roestlab/massformer \
            https://github.com/whitead/synspace \
            https://github.com/CrystalEye42/OpenChemIE

# From a file (one URL per line, '#' = comment), 8 workers, JSON dump
python CoScientist/alembic/run_benchmark.py \
    --repos-file repos.txt \
    --parallel 8 \
    --json-output bench.json
```

Results are written to `alembic_bench.md` and per-repo logs to `alembic_bench_logs/`.

---

## Project Structure

```
CoScientist/alembic/
├── agents.py          # Agent definitions (explorer, environment, coder, debugger, validator)
├── main.py            # Pipeline orchestrator (run_pipeline)
├── builder.py         # Programmatic build → commit → serve driver (shared by CLI + AlembicAgent)
├── start_chain.py     # CLI: build base image → run pipeline → commit → serve (thin wrapper over builder.py)
├── run_benchmark.py   # Parallel benchmark runner
├── common.py          # Shared Docker helpers (ensure_base_image, get_repo_name)
├── instructions/      # System prompts for each agent
├── docs/              # DESIGN.md + A2A.md (CoScientist AlembicAgent integration)
└── tools/             # Tool implementations (fs, shell, venv, invoke, paths)

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
