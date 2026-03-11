# chemical-mcp-server

Standalone MCP (Model Context Protocol) server for chemistry tools: SMILES, molecular properties, docking, molecule/reaction OCR, visualization.

Runs independently with **uv**, **Docker** or **docker-compose**.

## Requirements

- Python 3.11+
- Optional: [uv](https://docs.astral.sh/uv/) for install/run
- Optional: Docker & docker-compose for containerized run

## Setup (local with uv)

```bash
cd mcp-servers/chemical-mcp-server
cp .env.example .env
# Edit .env if needed (CHEM_SERVICES_HOST, CHEM_SERVICES_PORT)

uv sync
```

## Run (local)

```bash
# With uv
uv run python -m server.chemical_server

# Or after uv sync
chemical-mcp-server
```

Server listens on `http://0.0.0.0:7331/mcp`.

## Run with Docker

```bash
cp .env.example .env
docker compose up --build
```

## Run with Docker (one-off)

```bash
docker build -t chemical-mcp-server .
docker run -p 7331:7331 --env-file .env chemical-mcp-server
```

## Environment (.env)

| Variable | Description | Default |
|----------|-------------|--------|
| `CHEM_SERVICES_HOST` | Host of the chemistry API (OpenChemIE/docking) | `localhost` |
| `CHEM_SERVICES_PORT` | Port of the chemistry API | `8005` |
| `PROCESSED_IMG_STORAGE_PATH` | Directory for annotated OCR images | `/tmp/chemical_mcp_annotated` |

Copy `.env.example` to `.env` and adjust as needed.

## Tools exposed via MCP

- `name2smiles`, `smiles2name`, `smiles2prop`
- `visualize_molecule` (SMILES → HTML)
- `fetch_activity_data` (BindingDB / ChEMBL)
- `detect_molecules`, `extract_reactions` (images directory)
- `calculate_docking` (SMILES, PDB ID, output HTML path)

All file paths are passed as arguments; no path-related env vars are required for tool calls.
