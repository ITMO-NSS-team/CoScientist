# Alembic ↔ CoScientist — A2A Integration

Alembic's pipeline (a scientific GitHub repo → a deployable FastMCP server, see
[DESIGN.md](DESIGN.md)) is surfaced inside CoScientist as the **AlembicAgent**, a
first-class sub-agent of the orchestrator that is also exposed as a standalone
[A2A](../../docs/a2a.md) service. The orchestrator can now answer *"wrap repo X as
an MCP server"* by delegating to it, exactly as it delegates research, coding or
computation.

```
        OrchestratorAgent  (:8000)
              │  A2A / in-process AgentTool
              ▼
        AlembicAgent  (:8007)
              │  build_mcp_server(repo_url) / stop_mcp_server(container)
              ▼
   CoScientist.alembic.builder        ── Docker CLI ──▶  alembic pipeline
   (build → commit → serve)                              (ephemeral container)
              │
              ▼
        alembic-tool:<repo>  →  running FastMCP server at http://host:<port>/mcp
```

## What was added

| File | Role |
|---|---|
| `CoScientist/alembic/builder.py` | Programmatic driver for the Docker **build → commit → serve** flow. Returns structured results and **raises** (`AlembicBuildError`) instead of `sys.exit`, so it is callable from both the CLI and the agent. Single source of truth — `start_chain.py` is now a thin CLI over it. |
| `CoScientist/tools/alembic_tools.py` | `AlembicToolset` exposing `build_mcp_server` and `stop_mcp_server` as ADK tools. Runs the blocking Docker work in a worker thread; tracks live containers in session state. |
| `CoScientist/assembly/bindings.py` | Registers the `alembic` tool entry + its prompt `ToolDoc`s. |
| `CoScientist/agents/prompts/templates.py` | The `alembic` prompt template. |
| `CoScientist/agents/system.yaml` | Declares `AlembicAgent` (model `coder`, tool `alembic`, `a2a` key `alembic` / port `8007`) and adds it to the orchestrator's subordinates. |

Because everything is declared in `system.yaml`, the A2A card, ports, `run_all`,
the benchmark client and the orchestrator roster all pick the agent up
automatically — no per-agent server module.

## The tools

### `build_mcp_server(repo_url, serve=True) -> dict`
Runs the full pipeline in Docker and (when `serve=True`) launches the result.

- success → `{status, repo, image, tools, url, container, port, validation_summary}`
- failure → `{status: "error", error}`

`tools` is a best-effort list of the `@mcp.tool()` names extracted from the
coder's `server.md`; `validation_summary` is the (truncated) per-stage
PASS/FAIL report. Live containers are recorded in session state so a later
`stop_mcp_server` call (a separate A2A turn) can find them.

### `stop_mcp_server(container) -> dict`
Stops and removes a serve container previously returned by `build_mcp_server`.

## Running

```bash
# Serve the agent standalone (and benchmark it)
python -m CoScientist.a2a.serve alembic            # :8007
python -m CoScientist.a2a.benchmark --agent alembic \
    --text "Wrap https://github.com/whitead/synspace as an MCP server."

# Or as part of the whole stack
python -m CoScientist.a2a.run_all                  # alembic comes up on :8007
```

## Requirements & notes

- **Docker.** The agent shells out to the Docker CLI; the build itself runs
  inside an ephemeral container (the security boundary — pipeline agents may run
  arbitrary shell). When the A2A stack runs in a container, mount the Docker
  socket (`-v /var/run/docker.sock:/var/run/docker.sock`). Without Docker on
  `PATH`, `build_mcp_server` returns a structured error rather than crashing.
- **Cost.** A build can take minutes to tens of minutes (heavy ML deps), so the
  tool runs it off the event loop in a worker thread, and the prompt instructs
  the agent to build **once per repository** and trust the result.
- **Secrets.** API keys passed at build time are scrubbed from the committed
  image (`pipeline.log` wiped, sensitive env vars blanked via `docker commit
  --change`) by `builder.build_image`, preserving `start_chain.py`'s behaviour.
- **Non-HITL.** The agent does not block on console input, so it works
  unattended over A2A.
