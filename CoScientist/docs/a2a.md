# A2A Mode — Agent-to-Agent Architecture

CoScientist can run as a set of independent **A2A** (Agent-to-Agent) HTTP
services instead of a single in-process agent tree. Each agent runs as its own
server; the orchestrator calls them over the [A2A protocol](https://google.github.io/A2A)
(JSON-RPC over HTTP) instead of in-process `AgentTool` calls.

The original in-process mode still works unchanged — A2A is additive.

---

## 1. How it works

```
                ┌────────────────────────┐
   user ───────▶│   OrchestratorAgent    │  :8000
                │  (LlmAgent + remotes)  │
                └───┬───┬───┬───┬───┬────┘
        A2A/HTTP    │   │   │   │   │
           ┌────────┘   │   │   │   └────────┐
           ▼            ▼   ▼   ▼            ▼
       Planner   Hypotheses Research TaskExec Medical
        :8001      :8002    :8003    :8004    :8005
```

- Every agent is wrapped as an **A2A FastAPI server** by `make_a2a_app()`.
- Each server publishes an **AgentCard** at `/.well-known/agent-card.json`
  (and the deprecated `/.well-known/agent.json`).
- The orchestrator holds a `RemoteA2aAgent` per sub-agent (instead of an
  in-process agent), each pointing at a sub-agent's card URL. From the
  orchestrator LLM's point of view they are still just tools — it decides when
  to call them based on each card's `description`.
- Every server attaches an **Opik tracer** so all events/callbacks land in the
  `adk-coscientist` Opik project.

### Key files

| File | Role |
|------|------|
| `a2a/config.py` | Ports + URLs for all agents (override via env vars) |
| `a2a/server.py` | `make_a2a_app()` — wraps any ADK agent as an A2A server + Opik tracing |
| `a2a/orchestrator.py` | Orchestrator using `RemoteA2aAgent` for each sub-agent |
| `a2a/servers/*.py` | One thin server module per agent (planner, hypotheses, research, task_execution, medical) |
| `a2a/run_all.py` | Launches all 6 servers in one process |
| `a2a/benchmark.py` | CLI client: live event streaming + latency testing |
| `agent.py` | `adk web` entry point — `A2A_MODE=1` switches to the A2A orchestrator |

---

## 2. Running

All commands run from the repo root (`/app`), **not** from `CoScientist/`
(running inside the package shadows the stdlib `logging` module).

### Everything at once

```bash
python -m CoScientist.a2a.run_all
```

Starts all six servers. Ports are printed on startup.

### A single agent (for development)

```bash
python -m CoScientist.a2a.servers.hypotheses    # :8002
```

### The orchestrator with a web UI

```bash
A2A_MODE=1 adk web --port 8000
# sub-agents must already be running (e.g. via run_all)
```

### Configuration

Ports default to 8000–8005 and are overridable:

```bash
HYPOTHESES_PORT=9002 A2A_HOST=0.0.0.0 python -m CoScientist.a2a.run_all
```

Env vars: `ORCHESTRATOR_PORT`, `PLANNER_PORT`, `HYPOTHESES_PORT`,
`RESEARCH_PORT`, `TASK_EXECUTION_PORT`, `MEDICAL_PORT`, `A2A_HOST`,
`A2A_DISABLE_OPIK` (set to `1` to turn off tracing).

> **Restart after edits.** A running `run_all` holds the old code in memory.
> After changing an agent or its card, restart the process.

---

## 3. Testing & observing

`a2a/benchmark.py` is a CLI client. By default it streams every internal event
(status transitions, agent thoughts, tool calls/results, artifacts) live:

```bash
# Watch everything happen, in real time:
python -m CoScientist.a2a.benchmark --agent hypotheses \
    --text "Generate a hypothesis about why sleep aids memory."

# Latency only, 5 runs:
python -m CoScientist.a2a.benchmark --agent research \
    --text "What is CRISPR?" -n 5 --no-stream
```

`--agent` accepts: `orchestrator`, `planner`, `hypotheses`, `research`,
`task_execution`, `medical`.

Traces also appear in the **Opik dashboard** under project `adk-coscientist`.

### Raw protocol check

```bash
# Agent card
curl http://localhost:8002/.well-known/agent-card.json

# Send a message
curl -X POST http://localhost:8002/ -H "Content-Type: application/json" -d '{
  "jsonrpc":"2.0","id":"1","method":"message/send",
  "params":{"message":{"messageId":"m1","role":"user",
    "parts":[{"kind":"text","text":"Hello"}]}}}'
```

---

## 4. Adding a new agent

The workflow is the same as before — **define the agent in `agents/`** — plus
two small steps to expose it over A2A.

### Step 1 — Define the agent (as usual)

Create `agents/my_agent.py`:

```python
from google.adk.agents.llm_agent import LlmAgent
from CoScientist.agents.common import agent_tools, make_llm
from CoScientist.agents.prompts import my_instruction

my_agent = LlmAgent(
    name="MyAgent",
    model=make_llm(),                 # shared model/config from common.py
    instruction=my_instruction,       # your prompt
    description="What this agent does — the orchestrator reads this to route.",
    output_key="my_results",
    tools=agent_tools(my_toolset, hitl=True),  # hitl=True adds approval tools
)
```

`agents/common.py` gives you `make_llm()`, `agent_tools()`, `MODEL`,
`hitl_enabled` so every agent shares one settings load.

(Optional) re-export it from `agents/__init__.py` and `agents/agents.py` for
in-process use.

### Step 2 — Add a port

In `a2a/config.py`, add an entry to `AGENT_PORTS`:

```python
"my_agent": int(os.getenv("MY_AGENT_PORT", "8006")),
```

`AGENT_URLS` and `AGENT_CARD_URLS` are derived automatically.

### Step 3 — Create the server module

Create `a2a/servers/my_agent.py` (copy an existing one):

```python
import uvicorn
from a2a.types import AgentCard, AgentCapabilities, AgentSkill
from CoScientist.a2a.config import AGENT_PORTS, AGENT_URLS
from CoScientist.a2a.server import make_a2a_app
from CoScientist.agents.my_agent import my_agent

PORT = AGENT_PORTS["my_agent"]

_card = AgentCard(
    name="MyAgent",
    description="What this agent does",
    url=AGENT_URLS["my_agent"],
    version="1.0.0",
    capabilities=AgentCapabilities(streaming=True),
    defaultInputModes=["text/plain"],
    defaultOutputModes=["text/plain"],
    skills=[AgentSkill(id="do", name="Do", description="...", tags=["..."])],
)

app = make_a2a_app(my_agent, _card, "my_agent")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
```

`make_a2a_app()` handles the Runner, A2A executor, and Opik tracing for you.

> **Note on imports:** splitting agents into per-file modules is for code
> organisation. It does **not** currently give import-time isolation between
> servers — the top-level `CoScientist/__init__.py` eagerly imports the full
> agent tree (incl. the RAG/FEDOT stack), so every server transitively loads
> all agents at startup. Making that lazy is a possible follow-up.

### Step 4 — Register with the orchestrator

In `a2a/orchestrator.py`, add a `RemoteA2aAgent` inside `_build_tools()`:

```python
RemoteA2aAgent(
    name="MyAgent",
    agent_card=AGENT_CARD_URLS["my_agent"],
    description="What this agent does — the orchestrator routes on this.",
),
```

### Step 5 — Add it to `run_all.py`

Import its `app` and add a `(label, app, port)` tuple to `_SERVERS`.

That's it. Run `run_all`, then
`python -m CoScientist.a2a.benchmark --agent my_agent --text "..."`.

---

## 5. Notes & gotchas

- **AgentCard required fields:** `name`, `description`, `url`, `version`,
  `capabilities`, `defaultInputModes`, `defaultOutputModes`, `skills`. Omitting
  any raises a pydantic validation error at import.
- **`streaming=True`** must be set on the card or the server rejects
  `message/stream` requests.
- **HITL agents block on console input.** `planner`, `research`,
  `task_execution` use `request_approval`/`request_selection`, which read from
  *their server's* stdin — over A2A that means the request hangs. For automated
  testing set `HITL__ENABLED=false` in `.env`, or test non-HITL agents
  (`hypotheses`, `medical`).
- **`task_execution`** needs the RAG DB + FEDOT.MAS reachable for its
  tool-discovery step.
- **JSON-RPC errors return HTTP 200** with the error in the body (per spec) —
  don't treat 200 as unconditional success.
- **Agent card endpoint:** prefer `/.well-known/agent-card.json`;
  `/.well-known/agent.json` is deprecated in the A2A SDK and will be removed.
