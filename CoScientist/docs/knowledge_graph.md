# CoScientist Knowledge Graph

## Concept

The multi-agent system now builds an explicit **graph of the research process**
instead of leaving it implicit in agent chatter. The thesis (Proposition 1 —
"MAS for science, differentiated by the graph structure of research"): making the
process a first-class typed graph buys us three things —

1. **Interpretability** — you can see exactly what each agent did and why.
2. **Reproducibility** — the graph is a deterministic, replayable protocol of a run.
3. **Performance (graph memory)** — facts established in earlier runs are retrieved
   and fed back into later runs, so the system builds on prior findings instead of
   rediscovering them. This is the lever we expect to move the benchmark metrics.

It is exposed in three **views** of one underlying structure:

| View | Scope | Answers | How it's built |
|------|-------|---------|----------------|
| **execution** | one run | *what the system did* | recorded live from agent events |
| **knowledge** | one run | *what we learned this run* | deterministic projection (no LLM) |
| **memory** | all runs | *what we know overall* | LLM entity/relation extraction, accumulated |

## Architecture

```
USER QUERY
  │  (execution graph, live)
  ▼
goal ── OrchestratorAgent ── agent_call: TaskExecutorAgent ── agent_call: ExperimentAgent ── tool_call: dock(...)
              │                                                                                     │
              └── result  ◀──────────────────────────────────────────────────────────────────────┘
                    │
                    │  (semantic extraction, LLM, on the final answer)
                    ▼
        Entities/Relations  ──ingest──►  KNOWLEDGE MEMORY (persistent, cross-run)
                                              │  target:GSK-3beta, molecule:CCO {docking:-9.2}, …
                                              │
   next run ◀── inject "Known from prior runs: …" ◀── relevant(query)
```

**The cross-run loop is the point:** run *N* produces a final answer → we extract
typed domain entities/relations from it → merge them into a persistent memory
graph (dedup by canonical key, accumulate mentions/sources). Before run *N+1*'s
orchestrator/planner act, we retrieve the entities relevant to the new query and
inject them into context. That is the "graph memory" the ablation study targets
(base vs +memory vs +MCP vs +Fedot).

## One data-driven template (no class-per-entity)

Everything is `Node` + `Edge` (`graph/models.py`). A node's *kind* and an entity's
*domain type* are **strings in data**, never Python subclasses, so new entity or
relation types need zero code changes:

- `Node.kind`: system | agent | goal | agent_call | tool_call | result | **entity**
- an entity node carries its domain type in `semantic.type` (`target`, `molecule`,
  `metric`, `paper`, `hypothesis`, `method`, …) and its fields in `input` (attrs).
- `Edge.type` is a free string — control-flow (`caused_by`, `delegated_to`,
  `produced`) and knowledge relations (`has_property`, `about`, `supports`, …).

The LLM extraction returns exactly this template:
`{entities:[{key,type,name,attrs}], relations:[{src,dst,type}]}` (`graph/semantic.py`).

## How agents interact with it

- **Root on start:** the orchestrator and planner get the graph root up front
  (every agent + capabilities + this session's trace) **plus relevant prior-run
  memory** — `inject_graph_root` → `{graph_root?}` in their prompts.
- **Read anytime:** every reasoning agent has graph tools
  (`read_research_graph` / `get_graph_history` / `get_agents_info`).
- **Grows automatically:** `GraphMemoryPlugin` (an ADK plugin on the Runner)
  records goals, delegations, tool calls and results — no agent bookkeeping.

## Visualization

Live web page `/graph` (vis-network, vendored — works offline): nodes are
**coloured by status** (running/success/failed) and **shaped by type**; click any
node for full detail incl. the prompt an agent was called with; a **view switch**
toggles execution / knowledge / memory; updates live (1.5 s poll). Also a CLI:
`python -m CoScientist graph viz|show|dot --view {execution,knowledge,memory}`.

## Code map

```
graph/
  models.py        Node/Edge/Semantic schema (the one template)
  store.py         NetworkX-backed store + JSON snapshots (per run)
  memory.py        in-process session graph + root seeding (agent roster)
  plugin.py        GraphMemoryPlugin — grows the execution graph from events,
                   and (when enabled) triggers semantic extraction on the result
  agent_tools.py   read-only graph tools given to every agent
  knowledge.py     execution → knowledge-view projection (deterministic, no LLM)
  semantic.py      Entity/Relation/Extraction template + LLM extractor (Option B)
  memory_store.py  KnowledgeMemory — persistent cross-run entities/relations
  viz.py           to_dot / interactive HTML (colour by status, shape by type)
agents/callbacks/tool_callbacks.py
                   inject_graph_root — root + relevant prior-run memory into prompts
web/app.py         /graph page + /api/graph?view={execution,knowledge,memory}
web/templates/graph.html   live, clickable, view-switching graph UI
```

Wiring: the `graph` tool and `inject_graph_root` callback are registered in
`assembly/bindings.py` and attached in `agents/system.yaml`; the plugin is added
to the Runner in `main.py`.

## Flags

| flag | default | effect |
|------|---------|--------|
| `LOG_AGENT_EVENTS=0` | on | turn the graph/logging off |
| `KG_SEMANTIC_ENABLED=0` | **on** | LLM entity extraction into memory (1 small LLM call per query); set 0 to disable |
| `KG_SEMANTIC_MODEL` | main model | model used for extraction |
| `KG_MEMORY_PATH` | `graph_runs/knowledge_memory.json` | persistent memory file |
| `GRAPH_SNAPSHOT_DIR` | `./graph_runs` | per-run execution-graph snapshots |

## Status & next steps

- Execution + knowledge views and the cross-run memory loop are implemented and
  tested (deterministically; semantic extraction verified with a mocked LLM).
- Semantic extraction is off by default — flip `KG_SEMANTIC_ENABLED=1` (needs the
  LLM/VPN) to populate the memory from real runs.
- For the paper/ablation: vary graph-growing strategy (plan-all-at-once vs
  incremental vs hybrid) and toggle the memory layer; measure task metrics,
  redundant-call count, and result variance (reproducibility) on the chem
  benchmarks where targets/scores are quantitatively checkable.
