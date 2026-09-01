# Merge main → hypothesis-subsystem-a2a-mcp + review fixes + test rewrite

## Context & git state (verified from `.git/`)

| Ref | Commit |
|-----|--------|
| HEAD / current branch `hypothesis-subsystem-a2a-mcp` | `09ee672` |
| `origin/hypothesis-subsystem-a2a-mcp` | `09ee672` (in sync, no unpushed commits) |
| local `main` | `4d622ac` (STALE) |
| `origin/main` | `477bac14` (newer — must be used for the merge) |
| `origin/hypothesis-mvp-branch` | `d7b649d` (source of the reviewed PR) |
| `origin/hypothesis-evolution` | `03bbb4d` |
| `origin/graph_refactoring` | `8b50d2e` |

Key implication: the review comments cite line numbers that do **not** exist on the
current branch (`bindings.py#L674-L698`, `generator_agent.py#L40/#L105/#L143-L154`,
`schema.py#L408-L417`, `validator.py#L138`, `queries.py#L146/#L216`,
`templates.py#L1301-L1306`, `system.yaml#L161/#L170/#L171`). Those files/lines live on
the **post-graph `main`** (`477bac14`). The merge will introduce the Research Graph
subsystem (`validator.py`, `queries.py`, node-permission schema, `ResultAggregator`).
All review fixes must therefore be **re-anchored to the merged code**, not applied
blindly to the current files.

Current branch inconsistencies to note:
- [`tests/unit/test_hypothesis_assembly.py`](tests/unit/test_hypothesis_assembly.py:44) already asserts a
  **target** state (`tools == [generate_via_moosechem, run_critic_loop]`, canonical
  callback list `[inject_state, before_get_task, inject_research_context]`, populated
  guard whitelist) that the current implementation does NOT satisfy — it is aspirational
  and currently failing. It encodes exactly the review's bugs #1 and #2.
- `HypothesisLoopCoordinator` (the critic loop) exists but is NOT wired into the
  generator; the generator only exposes `retrieve_validation_tools` + `generate_via_moosechem`.

## Decisions (confirmed with user)

1. **Merge target** = latest `origin/main` (`477bac14`). Local `main` is stale and must
   be fast-forwarded/updated first. Continue all work on `hypothesis-subsystem-a2a-mcp`.
2. **Critic loop wiring** = wire `run_critic_loop` as an internal ADK tool of the
   subsystem, **declared in `system.yaml` and registered in `bindings.py`** — the same
   fix as review #2, matching the expectation already encoded in
   `test_hypothesis_assembly.py`.
3. **Graph commit** = programmatically commit final ACTIVE hypotheses to the Research
   Graph, **best-effort via `CoScientist.graph.client`**, at the end of the critic loop,
   using the node kinds the merged `main` provides.

## Target data flow

```mermaid
flowchart LR
    U[User / Orchestrator] -->|delegate| H[HypothesesAgent AgentTool]
    H --> G[Generator LlmAgent]
    G --> R[retrieve_validation_tools]
    R -->|tool_catalog| C[Context enrichment]
    C --> M[generate_via_moosechem -> MooseChemMCPTool]
    M --> K[run_critic_loop -> HypothesisLoopCoordinator]
    K --> V[Critic critique + RAG enrichment]
    V --> ACT[Final ACTIVE hypotheses]
    ACT --> GR[(Research Graph - best effort)]
    ACT -->|HypothesisList| U
```

## Phases

### Phase 0 — git reconciliation
- Fetch latest `origin`; update local `main` to `origin/main` (`477bac14`).
- Confirm working tree on `hypothesis-subsystem-a2a-mcp` is clean.

### Phase 1 — merge main into current branch
- `git merge main` on `hypothesis-subsystem-a2a-mcp`; resolve ALL conflicts.
- Expected conflict hotspots: `CoScientist/agents/system.yaml` (HypothesesAgent vs
  main's graph/orchestrator/ResultAggregator agents), `assembly/assembler.py`,
  `assembly/bindings.py`, `assembly/schema.py` (node-permission schema), `agents/prompts/templates.py`.
- Gate: `python -m CoScientist.assembly` validates; `tests/unit/test_assembly.py` passes (no LLM calls).

### Phase 2 — review fixes, re-anchored to merged code
- **#1 callback list TypeError**: normalize `before_agent_callback` to ADK canonical
  list form (`[_inject, *callbacks]`) in `HypothesisSubsystemAgent.__init__`; drop the
  hand-rolled `composed_before_agent`.
- **#2 guard whitelist**: declare `generate_via_moosechem` + `run_critic_loop` in
  `system.yaml` (`tools:`) and register them in `bindings.py` so the `guard_unknown_tools`
  whitelist is populated; OR drop `after_model: [guard_unknown_tools]` for this agent.
- **#3 graph contract**: programmatic commit of final ACTIVE hypotheses to the Research
  Graph at the end of the critic loop.
- **Non-blocker**: match MooseChem scores to hypotheses by LLM-returned `index`
  (`moosechem_tool.py` positional-sort bug).
- **Non-blocker**: pass minimal `provenance` in the fallback path (`generator_agent.py`
  except-wrapper) to avoid `ValidationError`.

### Phase 3 — wire critic loop + tool retrieval
- Expose `run_critic_loop` as an internal FunctionTool; `HypothesisLoopCoordinator`
  runs generator output through critic (critique + RAG context enrichment).
- Ensure `retrieve_validation_tools` enriches context (tool catalog) before generation.

### Phase 4 — rewrite the two tests
- **Test 1 (structure)**: generator → MooseChem MCP tool; critic critiques and enriches
  context; context is enriched with tools from tool retrieval.
- **Test 2 (full pipeline)**: orchestrator delegates to the subsystem; the full
  generator→MCP→critic-loop runs; hypotheses return upward and proceed to verification.
- Remove/repair legacy broken tests (`test_hypothesis_assembly.py` mismatch; integration
  scripts with wrong `PROJECT_ROOT`, typo dir, missing `test_cases.json`).

### Phase 5 — verification
- Full `pytest` green (unit no-LLM + the two new tests where MCP/graph are mocked).
- Smoke: `build_system()` succeeds; commit on `hypothesis-subsystem-a2a-mcp`.
