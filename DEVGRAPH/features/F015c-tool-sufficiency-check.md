---
id: F015c
title: MCP inventory index + per-step tool-sufficiency / capability-gap callable
type: feature
status: proposed
created: 2026-06-11
updated: 2026-06-11
owners: [SoloWayG]
derives_from: [F015]
depends_on: [F009, F005]
sources: [S019, S020, S021]
tags: [tool-selection, capability-gap, mcp, retrieval, reliability]
code:
  - CoScientist/tools/retrieval_tools.py:RetrievalToolSet   # retrieve_tools (live MCP index)
  - CoScientist/tools/servers_web_search.py:search_mcp_servers
benchmarks: []
---

## Goal
The shared, standalone callable at the heart of the АМ: given a plan step's
`required_tools` and the **live MCP inventory**, decide **sufficient** (dispatch) vs a
concrete **capability gap** (→ repo-search F015d → Alembic F015e). Built FIRST; consumed by
F015a (inventory injection), F015b (its deterministic oracle), and F015g (dispatch gate).

## ⚠ REUSE the existing RAG-over-tools seam (dedup, 2026-06-12)
The substrate already exists and is wired end-to-end **per whole task**: the master
orchestrator calls RAG-over-tools (`retrieval_tools.py:retrieve_tools` → writes
`state['filtered_tools']`), and `fedotmas_tools.py:fedot_tool` already consumes it
(`filtered_tools` → `server_ids` → Postgres lookup → `servers_payload`). F015c does NOT build a
new retrieval stack — it **drives this existing seam per plan-step** and adds what's missing:
the sufficiency verdict (sufficient / Type-A gap / Type-B gap), fail-closed on
backend-unavailable, the timeout-guarded probe, and the enriched index representation.

## Best practices to adopt
- **MCP-Zero — request-driven hierarchical routing [S019]:** treat each `required_tool` as a
  structured capability REQUEST; two-stage server-filter → tool-rank over the live inventory
  (reuse F009 RAG + F005 discovery). Tools bound from the live inventory at dispatch → a
  sub-agent never invents a name. (~98% context-token reduction across ~3000 tools.)
- **RAG-MCP — probe + chance-corrected threshold [S020]:** last gate before "sufficient" =
  ONE sandboxed, **timeout-guarded** probe call confirming reachability + signature; calibrate
  the threshold vs a distractor baseline (not a magic cosine). Retrieve **per required_tool**;
  size the handed subset adaptively.
- **AnyTool — Type-A vs Type-B gap + bounded re-retrieval [S021]:** classify each gap —
  Type A (no candidate tool → F015d/F015e build) vs Type B (tools exist, plan wrong → back to
  F015b). Before declaring a true Type-A gap, re-retrieve with a broadened/synonym query;
  only escalate if still empty. Ground in retrieval EVIDENCE, never an LLM "can you?" self-report.

## ⚠ Risks / open questions (adversarial review — read this)
- **F015c alone does NOT kill "Tool 'X' not found" (claim re-scoped):** F014's failure is
  raised INSIDE the FEDOT.MAS `molecule_generator` sub-agent because FEDOT.MAS's own
  meta-agent (`mas_gen.py`) under-equips an internally-generated worker; tools bind at
  **server granularity** (`registry.py:create_toolset`). A pre-dispatch check at the
  CoScientist layer ensures the right **servers** reach FEDOT.MAS and detects true gaps, but
  the meta-agent can still name a tool from a server it wasn't given. Honest scope: F015c =
  "right servers + true-gap detection"; actually bounding the sub-agent's kit needs F015g's
  granularity decision (F015g.D1) **or** enriching CoScientist-owned server **descriptions**
  (allowed per F014.D1; not third-party). 
- **Retrieval-oracle fail-closed:** `retrieve_tools` returns `status:'unavailable', result:[]`
  when the rag_tools/Postgres backend is down (observed F014.A2, `:5432` refused). A naive
  oracle would then declare EVERYTHING a Type-A gap → needless infra-heavy Alembic builds.
  Must distinguish "no match" from "backend unavailable" and **fail closed to HITL/abort**.
- **Probe must be timeout-guarded** (cf. F003.A2: an SSE/`list_tools` hang killed a whole run).
- **Index representation:** third-party MCPs have thin docs → embed name+description+synthetic
  example calls. You may rewrite the INDEX representation but NOT the tool (F014.D1).
- **Serves the planner's inventory + normalizes server names (found in F015a.A2):** the plan
  carries `tool_servers: [{server, tools}]` (F015a, `plan.py:ServerTools`); F015c supplies the
  **server-grouped** inventory the planner reads AND resolves each `(server, tool)` against the
  live index. With a static stand-in the planner mis-named a server (`chemical-mcp` vs exact
  `chemical-mcp-server`). Treat a near-miss server name as a **correctable** gap (fuzzy-resolve
  to the real server), not a hard failure; an unresolvable one is a true Type-A gap.

## ✅ TODO
- [ ] Build the live-MCP inventory index (name+description+synthetic calls) as the shared substrate.
- [ ] `sufficiency(step) -> {sufficient | gap: TypeA|TypeB, evidence}` callable (fail-closed on backend-unavailable).
- [ ] Timeout-guarded reachability probe; threshold calibrated vs distractors; bench on dataset_S (F015h).

## Symbols
- `CoScientist/tools/retrieval_tools.py:RetrievalToolSet` — `retrieve_tools` (note: `[]` on backend down).
- `CoScientist/tools/servers_web_search.py:search_mcp_servers` — web-registry discovery.
