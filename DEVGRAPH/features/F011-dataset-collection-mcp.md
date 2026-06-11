---
id: F011
title: Dataset-collection MCP server
type: feature
status: done
created: 2026-06-11
updated: 2026-06-11
owners: [ITMO-NSS-team]
derives_from: [F000]
depends_on: [F000, F012]
sources: []
tags: [dataset, mcp, papers]
code:
  - mcp-servers/dataset-collection-mcp-server/dataset_collection_server.py
  - mcp-servers/dataset-collection-mcp-server/prompt.py
benchmarks: []
---

## Goal
An MCP server that builds datasets from papers (collect → structure), so agents
can assemble datasets without bespoke code.

## Current state
Shipped in #196 (commit `875487b`) alongside a papers-search fix; later patched in
#269 (`b5e30cc`). Lives in `mcp-servers/dataset-collection-mcp-server/`.

## Attempts
### F011.A1 — dataset-collection MCP (#196) · earlier · outcome: success
- **Method:** stand up an MCP server that turns paper content into datasets.
- **Evidence:** commit `875487b` (#196); `dataset_collection_server.py`.
### F011.A2 — fix in dataset-collection MCP (#269) · 2026-06-10 · outcome: success
- **Method:** bug fix in the dataset-collection server.
- **Evidence:** commit `b5e30cc` (#269).

## ✅ TODO
- [ ] "Create dataset from papers" is still unchecked in tools_checklist.md — confirm
      coverage/wiring end-to-end.

## ⚠ Pitfalls / Known problems
- Depends on papers-search (F012) for inputs; an empty/failed search yields empty datasets.

## Symbols
- `mcp-servers/dataset-collection-mcp-server/dataset_collection_server.py` — MCP entrypoint.
