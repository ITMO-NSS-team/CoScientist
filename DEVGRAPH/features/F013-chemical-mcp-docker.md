---
id: F013
title: Chemical MCP server + docker infra
type: feature
status: done
created: 2026-06-11
updated: 2026-06-11
owners: [ITMO-NSS-team]
derives_from: [F000]
depends_on: [F000]
sources: []
tags: [chemistry, mcp, docker, rdkit, docking, retrosynthesis]
code:
  - mcp-servers/chemical-mcp-server/
  - mcp-servers/chemical-mcp-server/docker-compose.yml
  - CoScientist/chemical_utils/
benchmarks: []
---

## Goal
Expose chemistry capabilities (docking, retrosynthesis, reaction prediction/
classification, OCR from figures, RDKit props, IUPAC↔SMILES, ChEMBL/BindingDB
activities) as a dockerized MCP server.

## Current state
First MCP server + docker-compose shipped in #187 (commit `21a6ed4`); S3 utils
added in #197 (`45db27b`). The bulk of verified MCP-covered capabilities in
`tools_checklist.md` come from here. In-repo: `mcp-servers/chemical-mcp-server/`
(with `server/`, `docker-compose.yml`) + `CoScientist/chemical_utils/`.

## Attempts
### F013.A1 — chemical MCP server + docker-compose (#187) · earlier · outcome: success
- **Method:** package chemistry tools as an MCP server with docker-compose.
- **Evidence:** commit `21a6ed4` (#187); `mcp-servers/chemical-mcp-server/`.
### F013.A2 — S3 utils for chemical MCP (#197) · earlier · outcome: success
- **Method:** add S3 storage utilities for the chemical server.
- **Evidence:** commit `45db27b` (#197).

## ✅ TODO
- [ ] Extract molecules/reactions **from PDF** still uncovered (only "from figure" is) — tools_checklist.md.
- [ ] "Training models for molecule generation" + AutoML property prediction not covered.

## ⚠ Pitfalls / Known problems
- Heaviest infra: needs docker + (per README) OpenChemie/Chroma services. Several
  chemistry features depend on these containers being up.

## Symbols
- `mcp-servers/chemical-mcp-server/docker-compose.yml` — service composition.
- `CoScientist/chemical_utils/` — molecule/reaction/docking/OCR utilities.
