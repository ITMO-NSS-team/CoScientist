---
id: F007
title: Paper analysis & parsing pipeline (+ MCP server)
type: feature
status: done
created: 2026-06-11
updated: 2026-06-11
owners: [ITMO-NSS-team]
derives_from: [F000]
depends_on: [F000]
sources: []
tags: [papers, parsing, marker, chroma, rag, mcp]
code:
  - CoScientist/paper_parser/parse_and_split.py
  - CoScientist/paper_parser/s3_connection.py
  - CoScientist/paper_analysis/chroma_db_operations.py
  - CoScientist/paper_analysis/question_processing.py
  - CoScientist/paper_analysis/research_taxonomy.py
  - mcp-servers/paper-analysis-mcp-server/paper_analysis_server.py
benchmarks: []
---

## Goal
Parse scientific PDFs (Marker), chunk semantically, store in Chroma + S3, and
answer questions over papers — exposed as an MCP server for agents to call.

## Current state
Built across #204 (paper-analysis MCP server, `26affcf`), #239 (new paper
analysis, `9db0007`), and #256 (updated paper analysis, `9e0a0ec`).
`paper_parser/` handles parsing/splitting and S3; `paper_analysis/` handles Chroma
ops, question processing, domain metadata, and research taxonomy; the MCP server
(`paper_analysis_server.py`) exposes it. Consumed by ResearchAgent (F003).

## Attempts
### F007.A1 — Paper analysis MCP server (#204) · earlier · outcome: success
- **Method:** stand up an MCP server wrapping parse → chunk → store → query.
- **Evidence:** commit `26affcf` (#204); `mcp-servers/paper-analysis-mcp-server/`.
### F007.A2 — New + updated paper analysis (#239, #256) · 2026-06 · outcome: success
- **Method:** rework chunking/QA + Chroma operations and taxonomy/metadata.
- **Evidence:** commits `9db0007` (#239), `9e0a0ec` (#256); `paper_analysis/*`.

## ✅ TODO
- [ ] Extract molecules/reactions **from PDF** is still uncovered (see tools_checklist.md).
- [ ] No retrieval-quality eval over the paper QA.

## ⚠ Pitfalls / Known problems
- Marker parsing + LLM cleanup is slow and S3-dependent; failures here cascade into
  empty ResearchAgent results (F003). Check S3 + Marker before blaming the agent.

## Symbols
- `CoScientist/paper_parser/parse_and_split.py` — Marker parse + semantic chunking.
- `CoScientist/paper_analysis/chroma_db_operations.py` — vector store ops.
- `CoScientist/paper_analysis/question_processing.py` — QA over chunks.
- `mcp-servers/paper-analysis-mcp-server/paper_analysis_server.py` — MCP entrypoint.
