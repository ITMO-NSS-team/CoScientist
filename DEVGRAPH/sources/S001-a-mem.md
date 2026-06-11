---
id: S001
type: paper
title: "A-MEM: Agentic Memory for LLM Agents"
url: https://arxiv.org/abs/2502.12110
venue: NeurIPS 2025
trust: inspirational
used_by: [DEVGRAPH-design]
tags: [memory, zettelkasten, knowledge-graph]
---

## Idea extracted
Store agent memory as **atomic, linked notes** (Zettelkasten). Each note carries
structured attributes — a contextual description, keywords, tags — and is linked
to related notes. Links and note contents **evolve**: adding a new note can update
the descriptors of older, related notes. This beats fixed store/retrieve schemas
because organization adapts to the task.

## How we used it
DEVGRAPH's core shape: feature/source nodes as small Markdown files with YAML
frontmatter (= structured attributes), typed links between IDs, and an explicit
"update related nodes when you learn something" protocol (README §7). We dropped
A-MEM's automatic embedding-driven linking in favor of explicit IDs + `grep`,
because a coding repo wants deterministic, diff-able, reviewable edges over a
vector index.

## Verification log
- 2026-06-11 — set `inspirational`: design influence on the note/link model; we did
  not implement A-MEM's algorithm, so there's nothing to empirically verify here.
