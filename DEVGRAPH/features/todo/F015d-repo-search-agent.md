---
id: F015d
title: Repo-search agent — find code repos (literature-cited links first, then public platforms)
type: feature
status: proposed
created: 2026-06-11
updated: 2026-06-11
owners: [SoloWayG]
derives_from: [F015]
depends_on: [F003, F005]
sources: [S022, S023]
tags: [repo-discovery, literature, code-search, alembic-upstream]
code: []
benchmarks: []
---

## Goal
On a true Type-A capability gap (F015c), find candidate code repositories — **priority:
links cited in the collected literature** (F003), then public platforms by keyword — and
hand a ranked, applicability-annotated list to Alembic (F015e). Currently ABSENT (Alembic
is handed a bare `repo_url`).

## Best practices to adopt
- **AutoSOTA — tiered discovery cascade + cheap actionability selector [S022]:** (1) extract
  repo links from paper front matter (GitHub + `github.io` project pages); (2) full-document
  scan (abstract → data-availability → body → references); (3) keyword search by title +
  arXiv id + DOI. Rank multiple candidates with an LLM "actionability" selector over only
  {abstract, shallow file tree, README head}; dedup forks; drop template repos. **Do not rely
  on Papers-With-Code live API (retired)** — archived dump as a static seed only.
- **SUPER / Sci-Reproducer / Installamatic — grounded inspection before spending build budget
  [S023]:** a minimal grounded action set (AST def lookup, file search, presence-check of
  README/requirements/setup.py/pyproject/Dockerfile/CI) to decide IF a repo implements the
  method; prefer repos whose docs/CI actually exist (cheap installability pre-filter).
  Always VERIFY the matched repo corresponds to the paper (README cites the title) — reference
  extraction is false-positive-prone.

## ⚠ Risks / open questions
- Best end-to-end research-repo setup rates are low (SUPER ~16%) → F015d must surface **"no
  usable repo found"** as a first-class planner outcome (HITL handoff), not a loop.
- **Provenance carry-through (gap):** must pass `{hypothesis, citing-paper, capability-desc}`
  from the triggering plan step into F015d→F015e so the built tool is traceable — add a
  `provenance` field to the F015a step schema (currently no home for it).

## ✅ TODO
- [ ] Literature-first cascade (link extraction from F003 papers → keyword/public search via F005).
- [ ] Actionability ranker over {abstract, file tree, README}; repo↔paper match verification.
- [ ] "No repo found" → HITL outcome; carry provenance into F015e.
