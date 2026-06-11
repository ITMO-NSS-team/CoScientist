---
id: S009
type: internal
title: "ТП НИРСИИ — Агентный модуль автоматизации планирования и проведения вычислительных экспериментов (RU.СНАБ.01074-01)"
url: CoScientist/ТП_НИРСИИ от 02.06_согласован.pdf
venue: Approved technical project, 2026-06-02 (30 pp.)
trust: unverified
used_by: [F015]
tags: [spec, requirements, experiments-module, alembic]
---

## Idea extracted
The approved design for the experiments module (АМ) that replaces the current
FEDOT.MAS path. Core ideas:
- **Plan / execute separation:** an orchestrator emits a structured step-by-step JSON
  plan (each step = subtask + required tools + run params + expected artifacts);
  steps are dispatched to FEDOT.MAS **one at a time**, not as one overloaded query.
- **Plan critic** with a bounded iteration budget before execution.
- **Per-step tool-sufficiency check**; if insufficient → repo-search → **Alembic**
  builds a new containerized MCP server (explorer→environment→coder→validator,
  docker commit, streamable-http, registered & reusable) — **without modifying
  existing tools or AM code**.
- Control info as text/JSON between agents; file artifacts in S3 as presigned URLs.

The ТП explicitly names the current defect: handing FEDOT.MAS one excessive query
raises the rate of planning errors / failed experiments *even when all needed tools
exist* — i.e. the F014 failure modes.

## How we used it
Grounds feature [[F015]] (the experiments-module redesign) and the fix direction in
[[F014]] (F014.D1): fix the orchestration, not the MCP tools.

## Verification log
- 2026-06-11 — `unverified`: this is an approved *design/requirements* doc, not an
  empirical result. Flip toward `verified`/`partial` once F015 is implemented and
  benched against dataset_S (does plan+sufficiency actually cut loops & tool-not-found?).
