---
id: S002
type: paper
title: "Voyager: An Open-Ended Embodied Agent with Large Language Models"
url: https://arxiv.org/abs/2305.16291
venue: arXiv 2305.16291 / TMLR
trust: partial
used_by: [F002]
tags: [skill-library, lifelong-learning, code-as-skill]
---

## Idea extracted
Three parts: (1) an **automatic curriculum** that proposes next tasks, (2) an
**ever-growing skill library** where skills are stored as *executable code* and
retrieved by embedding over their description, (3) iterative prompting that feeds
execution errors + self-verification back into program refinement.

## How we used it
F002 (CoderAgent) adopts the **skills-are-runnable-code** stance: the agent writes
and runs real code in a sandbox rather than emitting prose plans. We did **not**
adopt the automatic curriculum or the persistent, retrieval-indexed skill DB —
F002's sandbox work is currently one-shot, not accumulated as reusable skills.

## Verification log
- 2026-06-11 — `inspirational` → `partial`: the code-as-skill idea shipped in F002
  (commit `f863802`); curriculum + skill-library-retrieval remain unimplemented.
  Re-open toward `verified` if/when F002's "persist sandbox scripts as reusable
  skills" TODO lands with an eval.
