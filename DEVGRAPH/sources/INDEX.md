# DEVGRAPH — Sources (citation sub-graph)

> Inspirations & citations: papers, repos, docs. Each has a **trust** value that
> changes as ideas are actually tried in this repo (see README §6). A source gets
> its own file once it has a real extracted idea / trust history / multiple users;
> otherwise a registry row here is enough.

**Trust:** `verified` · `partial` · `refuted` · `unverified` · `inspirational`.

| ID | Type | Title | Trust | Used by | Link |
|----|------|-------|-------|---------|------|
| [S001](./S001-a-mem.md) | paper | A-MEM: Agentic Memory for LLM Agents (NeurIPS 2025) | inspirational | DEVGRAPH design | arXiv:2502.12110 |
| [S002](./S002-voyager.md) | paper | Voyager: Open-Ended Embodied Agent w/ LLMs | partial | F002 | arXiv:2305.16291 |
| S003 | paper | AriGraph: KG world models w/ episodic memory | inspirational | DEVGRAPH design | arXiv:2407.04363 |
| S004 | doc | Architecture Decision Records (ADRs) | inspirational | DEVGRAPH design | https://adr.github.io |
| S005 | paper | Codebase-Memory: tree-sitter KG over codebase via MCP | inspirational | DEVGRAPH design | arXiv:2603.27277 |
| S006 | paper | From Experience to Strategy: Trainable Graph Memory | inspirational | DEVGRAPH design | arXiv:2511.07800 |
| S007 | blog | Google "AI co-scientist" (generate→critique→rank→evolve) | inspirational | F000 | research.google / Nature 2026 |
| S008 | doc | HITL control patterns in agentic systems (PLACEHOLDER — replace w/ the real paper/repo F001 drew from) | unverified | F001 | — |
| [S009](./S009-tp-experiments-module.md) | internal | ТП НИРСИИ — experiments module (АМ) design + Alembic | unverified | F015, F014 | `CoScientist/ТП_НИРСИИ от 02.06_согласован.pdf` |

Next free ID: **S010**.

## Notes
- **S008 is a placeholder.** Whoever continues F001 should replace it with the
  actual source the HITL design was taken from (paper or repo) and set trust
  from evidence.
- Reserve full files for sources whose *idea you implement* — that's where the
  verified/refuted lifecycle earns its keep.
