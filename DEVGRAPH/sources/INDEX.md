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

## АМ / experiments-module (F015–F015h) best-practice sources

> ⚠ URLs/ids below were surfaced by research agents (workflow `am-module-decomposition-research`,
> 2026-06-11) or taken from the ТП bibliography. Treat as **leads** — `trust: unverified`,
> verify before external citation. Each informs the sub-feature in "Used by".

| ID | Type | Title (short) | Used by | Link |
|----|------|---------------|---------|------|
| S010 | paper | Planning with Multi-Constraints via Collaborative Language Agents (ТП[1]) | F015a, F015b | COLING 2025 |
| S011 | paper | A Survey on the Feedback Mechanism of LLM-based AI Agents (ТП[2]) | F015b | IJCAI 2025 |
| S012 | paper | Creativity in LLM-based Multi-Agent Systems: A Survey (ТП[3]) | F015 | EMNLP 2025 |
| S013 | paper | Routine (planner names exact tools; large planning gains) | F015a | arxiv.org/html/2507.14447 |
| S014 | paper | ReWOO (plan as variable-binding DAG, ~5× fewer tokens) | F015a | arxiv.org/abs/2305.18323 |
| S015 | paper | Data Interpreter (runtime sub-DAG re-planning) | F015a | arxiv.org/abs/2402.18679 |
| S016 | paper | SOPStruct (deterministic structural plan gate) | F015b | arxiv.org/html/2504.00029 |
| S017 | paper | LLMs Cannot Self-Correct Reasoning Yet (Huang, ICLR 2024) | F015b | arxiv.org/abs/2310.01798 |
| S018 | paper | Self-Refine (bounded iterative refinement) | F015b | arxiv.org/abs/2303.17651 |
| S019 | paper | MCP-Zero (request-driven hierarchical tool routing) | F015c, F015h | arxiv.org/abs/2506.01056 |
| S020 | paper | RAG-MCP (retrieve per tool; probe; threshold) | F015c, F015g | arxiv.org/abs/2505.03275 |
| S021 | paper | AnyTool (Type-A/B gap + bounded re-retrieval) | F015c | arxiv.org/abs/2402.04253 |
| S022 | paper | AutoSOTA (tiered repo discovery + actionability selector) | F015d | arxiv.org/abs/2604.05550 |
| S023 | paper | SUPER / Sci-Reproducer (grounded repo inspection) | F015d | arxiv.org/abs/2504.00255 |
| S024 | paper | ToolMaker / Code2MCP (env-reset-on-retry, thin wrapper) | F015e | arxiv.org/abs/2502.11705 |
| S025 | paper | ResearchEnvBench (validator readiness ladder) | F015e | arxiv.org/abs/2603.06739 |
| S026 | paper | Repo2Run / EnvBench (deterministic-install-first, commit/rollback) | F015e | arxiv.org/abs/2502.13681 |
| S027 | paper | ScaleMCP (CRUD auto-sync registration index) | F015f | arxiv.org/abs/2505.06416 |
| S028 | paper | AutoMCP (4-gate verify-before-register) | F015f | arxiv.org/abs/2507.16044 |
| S029 | doc | MCP Security Explained (untrusted servers, sandboxing) | F015f | docker.com/blog/mcp-security-explained |
| S030 | paper | LLMCompiler (Task-Fetching Unit scheduler) | F015g | arxiv.org/abs/2312.04511 |
| S031 | paper | CodeAct — Executable Code Actions (Wang, ICML 2024) | F015a, F015g | arxiv.org/abs/2402.01030 |
| S032 | paper | FrugalGPT (model cascades / tiering for cost) | F015 | arxiv.org/abs/2305.05176 |

Next free ID: **S033**.

## Notes
- **S008 is a placeholder.** Whoever continues F001 should replace it with the
  actual source the HITL design was taken from (paper or repo) and set trust
  from evidence.
- Reserve full files for sources whose *idea you implement* — that's where the
  verified/refuted lifecycle earns its keep.
