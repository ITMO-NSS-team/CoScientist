---
id: project_card
type: project_card
name: CoScientist
version: 1.0.0            # README changelog; no separate release tag verified
updated: 2026-06-11
source_of_truth:         # where each claim below is verifiable in-repo
  agents: CoScientist/agents/catalog.py
  tools: CoScientist/tools/__init__.py
  mcp_coverage: tools_checklist.md
---

# Project Card — CoScientist

> A2A-style capability descriptor: **what the system can do right now**, grounded
> in code. Update this when a capability is added/removed. Don't list aspirations
> here — those are `proposed` features in `features/`.

## Identity
- **What it is:** a multi-agent system for scientific discovery, built on Google
  ADK, FEDOT.MAS, and RAG-based tool retrieval (README.md).
- **Entry points:** `python -m CoScientist.main` (CLI); `create_manager()` API.
- **Primary model wiring:** LiteLLM over OpenRouter, with provider pinning to a
  known-good set (deepinfra/groq/together/fireworks) + retries — added to stop
  flaky empty responses (commit `79cb9c6`, `config/settings.py:pinned_providers`).

## Agents (delegatable) — `CoScientist/agents/catalog.py`
| Agent | Does | Enabled |
|-------|------|---------|
| OrchestratorAgent | Plans minimal steps, delegates, combines results (root). | yes |
| PlannerAgent | Step-by-step roadmap for complex tasks; called first. | gated by `settings.orchestrator.use_planner` |
| HypothesesAgent | Generates ideas/hypotheses. | yes |
| ResearchAgent | Literature + web + RAG retrieval; can find/download papers. | yes |
| TaskExecutorAgent | Runs computation via **existing** MCP tools (retrieve→execute). | yes |
| CoderAgent | Writes/runs code, shell/git, deps, long jobs in a sandbox (#268). | yes |
| MedicalAgent | PubMed/PICO, study taxonomy, DICOM image analysis. | yes |
| CriticAgent | Pre-action critic over the agent roster (`critic_agent.py`). | yes |

## Tools / toolsets — `CoScientist/tools/__init__.py`
FEDOT.MAS (`fedot_toolset_instance`), web search (`websearch_toolset_instance`),
paper analysis (`paper_analysis_toolset_instance`), paper search
(`papers_search_toolset_instance`), RAG retrieval (`retrieval_toolset_instance`),
MCP-server discovery via RAG (`search_mcp_servers`), medical (`med_toolset_instance`),
coder/sandbox (`coder_toolset_instance`).

## MCP servers (in-repo) — `mcp-servers/`
- `chemical-mcp-server` — chemistry ops (docking, retrosynthesis, OCR, RDKit props, etc.).
- `paper-analysis-mcp-server` — parse/analyze papers.
- `papers-search-mcp-server` — OpenAlex search/download.
- `dataset-collection-mcp-server` — build datasets from papers.
- External: Tavily MCP (web search) — **currently DISABLED** (VPN/TLS hang to
  `mcp.tavily.com`; see feature F003.A2). No live web search until re-enabled.
- Plus: MCP servers discovered at runtime — via public registries
  (`search_mcp_servers`, F005) and via the RAG DB (`RetrievalToolSet`, F009).

## Capabilities (verified-present, from `tools_checklist.md`)
Covered by an MCP tool today: molecule generation (GAN+Transformer, 6 cases),
retrosynthesis route search, forward reaction product prediction, reaction
classification, docking, molecule/reaction extraction **from figures**, IUPAC↔SMILES,
RDKit property extraction, SMILES HTML viz, ChEMBL/BindingDB activity fetch.

**Known *gaps* (tool exists but no MCP coverage / not wired):** training models for
molecule generation, AutoML property prediction (+training), paper-DB Q&A,
uploaded-paper Q&A, relevant-paper search in DB, dataset-from-papers, OpenAlex
download, nanoparticle synthesis/property/shape prediction & imaging, molecule/
reaction extraction **from PDF**. (Full list: `tools_checklist.md`.)

## Benchmarks / evals
**None recorded.** Only one integration test found (`tests/integration/test_paper_analysis.py`),
and integration tests need ITMO VPN + hosted services (README). There is **no
benchmark suite or eval harness** in-repo as of seeding. → tracked as a gap; any
agent that adds evals must record results here and in the relevant feature's
`benchmarks:`.

## Connect / run notes
- Git deps not on PyPI: `rag_tools` (GitHub) and `fedotmas` (ITMO, SSH) — see README.
- Secrets via `.env` (LLM keys, Tavily, OpenAlex, Postgres for rag_tools).
