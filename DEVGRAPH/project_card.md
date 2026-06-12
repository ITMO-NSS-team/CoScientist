---
id: project_card
type: project_card
name: CoScientist
version: 1.0.0            # README changelog; no separate release tag verified
updated: 2026-06-13
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

## Where things live — data & control map
A map so the FEDOT-vs-tools / "where's the case list?" confusion doesn't recur.

**Tool & server metadata (the registry):**
- Stored in the **rag_tools DB** (Postgres + Qdrant), populated by `scripts/rag_tools`. A tool
  entry = `{name, server_id, description, …}`; a server row = `{server_id, name, url, protocol,
  description, …}` — **a server row does NOT enumerate its tools** (`MCPServer` has no tools field).
- Read at **plan time** via RAG: `list_available_tools(query)` (the orchestrator's tool) and
  `retrieve_tools(query)` (ToolRetriever/TaskExecutor path) — both return per-tool
  `{name, server_id, description, score}`, top-k RAG-ranked. The **tool description** is what the
  orchestrator sees to ground a request; it is now returned **in full** (untruncated — F009.A3).
- `get_server_info(server_id)` → server-level metadata (url/description), NOT a tool list.

**Description (plan time) vs execution (run time):**
- The orchestrator/planner only **read descriptions** from the registry — they do NOT call MCP
  tools directly. To actually RUN a tool, the orchestrator delegates to **TaskExecutorAgent →
  FEDOT.MAS**, which binds the real `McpToolset` and executes. So a tool's *live* output (e.g.
  `list_generative_train_cases` returning the real cases, or a 404 listing available datasets)
  appears only on the execution path, inside FEDOT.

**FEDOT.MAS dispatch (experiments executor):**
- `fedot_tool(task_description)` is the seam: it pulls **servers** from session state
  (`state['filtered_tools']` from RAG + `state['deployed_mcps']`) → `servers_payload =
  {server: HttpMCPServer(url, desc)}` → `mas.run(task)`. FEDOT.MAS's meta-agent reads only
  **server descriptions**, autonomously generates a worker roster (`routing_meta_agent` +
  workers), and binds tools **per server** (whole toolset, not per-tool). See F015g.D1.

**Artifacts & datasets (S3):** computational artifacts, and some MCP servers' datasets/models,
live in **S3** (e.g. the remote generative MCP's training files at
`molecule-generative-mcp.s3.amazonaws.com/train/…`; presigned URLs, ТП §2.4). A missing
dataset/model there → "false success" (F015h). **GENERATION RESULTS are also S3, not inline:**
`generate_mols`/`generate_case_mols` return `results_presigned_url` (+ `results_s3_key`) to a results
CSV — the molecules live behind the link. ⚠ Today the FEDOT.MAS `molecule_generator` sub-agent
**paraphrases that result and drops the link** (state `output_key` = LLM text, not the raw tool
payload), so the real molecules never reach the orchestrator (F010.A3; fix → F015g). The `vault` MCP
(`http://10.32.11.45:8000/mcp`) is the intended helper for pulling/holding artifact links.

**LLM:** LiteLLM over OpenRouter; model from `settings.llm.main_model`; provider pinning via
`extra_body` (`agents.make_llm` / `experiments/planner`). The **per-run** model is recorded in
the **Opik trace metadata** — that is ground truth, not the live `.env` (F014.A1).

**Observability:** all agent/LLM/tool spans → **Opik** (Comet cloud, workspace `itmo-nss`,
project `adk-coscientist`) — the only reliable window into sub-agent / FEDOT internals.

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
