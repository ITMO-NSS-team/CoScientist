---
id: F016
title: Benchmark evaluation — which science-agent benchmarks CoScientist can solve
type: feature
status: proposed
created: 2026-06-12
updated: 2026-06-12
owners: [SoloWayG]
derives_from: [F000]
depends_on: [F003, F009, F010, F002]
relates_to: [F014, F015h]
sources: []
tags: [benchmark, eval, research-agent, capability-map]
code:
  - benchmarks/list.md
  - benchmarks/Презентация.pdf
  - benchmarks/Chem bench.pdf
benchmarks: []
---

## Goal
Decide **which external science-agent benchmarks CoScientist can actually be
evaluated on**, map each to the capability it requires, separate "runnable now"
from "needs work" from "doesn't fit", and drive the work to (a) verify the
runnable ones and (b) close gaps for the ones that fit.

## Current state
`benchmarks/` holds a **survey** of ~30 benchmarks (presentation + `Chem bench.pdf`;
`list.md` only names EXP-Bench). There is **no adapter/harness** that feeds a
benchmark task into `manager.run()` and grades the output — `scripts/opik_eval/`
only reads reliability metrics from existing Opik traces (F015h/F014), it does not
drive the pipeline against a benchmark. So today: **0 benchmarks actually run**;
the verdicts below are an *assessment* (from the survey + project_card capabilities
+ this session's e2e findings), not yet empirical — hence the verify-TODOs.

CoScientist capabilities used for the mapping (project_card.md): ResearchAgent
(papers-search/OpenAlex + paper-analysis RAG; **no live web search** — Tavily
disabled, F003.A2), HypothesesAgent, TaskExecutorAgent (run existing MCP tools via
FEDOT.MAS), CoderAgent (sandbox code/shell/git, F002), MedicalAgent, chemical MCP
(docking/retro/RDKit/IUPAC↔SMILES/figure-OCR/ChEMBL), RAG tool-retrieval +
MCP orchestration (F009/F005).

## Capability map (benchmark → requirement → fit)

Fit: ✅ runnable now (capability present + open dataset/grader) · 🟡 partial
(capability exists but heavy infra / a gap) · ❌ doesn't fit now.

### ✅ Can run now (build an adapter + run)
| Benchmark | Tests | Required capability | What's needed |
|---|---|---|---|
| **MCP-Bench** (2508.20453) / **MCPAgentBench** (2512.24565) | multi-step MCP tool retrieval + orchestration (28 / 250 tools) | RAG tool-retrieval + orchestration (F009/F005/TaskExecutor) — **the system core** | adapter to drive their tool tasks through the orchestrator; not science-specific (no domain GT). Directly measures the routing/retrieval we're debugging. |
| **LAB-Bench / LitQA2** (2407.10362) | grounded literature MCQ | ResearchAgent + papers-search/RAG (verified e2e F003.A3) | adapter + LitQA2 set + MCQ auto-grader. FigQA/TableQA/DNA subsets = gaps. |
| **HypoBench** (2504.11524) | hypothesis generation/ranking | HypothesesAgent | adapter; **grader ships with it** (HDR/FDR/generalizability). |
| **ChemBench** (Nature s41557-025-01815-x) | chemistry Q&A across subfields | chemical MCP (RDKit, IUPAC↔SMILES, …) + LLM | adapter + open HF dataset; wire chemical MCP for tool-requiring items. |
| **ResearchBench** (2503.21248) / **AI Idea Bench 2025** (2504.14191) | inspiration-retrieval + hypothesis composition/rank | ResearchAgent + HypothesesAgent | adapter; auto/partial grader included. |
| **AutoResearchBench** (2604.25256) / **ResearcherBench** (2507.16280) | find paper by partial cues / list-papers / tech brief | ResearchAgent (literature retrieval) | adapter; ResearcherBench is open-ended → needs an LLM-judge. |

### 🟡 Partial (capability exists, but heavy infra or a gap)
| Benchmark | Gap / why partial |
|---|---|
| **MLE-Bench** (2410.07095), **MLAgentBench** (2310.03302), **ScienceAgentBench** (2410.05080) | CoderAgent + FEDOT.MAS can attempt, but need real compute, datasets, mature sandbox. |
| **EXP-Bench** (2505.24785), **SciReplicate-Bench** (2504.00255), **LMR-Bench** (EMNLP'25) | reproduce-algorithm-from-paper; CoderAgent path, ambitious; EXP-Bench is the one named in `list.md`. |
| **PaperArena** (2510.10909), **DeepResearch Bench** (2506.11763), **MMDeepResearch** (2601.12346) | ResearchAgent exists but **handicapped by no live web search**; MMDeep is multimodal. |
| **FML-Bench** (2510.10472) | ML-research tasks; heavy compute. |
| **SciVisAgentBench** (2603.29139) | CoderAgent viz; needs viz stack. |
| **MaCBench** (2411.16955) | multimodal chem (tables/plots/AFM/spectra); only figure-OCR partly applies. |

### ❌ Doesn't fit now
| Benchmark(s) | Why |
|---|---|
| Full-pipeline: **AstaBench** (2510.21652), **MLR-Bench** (2505.19955), **PaperBench** (2504.01848), **RE-Bench** (2411.15114), **ResearchGym** (2602.15112), **FIRE-Bench** (2602.02905), **HeurekaBench** (2601.01678), **InnovatorBench**, **ReplicatorBench** (2602.11354) | require full research-agent autonomy + heavy infra (GPU/Rust/CUDA, end-to-end paper writing). |
| Biology: **BioMysteryBench** (Anthropic), **CompBioBench**, **sc-HeurekaBench** | no bioinformatics/genomics tooling in the system. |
| **ChemPro** (2602.03108) not yet released · **GPQA** (2023) outdated · **FrontierScience** (OpenAI) closed · **LongCoT** (2604.14140) general-LLM, not agentic |

## ✅ TODO
- [ ] **Build the benchmark adapter harness** (shared prerequisite for ALL): task → `manager.run()` → grader, traces read from Opik. None exists yet; reuse `scripts/opik_eval/metrics.py` for reliability, add per-benchmark graders.
- [ ] **Verify the ✅ "can-run-now" set empirically** (the user's "проверить что уже можно"):
  - [ ] MCP-Bench / MCPAgentBench — run N tasks, score tool-retrieval/orchestration success.
  - [ ] LAB-Bench LitQA2 — run the MCQ subset, report accuracy.
  - [ ] HypoBench — run with its HDR/FDR grader.
  - [ ] ChemBench — run open subset, report accuracy.
  - [ ] ResearchBench / AI Idea Bench / AutoResearchBench / ResearcherBench — run, judge.
  - (For each: confirm dataset/harness is actually available — verdicts above are from the survey, not the repos.)
- [ ] **Develop toward the 🟡 partial set** (only the ones that fit the mission):
  - [ ] re-enable/replace live web search (F003.A2) → unlocks PaperArena / DeepResearch.
  - [ ] mature CoderAgent+FEDOT.MAS compute path (datasets, sandbox, GPU?) → MLE/MLAgent/ScienceAgentBench, EXP-Bench.
  - [ ] multimodal/figure understanding → MaCBench, MMDeepResearch.
- [ ] **Do NOT pursue** the ❌ set without a capability decision (bio tooling, full autonomy, GPU infra) — record any decision as F016.D1.
- [ ] Decide whether each ✅/🟡 adapter is its own sub-feature (F016a…) once the harness exists.

## ⚠ Pitfalls / Known problems
- **No harness exists** — every benchmark needs an adapter + grader; budget that first.
- **Orchestrator flakiness will skew scores** (F014 + this session: `accumulated_tools` crash [now guarded], literature→TaskExecutor mis-routing, critic over-rejection, runaways). Scores would reflect bugs, not capability — stabilize first or scores are noise.
- **No live web search** (Tavily disabled, F003.A2) caps every deep-research/web benchmark.
- **ITMO VPN + hosted services** required for the MCP/RAG/FEDOT.MAS paths.
- Verdicts here are **assessed, not measured** — verify dataset/harness availability per repo before committing (no web at analysis time).

## Symbols
- `benchmarks/` — the survey (presentation + `Chem bench.pdf` + `list.md`); the source-of-truth list this feature maps.
