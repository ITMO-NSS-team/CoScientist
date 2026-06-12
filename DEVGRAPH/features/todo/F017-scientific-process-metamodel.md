---
id: F017
title: Scientific-process meta-model (ontology) + runtime research knowledge graph
type: feature
status: proposed
created: 2026-06-12
updated: 2026-06-12
owners: [SoloWayG]
derives_from: [F000, F015]
depends_on: [F001, F002, F003, F009, F015]
sources: [S033]
tags: [meta-model, ontology, knowledge-graph, orchestration, validation, ai4s]
code: []   # net-new; no module yet. Affects orchestrator + every module's I/O.
benchmarks: []
---

## Goal
Integrate a **formal ontology of the scientific research process** (6 layers of entities +
relations + navigation rules) and a **runtime "research graph"** — a per-study directed,
labeled graph instance the system builds and the orchestrator **queries to decide what to do
next**. Purpose: give the models explicit context, record each module's strict I/O, validate
results and process (dependency checks, provenance, budget/economic stopping), and accumulate
reusable experience. Spec: [[S033]] (Мета-модель научного процесса, .docx).

> ⚠ **Not the same graph as DEVGRAPH.** DEVGRAPH = meta-model of the *repo/development*
> (features, attempts). This = a runtime meta-model of the *science being done* (question →
> hypothesis → evidence → conclusion). Two different graphs; don't conflate.

## The ontology (6 layers, each entity → a CoScientist module + status)
- **L1 Epistemic objects:** Research question (orchestration; *now unstructured*) → Hypothesis
  (Module 2, structured: statement, reasoning, needed evidence, **required_tools**, verification
  plan, confidence, source — central entity) → Evidence (literary/experimental/computational/
  expert/meta; Modules 1/3/HITL) → Conclusion (Module 4 validator; +validity boundary).
- **L2 Methodology:** Verification methods (registry; inputs/outputs/limits/cost; chosen by the
  experiment planner) · Confirmation conditions (formal sufficiency criteria; HITL-approved).
- **L3 Feasibility / resources:** **3.1 Tool capabilities** (status: *available / needs-adaptation
  / not-integrated / needs-creation / not-exists* — found by the tool-search/orchestration layer,
  created by Module 3.1/Alembic) · 3.2 Consumable resources (budgets: GPU-hours, tokens, time,
  expert-hours) · 3.3 Empirical base (datasets, lit corpus, prior results, **trusted corpus**) ·
  3.4 Cognitive/normative constraints (Research **profile**, methodological norms, theoretical
  frameworks, domain standards, ethics/regulatory, expert knowledge, participant roles — most
  *not yet implemented / now implicit*).
- **L4 Artifacts:** code/models, generated data, report, publication, ТЗ, efficiency justification
  (Modules 3–4).
- **L5 Navigation:** **Transition triggers** (as graph queries) · Information dependencies (partial
  order: "no experiment without a hypothesis", "no conclusion without evidence→conditions") ·
  Completion criteria (exhaustive/pragmatic/resource/economic) · **AI-application model** (autonomy:
  lab-assistant → assistant → copilot → architect).
- **L6 Economics:** Cost model (per-step cost, before-run estimate) · Efficiency metric
  (time/cost/quality).

## The runtime research graph (per study)
A concrete instance of the ontology: nodes = concrete entities (this question, this hypothesis,
this result) with `{type, content(attrs), status, timestamp, source=module|HITL}`; edges = typed
relations. Starts at the question, grows; non-linear (branches, cycles, **dead-ends kept as
negative results**). Six phases: Init (root + feasibility context = star shape) → Recon (M1:
literature evidence) → Branching (M2: hypotheses + critic + HITL ranking) → Deepening (M3:
experiments → evidence; M3.1/Alembic builds missing tools, status updated) → Folding (M4: validator
→ conclusions; may spawn a new question) → Codification (M4: artifacts).

## The orchestrator's role = querying the graph (the key behavioral shift)
The master orchestrator stops being rule-based and becomes a **cost-aware planner over the graph**:
- **Triggers as queries:** "is there a hypothesis `formulated` with ALL tools `available`?" → run
  it. "Is there evidence that refutes a hypothesis?" → revise / move on.
- **Dependency validation as path existence:** can't make a Conclusion unless a path
  hypothesis→evidence→met-conditions exists.
- **Progress / blocked branches** visible (waiting-for-tool, waiting-for-human).
- **Budget / economic stop:** consumable nodes hold remaining budgets; if marginal cost > expected
  value → propose finishing.
- **Optimal path:** cheap checks first; expensive only with budget.
- **Provenance chain** artifact→…→question (auto "Methods"/"Results" sections).
- **Reuse:** completed graphs saved as experience; a similar new question reuses one as a seed.

## How this relates to the user's intended flow & to F015
This IS the formalization of the desired orchestrator flow: scope the question → ground tools
(L3.1 status check) → if unclear, Research (enrich domain, find repos for Alembic) → Hypotheses
(L1) → run the experiments module. The **experiments orchestrator (F015a)** is "Plan generator /
experiment planner (Module 3)": it consumes {literature, hypothesis, task, live MCP tool list},
does its own L3.1 sufficiency check, calls Alembic on a gap, and forms the FEDOT.MAS plan. So
F017 is the *connective tissue / contracts* around F015; F015's pieces are the executors.

## What exists vs missing (from the doc's own status column)
- **Exists:** hypothesis gen (M2), literature (M1), validator (M4), report (M4), FEDOT executor
  (M3), HITL, tool-search (existing), data-constructor.
- **Missing / not-implemented:** the **research-graph store** itself; structured research-question
  object; **context-init agent** (profile, methodological norms, theoretical frameworks, domain
  standards, ethics, roles, budgets, trusted-corpus tagging); **cost estimator** (L6); transition
  triggers as *formal graph queries*; AI-application-model (autonomy) switching.

## ✅ TODO (decomposition candidates — likely an epic)
- [ ] Define the entity schemas (Pydantic) per layer + the research-graph store (nodes/edges/status).
- [ ] Module **I/O contracts**: what each module consumes/produces as graph nodes (validation hook).
- [ ] Orchestrator: transition triggers as graph queries + info-dependency (path) checks.
- [ ] Context-init agent (profile/norms/standards/ethics/budgets/roles) — most are net-new.
- [ ] Cost model + economic completion (L6); reuse of completed graphs.
- [ ] Reconcile with F015 (experiment planner = Module 3) and F001 (HITL points per role).

## ⚠ Pitfalls / open questions
- Huge scope — this is an **epic**; don't try to build it all. The near-term, high-leverage slice
  is: structured hypothesis (required_tools + verification plan) + tool-status (L3.1) + the
  orchestrator's "run only a hypothesis whose tools are all available" trigger. The rest (profile,
  norms, cost model, full graph store) can follow.
- Don't conflate with DEVGRAPH (dev graph). They can share *format ideas* (typed nodes, status,
  provenance, negative results kept) but are separate stores.

## Symbols
- (none yet) — net-new. Building blocks it will wrap: orchestrator (`agents/agents.py`),
  HypothesesAgent/ResearchAgent/CriticAgent, F015a planner (`CoScientist/experiments/`).
