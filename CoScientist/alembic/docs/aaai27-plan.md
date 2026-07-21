# Alembic → AAAI-27 main-track: research, decisions, and execution plan

_Compiled 2026-07-15. Extends the EMNLP-2026 demo (`docs/paper/`) into a full
main-track paper. This doc is the single source of truth for the extension;
update it as decisions change._

## 0. Venue, timeline, decision

- **Target: AAAI-27 main technical track**, with a hedge.
- CFP (verify reference-page cap on the official page — two AAAI pages disagree):
  - **Abstract due 2026-07-21** (mandatory, gates the paper).
  - **Full 7-page paper due 2026-07-28.** Supplementary/code 2026-07-31.
  - Double-blind, on **OpenReview**; mandatory reproducibility checklist.
  - Phase-1 reject 2026-09-24; rebuttal 2026-10-19..25; notification 2026-11-30;
    camera-ready 2026-12-14. Conference Feb 16–23 2027, Montréal.
- **Decision (HEDGE):** file the July-21 abstract as insurance; push experiments;
  if not credible by July 28, **fall back to ICLR-2027** (~Sept 24 full paper).
  NeurIPS D&B 2026 already closed. ICLR fallback is where the *goldified
  benchmark* (see §6) becomes the centerpiece.
- Benchmark/eval papers ARE in scope for AAAI main track (submit under an area's
  "Datasets & Benchmarks" topic, e.g. NLP or DMKM; precedent: TableBench AAAI-25,
  RGB/CELLO AAAI-24, GitTaskBench AAAI-26). No separate D&B track. Lead with the
  *methodological/empirical* contribution, not the artifact.

## 1. Problem formulation (M2)

Scientific software is written for humans, not agents, and the long tail of
valuable-but-obscure research repositories is exactly where frontier agents are
weakest — under-documented, hard to set up, absent from pretraining. The
prevailing answer (point a sandbox coding agent at the repo each time) re-pays a
stochastic, expensive setup-and-solve cost on every invocation and degrades
precisely on the repositories that most need help. We ask whether converting a
repository **once** into a gate-validated, always-split-environment MCP tool
yields a capability that is more correct, deterministically reusable, and cheaper
to amortize than re-deriving it ad-hoc — and we show the benefit is largest
exactly where the model has not memorized the repo.

**Contribution reframe:** from "system demo" → an empirical study of *when and why*
repo→MCP conversion beats ad-hoc coding agents on niche scientific repos, with
Alembic (correctness-by-construction gates + always-split envs) as the instrument
and the *explanation* for the reliability gap.

## 2. Research questions (M4)

- **RQ1 — Capability.** Does one-shot repo→validated-MCP conversion produce more
  correct/complete tools on niche scientific repos than a frontier coding agent
  solving the same task ad-hoc, on identical held-out tests?
- **RQ2 — Reliability.** Across repeated invocations, is a frozen validated tool
  more consistent than re-deriving the solution each run? (pass^k; frozen tool is
  deterministic ⇒ pass^k ≈ pass@1.)
- **RQ3 — Economy.** How many invocations amortize the one-time conversion cost
  vs paying full agent cost every call? (token/latency break-even.)
- **RQ4 — Contamination.** Is conversion most valuable where the model has least
  memorized the repo — does the Alembic-over-agent lift grow with obscurity?
- **RQ5 (optional) — Ablation.** Do the gates / split-envs, not the prompt, drive
  the reliability advantage?

## 3. Experiment matrix (M3)

All runs same-model (**GLM-5.2**, `openrouter/z-ai/glm-5.2`). Operational limits
(from prior runs): `--parallel 4` (DNS), `docker rmi -f alembic-tool:<repo>` after
each repo's `[bench] ↓ done` (disk), watch OpenRouter spend before blaming code.

- **E0 — Self-metrics backbone (done).** 14-repo delivery table (`docs/paper/
  tables/alembic_selfmetrics_body.tex`), run `runs/2026-07-10_tmbench-all-v2/`.
- **E1 — Head-to-head (RQ1–3).** Systems: Alembic (frozen MCP, v2 artefacts) vs
  **OpenHands** (ad-hoc) vs **ToolMaker** (in-progress GLM-5.2 rerun — breaks
  constantly, itself evidence for C1). Each task **K=3×** per system; score on the
  identical TM-Bench held-out pytest (matched denominators); log tokens+latency.
  Alembic tool invoked 3× (deterministic). OpenHands re-solves 3× (pass^k +
  variance). Panels: capability (pass rate), reliability (pass^k over k=1..3,
  variance), economy (per-invocation cost + break-even #invocations).
- **E2 — Contamination/popularity (RQ4).** Axis = GitHub stars + release-vs-cutoff
  + memorization proxy (does base GLM-5.2 reproduce each target symbol from its
  name alone, à la SWE-Bench Illusion, verbatim n-gram overlap). Show OpenHands
  success ∝ popularity/memorization while Alembic stays flat ⇒ lift concentrates
  on obscure repos. Cheap analysis; high value.
- **E3 — Expert-MCP gold case study (bio/chem-skewed).** Compare Alembic's
  *auto-generated* MCP vs a real *expert-written* community MCP on the same repo:
  tool coverage, correctness on shared inputs, robustness. Pairs:
  `gget ⟷ gget-mcp`, `scanpy ⟷ scanpy-mcp` (RDKit⟷mcp_rdkit as backup). Verify the
  expert MCPs run before locking.

**Repo set:** 14 TM-Bench + **RDKit, Biopython** (memorized controls, E2) +
**gget, scanpy** (expert-MCP gold, E3). TM-Bench pathologies handled explicitly:
cap nnU-Net's 1000-epoch training to a smoke run; mark single-tool-target /
pathological tasks as harness artefacts (also evidence for C1).

**Compute priority if the clock tightens:** E1-on-trustworthy-repos → E2
memorization-proxy → E3 → E1-full-suite.

## 4. Related work / positioning (F1, F2)

- **Only peer-reviewed competitor:** ToolMaker (ACL 2025, 2502.11705). Everything
  else preprint: **Code2MCP** (2509.05941), **ToolRosella** (2603.09290 — shipped
  validator is just `create_app()` import-smoke, weaker than its stated "≥3
  endpoints"), **Paper2Agent** (2509.06917, Stanford/Zou — paper+code→served MCP,
  bio, validated by reproducing paper results; **closest conceptual competitor,
  must position against**), **AutoMCP** (2507.16044, OpenAPI→MCP, non-scientific).
- **Open niche:** no benchmark scores repo→MCP conversion *quality* vs a gold
  standard — competitors validate by import-smoke or single-case reproduction;
  only ToolMaker uses held-out execution. Alembic's live-invocation-on-held-out
  validation is the differentiator.
- **Gold expert-MCP pairs** scarce, bio-skewed (~10–20 true repo-wrappers;
  BioContextAI registry: gget, scanpy, RDKit, Clair) → E3 is a focused case study,
  not a broad axis.
- **Adjacent benchmarks:** GitTaskBench (AAAI-26, repo-*use*), MCP-Bench (ICLR-26,
  usage; its rule-based tier could score a generated server's well-formedness),
  ToolArena (TM-Bench successor, containerized, human tests.py — the reuse target
  for the ICLR benchmark).

## 5. Evidence / citations for the motivation (I2, I3, I5)

- Agents fail on research repos: SUPER 16.3% (2409.07440, EMNLP-24), CORE-Bench
  21% (2409.11363), PaperBench ~21% (2504.01848, ICML-25), ToolMaker's OpenHands
  20% vs 80% (2502.11705), GitTaskBench 48% (2508.18993, AAAI-26, in-family).
- Token economy: Anthropic "Code execution with MCP" (2025-11-04) 98.7% token cut;
  progressive tool disclosure.
- Variance (I3): τ-bench pass^k <25%@k=8 (2406.12045), τ²-bench (2506.07982),
  temp-0 non-determinism ~15% acc (2408.04667).
- Contamination (I2): SWE-Bench Illusion 76%→<53% on unseen popular repos
  (2506.12286), LiveCodeBench temporal holdout (2403.07974, ICLR-25).
- **Caveat:** CodeAct (2402.01030, ICML-24) shows code-actions beat rigid
  tool-calling ⇒ frame Alembic as "freeze the winning code path once," NOT "tools
  beat code."

## 6. Deferred to ICLR-2027 fallback

If July 28 slips: build the **goldified benchmark** — tighten a ToolRosella
subset (or extend ToolArena) with human `tests.py` per repo so it scores repo→MCP
conversion *quality*; add the broader converter head-to-heads (Code2MCP,
ToolRosella, Paper2Agent). This is the unfilled niche and the stronger
main-track/ICLR contribution, but it does not fit 13 days.

## 7. Immediate actions

1. **By July 21:** file abstract + title (draft in §8; independent of results).
2. **Days 1–2:** stand up OpenHands harness on GLM-5.2; start E1 runs (compute is
   the long pole); continue ToolMaker GLM-5.2 rerun.
3. **Days 3–8:** E1/E2/E3 runs + memorization proxy.
4. **Days 6–10:** tables, figures, analysis.
5. **Days 8–12:** rewrite intro/related/eval around the empirical study; new
   experiment section; RQs; reproducibility checklist.
6. **Day 13:** polish, anonymize, submit.

## 8. Draft title + abstract (for July 21)

**Title options:**
1. _Convert Once, Call Reliably: Validated Repository-to-MCP Conversion Beats
   Ad-hoc Coding Agents on Scientific Software_
2. _From Repository to Reliable Tool: When One-shot MCP Conversion Outperforms
   Sandbox Coding Agents_

**Abstract (draft, ~150 words):**
Frontier coding agents can install and run some research repositories, but they
do so unreliably and re-pay a stochastic, expensive setup-and-solve cost on every
call — and they degrade precisely on the obscure, valuable repositories least
represented in pretraining. We study an alternative: converting a repository
*once* into a validated, served tool. We present Alembic, which turns a GitHub
URL into a Model Context Protocol server via a five-agent pipeline interleaved
with code-enforced gates that build isolated environments, write each tool as a
plain function, and validate it by live invocation on held-out inputs. On
fourteen scientific repositories plus controls, we compare one-shot conversion
against a same-model sandbox coding agent (OpenHands) and against ToolMaker.
Converted tools match or exceed ad-hoc solving on held-out tests while remaining
deterministic across repeated invocations (where the agent's pass^k collapses)
and cheaper to amortize — and the advantage grows with repository obscurity. We
release code, artefacts, and runs.
</content>
