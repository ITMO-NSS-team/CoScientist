# Alembic — Status Summary

Concise, current-state digest. Full history/evidence/testing detail lives in
[IMPROVEMENTS_SPEC.md](./IMPROVEMENTS_SPEC.md) (F1–F40 specs) and
[DEMO_READINESS_TODO.md](./DEMO_READINESS_TODO.md) (run-by-run narrative) —
**both left untouched**; this file is a pointer/summary, not a replacement.
Last synced against those docs: 2026-07-08.

---

## 1. Implemented & working

**Logging honesty / stage mechanics (F14–F23)** — the original 10-bug batch
caught in the 2026-06-30 baseline run, all confirmed fixed by rerun:
- **F14** Validator no longer blanket-skips every tool when one pytest case fails.
- **F15** Debugger requests always carry `repo_url`.
- **F16** Per-debugger-call timeout (600s), separate from the stage timeout.
- **F17** Transient LLM/API errors (empty-body JSONDecodeError) retried once.
- **F18** Explorer/Coder use realistic, domain-sized sample inputs, not toy arrays.
- **F19** Fixed `_UnknownToolStub` missing ADK `BaseTool` attributes.
- **F20** Stage-timeout paths no longer falsely log "pipeline complete."
- **F21** Coder verifies sample file paths exist before using them.
- **F22** Silent OpenRouter/LiteLLM `finish_reason:"error"` faults detected + retried.
- **F23** All blocking tool functions converted `async def` + `asyncio.to_thread`,
  so `asyncio.wait_for` timeouts can actually fire.

**Validator/debugger protocol (instruction-level, probabilistic — not code-enforced)**
- **F24** Validator independently re-invokes tools after every debugger fix
  instead of trusting its self-report; added a "sample is wrong, not code"
  corrected-args path (Class E) and cross-tool sibling-fix propagation.
  Confirmed working most of the time; one real miss found (Class E field
  omitted) and hardened 2026-07-07.
- **F25** SKIP is now a code-enforced gate (parsed from `server.md`,
  `invoke_mcp_tool` itself refuses SKIP-listed tools), not an LLM convention;
  Coder tries a cheap parameterization before falling back to SKIP.
  *(Item 2 — a decoupled "extended validation" pass for heavy tools — still deferred.)*
- **F29** Debugger's internal tool calls/reasoning now logged per debug round
  (previously a black box).
- **F30** Successful `invoke_mcp_tool` results capped (20 list items / 2000
  chars) to stop unbounded result blobs bloating context.

**Tool-selection & static checks**
- **F26** Tools with no checkable output (GUI/notebook launchers) no longer proposed.
- **F27** Tool parameter count capped (~4–5); prefer thin CLI-mirroring tools.
- **F28** `validate_syntax` now also statically checks `helpers/*.py`, catching
  hallucinated imports before any live invocation.
- **F38** Helpers now resolve relative sample paths against `REPO_PATH`
  internally, fixing false `FileNotFoundError`s on real files.
- **F39** F28's import-safety check hardened against a 3rd helper-script shape
  (flat `sys.argv` indexing, no argparse) that used to defeat it.

**Crash fixes & infra hardening**
- **F33** Fixed unhandled `IsADirectoryError` in `read_file`/`read_output_file`.
- **F34** `clone_repo` now fetches git submodules (`--recurse-submodules`).
- **F35** Dedicated fallback `reporter_agent` guarantees *some* report gets
  written even if the validator hard-times-out or exhausts guard retries.
- **F36** Fixed `MAX_TOOL_CYCLE` guard unjustly aborting tools whose own
  instructions require repeated identical-args calls.
- **F37** `invoke_mcp_tool` capped at 120s, kills the whole process group (not
  just the immediate child) on timeout.
- **F32 (mitigation 1 of 2)** Soft-deadline nudge forces a "write what you
  have" report in a stage's last 15% of budget. Known gap: can't preempt one
  very long in-flight tool call — that's what F35 covers.

**Metrics & observability**
- **F12** Structured per-run JSON metrics + failure-class taxonomy
  (`reports/metrics.json`), aggregated across a benchmark into stage
  pass-rates and error-distribution tables.

**External resources**
- **F6 (partial)** Gated HuggingFace weights (`HF_TOKEN`) pre-downloaded
  during the Environment stage — confirmed working live against `CONCH`
  (`MahmoodLab/conch`). Dataset-mount convention (part b) still deferred, no
  repo in the current subset needs it yet.

**Today's fix (2026-07-08, not yet written into either doc above)**
- Fixed a circular-import bug in `tools/invoke.py`
  (`from CoScientist.alembic.main import INVOKE_TIMEOUT` — wrong package
  prefix for the container, and would circular-import even with the right
  one) that broke **every single pipeline run** at Docker-container startup.
  Now a deferred, function-local import matching the existing pattern in
  `shell.py`/`venv.py`. Rebuilt `alembic-base:latest`; a 7-worker run over
  the full `toolmaker_subset.txt` (14 repos) is currently in progress.

**Reverted (tried, abandoned — not active)**
- **F31** `check_ancient_pins` upfront wheel-availability check. Worked as
  designed but didn't prevent the real failure class (ABI incompatibility) it
  targeted; reverted, not worth the complexity.
- **F40** Env-var timeout-multiplier knob. Built, then judged too much
  indirection; replaced with plain constants centralized in `main.py`.

---

## 2. Left to implement

Optional robustness backlog (ranked by payoff in IMPROVEMENTS_SPEC.md):

| # | What | Why it'd help |
|---|---|---|
| F1 | Static AST import/symbol gate, Coder→Validator | Catch hallucinated symbols before any run |
| F4 | Bounded, AST-verified tool selection + real param names into Coder | Pre-empt wrong-kwarg `TypeError`s |
| F2 | Semantic output-correctness gate (LLM judge on return values) | Catches "ran clean but returned garbage" |
| F3 | Held-out validation invocation (2 distinct samples per tool) | Stops overfit-to-demo-args tools passing |
| F5 | First-class conda/`environment.yml` path | Helps rdkit/openbabel/pinned-CUDA repos |
| F7 | Declared, allowlisted repo-secret injection beyond HF_TOKEN | Unblocks other token-gated repos |
| F8 | Reproducible image from a recorded `install.sh`, not `docker commit` | Auditable, reproducible builds |
| F9 | Fresh-checkpoint isolation per validate/debug attempt | Stops side-effects masking/faking failures |
| F10 | Persistent failure-memory across debug iterations | Stops oscillating fixes |
| F11 | Tiered model routing (strong model for planning, cheap for repair) | Cost/latency |
| F13 | Success-memoized, resumable benchmark runner | Cheaper re-benchmarking |
| F25 item 2 | Decoupled "extended validation" pass for SKIP-marked heavy tools | Real coverage for training/inference tools |
| F32 item 2 | Explicit `abort_reason: upstream_stage_timeout` tagging | Distinguish "hard repo" from "starved by earlier timeout" |

**F32 (main gap, unresolved)** — Stage timeouts (Explorer 900s, Validator
1800s) are currently the **single biggest source of lost signal**: the
2026-07-07 full rerun lost all validator signal on 4/10 repos and all
explorer signal on 2/11, purely to timeout, not unfixable bugs. Mitigation 1
(soft-deadline nudge) is done; a full fix (bigger budgets, or genuinely
parallelizing stages) is still an open decision.

**Two flagged, not-yet-actioned findings** (from the CONCH/toolmaker run):
1. Generated tools reload HF weights live via `hf-hub:` reference instead of
   the pre-downloaded local checkpoint path — partially undercuts F6.
2. A `uv`-quoting mistake can silently fall back to a doomed slow-plain-pip
   install path on any repo with heavy ML deps (root cause of `UNI`'s
   Environment-stage failure).

---

## 3. Demo readiness (deadline 2026-07-10, i.e. **2 days out**)

**Must-have, still open:**
- [ ] `sections/evaluation.tex` — not yet written up with real numbers. Data
  exists across 4 full benchmark runs (baseline → rerun2 → final-eval →
  full-rerun). Recommended headline metric: **tool-invocation pass rate**
  (40–55% band across runs), not the noisy Overall-PASSED-repo count
  (low-n, swings with LLM-regeneration variance run to run).
- [ ] `sections/limitations.tex` — update for failure classes found since it
  was last written (stage-timeout cascade, F32 in particular).
- [ ] Paper mechanics: real author/affiliation/emails, `\repourl`/`\demourl` +
  ≤2.5 min screencast, trim comparison docs into a 2-page appendix,
  acknowledgments, `preprint` → camera-ready package switch.
- [ ] TM-Bench/ToolArena head-to-head comparison — **decision recorded, not
  executed**: recommendation is to mark as future work (no LICENSE on
  ToolMaker's repo, task drift vs. the paper's original 15, and a per-task
  shim would be real engineering) rather than attempt full parity in the
  time left.

**Operational risk — OpenRouter reliability:** this week alone hit two
distinct failure modes that silently zero out a whole benchmark run:
1. **Total spend limit exhausted** (`"Key limit exceeded (total limit)"`,
   clean 403 on the very first LLM call, every repo, ~10s each) — a billing
   issue, not a pipeline bug; check key status at
   `openrouter.ai/workspaces/default/keys/<id>` before any run that matters.
2. **Unexplained mid-run block** (`"Access denied by security policy."`, hit
   6/13 repos on one run, not from the start) — root cause never pinned down
   (WAF? rate limit?); distinguish this from #1 by the exact error string
   before assuming a pipeline bug.

**Right now:** after today's import-bug fix, a 7-worker run over the full
`toolmaker_subset.txt` (14 repos) is in progress — its result determines
whether this becomes the Evaluation section's data or needs another pass.
