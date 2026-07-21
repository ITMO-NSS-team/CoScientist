# 01 — Architecture audit & design-choice re-evaluations

Scope: the `CoScientist/alembic` module as of 2026-07-06 (branch
`alembic-paper-stas`). Focus on things **not** already in
[IMPROVEMENTS_SPEC.md](../IMPROVEMENTS_SPEC.md).

## 1. What the system is (verified from source)

A 4-agent sequential pipeline on Google ADK + LiteLLM, one shared model
(`openrouter/qwen/qwen3-235b-a22b-2507`), run inside an ephemeral Docker
container that is the security boundary:

```
Explorer → Environment → Coder → Validator(→ Debugger sub-agent)
   read      build venv    write     invoke every tool live,
   repo,     (1 or 2)      server.py debug on failure, commit → alembic-tool:<repo>
   propose                 + tests
   1–5 tools
```

Stages hand off **only via `reports/*.md`** (LLM-written, LLM-read). Runtime
hardening (`agent_runtime.py`) is solid: guard-retries, loop-breakers,
per-stage + per-debugger timeouts, a taxonomy classifier, and three ADK/LiteLLM
monkeypatches (unknown-tool stub, silent-fault detector, async tool offload).

**Code health:** good. `agent_runtime.py`/`main.py` split is clean, the F12
metrics plumbing is real, error handling is honest (F20). The concerns below
are architectural, not hygiene.

## 2. Design choices worth re-evaluating

### 2.1 No sampling determinism — the load-bearing issue

**Finding (confirmed):** `agents.py` builds every agent as
`LiteLlm(model=MODEL)` with **no `temperature`, `top_p`, or seed**. LiteLLM
passes provider defaults (OpenRouter/Qwen ≈ temperature 0.7–1.0). Combined with
the workdir being wiped each run, **every run is an independent sample**: server
code, chosen tools, sample args, and pass/fail all vary.

Your final-eval documents this without naming the cause: *"each run regenerates
`server.py` from scratch via the LLM; a repo passing in one run and failing in
the next is expected noise."* The three-run tool-pass spread is 31.9 / 45.5 /
54.5%.

**Re-evaluation:** set `temperature=0` (and a fixed `top_p`) via ADK's
generation config on each agent. This will not make runs bit-identical (provider
non-determinism remains) but will **sharply narrow** the spread. Then report the
benchmark as **mean ± std over k≥3 seeds per repo**, not a single number. This is
the honest metric for a stochastic system and pre-empts the reviewer question.

Effort: ~1h. Impact: highest in this document. See [02](./02-stability.md) N1.

### 2.2 The Validator trusts the Debugger's self-report (never re-checks)

`instructions/validator.py` Step 4, verbatim: *"Trust its summary; do NOT call
`invoke_mcp_tool` again for the same tool unless the debugger reports it could
not verify."* F24 already documented the consequence — **every PASSED verdict in
every run to date is the debugger's own claim, asserted from inside a nested
`AgentTool` the validator has no visibility into.**

**Re-evaluation (cheaper than F24's full protocol redesign):** flip that one
rule. After the debugger returns "fixed", the validator re-runs
`invoke_mcp_tool` once with the same args and records **its own** verdict. It
already has the tool. Cost: one extra invocation per fixed tool. Benefit: every
green result becomes independently verified — the difference between a
believable Evaluation table and one a reviewer can dismiss.

A deeper version: **flatten the nesting** — let the Debugger only edit/install
and return, and make the Validator the sole owner of `invoke_mcp_tool`. That
removes the opacity entirely but is a bigger change; the instruction flip
captures ~80% of the value at ~5% of the cost.

Effort: ~1h (instruction). Impact: paper credibility. See [02](./02-stability.md) N2.

### 2.3 Strictly sequential stages — Environment ‖ Coder is free latency

`main.py`'s `run_pipeline` awaits `_run_stage` for explorer → environment →
coder → validator in strict order. But DESIGN.md itself says the Coder *"runs
against the explorer report; env builds in parallel conceptually."* Verified:
the Coder reads repo files (`bash`/`read_file`/`grep`) and writes `output/`
artifacts; the Environment writes `.venv`/`.venv-repo`. **They share no input or
output** — Coder depends only on the Explorer report, Environment depends only
on the Explorer report. The Validator is the first stage needing both.

**Re-evaluation:** `asyncio.gather` Environment and Coder. On the budgets you
set (env 40 min, coder 25 min), the serial critical path through both is up to
~65 min; in parallel it is `max(env, coder)`. Even at typical durations
(env ~40–250s, coder ~250–700s in the biotite metrics) you reclaim the smaller
of the two per repo.

Caveats to validate: both stages issue shell commands in the same container
(CPU/IO contention, not correctness), and the F19 unknown-tool patch + F22/F23
runtime support are global, so they compose. Recommend gating behind a flag and
measuring on 2–3 repos first.

Effort: ~3h. Impact: throughput, especially at `--parallel`.

### 2.4 Fixed wall-clock timeouts kill slow-but-working repos

`STAGE_TIMEOUT` is absolute wall-clock. AgML's validator hit 3164s and was
killed while genuinely making progress (final-eval). Before F23 you *couldn't*
do better — a frozen event loop meant no timer fired. **Post-F23 the loop stays
responsive**, so you can now distinguish "hung" from "slow."

**Re-evaluation:** add a **no-progress timeout** — reset a timer on every ADK
event; abort only if `T_idle` (e.g. 8 min) elapses with no new event, keeping a
generous absolute ceiling as a backstop. This stops penalizing hard repos that
are advancing and gives a truer denominator in the benchmark.

Effort: ~4h (wrap the `run_async` loop with an idle watchdog). Impact:
fewer spurious timeouts → less noise in the eval table.

### 2.5 Reports-only handoff vs. structured contract — the strategic choice

Everything machine-critical (the tool list, signatures, the `samples:` block,
the SKIP set, venv layout) is currently embedded in **LLM-authored markdown that
downstream code cannot reliably parse.** This is the root cause behind several
F-items: F25's SKIP gap exists *because* SKIP is free-text the validator LLM has
to re-read correctly every time; F1/F4 stay "optional" partly because there is
no structured tool list to gate on.

**Re-evaluation:** make the inter-agent contract **dual** — a small
JSON/pydantic sidecar (`reports/*.json`) for machine fields, the markdown kept
for humans and for LLM context. The Coder emits `tools: [{name, signature,
sample_args, skip:bool, skip_reason}]`; `main.py` parses it and hands the
validator a **computed** invoke/SKIP split (F25.1 becomes trivial), and F1/F4
gates operate on a real list. This is the one item here that is a genuine
re-architecture, but it is the enabler for a whole cluster of the backlog.

Effort: 1–2 days. Impact: unlocks F1/F4/F25 as *code*, not *prompting*.

### 2.6 Provider reliability is treated as a symptom, not a cause

F17 (retry on empty body) and F22 (detect unmapped `finish_reason:"error"`) are
both **mitigations for OpenRouter routing to a flaky backend.** OpenRouter lets
you pin the provider (`provider: {order:[...], allow_fallbacks:false}`) or you
can hit a first-party endpoint directly. For benchmark runs especially, pinning
removes the fault at the source instead of paying retry latency for it.

Effort: ~1h (env/config). Impact: fewer dead minutes, cleaner logs, and one
fewer confound in the eval.

### 2.7 `docker commit` captures the filesystem, including any leaked secret

Minor but worth a line for a paper that sells "reproducible artifact." You blank
secret **env vars** via `--change` and scrub `pipeline.log`, but `docker commit`
freezes the whole filesystem — any key an agent wrote to a file (a stray `.env`
under `output/`, a pip/hf cache, a shell history) persists in the shipped image.
A one-shot secret-scan (or building from F8's `install.sh` instead of
committing the dirty container) closes this. Low priority, but cheap to note in
the Ethics/Availability section pre-emptively.

## 3. Position vs. competition (condensed)

The three-way table in [TOOLMAKER_COMPARISON.md](../TOOLMAKER_COMPARISON.md) is
accurate; I won't repeat it. What matters for the paper:

**Alembic's durable advantages** (keep emphasizing): autonomous multi-tool
discovery, served MCP over HTTP, two-venv (hosts old-Python repos ToolMaker's
single Py3.12 and ToolRosella's ≥3.10 floor cannot), and real per-tool
invocation in the loop.

**Where ToolMaker is genuinely ahead, and why it reaches 80%:** it is handed a
**fixed target signature** and validated against **held-out gold unit tests**.
That is a fundamentally easier, more verifiable problem than "discover the right
tools from a URL." Two of its edges are things you can adopt without surrendering
autonomy:

- **Correctness signal** (its unit tests / your F2+F3): alembic's `{ok}` flag
  says "ran without raising," not "produced the right answer." A reviewer *will*
  ask how you know a PASSED tool is correct. N2 (independent re-invoke) is the
  floor; F2/F3 (semantic judge + held-out input) is the real answer; report 03
  shows how TM-Bench's typed `returns` + held-out `test_cases` give you a
  ready-made template for it.
- **Reproducibility**: ToolMaker's target is fixed and its image is built from a
  recorded script (F8). Alembic's per-run variance (§2.1) is the mirror-image
  weakness. N1 + persisting the generated `server.py` as an artifact answers it.

**Net:** you are not behind on capability; you are behind on **evidence**. The
cheap N-items convert capability into evidence.
