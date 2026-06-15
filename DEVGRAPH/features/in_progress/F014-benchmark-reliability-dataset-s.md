---
id: F014
title: Benchmark reliability on dataset_S.xlsx (drug-design molecule generation)
type: feature
status: in_progress
created: 2026-06-11
updated: 2026-06-13
owners: [SoloWayG]
derives_from: [F000]
depends_on: [F000, F003, F006, F010]
sources: []
tags: [benchmark, reliability, openrouter, fedotmas, orchestrator-loop, eval]
code:
  - CoScientist/config/settings.py:LLMSettings
  - CoScientist/agents/agents.py:provider_routing
  - CoScientist/tools/fedotmas_tools.py:FedotMASToolset
  - CoScientist/dataset_S.xlsx
  - scripts/opik_eval/opik_history.py
  - scripts/opik_eval/opik_deep.py
  - scripts/opik_eval/provider_pinning_microtest.py
  - scripts/opik_eval/dump_traces.py
  - scripts/opik_eval/analyze_dump.py
  - scripts/opik_eval/trace_locator.py
benchmarks: []
---

## Goal
Make CoScientist run the `dataset_S.xlsx` benchmark reliably. The dataset is a
**drug-design / molecule-generation** suite: columns `case`, `content`,
`decomposers_tasks`, `is_correct`; cases incl. alzheimer (GSK-3β), dyslipidemia
(PCSK9), lung cancer (KRAS G12C), sclerosis (BTK), Parkinson (ABL), Drug_Resistance
(STAT3). Tasks are NL "generate molecules that inhibit X with property Y" prompts.

Three failure modes were reported from runs on branch `refactoring_full_pipe`:
(1) OpenRouter returns empty strings from the LLM; (2) non-existent tools passed to
FEDOT.MAS; (3) the orchestrator loops. Two fixes were tried — provider pinning, and
swapping `gpt-oss-120b` → `qwen3-235b` — with unclear conclusions.

## Current state (after F014.A1 audit + F014.A2 Opik analysis + F014.A3 live test)
Ground truth now comes from **Opik traces** (each records `metadata.main_model`),
which corrected an earlier wrong inference:
- **Yesterday's benchmark (2026-06-10) actually ran on `qwen3-235b`**, not
  gpt-oss-120b — Opik proves it. (F014.A1 guessed "never swapped to qwen" from the
  *current* `.env`; that was wrong — the config changed over time. Today's runs
  reverted to gpt-oss-120b.) Lesson: trust per-run Opik metadata, not the live `.env`.
- **All three failures are evidenced** in real Opik traces (see F014.A2), all on
  the **qwen3** benchmark runs: empty/`finish=None` responses, hallucinated
  tool-names (`Tool 'smiles2props'/'predict_ml' not found`), and severe runaways
  (up to **81 LLM calls**, several hitting a hard **700s ceiling**).
- **Provider pinning works mechanically** (F014.A3, live): pin-ON confines
  gpt-oss-120b to DeepInfra; pin-OFF scatters across Novita/DigitalOcean/Parasail/
  DekaLLM, and a flaky off-list provider (DekaLLM) was caught returning ~empty
  (`ctok=1`). But pinning is currently **OFF** (`pinned_providers=[]`) and applies
  to no model in use → the empty-response fix is not engaged in production.
- **Switching to qwen does NOT fix flakiness**: qwen's routing scatters across even
  more providers (5 distinct in 5 calls), and the real qwen benchmark still emitted
  empties + `litellm.APIError: OpenRouter`; its runaways were the worst observed.

## Attempts
### F014.A1 — Forensic config + log audit (no live run) · 2026-06-11 · outcome: partial
- **Method:** instead of burning OpenRouter credits on a live VPN-dependent run,
  audit the actual config + the only persisted log (`logs/app.log`) and the dataset.
- **Result / findings:**
  - **CONFIRMED — model under test = `gpt-oss-120b`.** `logs/app.log` (2026-06-11
    run, and 2026-05-28) shows `LiteLLM completion() model= openai/gpt-oss-120b`
    repeatedly. `.env`: `LLM__MAIN_MODEL=openrouter/openai/gpt-oss-120b`; only
    `SCENARIO_MODEL` and `CODER_MODEL` are qwen3-235b. → **the main agentic model
    was never actually swapped to qwen**, so "did qwen help?" is unanswered.
  - **CONFIRMED — pinning disabled for the model that needs it.**
    `config/settings.py:32` `pinned_providers: List[str] = []`; comment (lines
    22–31): the 4-provider set `["deepinfra","groq","together","fireworks"]` is for
    `gpt-oss-120b`, was disabled when the team assumed the active model is qwen3
    (Alibaba/WandB). But `.env` still runs `gpt-oss-120b`. → **config drift**: the
    empty-string fix (pinning) is not engaged for the model in use, so "did pinning
    help?" is **untestable in this state** — it isn't on.
  - **CONFIRMED — orchestrator loop.** `logs/app.log:518-544` (07:11:32→07:13:11):
    **9** back-to-back `ResearchAgent` delegations re-issuing near-duplicate queries
    ("components used in the synthesis of BASHY dye" → "BASHY dyes synthesis
    components" → keywords "BASHY dye" → …). Two `keywords` args are **identical
    except for a different Unicode hyphen** (`‐` U+2010 vs `‑` U+2011) — so any
    exact-string dedup would miss them. Classic reformulation loop.
  - **UNCONFIRMED — empty OpenRouter strings.** Not found in `logs/app.log`; the
    only error there is a `httpcore.ConnectTimeout` (2026-05-28, network/VPN), not
    empty content. Reported by user; yesterday's logs not persisted. Per the
    anti-entrenchment rule: recorded as a hypothesis, not a fact.
  - **UNCONFIRMED — non-existent tools to FEDOT.MAS.** Not in the log. Likely origin
    (where to look): `fedotmas_tools.py:fedot_tool` builds `servers_payload` from
    `state['filtered_tools']` (Postgres lookup, None-filtered at line 58) **and**
    `state['deployed_mcps']` (web MCPs, **not** reachability-validated). A stale /
    unreachable entry in either would reach FEDOT.MAS as a "tool that doesn't exist".
- **Evidence:** `logs/app.log` (lines cited); `config/settings.py:20-33`;
  `.env` (MAIN_MODEL / SCENARIO_MODEL / CODER_MODEL); `fedotmas_tools.py:35-68`.
- **Next:** F014.A2 — a *controlled* live re-run (below), because A1 can't confirm
  the LLM-level failures without persisted logs.

### F014.A2 — Opik trace analysis of the real benchmark runs · 2026-06-11 · outcome: success
- **Method:** read agent traces from Opik (Comet cloud, workspace `itmo-nss`,
  project `adk-coscientist`) instead of stdout — see memory [[opik-tracing-access]].
  Scripts: `scripts/opik_eval/opik_history.py`, `opik_deep.py`. Surveyed 30 traces,
  deep-analyzed 11 (errors + runaways).
- **Findings (all evidenced by trace id/timestamp + span error_info):**
  - **Model attribution:** 2026-06-10 benchmark = `qwen3-235b` (all drug-design
    cases); 2026-06-11 = `gpt-oss-120b`. From per-trace `metadata.main_model`.
  - **Empty responses — CONFIRMED:** in the long qwen3 runs, ~1 LLM span per run
    has empty content with `finish_reason=None` (e.g. 17:15:28 `empty=1/45`,
    17:03:48 `empty=1/50`). Hard variants: `litellm.APIError: OpenRouter` on
    15:35:22 & 15:31:03; `BadRequestError` on 13:46:40.
  - **"Non-existent tools" — CONFIRMED, located & re-rooted:** raised by **ADK**
    (`google/adk/flows/llm_flows/functions.py:1000`: `Tool '{name}' not found.
    Available tools: ...`) when an LLM emits a function-call to an unregistered name.
    The error span is `invoke_agent molecule_generator` — i.e. **inside the FEDOT.MAS
    `molecule_generator` sub-agent** spawned by `fedot_tool`, NOT the CoScientist
    orchestrator. That sub-agent is equipped with only 5 *generation* tools
    (`list_generative_train_cases, get_state_from_server,
    start_generative_model_training, generate_mols, generate_case_mols`), but the
    benchmark tasks demand generate→**dock**→filter-by-**properties**. So its LLM
    calls evaluation tools that exist elsewhere (chemical MCP) but are NOT wired to
    it: `calculate_docking` (16:00:47), `smiles2props` (17:34:48), `predict_ml`
    (17:27:08), `get_structure_similarity` (17:03:48). → A **tool-roster/wiring
    mismatch** (sub-agent under-equipped for the multi-step task), not a
    server-reachability/payload issue (revises the F010/F014.A1 guess). Separate
    related cases at orchestrator level — `tavily_search` (Tavily disabled, F003),
    `request_approval` (HITL callbacks commented out, F001) — same family (call to a
    tool not attached), different locus.
  - **Runaway loops — CONFIRMED & quantified:** single queries with 28–81 LLM
    calls; several pinned at the **700s ceiling** (17:27, 17:15, 17:03). The
    repeated calls are `write_file`/`execute_bash`/`fedot_tool` — the agent
    thrashing (e.g. `execute_bash {"command":"env | grep -i mcp"}`).
  - **Infra noise (not LLM):** `McpError Timed out ... 300.0s` (17:38), Postgres
    `:5432` refused locally (today), `WinError 1225` (yesterday was a Windows box).
  - **Limitation:** Opik records only `provider=openrouter`, NOT the upstream
    OpenRouter sub-provider → pinning's routing effect is NOT checkable from Opik.
- **Evidence:** Opik traces listed above; reproduced via the saved scripts.

### F014.A3 — Live provider-pinning micro-experiment · 2026-06-11 · outcome: success
- **Method:** the full pipeline can't run here (local Postgres `:5432` down → runs
  crash after ~1 LLM call, cf. trace 04:41). So test the pinning hypothesis directly
  with raw `litellm.completion` calls that expose OpenRouter's *served provider*
  (response top-level `provider`). 5 calls × 3 conditions.
  Script: `scripts/opik_eval/provider_pinning_microtest.py`.
- **Result:**
  - **A — gpt-oss-120b PIN-ON** `["deepinfra","groq","together","fireworks"]`:
    served provider = **DeepInfra ×5** (confined to the pinned set). ✓ pinning routes.
  - **B — gpt-oss-120b PIN-OFF:** providers = DeepInfra, **Novita×2, DigitalOcean,
    Parasail** (run 1 also hit **DekaLLM `ctok=1`** — a real near-empty). Default
    routing is a grab-bag incl. off-list providers.
  - **C — qwen3-235b PIN-OFF:** **5 distinct providers in 5 calls** (WandB,
    DeepInfra, StreamLake, Alibaba, Together); clean `ctok=2` on the trivial prompt.
  - **Conclusion:** pinning **deterministically** confines routing and removes
    exposure to flaky providers (we directly observed a flaky near-empty on pin-off);
    qwen scatters across *more* providers, so it doesn't fix flakiness.
  - **⚠ Honesty caveats:** (1) N=5 is too small to quantify an empty-*rate*
    difference — the deterministic routing change is the solid result, not a rate.
    (2) Some "empty" flags here are a test artifact: `max_tokens=64` on a reasoning
    model (gpt-oss) gives `finish=length` with all tokens spent on reasoning. Only
    `finish=stop` + near-zero `ctok` (DekaLLM) is a genuine provider near-empty.
- **Evidence:** `/tmp/microtest.out` this session; rerun the saved script to reproduce.

### F014.A4 — Full-pipeline model A/B (qwen vs gpt-oss) + pinning re-test · 2026-06-12 · outcome: partial
- **Method:** with **Bug D fixed** (the `accumulated_tools` guard, see [[F009]].A3) so
  dataset_S runs no longer crash early on the TaskExecutor path, ran the **actual
  orchestrator** on **5 dataset_S queries** (GSK-3β, KRAS, STAT3, BTK, multi-target)
  per condition; model swapped via `os.environ['LLM__MAIN_MODEL']`; each trace scored
  with `scripts/opik_eval/metrics.py`. Conditions: gpt-oss-120b **unpinned**,
  gpt-oss-120b **pinned** (deepinfra/groq/together/fireworks), qwen3-235b.
- **Result (n=5 each):**
  - **qwen3-235b: 0 empties** (all `finish=STOP`), 0 errors, 1 runaway; but **~2×
    slower** (avg 324s vs 172s) and chattier (avg 17 vs 11.6 LLM calls).
  - **gpt-oss UNPINNED: 4 empties** (`finish=None`, genuinely empty output `{}` —
    inspected: `content=None`, no `reasoning`/`tool_calls`, so NOT a reasoning/length
    artifact), 1 ValueError, 1 runaway.
  - **gpt-oss PINNED: 3 empties** (`finish=None`), 1 JSONDecodeError, 0 runaways →
    **pinning did NOT reduce full-pipeline empties (4→3, within noise).**
- **Interpretation — two prior F014 claims NOT supported by this run:** (1) "qwen is
  the empty-prone one / qwen worse" — here **qwen had 0 empties** vs gpt-oss's 3-4.
  (2) "re-pin gpt-oss fixes the empties" — pinning confined *routing* in the raw
  micro-test (A3) but **did not remove full-pipeline empties** (A4). So the gpt-oss
  empties are not (only) provider-routing flakiness fixable by `provider.only`; open
  question — a specific agent turn? the pinned providers themselves under agentic load?
- **Evidence:** Opik traces — gpt-oss unpinned `019ebc31/37/3d/44/4f`, qwen
  `019ebc32/38/3e/49/55`, gpt-oss pinned `019ebc75/78/7e/84/87`; empty-span inspection
  (`output={}`, `finish=None`). See [[opik-tracing-access]].
- **⚠ Caveats (anti-entrenchment):** n=5/condition is noisy; Opik logs
  `provider=openrouter` only, so traces can't confirm `provider.only` engaged for the
  pinned arm (A2/A3 limitation) — pinning may have routed correctly yet those providers
  still empty under load, OR not engaged. Does NOT flip F014's conclusion; recorded as
  **counter-evidence** to re-test with a larger, provider-logged run.
- **Config fix applied (variant 1):** `.env` now sets `LLM__PINNED_PROVIDERS`
  (previously only the dead `LLM__ALLOWED_PROVIDERS`); `provider_routing()` reads
  `pinned_providers`, so gpt-oss is now pinned **by default**. Closes the config-drift gap.

### F014.A5 — Full 160-trace offline taxonomy (06-12/06-13) vs DEVGRAPH coverage · 2026-06-13 · outcome: partial
- **Method:** dumped **every** Opik trace since 2026-06-12 (server-side `start_time`
  filter, full bodies+spans) to a local folder and mined it offline (no rate limits).
  Tooling (new): `scripts/opik_eval/dump_traces.py` (resumable dumper), `analyze_dump.py`
  (exception/notfound/empty/runaway taxonomy with a per-day split), `trace_locator.py`
  (server-side `thread_id` resolver + run→trace manifest, [[F008]].A2). Corpus: **160
  traces** (110 on 06-12, 50 on 06-13; gpt-oss 61 / qwen 96). Goal: verify DEVGRAPH
  tracks the real errors and that "fixed" items don't recur.
- **DEVGRAPH coverage — CONFIRMED tracked & still present (good):** hallucinated tool
  names exactly as F014.A2 (`predict_ml`×6, `smiles2props`, `get_state_from_server`,
  + new agent-name hallucinations `property_filter_agent`/`literature_agent`/`research_agent`
  and **target-parameterized** `gsk3beta_molecule_generator`); runaways (23/160 ≥25 LLM
  or ≥690s); gpt-oss empties (`finish=None`); McpError 300s; ExceptionGroup (FEDOT).
  `KeyError: 'request'` (F006.A3) and `EOFError` HITL (F001.A2) appear **only in 06-12**
  traces — no recurrence in the 06-13 sample, consistent with those fixes holding.
- **GAPS / under-tracked (evidence-backed):**
  1. 🔴 **OpenRouter 402 `Insufficient credits` — UNTRACKED, batch-invalidating.** 20
     occurrences across **all 10 `l_L1`/`l_L2` runs at 14:36 on 06-13** (qwen) — every
     one ended **no-answer** purely from a billing outage, not its experimental condition.
     F014 logs only generic `APIError: OpenRouter`; a 402 is **account-wide & persistent**,
     so the entire L1-vs-L2 A/B from that window is **invalid data**. Same misattribution
     family as the "trust trace metadata over `.env`" lesson.
  2. 🟠 **`KeyError: 'Context variable not found: accumulated_tools / accumulated_web_mcps'`.**
     The Bug-D guard ([[F009]].A3) seeds `{accumulated_tools}` but the **sibling
     `{accumulated_web_mcps}` prompt var is unseeded** → same crash; and `accumulated_tools`
     KeyError still appears in some `session_001` traces (verify pre/post-guard date — if
     post, the guard is incomplete). Raised on the orchestrator span.
  3. 🟠 **`asyncpg` connect **TimeoutError** ×47 (all 06-13)** — a DB-**timeout** variant
     distinct from the refused/`ConnectionError` that F009.A2 degrades; the high count
     suggests retry-hammering or a path the graceful wrapper misses.
  4. **get_state_from_server over-reach = ORCHESTRATOR, 06-12 only (corrected).** The
     error's `Available tools:` list is the orchestrator roster (`list_available_tools,
     list_server_tools, Hypotheses/Research/TaskExecutor/Coder/MedicalAgent`), so the
     over-reach is the **master orchestrator** calling an MCP tool. Genuine occurrences:
     **4×, all 06-12** (`ab_B` crash + 2× `session_001`). An earlier draft said it "recurs
     on 06-13" — **false positive**: `get_state` is a *legit* tool of the FEDOT
     `molecule_generator` sub-agent, so it shows up in other agents' `Available tools:`
     lists. Whether B2 truly eliminated it is **UNPROVEN** (06-13 runs died on credits
     before reaching execution). Cross-ref F000.A3 + reminder in F000 TODO.
- **06-13 is infra-tainted:** 24/50 traces carry credits/DB-timeout failures, so the
  aggregate 06-13 "regression" (errored 30%→54%, answer 41%→34%) is **mostly an infra
  artifact**, not a prompt/model regression — must isolate before concluding anything.
- **Evidence:** `opik_dump/traces_since_2026-06-12/` (gitignored) + `analyze_dump.py`
  output; credits batch = thread_ids `l_L1_0*_143608`/`l_L2_0*_143608`; KeyError span
  msg `'Context variable not found: accumulated_web_mcps'` (trace `457bdaaa…`); get_state
  not-found in `ab_S5_0*`/`l_Lplan_0*`/`l_Lnone_00` (qwen, 06-13). See [[opik-tracing-access]].
- **Next:** (1) pre-flight/abort-on-402 + DB-timeout tagging in the runner so tainted runs
  never enter an A/B; (2) seed `{accumulated_web_mcps}`; (3) re-confirm get_state guard.

## ✅ TODO
- [ ] **Abort/flag credit- & DB-tainted runs (F014.A5):** detect OpenRouter `402` and
      `asyncpg`/MCP timeouts per run; tag the trace and **exclude from A/B aggregates**
      (a 402 invalidated the whole `l_L1`/`l_L2` 06-13 batch).
- [ ] **Seed `{accumulated_web_mcps}`** like `{accumulated_tools}` (Bug-D guard sibling,
      F014.A5) and verify the `accumulated_tools` KeyError is gone post-guard.
- [x] **Re-pin gpt-oss-120b** — DONE (F014.A4, variant 1): `.env` now sets
      `LLM__PINNED_PROVIDERS=["deepinfra","groq","together","fireworks"]` and
      `provider_routing()` reads it. BUT the full-pipeline A/B showed pinning did **not**
      reduce empties (4→3) — so pinning is engaged but is **not** the empties fix.
- [ ] **Persist run logs to file per benchmark run** — superseded in practice by Opik
      (A2), but a local file log still helps when Opik/network is down.
- [ ] **Full controlled A/B re-run on real infra** (needs Postgres + VPN — i.e. the
      Windows box, not this Mac): arm (a) gpt-oss-120b + pinning; arm (b) qwen3 main +
      pinning off. Measure empty-rate, tool-not-found count, loop size, `is_correct`/case.
      Drive it with a per-run runner; read results from Opik (A2 scripts).
- [~] **Fix config drift:** PARTLY done (F014.A4) — `.env` now drives `pinned_providers`
      via `LLM__PINNED_PROVIDERS`; the dead `allowed_providers` is marked legacy. Still
      TODO: have settings warn/raise on a known-bad pairing (e.g. gpt-oss-120b + []), and
      decide one coherent (`main_model`, pin) choice (qwen needs pin=[]).
- [ ] **Fix via the experiments module F015, not by touching tools (decision F014.D1):**
      the `molecule_generator` tool-roster mismatch is resolved by F015's per-step
      **tool-sufficiency check** (don't dispatch a step whose tools are absent; build
      them via Alembic) + structured plan + critic. Do NOT "equip molecule_generator"
      by editing MCP tools — see F014.D1.
- [ ] **Orchestrator-level tool gaps:** stop exposing/expecting `tavily_search` (Tavily
      disabled, F003) and `request_approval` (HITL callbacks commented, F001) in prompts.
- [ ] **Orchestrator loop guard:** NFKC-normalize + dedup near-identical delegations;
      cap repeated same-agent/same-tool calls with no new info (ties to F006 critic).
      Worst offenders observed: `write_file` / `execute_bash` thrash on qwen3.
- [ ] **Ask Opik (or patch) to record the upstream OpenRouter provider** so future
      pinning A/B is checkable from traces, not just raw calls (A2 limitation).

## ⚠ Pitfalls / Known problems
- **qwen malformed tool-call JSON KILLS the run (F015a.A4, 2026-06-15):** the model sometimes emits broken
  JSON in `tool_call.function.arguments` (missing comma / truncation / extra data — worse for big payloads
  like a `submit_plan` plan). ADK parses it at **`lite_llm.py:1630` `json.loads(...)` UNGUARDED in the
  non-streaming path** (the streaming path is guarded at :2158-2159) → `JSONDecodeError` propagates and ends
  the whole run with no answer. Evidenced: traces `019ec80c` (char 441), `019ec85a` (char 2122). Same task can
  pass on retry → run-to-run variance, same family as the empties. **Fix: JSON-repair shim at :1630 + bounded
  re-prompt.**
- **MCP 300s timeout sinks the run as `len=0` (F015a.A4):** `search_papers` / `generate_case_mols` (and other
  non-async MCP tools) hit a 300s ceiling → `McpError` propagates to the orchestrator → empty answer even
  though work was happening. Needs a per-MCP-call timeout + partial-result delivery, not whole-run abort.
- **"False success" — `status:success` ≠ correct (trace `019eb27d`, 2026-06-12):** a
  GSK-3beta run returned 10 valid SMILES but training **404'd** on a missing dataset
  (`gsk3b_inhibitors_chembl.csv`; only `Alzheimer.csv`/`Test_mas_1.csv` exist) and
  `predict_ml` had no model — so the molecules are **generic, unscored**. Missing data
  **assets** (datasets/models the task assumes) are a distinct failure class from
  tool-not-found / runaways. Eval must score correctness, not the success flag (see F015h).
- **Trust Opik per-run metadata over the live `.env`** — the model changed across
  sessions (qwen3 yesterday, gpt-oss today); only the trace's `metadata.main_model`
  tells you what a given run actually used. F014.A1 got this wrong by reading `.env`.
- **An OpenRouter `402 Insufficient credits` invalidates a whole batch, not one call
  (F014.A5).** It is account-wide and persistent, so every run in that window fails
  the same way (the 06-13 `l_L1`/`l_L2` batch: 10/10 no-answer). It masquerades as
  "model/prompt got worse" in aggregate metrics. **Check for 402 (and DB/MCP timeouts)
  before comparing conditions**, and exclude tainted runs — same misattribution trap
  as reading `.env` instead of trace metadata.
- **Config drift is real:** pinning is OFF for the gpt-oss model in use. Any claim
  that "pinning/qwen helped" is invalid until the config actually applies them.
- **Pinning is the right lever, but qwen is not a flakiness fix** (F014.A3): qwen
  scatters across *more* providers; its real benchmark still had empties + APIErrors
  and the worst runaways.
  **⚠ Counter-evidence (F014.A4):** in a fresh full-pipeline A/B (n=5/condition) the
  *opposite* showed — **qwen had 0 empties**, gpt-oss had 3-4 (pinned OR unpinned), and
  **pinning did not reduce them**. So neither "qwen worse" nor "pinning fixes empties"
  is settled. Conditions differ (A3 = raw litellm, trivial prompt; A4 = full agentic
  load) and n is small — do a larger provider-logged A/B before trusting either.
- **The "non-existent tools" bug is tool-name hallucination**, not server
  reachability — fix by constraining the exposed/registered toolset.
- **Loop dedup must NFKC-normalize** — a live loop slipped a duplicate past on a
  U+2010 vs U+2011 hyphen difference alone.
- **Measuring empties needs care:** `finish=length` on a reasoning model is a
  token-budget artifact, not a provider failure; the real signal is `finish=stop`
  with near-zero completion tokens (the DekaLLM `ctok=1` case).

## Decisions
### F014.D1 — Fix experiment orchestration, not the MCP tools · 2026-06-11
- **Context:** the tool-not-found failures come from a sub-agent (`molecule_generator`)
  calling tools it isn't equipped with (docking/properties), inside FEDOT.MAS.
- **Constraint (from the user):** we must NOT change the *content* of MCP tools; we may
  edit descriptions only of MCP servers we build ourselves; and the system is meant to
  *use and search third-party MCPs*, which we cannot edit at all.
- **Options:** (a) wire missing eval tools into `molecule_generator` — **rejected**: in
  general that means changing tools/servers we don't own. (b) enable ADK
  `reflect_retry_tool_plugin` — useful band-aid, doesn't address planning. (c) build the
  proper **experiments module ([[F015]])** with per-step tool-sufficiency + plan + critic,
  extending tooling via **Alembic** (new MCP servers) without touching existing ones.
- **Choice:** (c). The real fix is orchestration (F015), grounded in [[S009]]. (b) may be
  enabled opportunistically as a safety net.
- **Consequence:** F014's tool-not-found and runaway TODOs are owned by F015; F014
  remains the *measurement/benchmark* feature that will verify the fix on dataset_S.

## Symbols
- `CoScientist/config/settings.py:LLMSettings` — `pinned_providers` (currently `[]`), `main_model`.
- `CoScientist/agents/agents.py:provider_routing` — builds the OpenRouter `provider.only` payload; returns None when `pinned_providers` is empty (→ no pinning).
- `CoScientist/tools/fedotmas_tools.py:FedotMASToolset.fedot_tool` — assembles `servers_payload` for FEDOT.MAS from session state.
- `scripts/opik_eval/opik_history.py` · `opik_deep.py` — read/analyze runs from Opik (A2).
- `scripts/opik_eval/provider_pinning_microtest.py` — live pinning A/B/C (A3).
