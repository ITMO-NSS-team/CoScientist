# 02 — End-to-end tool-test stability

Your stated pain: *"the stability is not great … sometimes files are missing …
performing the test itself is hard."* This report root-causes the instability by
source and gives cheap fixes, most of which are **not** in the F-backlog.

## The instability is five distinct sources, ranked by contribution

| # | Source | Symptom in your logs | Fix | Status |
|---|--------|----------------------|-----|--------|
| 1 | **LLM nondeterminism (no temperature set)** | Same repo passes then fails across "identical" runs; 31.9→54.5% spread | **N1** below | New |
| 2 | **Bad/missing sample args** | `FileNotFoundError` on invented paths; `signal too short`; "file cannot be read" | **N3** + **N4** below | New (extends F18/F21/F24) |
| 3 | **Fragile stdout→JSON contract** | `JSONDecodeError: Expecting value: line 1 column 1` (aizynthfinder, dalle-mini) | **N5** below | New |
| 4 | **Unverified debugger fixes** | PASSED that isn't really checked; sibling tool re-hits fixed bug | **N2** below | New (cheaper than F24) |
| 5 | Heavy tools / weights / GPU | SKIP-then-invoked training runs, timeouts | F25 + F6 | Already specced |

Sources 1–4 are all cheap and mostly untracked. Do them first.

---

## N1 — Kill the nondeterminism (config, ~1h) — ✅ IMPLEMENTED 2026-07-06

Confirmed: no agent set a sampling temperature. This is the largest single
contributor to "it passed yesterday and fails today."

**Implemented (code part):** `agents.py` now builds every agent via a `_model()`
factory = `LiteLlm(model=MODEL, temperature=MODEL_TEMPERATURE, top_p=MODEL_TOP_P)`,
defaulting to `temperature=0`, `top_p=1` (env-overridable via `MODEL_TEMPERATURE`
/ `MODEL_TOP_P` for seed/variance experiments). LiteLlm forwards these kwargs to
`litellm.completion()`, so they reach the provider. Still to do (run methodology,
not code): report the benchmark as mean ± std / pass@k over k≥3 runs, and persist
each run's generated `server.py` + samples as artifacts.

**Do:**
1. Set `temperature=0` and a fixed `top_p` on each agent's model (ADK
   generation config). Qwen at temp 0 is far more repeatable.
2. For the benchmark, run each repo **k≥3 times** and report per-repo
   pass **distribution** (mean ± std, or pass@k), so residual variance is
   *measured* instead of being a surprise.
3. Persist the generated `server.py` + `samples` for each run as artifacts, so a
   given result can be re-validated against the exact code that produced it.

This single change converts "flaky" from an unbounded worry into a bounded,
reported quantity — and it is what a reviewer will expect from a stochastic
system.

## N2 — Make PASSED mean "verified" (instruction, ~1h)

`instructions/validator.py` Step 4 currently says *do NOT re-invoke after the
debugger returns; trust its summary.* Flip it: after any debugger "fixed" claim,
the validator calls `invoke_mcp_tool` once itself and records its own verdict.
The validator already owns that tool. This closes the most damaging half of F24
(unverified self-reports) for the price of one extra call per fixed tool, and it
directly improves the trustworthiness of every number in your Evaluation table.

While editing that instruction, add the F24 sibling-propagation note as a
one-liner: *after a fix to one tool's helper, re-invoke any sibling tool that
shares the same helper pattern before trusting it* (this is what re-hit the
`ecg`→`ppg` argv bug).

## N3 — Deterministic sample gate before invocation (code, ~3h)

Today a bad sample (nonexistent file, too-small array) costs a full
`invoke_mcp_tool` subprocess spin-up **plus** a debugger round-trip before
anyone realizes the *sample* — not the code — was wrong. F18/F21 attack this
with prompting only, and F24 shows it recurs anyway.

**Add a deterministic pre-flight inside `invoke_mcp_tool` (or a thin wrapper):**
before running the tool, for every string arg that looks like a path, resolve it
against `REPO_PATH` and `cwd`; if it does not exist, return a **distinct
outcome**:

```json
{"ok": false, "kind": "bad_sample",
 "error": "sample file 'example.pdb' does not resolve under REPO_PATH"}
```

The validator treats `kind:"bad_sample"` differently from a runtime failure: it
does **not** call the debugger (the code may be fine) — it asks for a corrected
sample (or marks the tool "unverified — no valid sample" rather than FAILED).
Optionally extend the same gate to cheap precondition checks the coder recorded
(min length/duration), giving F24's "the sample is wrong, not the code" outcome a
concrete home. This is the durable version of F18/F21 the docs keep deferring to
"F1/F4 territory" — but as a ~30-line resolver it needs neither.

## N4 — Explorer must read the repo's own tests/examples (instruction, ~1h) — ✅ IMPLEMENTED 2026-07-06

**Implemented:** `instructions/explorer.py` now (1) lists the repo's own
`tests/`/`examples/`/`demo*.py` as a high-priority read with "READ AT LEAST ONE",
(2) requires extracting real call signatures + real fixture paths + real input
sizes from them, (3) points the "Examples" block at that harvest as the preferred
source, and (4) the final line no longer skips tests (only migrations/CI/internal
detail), and the call budget was raised 20→25 with the test/example read made
mandatory before writing the report.

Original finding — a direct contradiction in the prompts:

- `instructions/explorer.py` line 124: *"Skip: tests, migrations, CI configs …"*
- `instructions/coder.py` (F18/F21): *"Prefer real sample data the repo ships in
  its own `tests/`, `examples/`, or `data/` directories."*

The explorer is told to skip exactly the files that contain the real signatures,
real fixture paths, and real input sizes the coder is later told to prefer — and
then the coder invents `example.pdb` because nobody surfaced the real one.

**Do:** change the explorer priority list to **explicitly read 1–2 of the repo's
own test files and example scripts**, and to record, per proposed tool, a real
fixture path and a real invocation lifted from those tests. This one instruction
change feeds three separate open problems at once (F1 real symbols, F21 real
paths, F18 real sizes) and is the highest-leverage prompt edit available.

Pair with a minor budget tweak: the flat "20 tool calls / 7 files" cap starves
large repos (astropy, biopython). Scale it with repo size, or at least exempt
"read one test file" from the cap.

## N5 — Robust tool-result contract (code, ~2h)

`tools/invoke.py` parses the tool result as `stdout.splitlines()[-1]` → the last
line must be the JSON. This breaks whenever the wrapped repo prints anything
after the helper's `print(json.dumps(...))`: progress bars, `logging` to stdout,
library banners, `atexit`/thread output. That is the real origin of several
`JSONDecodeError` failures (and part of why `run_interactive_gui`-style tools —
F26 — fail so opaquely).

**Do:** change the helper contract from "JSON is the last line" to a
**sentinel-delimited block**:

```python
print("<<<ALEMBIC_RESULT>>>"); print(json.dumps(result))
```

and have the invoker extract the text after the last sentinel. (Alternative:
write the JSON to a dedicated file descriptor / a known temp file the invoker
reads, leaving stdout entirely to the repo.) Update the `coder.py` helper
template + `invoke_tool.py` together. Removes a whole class of "ran fine but I
can't read the answer" failures.

---

## How these connect to the benchmark

N3 and N4 are the same problem TM-Bench solves by *fiat*: it ships **gold,
typed, held-out invocation examples** per task, so "what do I call this with"
never depends on the LLM guessing. See [03-benchmarking.md](./03-benchmarking.md)
§4 — even before you build the full TM-Bench harness, adopting its *shape*
(typed `returns` schema + a second held-out invocation + structure/value
assertions) gives your internal validator a correctness signal and removes the
sample-guessing instability in one move. N1+N2+N3+N4+N5 make the current
validator trustworthy; TM-Bench makes it *gold-graded*.
