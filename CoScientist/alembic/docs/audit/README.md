# Alembic — Architecture & Benchmarking Audit (2026-07-06)

An independent audit of the `CoScientist/alembic` module: architecture, current
performance, design choices vs. competition, plus the two things you flagged as
painful — **end-to-end tool-test stability** and **benchmarking (TM-Bench)**.

This audit deliberately does **not** re-litigate the F1–F28 backlog in
[IMPROVEMENTS_SPEC.md](../IMPROVEMENTS_SPEC.md). Everything here is either (a) a
concrete item **not** in that list, or (b) a **re-evaluation of a design choice**
the F-list treats as fixed. Where an item extends an existing F-number, it says so.

## Reports

1. [01-architecture.md](./01-architecture.md) — architecture map, design-choice
   re-evaluations, position vs. ToolMaker/ToolRosella, what a reviewer will attack.
2. [02-stability.md](./02-stability.md) — the E2E-test instability, root-caused by
   source, with new cheap fixes. **Start here for your #1 headache.**
3. [03-benchmarking.md](./03-benchmarking.md) — TM-Bench / ToolArena integration,
   three corrections to the current plan, the dual-metric design, feasibility.
4. [04-environment-setup.md](./04-environment-setup.md) — audit of
   `environment.py`/`venv.py`/`shell.py`, grounded in an independent from-scratch
   14-repo run of the same `toolmaker_subset.txt` set: two unbounded-timeout gaps,
   a docstring that contradicts the no-editable-install rule, a self-backgrounding
   risk observed live in a different agent this same session, plus calibration
   data (conda-attempt ordering, stage-timeout sizing) from the independent run.

---

## Executive summary

Alembic is architecturally ahead of both named competitors on the axes that
matter for the paper's thesis (autonomous **multi-tool** discovery, **served
MCP**, **two-venv** for old-Python repos, **real in-loop invocation**). The
F14–F28 work has hardened the runtime honestly and the code quality is good.

**The single biggest problem is not a missing feature — it is measurement
non-determinism.** No agent sets a sampling temperature (confirmed: every
`LiteLlm(model=MODEL)` in `agents.py` uses provider-default sampling), so each
run regenerates different code, different samples, and different pass/fail
outcomes. Your own final-eval spread is **31.9% → 45.5% → 54.5% tool-pass across
three "identical" runs** — a swing larger than almost any fix in the backlog.
This is simultaneously the source of your "E2E stability" pain **and** the thing
a reviewer will hit hardest ("is this reproducible?"). Two of the top three
actions below address it directly and cost almost nothing.

## Top actions not already in the backlog

Ordered by (impact ÷ effort). "New" = not in IMPROVEMENTS_SPEC. Effort is
rough eng-hours.

| # | Action | Effort | Why it matters | Report |
|---|--------|:------:|----------------|:------:|
| **N1** ⏸️ | **Pin sampling (temperature) to cut run-to-run variance** — **POSTPONED** | ~1h | Would cut the variance that is both the stability pain and the reproducibility risk — BUT pinning the temperature (tried 0 and 0.2) reliably sent the Explorer into `read_file` repetition loops (cycling the same modules), even with dedup + a cycle-breaker backstop. **Reverted to provider default 2026-07-06; kept opt-in via `MODEL_TEMPERATURE` env for a later attempt.** | [02](./02-stability.md), [03](./03-benchmarking.md) |
| **N2** | **Validator must independently re-invoke a tool after every debugger "fixed" claim** | ~1h | Today *every* PASSED rests on the debugger's self-report (F24). One instruction flip + a tool it already has makes every green result trustworthy. | [01](./01-architecture.md), [02](./02-stability.md) |
| **N3** | **Deterministic pre-invocation sample gate** (resolve path-shaped args + check basic preconditions *before* running the tool) returning a distinct `bad_sample` outcome that routes to sample-repair, not the debugger | ~3h | Kills the "files missing / sample wrong" class deterministically instead of burning a live invocation + debugger round-trip on it. | [02](./02-stability.md) |
| **N4** ✅ | **Explorer reads the repo's own `tests/`+`examples/`** (it was told to *skip* them) | ~1h | Those files are the single best source of real signatures (F1), real fixture paths (F21) and real sample sizes (F18) — three open problems fed by one cheap instruction change. **Implemented 2026-07-06.** | [02](./02-stability.md) |
| **N5** | **Robust tool-result contract**: sentinel-delimited JSON (or a result FD), not "last line of stdout" | ~2h | Chatty repos (progress bars, warnings, atexit prints) corrupt the current parse → spurious `JSONDecodeError`. | [02](./02-stability.md) |
| **N6** | **Run Environment ‖ Coder concurrently** (they share no artifact; DESIGN.md already calls them "parallel conceptually") | ~3h | Saves up to a full Coder-stage (~25 min budget) of wall-clock per repo; the pipeline is strictly sequential today. | [01](./01-architecture.md) |
| **N7** | **Idle / no-progress timeout** alongside the wall-clock cap | ~4h | Now feasible post-F23 (event loop no longer freezes). Stops killing genuinely-slow-but-working repos (AgML) as if they were hung. | [01](./01-architecture.md) |
| **N8** | **Structured (JSON/pydantic) inter-agent contract** for machine-critical fields (tool list, signatures, sample args, SKIP set), prose reports kept for humans | ~1–2d | The strategic fix: makes F1/F4/F25 enforceable in code instead of by LLM good-behaviour, and removes the "validator misread the markdown" bug class (F25's SKIP gap). | [01](./01-architecture.md) |
| **N9** | **Pin OpenRouter provider routing / use a direct endpoint** for benchmark runs | ~1h | F17/F22 treat the *symptoms* of a flaky backend; pinning the provider removes the cause of the `finish_reason:"error"` and empty-body faults. | [01](./01-architecture.md) |
| **N10** | **Add a timeout to every `setup_venv`/`_pip_install` subprocess call** in `tools/venv.py` | ~1h | Currently unbounded — the one tool the Environment stage is told to try *first* is the one with no hang protection at all, unlike `bash_env`/`check_venv_compat`. | [04](./04-environment-setup.md) |
| **N11** | **Delete (or fix) `setup_venv`'s `pyproject_toml=` editable-install branch** | ~1h | Its own docstring example directly contradicts `environment.py`'s Critical Rule #3 (no editable installs) — a live footgun, not dead code, since ADK surfaces docstrings to the LLM. | [04](./04-environment-setup.md) |
| **N12** | **Reject self-backgrounded commands (`&`/`nohup`/`disown`) in `bash`/`bash_env`** | ~2h | Observed live this session in a different agent doing the same environment-setup work: backgrounding a slow install makes the tool report false success while the real work stalls unobserved. | [04](./04-environment-setup.md) |

N1–N5 are all cheap and all target the stability headache. If you do nothing
else this week, do **N1 + N2 + N4** (≈3 hours total, all instruction/config).
N10–N12 are the equivalent short list for the Environment stage specifically.

## One-line take on the competition

Keep the design. Alembic wins on output format and autonomy. The two places
ToolMaker is genuinely ahead — **gold-output correctness** (held-out unit tests)
and **reproducibility** (deterministic, single fixed target) — are exactly what
N1/N2 and the TM-Bench adoption in report 03 would close, without giving up
alembic's harder, more valuable problem.
