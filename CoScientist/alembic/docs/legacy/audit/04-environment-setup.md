# 04 — Environment-setup workflow audit

Scope: `instructions/environment.py`, `tools/venv.py`, `tools/shell.py` — the
three files driving the Environment stage (two-venv layout, `setup_venv`,
`check_venv_compat`, `bash`/`bash_env`). Like [01](./01-architecture.md)–[03](./03-benchmarking.md),
this deliberately does **not** re-litigate [IMPROVEMENTS_SPEC.md](../IMPROVEMENTS_SPEC.md)'s
F1–F40 backlog; everything below is either a concrete gap not in that list, or a
calibration check against fresh, independent evidence (see [Method](#method--evidence-base)).

## Ranked findings

| # | Finding | Impact | Effort | Status |
|---|---|:---:|:---:|---|
| **N10** | `setup_venv`'s subprocess calls have no timeout | High | Low | New |
| **N11** | `setup_venv`'s own docstring example contradicts Critical Rule #3 | High | Low | New |
| **N12** | No guard against the agent backgrounding a slow install itself | High | Low–Med | New |
| **N13** | A cancelled `bash_env` call isn't killed, just abandoned | Med | Med | New (generalizes F12) |
| **N15** | Decision-tree parsing in `environment.py` should be a deterministic tool | Med–High | Med | Concretizes N8 |
| **N14** | "NOT `--index-url`" torch guidance is stricter than necessary | Low | Trivial | New |

N10–N12 are all High-impact/Low-effort — do these first.

---

## N10 — `setup_venv`'s subprocess calls have no timeout

**Problem.** Every `subprocess.run` in `tools/venv.py` omits `timeout=`:
`_pip_install` (line 16) and all three calls inside `_setup_venv_sync` (venv
creation, `-r requirements.txt`, `-e pyproject`, and the final
`fastmcp`/`pytest`/`mcp`+`packages` install) can block forever. Compare:
`bash_env` gets 900s, `check_venv_compat` gets `240 * TIMEOUT_SCALE`. The one
tool `environment.py` calls "the fast path" and tells the agent to try
**first** (Step 3a Attempt 1) is the one tool in the Environment stage with an
unbounded hang risk — a stalled resolver, a flaky mirror, or (worse) a package
that drops into an interactive prompt (some `setup.py`-based builds do) hangs
with nothing to catch it except the outer 2400s `STAGE_TIMEOUT`. And even that
doesn't actually kill the process — see N13.

**Fix.** Add `timeout=int(600 * TIMEOUT_SCALE)` (or similar) to every
`subprocess.run` in `venv.py`, catch `subprocess.TimeoutExpired` the same way
`_run_shell` does in `shell.py`, and return a clear
`{"success": False, "error": "timed out after Ns"}` instead of letting it
propagate as an uncaught exception up through the ADK tool-call machinery.

**Effort.** Low — mechanical, no behavior change on the happy path.

---

## N11 — `setup_venv`'s own docstring example contradicts Critical Rule #3

**Problem.** `environment.py`'s Critical Rule #3 is unambiguous:

> **Never use `pip install -e .` or editable installs.** ... Editable installs
> of complex Cython/C-extension projects almost always fail and waste many
> retries.

But `venv.py`'s `setup_venv` docstring gives this as its **second example**:

```python
setup_venv("https://github.com/Roestlab/massformer",
           pyproject_toml="pyproject.toml", python_version="3.11")
```

and that code path (`venv.py:83-91`) unconditionally runs
`_pip_install(..., "-e", str(proj_path.parent))` — exactly the editable
install Rule #3 forbids. It's dormant today only because `environment.py`'s
Step 3a Attempt 1 tells the agent to sidestep this parameter entirely for
pyproject-only repos ("list the runtime deps from `[project].dependencies`
... **NOT** `pip install -e .`" — i.e. use `packages=[...]` instead). The
`pyproject_toml=` parameter is never legitimately invoked by the current
instruction text.

That makes it a live footgun, not dead code: ADK surfaces a tool's docstring
to the LLM as the function's description/schema. A model that's uncertain
mid-task, or a smaller/less-agentic model that weights "the tool's own
worked example" over a rule stated 40 lines earlier in a long prose
instruction, has a concrete, one-call path back into the exact failure class
this system already spent effort learning to avoid.

**Fix.** Either delete the `pyproject_toml` parameter/branch from `setup_venv`
entirely (current instructions never use it — confirmed by grep), or change
its behavior to a non-editable install (`uv pip install <path-to-project>`,
no `-e`) so the example and the rule agree. Deleting is cleaner: a
non-editable build-from-pyproject install is still slower/more fragile than
extracting `[project].dependencies` into `packages=`, which is what Rule #3
is actually steering toward.

**Effort.** Low — delete ~9 lines + the misleading docstring example.

---

## N12 — No guard against the agent backgrounding a slow install itself

**Problem.** `_run_shell` (`shell.py`) does
`subprocess.run(stripped, shell=True, ...)` with no check for a trailing
background operator. Nothing stops the agent from writing
`uv pip install ... &`, `nohup uv pip install ... &`, or `... ; disown` —
which would return control (and an empty/misleading `output`) almost
instantly while the real install keeps running, unobserved, in the
background. `check_venv_compat`/the debugger would then see "package still
missing" and misdiagnose it as an install failure rather than a race, burning
a debugger round on a problem that would have resolved itself given time.

This is not a hypothetical. **I hit this exact failure mode myself this
session**, running the same kind of environment-setup work by hand: several
of my own subagents chose to background a slow `pip`/`uv` install with a
`Monitor`-style wait, then ended their turn — silently stalling until I
noticed and manually resumed them. Alembic's own `bash_env` docstring
advertises "900s timeout ... for slow installs" specifically *because* the
default `bash()` timeout (15s) is too short for package installs — which
means a model that has already been burned by that 15s timeout once is a
very plausible candidate for "let me background it so nothing times out,"
not realizing bash_env already solves that. A materially smaller/less
agentic model seems, if anything, more likely to reach for this pattern, not
less — it's a common shell habit that doesn't require understanding the
tool-call contract it breaks.

**Fix.** In `_run_shell`, reject (with a clear error, not a silent no-op)
commands containing a trailing `&` (not part of `&&`), or `nohup`/`disown`/
`setsid` combined with backgrounding. A simple regex on the stripped command
is enough — this doesn't need to be bulletproof against deliberate evasion,
only to stop the default, unremarkable case. At minimum, add one explicit
line to `bash`/`bash_env`'s docstrings forbidding it, since the docstring is
literally the agent-facing contract for these tools.

**Effort.** Low–Medium (a regex guard + a test is Low; hardening against
creative evasion is Medium and probably not worth it given the container is
already the security boundary).

---

## N13 — A cancelled `bash_env` call isn't killed, just abandoned

**Problem.** `bash`/`bash_env` wrap a blocking `subprocess.run` in
`asyncio.to_thread` (F23's fix for the event-loop-freeze problem). That's
correct for keeping the loop responsive, but it has a side effect nobody
asked for: when the **stage's** outer `asyncio.wait_for(..., timeout=
STAGE_TIMEOUT[stage])` fires — not `bash_env`'s own 900s, the coarser stage
budget — the awaiting task is cancelled, but the worker thread executing
`subprocess.run` keeps running to completion, and the OS child process is
untouched entirely (`asyncio.to_thread`'s executor has no mechanism to kill a
thread mid-flight). A long `uv pip install` / `apt-get` / `huggingface-cli
download` in flight when the Environment stage's 2400s fires keeps consuming
CPU/network/disk for a repo the pipeline has already logically abandoned.

This is the same root cause you already caught once, one layer up: F12's
write-up describes "an earlier foreground invocation's 2-minute CLI timeout
killed the host-side orchestrator but left its Docker container running
detached and orphaned for 39+ minutes." That was the container-orchestration
version of this bug; `bash_env`'s subprocess is the in-process version, and
it hasn't been fixed here yet.

**Fix.** Replace the bare `subprocess.run` in `_run_shell` with
`subprocess.Popen` started in its own process group
(`start_new_session=True`), and wrap the await in a `try/finally` (or an
`asyncio.CancelledError` handler) that kills the process group
(`os.killpg`) if the awaiting coroutine is cancelled before the process
exits. This bounds the leak to "cancelled work stops promptly" instead of
"cancelled work runs to its own natural completion regardless."

**Effort.** Medium — touches the core shell-execution path, worth a
dedicated test (start a `sleep 999`, cancel the wrapping task, assert the
child PID is gone).

---

## N14 — "NOT `--index-url`" torch guidance is stricter than necessary

**Problem.** `environment.py`'s Attempt 2 says:

> `torch`, `torchvision`, `torchaudio` → install separately with
> `--extra-index-url https://download.pytorch.org/whl/cpu` (NOT
> `--index-url`).

I verified directly (fresh `uv venv` + `uv pip install torch --index-url
https://download.pytorch.org/whl/cpu`, nothing else in the same call):

```
Resolved 10 packages in 764ms
 + filelock==3.29.0  + fsspec==2026.4.0  + jinja2==3.1.6  + markupsafe==3.0.3
 + mpmath==1.3.0  + networkx==3.6.1  + setuptools==70.2.0  + sympy==1.14.0
 + torch==2.12.1+cpu  + typing-extensions==4.15.0
```

PyTorch's own CPU wheel index mirrors **all** of torch's transitive deps
(confirmed via `curl .../whl/cpu/` — sympy, jinja2, networkx, filelock,
typing-extensions all listed), not just torch/torchvision/torchaudio
themselves. So a bare `--index-url` resolves cleanly as long as torch is
installed in its own dedicated call — which is exactly what the current
recipe already does ("install separately"). The real `--index-url` gotcha
(it *replaces* the default index, breaking resolution of any *non-torch*
package requested in the *same* invocation) doesn't apply to a torch-only
call.

**Fix.** Not urgent, but worth simplifying: drop the "NOT `--index-url`"
constraint for the torch-only case. `--extra-index-url` isn't wrong to keep
(it's strictly safe), but the current wording adds a rule for a problem that
doesn't occur in the shape this recipe already uses it in — and it's the
form that contradicts what a model has seen a thousand times in training
data (pytorch.org's own "Get Started" command is bare `--index-url`),
inviting an unforced "correction" back to the simpler form for zero benefit
either way.

**Effort.** Trivial (wording only).

---

## N15 — Concretizing N8 for the Environment stage: a deterministic layout-decision tool

**Problem.** `environment.py` is ~310 lines carrying a full decision tree
("declares Python ≥ 3.10 → one-venv; < 3.10 → two-venv; no declared version
but conflicts look version-bound → promote to two-venv"), a 3-attempt
fallback ladder, a 7-row symptom→fix lookup table, and a whole separate
weights-download subsection — all as prose the LLM must correctly parse and
apply, every run, from a markdown exploration report. [03-benchmarking.md](./01-architecture.md)'s
N8 already names the general fix ("structured inter-agent contract... makes
F1/F4/F25 enforceable in code instead of by LLM good-behaviour") — this is
the concrete instance for environment.py specifically, which N8 didn't spell
out: the one-venv/two-venv decision is a **pure function of the exploration
report's declared Python constraint**, and doesn't need an LLM at all.

Given the target system explicitly runs a smaller, less agentic model than
the one auditing it, collapsing exactly this kind of "correctly execute a
multi-branch IF tree out of a long prose document" step into a deterministic
tool call removes the single largest source of avoidable variance in the
Environment stage — more so than any wording fix to the prose itself could.

**Fix.** Add a small deterministic tool, e.g.
`detect_python_layout(repo_url) -> {"layout": "one-venv"|"two-venv",
"server_python": "3.10", "repo_python": "3.8"|None, "source": "pyproject.toml
python_requires"}`: parse `setup.py`/`pyproject.toml`/`environment.yml`/
`tox.ini` with `packaging.specifiers.SpecifierSet` (reusing the parsing
approach F31's now-reverted `check_ancient_pins` already prototyped, minus
the wheel-availability check that made it not pay for itself). Call it
**before** Step 2 in `environment.py`, and replace the "decide layout"
prose with "trust `detect_python_layout`'s result; do not re-derive it
yourself" — the same pattern already used successfully for the SKIP/invoke
split in F25 (`main.py`'s `_build_validator_message` computing it in code
and handing the validator the authoritative answer).

**Effort.** Medium — the specifier-resolution logic is the part F31 already
paid the cost to get right twice (exact pins vs. ranges vs. `requires_python`
exclusion); this reuses that lesson without reintroducing the part that
didn't pay off (wheel-tag checking).

---

## Calibration notes (existing backlog — no new action, just fresh evidence)

- **F5 (conda promoted to a primary strategy).** Across the same 14-repo set
  used as this audit's evidence base, conda was needed **0/14** times — even
  dependencies with a history of needing conda (rdkit-family, SimpleITK,
  opencv) had current, working wheels via plain `uv`. Weak but real evidence
  that the existing 3-attempt ordering (uv → uv-no-pins → conda) is fine for
  repos published in the last ~2-3 years; F5 likely matters more for the
  older/legacy repos your own audit already surfaced (`BioSPPy`,
  `auto-sklearn`-era pins) than for the average case. Doesn't argue against
  F5, just against raising its priority further without repo-pool evidence.
- **`STAGE_TIMEOUT["environment"] = 2400s`.** Well-calibrated against real
  numbers: my 14 runs (same uv-based, CPU-only-torch approach) ranged
  84s–1644s end-to-end, comfortably inside budget even accounting for a
  materially less capable model needing more retries per repo.

## Method / evidence base

Findings above are grounded in a from-scratch, non-agentic (by me directly,
not delegated) environment-setup pass over the same 14 repos as
`benchmarks/alembic/toolmaker_subset.txt`, using the same core tool (`uv`) as
`venv.py`. Per-repo wall-clock and issue count:

| Repo | Time | Notable issue hit |
|---|---:|---|
| CONCH | 1644s | unpinned `transformers` resolved to an incompatible v5.x |
| MedSAM | 1568s | `setup.py`'s custom install hook doesn't fire under PEP 517 editable installs |
| MedSSS | 1234s | heavy/optional deps (`vllm`, `deepspeed`) correctly skipped CPU-only |
| ModernBERT | 1414s | repo ships training scripts only, not an installable package |
| MUSK | 1182s | same unpinned-`transformers` class as CONCH |
| nnUNet | 207s | transitive `torchvision` resolved non-CPU, ABI-incompatible with CPU torch |
| flowmap | 172s | needed `--recurse-submodules` (same class as F34) |
| cytopus | 150s | undeclared `matplotlib`/`numpy` deps; `setuptools<81` for `pkg_resources` |
| esm | 123s | undeclared `numpy` dep |
| RETFound_MAE | 105s | none — clean |
| UNI | 104s | none — clean |
| TabPFN | 176s | none — clean |
| STAMP | 171s | none — clean |
| PathFinderCRC | 84s | none — clean |

No pretrained weights were downloaded (several are HF-gated, matching your
own F6/F7 scope); all environments used CPU-only PyTorch. Full per-repo
`install.log`/`stats.json` available on request if useful as fixtures for
testing the N15 tool above.
