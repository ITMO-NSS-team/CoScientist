# Environment & Installation Upgrade — Plan

**Principle:** one *deterministic* install path, run identically on every repo;
the LLM is invoked only for genuine errors, never to decide the happy path or
create files. Always two-venv (dependency isolation, not a version decision).
Repo-appropriate Python. fastmcp leaves the environment stage entirely.

Motivation: the install job is today smeared across four components
(`setup_venv` installs deps but refuses the package; the env agent installs it
inconsistently; the env gate band-aids it; the debugger flails/fabricates
files). photutils looped three times because of exactly this scatter.

---

## 1. Unified repo install — `install_repo(venv, repo_dir)`

Deterministic, keyed on which files exist and on the import outcome — no LLM
judgment:

1. `requirements.txt` present → `uv pip install -r requirements.txt` (some repos
   declare deps only here).
2. `pyproject.toml`/`setup.py` present → `uv pip install -e .` — installs the
   package **and** its declared deps **and** builds any Cython/C extension.
   (Proven: the debugger's own `-e ./repos` on photutils returned exit 0.)
   On editable failure, retry once non-editable (`uv pip install .`).
3. Neither (script-only repo) → drop a `<repo>.pth` into the venv's
   `site-packages` (repo root importable).
4. Verify `import <top_module>` for each planned tool's target — **the single
   source of truth** for "did it work."

Only if step 4 still fails → **one** scoped debugger round: fix a *real* error
(relax an over-tight pin, `apt` a system lib). It may NOT create tools/tests/
`.py` files (already enforced in the env-gate prompt).

**Knob:** editable (`-e`) is the default — matches ToolMaker, forgiving of
sloppy research packaging, keeps repo source live for the debugger.

---

## 2. Two-venv model (always)

| | **main venv** | **server venv** |
|---|---|---|
| holds | repo (via `install_repo`) + its deps + **pytest** | **fastmcp** only |
| python | repo-appropriate (`requires-python`) | fixed ≥3.10 |
| runs | tool functions + tests (`run_function.py`, pytest) | `server.py` |
| built by | Environment stage | **Wrapper stage** (right before codegen) |

The server venv never touches the repo; `server.py` shells to the main venv via
`run_function.py` (unchanged). This retires the one-venv layout: the split is now
purely dep isolation, so it's constant — no `decide_layout` version branch.

**Naming decision (pick one):**
- **A. Rename (recommended):** `.venv` = main, `.venv-server` = fastmcp. Matches
  the mental model; touches `paths.py`, codegen `_PYTHON`, `serve.py`, gate.
- **B. Keep names:** `.venv` = server, `.venv-repo` = main (always created).
  Least churn, but "main = `.venv-repo`" stays counterintuitive.

---

## 3. Slimmed env gate (G2)

- **Hard (must pass):** main venv exists · every planned tool's module imports in
  it (step-4 smoke) · pytest present.
- **Soft (nudge, don't fail):** `check_venv_compat` conflicts.
- **Removed:** all fastmcp/mcp/server-venv checks (now the wrapper's G4 job).
- Keep the `.resolve()`→`.absolute()` fix (probe/install the venv, not the
  externally-managed base python).

---

## 4. Stage responsibility shifts

- `decide_layout` → always two-venv; `repo_python` from `requires-python`,
  `server_python` fixed (≥3.10).
- `setup_venv` → thin: create the venv (+ pytest); the repo/dep install moves to
  `install_repo`. Its "editable installs unsupported" stance is dropped.
- **Environment agent** → shrinks to the genuinely creative work: choose the
  Python version, `apt-get` system libs, download weights. The repo/dep install
  is deterministic, not prompted.
- **Wrapper stage** → first build the server venv + `ensure_server_packages`
  (fastmcp/mcp), then codegen `server.py` + G4. (Move `ensure_server_packages`
  here from the env gate.)

---

## 5. Files touched

`tools/venv.py` (new `install_repo`; thin `setup_venv`; move
`ensure_server_packages` call-site) · `tools/analysis.py` (`decide_layout` →
always two-venv) · `main.py` (`_env_gate` slim + `install_repo`; `_wrap` builds
server venv first) · `tools/paths.py` (`tools_python`/server-python per naming
choice) · `tools/codegen.py` + `docker/alembic/serve.py` (server-venv name) ·
`instructions/environment.py` (drop install micromanagement) · base image
`docker/alembic/Dockerfile` (Python bump if desired — orthogonal).

---

## 6. Out of scope / unchanged

- Harness JSON contract (`validation.json` etc.) — untouched.
- **TM-Bench strict export** (re-install into `python:3.12`) — a separate
  export-time concern, not the core env strategy. Native builds use the repo's
  proper version.
- One-venv mode — retired.
