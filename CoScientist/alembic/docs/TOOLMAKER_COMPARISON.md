# ToolMaker vs. Alembic — repo→tool comparison

Companion to [DESIGN.md](./DESIGN.md) and [TOOLROSELLA_COMPARISON.md](./TOOLROSELLA_COMPARISON.md). Source analyzed: `ToolMaker/toolmaker` (ACL 2025, KatherLab) + paper `arXiv:2502.11705v2`. ToolMaker does **not** emit MCP — it emits a standalone Python function — but it tackles the same install/generate/validate/repair problem, so the same six axes apply.

> **Biggest framing difference, read first.** ToolMaker is **task-specified, single-function**: a human writes a `task.yaml` giving a one-line description, the GitHub URL, the **exact function name + argument list**, and **one example invocation**. ToolMaker then produces *one* tool with that fixed signature. Alembic and ToolRosella are **repo-autonomous, multi-tool**: they decide *what* tools exist and their signatures from the repo alone. So ToolMaker trades autonomy for a much tighter, verifiable target — which is why several of its robustness mechanisms work as well as they do.

---

## 1. ToolMaker — system design

Three CLI stages (`install` → `create` → `run`), all executed **inside Docker** via an HTTP runtime server. LLM = OpenAI `gpt-4o` for most calls + a reasoning model for plan/initial-implementation. The agentic core is a formal state machine over `(conversation_history, environment_state)` with three component types: **LLM calls** (mutate only the conversation), **environment interactions** (read/write the container), and **agents** (chain the two).

```
install ─► agent sets up repo in container ─► record all write-actions as install.sh ─► docker build ─► toolmaker-tool:installed-<name>
                                                                                                              │
create  ─► make_plan (explore→pseudocode) ─► implement_function ──► CLOSED LOOP (≤30):                        │
              reset container to installed image ─► run function with EXAMPLE args ─► ASSESS (LLM judge)      │
              └ if not successful: DIAGNOSE (gather info, env reset after) ─► REIMPLEMENT ─► SUMMARISE ◄──────┘
run     ─► load installed image, mount input/output, execute implementation.py with invocation args (cached by hash)
```

### Stage-by-stage

| Stage | What it does |
|---|---|
| **install** (`install_repository`, ≤20–30 agent steps) | Agent in a fresh `python:3.12` container (CPU **or** CUDA, chosen by `definition.requires`) clones the repo, reads READMEs, **installs globally with pip** (told to avoid conda/venv/Docker-in-Docker), and **downloads pretrained models** — *but explicitly not datasets*. Every **write action** is captured as bash → concatenated into **`install.sh`** → baked into a reproducible image via `tool.Dockerfile` (bakes `.env` at build, `rm`s it after). Also emits `installed_repository.yaml` (path + summary). |
| **make_plan** | `explore_repository` agent (read-only — `WriteFile` removed from its action set) reads only relevant files; then a **reasoning model** writes high-level numbered pseudo-code. |
| **implement_function** | **Reasoning model** writes a single standalone `def {name}(...)` with type hints + docstring, heavy logging, full tracebacks, and a `run_and_stream_command` subprocess helper. Strong constraints: one function only, prefer wrapping the repo's CLI/functions over reimplementing. |
| **make_tool (closed loop, ≤30)** | Each iteration: `reset_runtime()` (fresh container from the installed image) → run the function with the **example** args via `/run` → `is_successful_execution` (LLM judge) → if not: `diagnose` (agent explores; **env is reset afterward so debug side-effects never persist**) → `rewrite_function` → `summarize_problem` (appended to a running `problem_summaries` memory fed into the next diagnosis). |
| **run** | Load installed image, mount input/output dirs, execute `implementation.py` with invocation args; **cache result by `sha256(tool:installed:args:mount)`**. |

**Runtime model:** `runtime.Dockerfile` builds `toolmaker-runtime` = a FastAPI server; actions are `POST /execute/<action>`, functions are `POST /run` (which shells out to `toolmaker_function_runner.py` that writes the code to disk, imports it, calls it with kwargs, dumps JSON). State restoration between loop iterations = reload the frozen installed image. Action set: `RUN_BASH_COMMAND, LIST_DIRECTORY, READ_FILE, WRITE_FILE, BROWSE, GOOGLE_DRIVE_LIST_FOLDER, GOOGLE_DRIVE_DOWNLOAD_FILE, RUN_IMPLEMENTATION`.

### Paper numbers
**TM-Bench**: 15 tasks (pathology, radiology, omics, + 3D vision/NLP/tabular), **42 test invocations**, **124 unit tests** (avg 8.3/task) asserting *structure / values / files / execution*. **Success criterion: a tool is correct iff it passes ALL unit tests of its test invocations** — and crucially those invocations use **different args than the creation example** (held out, to catch hard-coding). **ToolMaker ≈ 80% of tasks correct**, substantially beating the **OpenHands** SWE-agent baseline (SOTA on SWE-bench). Tools created with a closed-loop self-correction over ≤30 iterations.

---

## 2. Direct answers (a–f)

**a) Missing env keys the underlying repo requires** — **Handled, but author-declared (not auto-discovered).** The `task.yaml`'s `repo.env` lists required keys; `resolve_env()` resolves their values and injects them into the container at install/create/run. A `${env:VAR}` substitution mechanism with an **allowlist** (`ALLOWED_ENV_VARS = {"HF_TOKEN"}`) lets task definitions/arguments reference host secrets safely. The agent is explicitly told which vars exist (install + implement prompts). Secrets are baked into the build `.env` then deleted. → **Stronger than alembic & ToolRosella** (which inject no repo-specific secret), but still relies on a human declaring the key in `task.yaml`.

**b) Automatic environment setup** — A dedicated **agentic install stage** in a fresh container, README-driven, **global pip**, with the key twist that **all write-actions are recorded → `install.sh` → a reproducible `docker build`** (not a `commit` of a mutated container). CPU/CUDA base images. **Limitation: single global Python (3.12 base), no version negotiation, no two-venv** — so old-Python scientific repos are out of scope; it leans on the base image for system libs.

**c) External downloads (HF model, weights, README-only mention)** — **Best of the three.** The install agent is *instructed* to "download any necessary pretrained models. However, do NOT download any datasets," reads the README and acts, and has dedicated `BROWSE` + `GOOGLE_DRIVE_DOWNLOAD_FILE` actions plus bash (`wget`/`curl`/`huggingface-cli` with `HF_TOKEN`). README-mentioned model fetches are followed. **Datasets are deliberately not downloaded** — they enter per-invocation as **Docker mounts** (`Mounts.data_dir` / `input_mapping`).

**d) End-to-end tests with real tool calls + success criterion** — **Strong, real execution.** Every loop iteration **actually runs the function** in the installed container with the example args, and `ASSESS_TOOL_OUTPUT` is an **LLM judge** of *did it do the task* — `status==success` **and** result plausible **and** correct keys/types/shapes **and** no error signals in stdout/stderr. Held-out **unit tests** (124, on different inputs) define benchmark correctness. → vs alembic: alembic invokes each declared sample and checks a deterministic `{ok}` flag (no semantic output judging, no held-out inputs); ToolMaker adds an LLM correctness judgment but only on **one example** in-loop.

**e) Controlling how many tools per artifact** — **N/A by design: exactly one function per task.** Name, signature, and argument list are fixed by `task.yaml` up front. ToolMaker never discovers or caps tools. (Contrast: ToolRosella auto-caps at 12 with confidence tiers; alembic's explorer proposes 1–5.)

**f) Containerized usage** — **Deepest of the three.** Everything is containerized end-to-end: base CPU/CUDA images, an in-container FastAPI runtime, a **reproducible installed image** per tool, GPU via the CUDA image + `gpus`, input/output via Docker volumes, sandboxed from host, runnable on any machine. **But the deliverable is a Python function executed in-container, not an MCP/HTTP tool endpoint** — there is no served standardized interface (alembic's `docker commit` → MCP-over-HTTP is the differentiator here).

---

## 3. What alembic should borrow (stability/robustness)

Ordered by expected payoff.

1. **Build the installed image from a recorded `install.sh`, not `docker commit` of the mutated build container.** ToolMaker concatenates the env agent's successful write/bash actions into a script and does a clean `docker build`. This is more **reproducible, auditable, smaller, and reviewable** — a direct win for paper reproducibility and for trusting the artifact. Alembic already logs agent commands; emit them as a Dockerfile/`install.sh` and rebuild clean.

2. **Reset to a frozen install checkpoint before every validate/debug attempt.** ToolMaker reloads the installed image each iteration so debugging side-effects never accumulate or mask the real bug. Alembic's validator/debugger mutate one shared workdir — stale state can hide or fabricate failures. Adopt "fresh runtime per attempt from the frozen install image."

3. **Add an LLM output-correctness gate, not just `{ok}`/no-exception.** `ASSESS_TOOL_OUTPUT` asks whether the *returned value* is plausible and has the right keys/types/shapes. Scientific tools routinely run cleanly and return garbage; alembic's `invoke_mcp_tool` would pass them. A semantic "did it do the task" check closes a real robustness hole.

4. **Explicit resource policy: download model weights, mount datasets.** This is exactly the gap flagged in the ToolRosella comparison. Add a step that (a) lets the env/build agent fetch pretrained weights (README-driven, `huggingface-cli`/`wget`, with an `HF_TOKEN` channel) and (b) declares datasets/large inputs as **mounts / sample paths** rather than trying to bake them. Directly lifts success on heavy ML repos.

5. **Declared, allowlisted repo-secret injection (`repo.env` + `${env:VAR}`).** Alembic injects only its own LLM keys and scrubs them; it has no channel for a *repo-required* runtime secret (HF token, inference API key). Add a task/config-level `env:` declaration with allowlisted substitution so token-gated tools can be built and served, while keeping secrets out of the committed image (combine with alembic's existing scrub).

6. **Held-out validation invocation (different args than the generation sample).** ToolMaker's benchmark validates on inputs *different* from the creation example to catch hard-coded demo paths. Alembic validates with the same coder-declared samples that drove generation. Add ≥1 held-out invocation per tool → catches overfitting to the demo, a strong generalization/robustness claim for a paper.

7. **Persistent problem-summary memory across debug iterations + tiered models.** ToolMaker feeds a running `problem_summaries` list into each diagnosis ("avoid repeating the same mistakes") and uses a **reasoning model for plan/first-implementation, a cheaper model for the loop.** Alembic's debugger is largely stateless per call and uses one `MODEL` everywhere. Both changes (failure memory + model routing) improve first-pass quality, reduce oscillating fixes, and give clean cost/quality ablations.

## 4. Alembic strengths to keep and emphasize (vs ToolMaker)

- **Repo-autonomous, multi-tool, MCP-standardized output.** ToolMaker needs a human-written `task.yaml` (description + exact signature + example invocation) and yields **one** function. Alembic derives the tool set, signatures, and examples from the repo and ships a **served MCP endpoint** — agent-pluggable without a wrapper. Different, harder problem.
- **Two-venv layout for old-Python repos.** ToolMaker is single global Python 3.12 and cannot host tf-1.x / old-DGL repos; alembic can.
- **No per-task human spec required.** Alembic's explorer produces the "what to wrap" that ToolMaker assumes as input.
- **Real per-sample invocation already in-loop** (keep it) — extend it with ToolMaker's #3 (semantic judge) and #6 (held-out inputs) to be strictly stronger.

---

## 5. Three-way cheat sheet (for the paper)

| Axis | **Alembic** | **ToolRosella** | **ToolMaker** |
|---|---|---|---|
| Output | Served **MCP** server (HTTP), Docker image | **MCP** plugin folder + `mcp.json` (local venv) | Standalone **Python function** + installed Docker image |
| Autonomy | Repo→many tools, autonomous | Repo→many tools, autonomous (NL query→repo search too) | Task→**one** tool, human `task.yaml` (signature + example) |
| Orchestration | 4 ADK agents + debugger sub-agent | LangGraph 8-node DAG + RRF loop | Formal state machine, agentic install + ≤30-iter closed loop |
| **a) repo env keys** | none (only own LLM keys, scrubbed) | none | **declared `repo.env` + allowlisted `${env:VAR}`** |
| **b) env setup** | uv→venv (+conda), **two-venv**, apt-get sys libs | UV→conda→venv, ≥3.10 floor, `environment.yml` | agentic install→**reproducible `install.sh` image**, global pip, single Py3.12 |
| **c) downloads** | none | none | **models yes (agent), datasets via mounts** |
| **d) e2e + success** | real `invoke_mcp_tool` per sample, `{ok}` flag | **import-only smoke** (create_app) | **real run + LLM judges output**; benchmark = held-out unit tests |
| **e) tool count** | explorer proposes 1–5 (uncapped) | **cap 12, confidence tiers, AST-verified** | **always 1** (task-fixed) |
| **f) containerized** | **commit→serve MCP/HTTP**, secret-scrub | no (local folder) | **deep: all-Docker, CPU/CUDA, GPU, mounts** (no served endpoint) |

**Net read for the alembic paper:** alembic already wins on *standardized, served, multi-tool, old-Python-capable* output and on *real in-loop invocation*. Its weakest links — reproducible env capture, fresh-checkpoint isolation, semantic output verification, external-resource acquisition, repo-secret handling, and held-out generalization testing — are exactly the things ToolMaker (build-from-script, env-reset-per-iter, LLM output judge, model download policy, `repo.env`, held-out unit tests) and ToolRosella (deterministic AST gate, bounded/verified tool selection) demonstrate. Adopting those closes alembic's robustness gaps without giving up its MCP/multi-tool advantage.
