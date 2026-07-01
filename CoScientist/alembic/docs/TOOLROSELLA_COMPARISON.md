# ToolRosella vs. Alembic — repo→MCP comparison

Companion to [DESIGN.md](./DESIGN.md). Source analyzed: `ToolRosella-main/src/ToolRosella/mcp_construction` (the `code2mcp` workflow) + paper `arXiv:2603.09290v5`. Goal: understand how ToolRosella solves the same problem and what alembic should borrow to become more stable/robust for a paper.

---

## 1. ToolRosella's mcp_construction — system design

ToolRosella is a **3-agent pipeline**: *Tool-search* (find repos from an NL query via GitHub API + LLM) → **MCP-construction** (the part we care about) → *Planning* (write an `mcp.json` + task prompt, optionally drive Claude Code CLI). Only MCP-construction is the repo→MCP engine.

MCP-construction = **`code2mcp`, a LangGraph `StateGraph`** threading one dict-state through 8 nodes. Conditional edges short-circuit to `END` on `status=="failed"`; the `review` node forms a **Review-Revise-Fix (RRF)** loop back to `run`/`generate`.

```
download → analysis → env → generate → code_check → run → review → finalize
                                          ▲                    │
                                          └──── RRF loop ◄─────┘   (fix → re-run; or regenerate)
```

Output is a **local folder** `workspace/<repo>/mcp_output/`: `start_mcp.py`, `mcp_plugin/{mcp_service.py, adapter.py, main.py}`, `requirements.txt`, `README_MCP.md`, `workflow_summary.json`. No tests are emitted (generation block removed).

### Node-by-node

| Node | What it does |
|---|---|
| **download** | Full `git clone` into `source/`; makes `mcp_output/{mcp_plugin,tests_mcp,mcp_logs}`. Clone failure → *continues with empty* (non-fatal). |
| **analysis** | `gitingest` summary + **AST scan of public funcs/classes with parameter names** + package scan + entry-point scan (setup.py/pyproject `console_scripts`) + optional **DeepWiki** (via Jina `r.jina.ai`). LLM emits JSON: `core_modules`, `cli_commands`, `import_strategy ∈ {import, cli, blackbox}`, `dependencies`, `risk_assessment`. Falls back to static AST modules when no pyproject/packages. → `analysis.json` |
| **env** | Picks manager **UV → conda → venv** (first available). Infers Python from `requires-python`/`.python-version`/`runtime.txt`/README, **enforces ≥3.10**, fallback `[3.10,3.11,3.12]`. Installs base pkgs (`fastmcp,pytest,pytest-asyncio,papermill,nbclient,ipykernel,imagehash`) + repo deps by priority (`pip install -e .` → package name → `requirements.txt` → env.yml pip block). Conda path honors `environment.yml` channels + `--solver=libmamba`. Builds C/C++ (cmake/make/`build_ext`). Runs the repo's own `tests/` if present (best-effort). **If no env can be built → type `none`, pipeline continues degraded.** → `env_info.json` |
| **generate** | Prunes analysis to a **bounded tool set** (see Q-e), injects `__init__.py` along import paths, writes `start_mcp.py` (stdio/http via `MCP_TRANSPORT`), `mcp_service.py` (LLM with deterministic fallback template), `adapter.py` (import/cli/blackbox), `requirements.txt`, `README_MCP.md`. Strips ``` fences. |
| **code_check** | **Static AST gate**: every `from <mod> import <sym>` in generated code must reference a symbol actually exported by `source/`. Emits `missing_symbol`/`syntax_error`. LLM repair only if `TOOLROSELLA_CODECHECK_REPAIR=true`. Failure → routes to `review`. |
| **run** | Picks interpreter from env; ensures `fastmcp` importable. **Smoke test only**: `from mcp_service import create_app; create_app(); print('OK')`. Classifies error type. → `run_log.json`, `llm_statistics.json` |
| **review** (RRF) | LLM error analysis → `next_action`, `confidence`. `_apply_incremental_fixes`: infer offending file from traceback → ask LLM for **full-file replacement** → `ast.parse` validate → atomic write → loop back to `run`. Caps: fix-retries 5, generation-retries 5; stop if `confidence<0.3` or "environment_fix" needed. |
| **finalize** | Writes `workflow_summary.json`; `success` iff `run_result.success`. |

**Model layer:** LangChain `init_chat_model` wrapper (`utils.py`) supporting OpenAI/DeepSeek/Qwen/Claude/Bedrock/Ollama, with retry/backoff + token accounting. Repo memory: `MCPConstructionAgent` caches successful repos in `processed_repos.json` and **skips re-processing**.

### Paper numbers (for context)
122 repos / 35 subdisciplines / 6 domains → **1,580 tools**. **Success criterion: ≥3 validated tool endpoints an agent can correctly invoke with valid outputs.** Conversion: first-pass **33.6%** → **61.5%** after 3 RRF rounds (human 90.2%, LLM-only baseline 8.2%). **7.2 min/repo vs 31.6 min human (4.4× faster).** Dominant failures: **75.3% dependency/environment** (`ModuleNotFoundError` alone 30.9%), 24.7% code-gen. Downstream verified success avg **84.0%** (verifier = Claude Opus 4.6).

---

## 2. Direct answers (a–f)

**a) Missing env keys the underlying repo requires** — **Not handled at all.** ToolRosella manages only *its own* keys (LLM provider, optional `GITHUB_TOKEN`/`JINA_API_KEY`). There is no detection of, prompt for, or injection of secrets a repo needs at runtime (e.g. an inference API key, dataset token). Because `run` only imports `create_app()` and never invokes a tool, a missing repo-side key never even surfaces during construction. → **Alembic is on par (also no key handling), so this is an open problem for both — see §3.**

**b) Automatic environment setup** — **Strongest part of ToolRosella.** Manager cascade UV→conda→venv; Python-version inference from 5 sources with ≥3.10 floor; multi-source dependency install; first-class `environment.yml` (channels + libmamba); C/C++ build; degrades to a `none` env rather than aborting. **Limitation: single environment** — `fastmcp` and repo deps share one interpreter, hard-floored at Python 3.10. Old-Python scientific repos (tf-1.x, old DGL) can't be satisfied. No `apt-get`/system-library handling — which is exactly why their #1 failure class is `ModuleNotFoundError`/environment.

**c) External file downloads (HF model, weights, or README-only mention)** — **Not handled at all.** `grep` for `huggingface|hf_hub|from_pretrained|wget|gdown|download.*(model|weight|checkpoint)` over the source = **zero hits**. No README mining for "download X" steps. Whatever the repo's own `pip install`/`setup.py` pulls is all you get; anything gated behind a manual model/dataset fetch is unsupported.

**d) End-to-end tests with real tool calls + success criterion** — **Weak in the released loop.** Construction validation = (1) static import-symbol check (`code_check`) + (2) **import-only smoke test** (`create_app()` returns). **No `@mcp.tool` is ever invoked with arguments during construction.** The paper's stricter "≥3 endpoints correctly invocable with valid outputs" is a *benchmark-level* definition; its enforcement is not visible in the construction code (it appears to be measured downstream via the Planning Agent + LLM verifier). → **Alembic is materially stronger here:** its validator runs `invoke_mcp_tool` per declared sample and calls the debugger on real runtime failures.

**e) Controlling how many tools per MCP** — **Yes, explicitly.** `_prune_analysis_for_generation(max_total=12)` caps total tools at 12, with **per-module caps by import-confidence** (high→5, medium→3, low→1), and keeps only names that **intersect the real AST symbols** of the target file (excluding `test`/`example`/private). So the tool set is bounded *and* validated against actual code. → **Alembic only suggests "1–5 scenarios" in a prompt with no deterministic cap or AST check.**

**f) Containerized usage** — **No.** No Dockerfile, no image build, no `docker commit`. Deliverable is a local folder + a Planning-Agent `mcp.json` whose server `command` is the **local venv Python** running `start_mcp.py` over stdio (HTTP is possible via `MCP_TRANSPORT=http`, but unpackaged). → **Alembic is stronger:** it `docker commit`s a portable `alembic-tool:<repo>` image, serves over streamable-HTTP, and scrubs secrets from the committed image.

---

## 3. What alembic should borrow (to be more stable/robust)

Ordered by expected robustness payoff for the paper.

1. **Deterministic static import/symbol gate (from `code_check_node`).** Before any run, AST-verify every generated tool import resolves to a real exported symbol in the repo. Alembic's coder is *told* to "grep signatures" but nothing enforces it; a cheap deterministic gate kills hallucinated functions before the expensive validator/debugger loop. **Add it as a tool + a stage between coder and validator.**

2. **Bounded, confidence-ranked, AST-verified tool selection (from `_prune_analysis_for_generation`).** Replace the soft "1–5 scenarios" prompt with a hard cap + per-confidence caps + intersection against real public symbols (with their parameter names). This bounds cost, prevents over-generation, and — feeding **real signatures** into wrappers — pre-empts the wrong-kwarg `TypeError`s the alembic debugger currently fixes reactively.

3. **First-class conda / `environment.yml` path.** Today alembic reaches conda only as "Attempt 3". For scientific repos (rdkit, openbabel, pinned CUDA), promote conda + `environment.yml` (channels + `--solver=libmamba`) to a primary strategy alongside uv.

4. **Structured run metrics + error taxonomy.** Emit per-run `llm_statistics.json` (calls, tokens, retries) and classify every failure into a taxonomy (`ModuleNotFound`/`Import`/`Environment`/`Build`/`Syntax`/`Runtime`/…). This is the data the ToolRosella paper reports (Fig 3e) and what a paper on alembic will need for ablations and "stability" claims.

5. **Success-memoized, resumable benchmark runner (from `MCPConstructionAgent` + `processed_repos.json`).** Cache successful repos and skip them; alembic currently wipes the workdir each run. Combine with the existing `--resume <stage>` for cheaper re-benchmarking.

6. **(Consider, low priority) gitingest/DeepWiki enrichment for analysis** — but note its fragility (Jina rate limits, "Loading..." states, `verify=False`). Alembic's direct file-reading explorer is arguably *more* robust; treat external knowledge as optional augmentation, not a dependency.

## 4. Alembic strengths to keep and emphasize (they fix ToolRosella's documented failures)

- **Real in-loop tool invocation** (`invoke_mcp_tool` per sample) → enforce the paper's "≥3 invocable endpoints" criterion *at construction time*, not just downstream. This is alembic's headline differentiator vs. ToolRosella's import-only gate.
- **Two-venv layout** (server venv ≥3.10 for fastmcp + repo venv at the repo's own Python) → handles old-Python repos that ToolRosella's hard 3.10 floor cannot. Direct robustness edge on exactly the domains where ToolRosella scores lowest.
- **`apt-get` system-dependency handling** (env agent + debugger binary→package table) → directly attacks ToolRosella's #1 failure class (environment / `ModuleNotFound`), which they leave unaddressed.
- **Containerized, secret-scrubbed, HTTP-served artefact** → a deployable image vs. a local folder; matches the "operationalize / reproducible" framing better.
- **Runtime guardrails** (per-stage wall-clock timeouts, repeated-call/step loop-breakers, unknown-tool stub) → bounds cost on pathological repos.

### One-line takeaways still open for *both* systems (good "future work" for the paper)
- **Repo-side secret/env-key provisioning** (Q-a) — neither detects nor supplies keys a tool needs at runtime.
- **External artefact acquisition** (Q-c) — neither fetches HF models / weights / datasets, including README-only instructions. A "resource-acquisition" stage would lift success on heavy ML repos for both.
