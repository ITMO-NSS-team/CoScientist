Here are updates to current architecture that I want implemented.

R1. The target is the focus, not the budget

I want the tasks to be completed, not to get a graceful cheap fail whenever LLM felt like it. The current gates are good idea, 
but in benchmarks/alembic/runs/2026-07-09_remaster-qwen/logs/esm.log explorer failed to produce a report - this is unacceptable.
If the agent fails defined n times at the gate, we just roll back to the checkpoint start of the stage and try again with a short info about what failed previously - stochastics happens. Of course STAGE_RESET variable - by default 2 loops.

Stemming from the same idea - let's make timeouts for agent stages optional - and by default turned off.

R2. More reliable benchmark metrics

Right now benchmark metrics still rely on final report, and in benchmarks/alembic/runs/2026-07-09_remaster-qwen/summary.md 
there were STILL two fails because final report md was not accessible. - this is unacceptable. Partially fixed by R2
I want each stage's readiness, syntax / smoke / tool invocation (name, status, reason as currently) tests statistics 
to be excracted directly from run data, not from a report in the end. Therefore, i want all the data to be guaranteed extracted from the run data itself, the final report is not really needed.

R3. Structure update

I want to make it simpler to test and debug tools - by using ToolMaker's strategy of writing tool as a simple python function,
without the fastmcp server logic overhead. Split-venvs for pre-python3.10 scientific repos require that the main server script
is written in fresh python version, while the repo itself may require older version - we are good at setting these environments
in both 1 and 2-venv configs, however writing main script and helpers with argparse is a terrible overhead and breaks testing easily.

Also, the post-env gate is missing right now - Run check_venv_compat deterministically on .venv (and .venv-repo in two-venv mode) — no LLM discretion. Add a repo-import smoke test (import the top-level package(s) named in plan.tools). On failure → route to the debugger once, before the Coder (bounded), then re-check; if still broken, restart the stage checkpoint telling what happened before.

Coder should write simple .py files with just import, function with thorough docstring on its purpose, argument descriptions and with 
usage examples as now, and then write smoke / invocation tests for them (more on that in R6) (we should split them somehow, passing own-written smoke-tests is just a quick sanity check, while evidence-based invocation tests are a solid sign of performance). The gate is as now - AST, import, symbol, undefined-name checks, and correctness of smoke / invocation tests. 
the tools scripts should be imported stript.tool_name in smoke.invoc tests - make sure we don't get any errors there, tests are run within actual container checkpoint.

Writing reports is now needed only for explorer, all else is replaced by gates
Environment and coder read explorer report at the beginning (and not by calling a tool but just appended to prompt, this is explorer's report)

Debugger has an optional limit of DEBUGGING-ROUNDS=10 so that we are not stuck in an infinite loop, turned on by default.

After that, we give all that to mcp-wrapper agent that would implement the two-layer logic on the functions (this is purely artefacts now, we presume that tests for individual tools pass -> we assume that mcp server as valid) with 

    1.EXPLORE (with clone) 
    2. GATE: plan
    3. ENVIRONMENT 
    4. [GATE: static] 
    5. CODER 
    6. [GATE: scripts + tests available]
    7. VALIDATION (static) - run all tests, call debugger to fix all at once if encountered real errors - parallelism required to propagate errors within neighboring tools 
    + TMBench tests integration if we run with it (R4) - this only gets called on tests that fail, if a invocation test got a metric than we don't mark that as a failed test.
    8. MCP WRAPPER
    9. [Gate for server and helper scripts present and compile]

so still no validation agent, but a new wrapper agent

the final artefacts that we clone into [run's folder]/output/<reponame> are the server, helper scripts and setup.sh
all else tracked from benchmark is the same + all the new points of validation.


R4. TMBench compatability

In practice I want our agent to be fully compatible with the TMBench - and that is partially implemented right now. 
I want to be sure that in this case the explorer gets the context of the tools he is required to discover, and writing that in plan
with the files that are available being copied to container

Environment embeds TMBench task description into report if we're running with it, stating that this is the

    A "required tool" is described as a task definition YAML (benchmark/tasks/*.yaml), validated against the
    ToolDefinition pydantic model (toolmaker/definition.py:58-70). Here's the actual file for conch_extract_features:

    name: conch_extract_features
    repo:
        name: CONCH
        url: "https://github.com/mahmoodlab/CONCH"
        env:
        HF_TOKEN: "${env:HF_TOKEN}"          # required for downloading models
    papers: [lu2024conch]
    category: pathology
    description: Perform feature extraction on an input image using CONCH.
    arguments:
        input_image:
        description: Path to the input image
        type: str
    returns:
        features:
        description: The feature vector extracted from the input image, as a list of floats
        type: list
    example:
        arguments:
        input_image: /mount/input/TUM/TUM-TCGA-ACRLPPQE.tif
        mount:
        kather100k/CRC-VAL-HE-7K/TUM/TUM-TCGA-ACRLPPQE.tif: TUM/TUM-TCGA-ACRLPPQE.tif
    test_cases:
        kather100k_muc:
        arguments: {...}
        mount: {...}
        tcga_brca_patch_png: {...}
        tcga_brca_patch_jpg: {...}

One more thing - there are two task in TMBench that require the same stamp_train_classification_model and stamp_extract_features - let's add a option to run these two simultaneously, meaning explorer has to find all that is needed for both tools and coder has to also code both.

I understand that this might be tough but it would really showcase the actual Repo-to-MCP evolution of our project. If it fails, we'll just run them separately.

the actual tests: 

  For run_tool() (toolmaker/run.py:49-119) to execute a test
  invocation, it needs:

  1. A pre-built Docker checkpoint image — toolmaker-runtime:installed-<name>. Must already exist (allow_build=False —
  see last answer); nothing gets built on the fly.
  2. code.py — read as raw text (tool_folder.joinpath("code.py").read_text(), toolmaker/run.py:76). And yes, in
  practice it's exactly what you said: one standalone Python function, self-contained with its own imports inside the
  function body (that's an explicit requirement in the codegen prompt — see
  toolmaker/tasks/implement_function.py:24-35, "You are only allowed to write a single python function"). No class, no
  decorator, nothing else in the file.
  3. task_definition.yaml → parsed into a ToolDefinition, which supplies the test_cases (each a ToolInvocation =
  {arguments: {...}, mount: {...}}).
  4. The actual input data files, copied into run_folder/input per the mount mapping (Mounts.reset(),
  toolmaker/runtime/client.py:121-146), sourced from benchmark/data.

- when we run TMBench this is included in validation-debugger loop, before our other tests. If debugger is exhausted on TMBench tests, it gets the same amount of retry attempts on this.

R5. ToolMaker-inspired checkpoints

Stemming from the R1, we want to remember the current condition of our run if we execute agent reset, after each stage 
completion gate is passed.

In ToolMaker:S

    1. Checkpoint = docker commit. Once the agent finishes, runtime.save_checkpoint(tag=f"installed-{name}")
    (toolmaker/cli.py:108) calls:
    def save_checkpoint(self, tag: str) -> None:
        container = get_docker().containers.get(self.name)S
        container.commit(repository=self.repository, tag=tag)
    2. (toolmaker/runtime/client.py:290-295) — this snapshots the container's entire filesystem (all installed packages,
    cloned repo, downloaded models) into a new image toolmaker-runtime:installed-<name>. That's the checkpoint; nothing
    fancier than docker commit.
    3. Restoring a checkpoint. Whenever a tool needs to be built or invoked (create_tool, run_tool),
    DockerRuntimeClient.load_checkpoint(name, tag="installed-<name>") (toolmaker/runtime/client.py:297-321) just calls
    .create() again, pointing image:tag at the committed image — i.e. it spins up a brand-new container from the
    checkpoint image, so the expensive install step never has to be repeated. run_tool in toolmaker/run.py:89-98 does
    exactly this for every test invocation, and _get_run_dir looks up which checkpoint tag to use from install.json's
    "installed" field.
    4. Reproducible/portable variant. toolmaker import (toolmaker/cli.py:331-335) builds the checkpoint a different way
    — instead of docker commit-ing a live container, it does a real docker build from docker/installed.Dockerfile, which
    layers the recorded install.sh transcript (the exact bash commands the agent ran, saved by installed_state.bash())
    on top of toolmaker-runtime:latest. This produces the same installed-<name> tag reproducibly from source rather than
    from container state, so a checkpoint can be shared/rebuilt without shipping the actual committed image.

    So there are effectively two ways to arrive at a checkpoint image (live commit vs. Dockerfile rebuild from the saved
    install.sh), but both just produce a tagged Docker image that later runs are started from — there's no custom
    snapshot/restore format beyond the Docker image layer cache.

Basically I want that also - after checking that both the venv compat is done, the agent must also write the setup.sh
script. We return it just as an artefat of the run, not digging much into it now - will be used to run our MCP server
later.

R6. Tool invocations - aimed at result but aware of big possible time consumprtion
Stemming from TMBench compatability - we test all our tools to the possible extent - the ones with GUI or that do not produce visible 
output can only be tested for stable invocation. 

The standard two-level testing procedure is as in TMBench:

  1. Execution-level status (shallow — just "didn't crash")

  Set in toolmaker/runtime/server.py:80-91, purely from the subprocess exit code of toolmaker_function_runner.py:
  if cmd.return_code == 0:
      return FunctionCallResult(status="success", result=..., stdout=cmd.output)
  else:
      return FunctionCallResult(status="error", result=f"Process failed with return code {cmd.return_code}", ...)
  This says nothing about correctness — a function that runs to completion and returns garbage still gets status:
  "success" as long as it doesn't raise and exits 0.

For the time-constraining cases, we time limit OUR test suite for 120 secs - if the 
running process did not finish successfully or return an error - we mark that tool as runtime success

However as we generate tests, we could follow these principles from TMBench:

  2. Per-invocation correctness (pytest, hand-written per tool)

  Patterns I found across tools:
  - Exact/near-exact regression against precomputed reference outputs — np.testing.assert_allclose(features,
  np.load(expected_features_file), atol=1e-3) (conch), assert_almost_equal(result["p_value"], 0.181042) (pathfinder)
  - Structural/type checks — keys present, isinstance(...), list length (e.g. 512-dim feature vector)
  - Fixed scalar checks — result["num_params"] == 3552258 (stamp)
  - Output-artifact checks — file exists, non-zero size, correct image dimensions (medsam), trained-model directory
  non-empty (nnunet)

  So "correct" for a single invocation = passes test_status and every other test function pytest collected for that
  invocation's parametrization.

so we choose how to write invocation tests for function based on observed functionality we are SURE on, if this was
shown in readme or tests, if the files were not found or the function timed out - it still remains a runtime success
Explorer should document the basis for per-tool tests on its stage.

These two are reported separately, so for each tool there is its execution status and if there is a test for tool - there
is invocation correctness,

R7. Ability to turn off permissions for datasets downloads
For TMBench we would want to limit env agent's ability to download external data:

    Allowed: any downloads needed to set up the repo per its README — dependencies, configs, and in particular
  ▎ pretrained model weights/checkpoints needed so the tool works standalone.
  ▎
  ▎ Not allowed: datasets — the install agent is explicitly told "do NOT download any datasets," even if the README's
  ▎ setup instructions call for one.

so that would modify env agent's prompt, even if explorer (which must document all potential required data) said so.

R8. Remove the need to constanlty mention repo url in every tool call except git clone
This just wastes tokens, we don't really need multi-repo action now. Only the very first git clone it is needed.


Remember to keep all code concise, well structured, readable and not over-documented.