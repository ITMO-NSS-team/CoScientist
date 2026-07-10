"""Agent definitions.

Five LLM agents — explorer, environment, coder, debugger, wrapper — all on one
MODEL. The validator is NOT an agent: validation is a deterministic code loop
in main.py that calls the debugger as a subroutine. The wrapper agent is a
fallback only: server.py is rendered deterministically (tools/codegen.py) and
the agent runs solely when the compile/import gate fails.
"""
import litellm

litellm.suppress_debug_info = True

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

from alembic import config
from alembic.tools import (
    bash, bash_env, check_venv_compat, clone_repo, invoke_tool_function,
    read_file, read_output_file, run_tool_tests, search, setup_venv,
    update_file, write_file, write_report,
)
from alembic.instructions import (
    coder_instruction, debugger_instruction, environment_instruction,
    explorer_instruction, wrapper_instruction,
)


def _model() -> LiteLlm:
    """One model for every agent. Sampling params only when explicitly set via
    env (leaving them unset avoids the temperature-0 loops seen on qwen)."""
    sampling = {}
    if config.MODEL_TEMPERATURE is not None:
        sampling["temperature"] = float(config.MODEL_TEMPERATURE)
    if config.MODEL_TOP_P is not None:
        sampling["top_p"] = float(config.MODEL_TOP_P)
    return LiteLlm(model=config.MODEL, **sampling)


def _const(text: str):
    """Wrap a static instruction as an InstructionProvider. ADK runs `{var}`
    session-state templating on *string* instructions and raises KeyError on
    any literal brace-identifier — our prompts are full of `{}` code examples,
    so pass a callable, which ADK does NOT template (bypass_state_injection)."""
    return lambda _ctx: text


explorer_agent = Agent(
    name="explorer",
    model=_model(),
    description="Clones a scientific GitHub repo and reports its functionality, environment needs, and proposed tools.",
    instruction=_const(explorer_instruction),
    tools=[clone_repo, read_file, bash, search, write_report],
)

environment_agent = Agent(
    name="environment",
    model=_model(),
    description="Builds the Python virtual environment(s) for the repository from a computed layout.",
    instruction=_const(environment_instruction),
    tools=[setup_venv, bash_env, bash, check_venv_compat],
)

coder_agent = Agent(
    name="coder",
    model=_model(),
    description="Implements each verified tool as a plain Python function with smoke and invocation tests.",
    instruction=_const(coder_instruction),
    tools=[bash, read_file, write_file, read_output_file, update_file],
)

debugger_agent = Agent(
    name="debugger",
    model=_model(),
    description="Fixes a batch of reported failures — installs missing deps or edits tool/test files — and re-runs them to confirm.",
    instruction=_const(debugger_instruction),
    tools=[read_output_file, update_file, bash, bash_env,
           invoke_tool_function, run_tool_tests],
)

wrapper_agent = Agent(
    name="wrapper",
    model=_model(),
    description="Fallback fixer for the generated FastMCP server when its compile/import gate fails.",
    instruction=_const(wrapper_instruction),
    tools=[read_output_file, update_file, bash],
)
