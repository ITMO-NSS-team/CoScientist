"""Agent definitions.

Four LLM agents — explorer, environment, coder, debugger — all on one MODEL.
The validator is NOT an agent: validation is a deterministic code loop in
pipeline.py that calls the debugger as a subroutine (so its steps log normally
and it can never end without a rendered report). The old AgentTool nesting,
its callbacks, and the fallback reporter are gone with that change.
"""
import litellm

litellm.suppress_debug_info = True

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

from alembic import config
from alembic.tools import (
    clone_repo, read_file, bash, bash_env, search,
    read_report, write_report, write_file, read_output_file, update_file,
    validate_syntax, run_tests, setup_venv, check_venv_compat, invoke_mcp_tool,
)
from alembic.instructions import (
    explorer_instruction, environment_instruction,
    coder_instruction, debugger_instruction,
)


def _model() -> LiteLlm:
    """One model for every agent. Sampling params only when explicitly set via
    env (leaving them unset avoids the temperature-0 loops seen on qwen; see
    config.MODEL_TEMPERATURE)."""
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
    description="Clones a scientific GitHub repo and reports its functionality, environment needs, and proposed MCP tools.",
    instruction=_const(explorer_instruction),
    tools=[clone_repo, read_file, bash, search, write_report],
)

environment_agent = Agent(
    name="environment",
    model=_model(),
    description="Builds the Python virtual environment(s) for the repository from a computed layout.",
    instruction=_const(environment_instruction),
    tools=[read_report, setup_venv, bash_env, check_venv_compat, write_report],
)

coder_agent = Agent(
    name="coder",
    model=_model(),
    description="Implements a FastMCP server with helper scripts and pytest tests from the verified plan.",
    instruction=_const(coder_instruction),
    tools=[read_report, bash, read_file, write_file, write_report],
)

debugger_agent = Agent(
    name="debugger",
    model=_model(),
    description="Fixes one reported failure — installs a missing dep or edits server.py/helpers — and re-runs the tool to confirm.",
    instruction=_const(debugger_instruction),
    tools=[read_output_file, update_file, bash, bash_env, invoke_mcp_tool],
)
