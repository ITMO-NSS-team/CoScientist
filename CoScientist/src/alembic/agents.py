import os
import litellm

litellm.suppress_debug_info = True

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.agent_tool import AgentTool
from alembic.tools import (
    clone_repo, read_file, bash, bash_env, search,
    read_report, write_report,
    write_file, read_output_file, update_file,
    validate_syntax, run_tests, setup_venv, check_venv_compat,
    invoke_mcp_tool,
) 
from alembic.instructions import (
    explorer_instruction, coder_instruction,
    debugger_instruction, validator_instruction,
    environment_instruction,
)
MODEL = os.environ.get("MODEL", "openrouter/qwen/qwen3-235b-a22b-2507")

explorer_agent = Agent(
    name="explorer",
    model=LiteLlm(model=MODEL),
    description="Clones a scientific GitHub repo and writes a Markdown report of its functionality and MCP usage scenarios.",
    instruction=explorer_instruction,
    tools=[clone_repo, read_file, bash, search, write_report],
)

environment_agent = Agent(
    name="environment",
    model=LiteLlm(model=MODEL),
    description="Reads the explorer report and sets up the Python virtual environment for the repository, retrying until successful.",
    instruction=environment_instruction,
    tools=[read_report, setup_venv, bash_env, check_venv_compat, write_report],
)

coder_agent = Agent(
    name="coder",
    model=LiteLlm(model=MODEL),
    description="Reads an explorer report and implements a FastMCP server with pytest tests for the repository.",
    instruction=coder_instruction,
    tools=[read_report, bash, read_file, write_file, write_report],
)

debugger_agent = Agent(
    name="debugger",
    model=LiteLlm(model=MODEL),
    description="Receives a repo URL and an error message, fixes the bug — either by installing a missing system/pip dep or by editing server.py/helpers — and re-runs the failing tool to confirm.",
    instruction=debugger_instruction,
    tools=[read_output_file, update_file, bash, bash_env, invoke_mcp_tool],
)

validator_agent = Agent(
    name="validator",
    model=LiteLlm(model=MODEL),
    description="Validates the generated MCP server via syntax checks, pytest, and real tool invocations, calling the debugger agent on failures, then writes a validation report.",
    instruction=validator_instruction,
    tools=[read_report, validate_syntax, run_tests, invoke_mcp_tool, write_report, AgentTool(agent=debugger_agent)],
)
