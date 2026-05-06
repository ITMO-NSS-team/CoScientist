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
    validate_syntax, run_tests,
    setup_venv, check_venv_compat,
    build_docker_image,
)
from alembic.instructions import (
    explorer_instruction, coder_instruction,
    debugger_instruction, validator_instruction,
    environment_instruction, docker_instruction,
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
    description=(
        "Reads the explorer report, creates a local virtual environment with all repo "
        "dependencies installed, and writes an environment report documenting the result."
    ),
    instruction=environment_instruction,
    tools=[read_report, setup_venv, bash_env, check_venv_compat, write_report],
)

coder_agent = Agent(
    name="coder",
    model=LiteLlm(model=MODEL),
    description="Reads explorer and environment reports and implements a FastMCP server with pytest tests for the repository.",
    instruction=coder_instruction,
    tools=[
        read_report,
        validate_syntax,
        bash,
        read_file,
        write_file,
        write_report,
    ],
)

debugger_agent = Agent(
    name="debugger",
    model=LiteLlm(model=MODEL),
    description="Receives a repo URL and an error message, reads the offending file, fixes the bug, and returns a summary of what was changed.",
    instruction=debugger_instruction,
    tools=[read_file, update_file, bash],
)

docker_agent = Agent(
    name="docker",
    model=LiteLlm(model=MODEL),
    description=(
        "Reads the environment and server reports, writes a Dockerfile at the clone root, "
        "builds the Docker image, and records the image tag so the validator can run pytest "
        "inside the container."
    ),
    instruction=docker_instruction,
    tools=[read_report, read_file, write_file, build_docker_image, write_report],
)

validator_agent = Agent(
    name="validator",
    model=LiteLlm(model=MODEL),
    description="Validates the generated MCP server via syntax checks and pytest inside Docker, calling the debugger agent on failures, then writes a validation report.",
    instruction=validator_instruction,
    tools=[read_report, validate_syntax, run_tests, write_report, AgentTool(agent=debugger_agent)],
)
