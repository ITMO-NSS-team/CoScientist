import os

import litellm
litellm.suppress_debug_info = True

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.agent_tool import AgentTool
from alembic.tools import (
    clone_repo, read_file, bash, search,
    read_report, write_report,
    write_file, update_file,
    validate_syntax, run_tests,
    build_docker_image, test_mcp_launch,
)
from alembic.instructions import (
    explorer_instruction, coder_instruction,
    debugger_instruction, validator_instruction,
    docker_instruction,
)

MODEL = os.environ.get("MODEL", "openrouter/qwen/qwen3-235b-a22b-2507")

explorer_agent = Agent(
    name="explorer",
    model=LiteLlm(model=MODEL),
    description="Clones a scientific GitHub repo and writes a Markdown report of its functionality and MCP usage scenarios.",
    instruction=explorer_instruction,
    tools=[clone_repo, read_file, bash, search, write_report],
)

coder_agent = Agent(
    name="coder",
    model=LiteLlm(model=MODEL),
    description="Reads the explorer report and implements a FastMCP server with pytest tests for the repository.",
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
        "Reads the explorer and coder reports, analyses repository dependencies, fixes "
        "requirements files if necessary, writes a Dockerfile, builds the Docker image, "
        "and verifies the MCP server launches successfully inside the container."
    ),
    instruction=docker_instruction,
    tools=[
        read_report, read_file, write_file, update_file,
        bash, search,
        build_docker_image, test_mcp_launch,
        write_report,
    ],
)

validator_agent = Agent(
    name="validator",
    model=LiteLlm(model=MODEL),
    description=(
        "Validates the generated MCP server via syntax checks and pytest inside Docker. "
        "Calls the debugger agent for Python code bugs and the docker agent for "
        "environment/dependency issues. Task is complete only when both MCP launches "
        "and all tests pass."
    ),
    instruction=validator_instruction,
    tools=[
        read_report, read_file,
        validate_syntax, run_tests,
        write_report,
        AgentTool(agent=debugger_agent),
        AgentTool(agent=docker_agent),
    ],
)
