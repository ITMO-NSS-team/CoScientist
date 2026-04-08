import os

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from alembic.tools import clone_repo, read_file, bash, search, write_report, read_report, write_file
from alembic.instructions import explorer_instruction, coder_instruction

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
    description="Reads an explorer report and implements a FastMCP server with pytest tests for the repository.",
    instruction=coder_instruction,
    tools=[read_report, clone_repo, read_file, bash, search, write_file],
)
