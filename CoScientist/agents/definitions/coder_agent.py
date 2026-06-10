"""CoderAgent — general-purpose coder / sandbox engineer (dedicated model)."""
from google.adk.agents.llm_agent import LlmAgent

from CoScientist.agents.common import agent_tools, make_coder_llm
from CoScientist.agents.prompts import coder_instruction
from CoScientist.tools import coder_toolset_instance

coder_agent = LlmAgent(
    name="CoderAgent",
    model=make_coder_llm(),
    instruction=coder_instruction,
    description=(
        "General-purpose coder / sandbox agent. Writes and runs code, executes "
        "shell and git commands (clone/commit/push), manages files, installs "
        "dependencies, collects and processes data, and runs long jobs in an "
        "isolated workspace. Use it whenever a task requires doing software/data "
        "engineering rather than calling a ready-made service."
    ),
    output_key="coder_results",
    tools=agent_tools(coder_toolset_instance, hitl=False),
)

__all__ = ["coder_agent"]
