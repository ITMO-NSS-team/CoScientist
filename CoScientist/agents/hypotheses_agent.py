"""HypothesesAgent — generates scientific hypotheses for a given task."""
from google.adk.agents.llm_agent import LlmAgent

from CoScientist.agents.common import agent_tools, make_llm
from CoScientist.agents.prompts import hypotheses_instruction

hypotheses_agent = LlmAgent(
    name="HypothesesAgent",
    model=make_llm(),
    instruction=hypotheses_instruction,
    description="Agent to generate scientific hypotheses and ideas for given task",
    output_key="hypotheses",
    tools=agent_tools([], hitl=False),
)

__all__ = ["hypotheses_agent"]
