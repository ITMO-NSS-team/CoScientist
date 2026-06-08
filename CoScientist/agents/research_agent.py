"""ResearchAgent — literature and web search for scientific questions."""
from google.adk.agents.llm_agent import LlmAgent

from CoScientist.agents.common import agent_tools, make_llm
from CoScientist.agents.prompts import research_instruction
from CoScientist.tools import websearch_toolset_instance

research_agent = LlmAgent(
    name="ResearchAgent",
    model=make_llm(),
    instruction=research_instruction,
    description="Agent to answer questions and knowledge mining using Literature and Web Search.",
    output_key="search_results",
    tools=agent_tools(websearch_toolset_instance, hitl=True),
)
