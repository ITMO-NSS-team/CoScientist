"""ResearchAgent — literature and web search for scientific questions."""
from google.adk.agents.llm_agent import LlmAgent

from CoScientist.agents.callbacks import print_research_agent_tool_call
from CoScientist.agents.common import agent_tools, make_llm
from CoScientist.agents.prompts import research_instruction
from CoScientist.agents.research_callbacks import papers_agent_before_model
from CoScientist.tools import (
    paper_analysis_toolset_instance,
    papers_search_toolset_instance,
    websearch_toolset_instance,
)

research_agent = LlmAgent(
    name="ResearchAgent",
    model=make_llm(),
    instruction=research_instruction,
    description="Agent to answer questions and knowledge mining using Literature and Web Search.",
    output_key="search_results",
    # Drop any optional MCP toolsets that aren't configured (None) so the agent
    # still builds when their URLs are unset.
    tools=agent_tools(
        [
            t
            for t in [
                websearch_toolset_instance,
                paper_analysis_toolset_instance,
                papers_search_toolset_instance,
            ]
            if t is not None
        ],
        hitl=True,
    ),
    before_model_callback=papers_agent_before_model,
    after_tool_callback=print_research_agent_tool_call,
)

__all__ = ["research_agent"]
