"""OrchestratorAgent — top-level coordinator (in-process ADK mode).

For A2A mode (each sub-agent as a remote HTTP service) see:
  CoScientist/a2a/orchestrator.py
"""
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.agent_tool import AgentTool

from CoScientist.agents.common import agent_tools, make_llm
from CoScientist.agents.critic_agent import post_action_critique, pre_action_critique
from CoScientist.agents.hypotheses_agent import hypotheses_agent
from CoScientist.agents.med_callbacks import before_model_modifier as med_before_model
from CoScientist.agents.medical_agent import medical_agent
from CoScientist.agents.planner_agent import planner_agent
from CoScientist.agents.prompts import orchestrator_instruction
from CoScientist.agents.research_agent import research_agent
from CoScientist.agents.task_execution_agent import task_execution_agent

orchestrator_agent = LlmAgent(
    name="OrchestratorAgent",
    model=make_llm(),
    instruction=orchestrator_instruction,
    description="Main Orchestrator Agent",
    before_model_callback=med_before_model,
    after_model_callback=pre_action_critique,
    after_tool_callback=post_action_critique,
    tools=agent_tools(
        [
            AgentTool(agent=planner_agent),
            AgentTool(agent=hypotheses_agent),
            AgentTool(agent=research_agent),
            AgentTool(agent=task_execution_agent),
            AgentTool(agent=medical_agent),
        ],
        hitl=True,
    ),
)
