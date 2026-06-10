"""PlannerAgent — generates a research roadmap with optional HITL review."""
from google.adk.planners import PlanReActPlanner

from CoScientist.agents.common import hitl_handler, make_llm
from CoScientist.agents.prompts import planner_instruction
from CoScientist.hitl.session_agent import SessionAgent

planner_agent = SessionAgent(
    name="PlannerAgent",
    model=make_llm(),
    instruction=planner_instruction,
    description="Generates a roadmap for solving the task",
    output_key="planner_roadmap",
    plan_file_path="roadmap.txt",
    planner=PlanReActPlanner(),
    hitl_handler=hitl_handler,
)

__all__ = ["planner_agent"]
