"""PlannerAgent — generates a research roadmap with optional HITL review."""
from google.adk.planners import PlanReActPlanner

from CoScientist.agents.common import make_llm, hitl_enabled
from CoScientist.agents.prompts import planner_instruction
from CoScientist.hitl.handler import ConsoleHITLHandler
from CoScientist.hitl.session_agent import SessionAgent

_hitl_handler = ConsoleHITLHandler() if hitl_enabled else None

planner_agent = SessionAgent(
    name="PlannerAgent",
    model=make_llm(),
    instruction=planner_instruction,
    description="Generates a roadmap for solving the task",
    output_key="planner_roadmap",
    plan_file_path="roadmap.txt",
    planner=PlanReActPlanner(),
    hitl_handler=_hitl_handler,
)
