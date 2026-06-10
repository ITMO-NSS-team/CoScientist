"""A2A server for PlannerAgent."""
import uvicorn
from a2a.types import AgentCard, AgentCapabilities, AgentSkill

from CoScientist.a2a.config import AGENT_PORTS, AGENT_URLS
from CoScientist.a2a.server import make_a2a_app
from CoScientist.agents import planner_agent

PORT = AGENT_PORTS["planner"]

_card = AgentCard(
    name="PlannerAgent",
    description="Generates a step-by-step roadmap for solving a scientific task",
    url=AGENT_URLS["planner"],
    version="1.0.0",
    capabilities=AgentCapabilities(streaming=True),
    defaultInputModes=["text/plain"],
    defaultOutputModes=["text/plain"],
    skills=[
        AgentSkill(
            id="plan",
            name="Plan",
            description="Produce a research roadmap for a scientific task",
            tags=["planning", "roadmap", "science"],
        )
    ],
)

app = make_a2a_app(planner_agent, _card, "planner")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
