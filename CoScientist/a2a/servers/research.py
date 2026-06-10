"""A2A server for ResearchAgent."""
import uvicorn
from a2a.types import AgentCard, AgentCapabilities, AgentSkill

from CoScientist.a2a.config import AGENT_PORTS, AGENT_URLS
from CoScientist.a2a.server import make_a2a_app
from CoScientist.agents import research_agent

PORT = AGENT_PORTS["research"]

_card = AgentCard(
    name="ResearchAgent",
    description="Answers scientific questions and mines knowledge via literature and web search",
    url=AGENT_URLS["research"],
    version="1.0.0",
    capabilities=AgentCapabilities(streaming=True),
    defaultInputModes=["text/plain"],
    defaultOutputModes=["text/plain"],
    skills=[
        AgentSkill(
            id="research",
            name="Research",
            description="Search scientific literature and the web to answer research questions",
            tags=["research", "literature", "web-search", "knowledge"],
        )
    ],
)

app = make_a2a_app(research_agent, _card, "research")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
