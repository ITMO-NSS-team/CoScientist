"""A2A server for HypothesesAgent."""
import uvicorn
from a2a.types import AgentCard, AgentCapabilities, AgentSkill

from CoScientist.a2a.config import AGENT_PORTS, AGENT_URLS
from CoScientist.a2a.server import make_a2a_app
from CoScientist.agents import hypotheses_agent

PORT = AGENT_PORTS["hypotheses"]

_card = AgentCard(
    name="HypothesesAgent",
    description="Generates scientific hypotheses and research ideas for a given task",
    url=AGENT_URLS["hypotheses"],
    version="1.0.0",
    capabilities=AgentCapabilities(streaming=True),
    defaultInputModes=["text/plain"],
    defaultOutputModes=["text/plain"],
    skills=[
        AgentSkill(
            id="hypothesize",
            name="Hypothesize",
            description="Generate novel scientific hypotheses for a research question",
            tags=["hypothesis", "science", "ideation"],
        )
    ],
)

app = make_a2a_app(hypotheses_agent, _card, "hypotheses")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
