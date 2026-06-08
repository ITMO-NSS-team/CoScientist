"""A2A server for MedicalAgent."""
import uvicorn
from a2a.types import AgentCard, AgentCapabilities, AgentSkill

from CoScientist.a2a.config import AGENT_PORTS, AGENT_URLS
from CoScientist.a2a.server import make_a2a_app
from CoScientist.agents.medical_agent import medical_agent

PORT = AGENT_PORTS["medical"]

_card = AgentCard(
    name="MedicalAgent",
    description=(
        "Handles medical and clinical questions: "
        "PubMed literature search, PICO extraction, study taxonomy, DICOM image analysis."
    ),
    url=AGENT_URLS["medical"],
    version="1.0.0",
    capabilities=AgentCapabilities(streaming=True),
    defaultInputModes=["text/plain"],
    defaultOutputModes=["text/plain"],
    skills=[
        AgentSkill(
            id="medical_research",
            name="Medical Research",
            description="PubMed search, PICO extraction, study taxonomy, and DICOM analysis",
            tags=["medical", "pubmed", "dicom", "clinical"],
        )
    ],
)

app = make_a2a_app(medical_agent, _card, "medical")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
