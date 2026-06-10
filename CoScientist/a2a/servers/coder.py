"""A2A server for CoderAgent."""
import os

# Over A2A each orchestrator delegation is a separate session; pin one shared
# sandbox workspace so multi-step work (clone, then build on it) persists across
# calls. Override CODER_WORKSPACE_ID to isolate concurrent runs.
os.environ.setdefault("CODER_WORKSPACE_ID", "a2a_shared")

import uvicorn
from a2a.types import AgentCard, AgentCapabilities, AgentSkill

from CoScientist.a2a.config import AGENT_PORTS, AGENT_URLS
from CoScientist.a2a.server import make_a2a_app
from CoScientist.agents import coder_agent

PORT = AGENT_PORTS["coder"]

_card = AgentCard(
    name="CoderAgent",
    description=(
        "General-purpose coder / sandbox engineer: writes and runs code, "
        "executes shell and git commands, manages files, installs dependencies, "
        "processes data, and runs long jobs in an isolated workspace."
    ),
    url=AGENT_URLS["coder"],
    version="1.0.0",
    capabilities=AgentCapabilities(streaming=True),
    defaultInputModes=["text/plain"],
    defaultOutputModes=["text/plain"],
    skills=[
        AgentSkill(
            id="code",
            name="Code & Run",
            description="Write/run code, shell & git operations, build and process data in a sandbox",
            tags=["coding", "sandbox", "shell", "git", "data"],
        )
    ],
)

app = make_a2a_app(coder_agent, _card, "coder")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
