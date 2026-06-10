"""A2A server for TaskExecutionAgent (tool discovery + experiment running)."""
import uvicorn
from a2a.types import AgentCard, AgentCapabilities, AgentSkill

from CoScientist.a2a.config import AGENT_PORTS, AGENT_URLS
from CoScientist.a2a.server import make_a2a_app
from CoScientist.agents import task_execution_agent

PORT = AGENT_PORTS["task_execution"]

_card = AgentCard(
    name="TaskExecutorAgent",
    description=(
        "Completes experiments and computational tasks. "
        "Discovers and deploys MCP tools, then runs the experiment pipeline."
    ),
    url=AGENT_URLS["task_execution"],
    version="1.0.0",
    capabilities=AgentCapabilities(streaming=True),
    defaultInputModes=["text/plain"],
    defaultOutputModes=["text/plain"],
    skills=[
        AgentSkill(
            id="execute_task",
            name="Execute Task",
            description="Discover MCP tools and run computational experiments to solve a task",
            tags=["experiment", "mcp", "computation", "ml"],
        )
    ],
)

app = make_a2a_app(task_execution_agent, _card, "task_execution")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
