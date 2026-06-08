import os

A2A_HOST = os.getenv("A2A_HOST", "localhost")

AGENT_PORTS: dict[str, int] = {
    "orchestrator":   int(os.getenv("ORCHESTRATOR_PORT",   "8000")),
    "planner":        int(os.getenv("PLANNER_PORT",        "8001")),
    "hypotheses":     int(os.getenv("HYPOTHESES_PORT",     "8002")),
    "research":       int(os.getenv("RESEARCH_PORT",       "8003")),
    "task_execution": int(os.getenv("TASK_EXECUTION_PORT", "8004")),
    "medical":        int(os.getenv("MEDICAL_PORT",        "8005")),
}

AGENT_URLS: dict[str, str] = {
    name: f"http://{A2A_HOST}:{port}/"
    for name, port in AGENT_PORTS.items()
}

# A2A well-known agent card URLs used by RemoteA2aAgent
AGENT_CARD_URLS: dict[str, str] = {
    name: f"http://{A2A_HOST}:{port}/.well-known/agent.json"
    for name, port in AGENT_PORTS.items()
}
