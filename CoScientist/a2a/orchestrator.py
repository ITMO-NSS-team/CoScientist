"""Orchestrator agent that calls sub-agents via the A2A protocol.

Mirrors the in-process orchestrator in ``CoScientist.agents.agents`` — same
catalog-driven roster, prompt, and critic wiring — but each sub-agent is a
remote A2A service (``RemoteA2aAgent``) instead of an in-process ``AgentTool``.
"""
import litellm
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.agent_tool import AgentTool

from CoScientist.a2a.config import AGENT_CARD_URLS
from CoScientist.agents import catalog
from CoScientist.agents.callbacks import pre_action_critique
from CoScientist.agents.callbacks import before_model_modifier as med_before_model
from CoScientist.agents.prompts import build_orchestrator_instruction
from CoScientist.config import get_settings

settings = get_settings()
MODEL = settings.llm.main_model
litellm.api_key = settings.llm.openai_api_key

# Catalog agent name (ADK name) -> A2A config key (see a2a/config.py).
_NAME_TO_KEY = {
    "PlannerAgent": "planner",
    "HypothesesAgent": "hypotheses",
    "ResearchAgent": "research",
    "TaskExecutorAgent": "task_execution",
    "CoderAgent": "coder",
    "MedicalAgent": "medical",
}


def _build_tools() -> list:
    """One RemoteA2aAgent per enabled catalog agent, in catalog order."""
    tools = []
    for spec in catalog.enabled_agents():
        key = _NAME_TO_KEY.get(spec.name)
        if key is None:
            raise ValueError(
                f"Catalog agent {spec.name!r} has no A2A port mapping in "
                "a2a/orchestrator.py (_NAME_TO_KEY). Add it there or to a2a/config.py."
            )
        remote = RemoteA2aAgent(
            name=spec.name,
            agent_card=AGENT_CARD_URLS[key],
            description=spec.description,
        )
        tools.append(AgentTool(agent=remote))
    return tools


orchestrator_a2a_agent = LlmAgent(
    name="OrchestratorAgent",
    model=LiteLlm(model=MODEL),
    instruction=build_orchestrator_instruction(),
    description="Main CoScientist Orchestrator (A2A mode)",
    before_model_callback=med_before_model,
    after_model_callback=pre_action_critique,
    tools=_build_tools(),
)
