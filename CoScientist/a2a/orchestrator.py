"""Orchestrator agent that calls sub-agents via A2A protocol (RemoteA2aAgent)."""
import litellm
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.agent_tool import AgentTool

from CoScientist.a2a.config import AGENT_CARD_URLS
from CoScientist.agents.critic_agent import post_action_critique, pre_action_critique
from CoScientist.agents.med_callbacks import before_model_modifier as med_before_model
from CoScientist.agents.prompts import orchestrator_instruction
from CoScientist.config import get_settings
from CoScientist.hitl.tool import get_hitl_tools

settings = get_settings()
MODEL = settings.llm.main_model
litellm.api_key = settings.llm.openai_api_key
hitl_enabled = settings.hitl.enabled


def _build_tools() -> list:
    remote_agents = [
        RemoteA2aAgent(
            name="PlannerAgent",
            agent_card=AGENT_CARD_URLS["planner"],
            description="Generates a step-by-step roadmap for solving the task",
        ),
        RemoteA2aAgent(
            name="HypothesesAgent",
            agent_card=AGENT_CARD_URLS["hypotheses"],
            description="Generates scientific hypotheses and research ideas for a given task",
        ),
        RemoteA2aAgent(
            name="ResearchAgent",
            agent_card=AGENT_CARD_URLS["research"],
            description="Answers questions and mines knowledge via literature and web search",
        ),
        RemoteA2aAgent(
            name="TaskExecutorAgent",
            agent_card=AGENT_CARD_URLS["task_execution"],
            description=(
                "Completes experiments and computational tasks. "
                "Discovers MCP tools and runs the experiment pipeline."
            ),
        ),
        RemoteA2aAgent(
            name="MedicalAgent",
            agent_card=AGENT_CARD_URLS["medical"],
            description=(
                "Handles medical and clinical questions: "
                "PubMed search, PICO extraction, study taxonomy, DICOM analysis."
            ),
        ),
    ]
    tools = [AgentTool(agent=ra) for ra in remote_agents]
    if hitl_enabled:
        tools.extend(get_hitl_tools())
    return tools


orchestrator_a2a_agent = LlmAgent(
    name="OrchestratorAgent",
    model=LiteLlm(model=MODEL),
    instruction=orchestrator_instruction,
    description="Main CoScientist Orchestrator (A2A mode)",
    before_model_callback=med_before_model,
    after_model_callback=pre_action_critique,
    after_tool_callback=post_action_critique,
    tools=_build_tools(),
)
