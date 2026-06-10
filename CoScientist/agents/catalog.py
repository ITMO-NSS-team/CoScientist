"""Single source of truth for the orchestrator's delegatable agents.

To add/remove an agent the orchestrator can call, edit ONE place — this list.
Each entry carries the agent's ADK name (exactly what the model calls and what
the critic sees), a description, optional routing guidance, and an `enabled`
flag. From this list we render:

  * the orchestrator prompt's "available agents" + routing sections,
  * the pre-action critic prompt's agent roster,

and, in agents.py, we attach the enabled agents as tools (mapping name -> the
LlmAgent instance). Nothing about an agent is duplicated across prompts.

`name` MUST match the corresponding LlmAgent's `name=` in agents.py.
"""

from dataclasses import dataclass
from typing import List

from CoScientist.config import get_settings

settings = get_settings()


@dataclass(frozen=True)
class AgentSpec:
    name: str            # ADK agent name — what the model calls (e.g. "CoderAgent")
    description: str      # what it does (one bullet in the orchestrator prompt)
    routing: str = ""     # when to pick it (routing guidance); "" => no routing line
    enabled: bool = True  # include in prompts AND attach as an orchestrator tool


# Order here is the order shown in prompts and in the tool list.
ORCHESTRATOR_AGENTS: List[AgentSpec] = [
    AgentSpec(
        name="PlannerAgent",
        description="produces a step-by-step roadmap for a complex, multi-step task; "
                    "call it first, then execute the roadmap by delegating each step.",
        routing="for a complex multi-step task, call this FIRST to produce a roadmap, "
                "then follow it step by step.",
        enabled=settings.orchestrator.use_planner,
    ),
    AgentSpec(
        name="HypothesesAgent",
        description="generates ideas and hypotheses.",
        routing="when the direction is unclear or multiple approaches need to be proposed.",
    ),
    AgentSpec(
        name="ResearchAgent",
        description="retrieves scientific knowledge and searches/downloads literature "
                    "(web, RAG, paper search).",
        routing="when external knowledge or literature is required, the problem cannot be "
                "solved computationally, or claims need literature validation. If it returns "
                "no/empty/insufficient results, escalate by reformulating into "
                '"find and download papers about <expanded topic>".',
    ),
    AgentSpec(
        name="TaskExecutorAgent",
        description="runs computations by orchestrating EXISTING ready-to-use MCP "
                    "tools/services (property estimation, docking, simulations, inference "
                    "with available models). Does NOT write code or clone repos — it only "
                    "solves a task if the capability already exists as an MCP tool.",
        routing="when the result can be obtained by RUNNING AN EXISTING tool/service. "
                'Hallmark: "compute/run X using tools we already have."',
    ),
    AgentSpec(
        name="CoderAgent",
        description="general-purpose coder/sandbox engineer: writes and runs code, shell "
                    "and git commands (clone/commit/push), manages files, installs "
                    "dependencies, processes data, and runs long jobs in an isolated "
                    "workspace.",
        routing="when the task requires DOING engineering work: writing/running code, "
                "shell/git operations, building or processing data, environment setup, or "
                'any multi-step build/run no existing tool provides. Hallmark: "write code '
                '/ run a script / clone a repo / assemble a dataset."',
    ),
    AgentSpec(
        name="MedicalAgent",
        description="answers medical/clinical questions: PubMed literature search, PICO "
                    "extraction, study taxonomy, and DICOM image analysis.",
        routing="when the task involves clinical questions, medical literature, or patient "
                "data; if the user uploaded a medical image (DICOM/scan), pass its artifact_id.",
    ),
]


# ── queries ──────────────────────────────────────────────────────────────────
def enabled_agents() -> List[AgentSpec]:
    return [a for a in ORCHESTRATOR_AGENTS if a.enabled]


def is_enabled(name: str) -> bool:
    return any(a.name == name and a.enabled for a in ORCHESTRATOR_AGENTS)


# ── renderers (used to fill prompt placeholders) ─────────────────────────────
def render_agent_bullets() -> str:
    """Bulleted agent list for the orchestrator's 'available agents' section."""
    return "\n".join(f"* **{a.name}** — {a.description}" for a in enabled_agents())


def render_routing_bullets() -> str:
    """Indented routing list ('which agent for which job') for the orchestrator."""
    return "\n".join(f"    - {a.name} — {a.routing}"
                     for a in enabled_agents() if a.routing)


def render_critic_roster() -> str:
    """Compact agent roster for the pre-action critic prompt."""
    return "\n".join(f"  - {a.name}: {a.description}" for a in enabled_agents())
