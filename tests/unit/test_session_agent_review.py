import asyncio
from typing import AsyncGenerator, List

from google.adk.agents.llm_agent import LlmAgent
from google.adk.events.event import Event
from google.genai import types

from CoScientist.experiments.review import ExperimentReviewSessionAgent
from CoScientist.hitl.models import HITLAction, HITLResponse
from CoScientist.hitl.session_agent import SessionAgent, render_task_plan


class _FakeSession:
    def __init__(self):
        self.state = {}
        self.events: List[Event] = []


class _FakeSessionService:
    async def append_event(self, session, event):
        session.events.append(event)


class _FakeContext:
    def __init__(self):
        self.session = _FakeSession()
        self.session_service = _FakeSessionService()
        self.invocation_id = "inv-1"
        self.branch = None
        self.user_content = types.Content(role="user", parts=[types.Part(text="ask")])

    def set_agent_state(self, name):
        return None


def _non_final_plan_event(author: str, text: str) -> Event:
    return Event(
        invocation_id="inv-1",
        author=author,
        content=types.Content(role="model", parts=[types.Part(text=text)]),
    )


def test_render_task_plan_contains_only_authoritative_registered_tasks():
    tasks = [{
        "id": "TASK-1",
        "title": "Generate both inhibitor sets",
        "description": "Use generate_mols and rank each target separately.",
        "assignee": "TaskExecutorAgent",
        "status": "TODO",
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
    }]

    review = render_task_plan(tasks)

    assert "Generate both inhibitor sets" in review
    assert "TaskExecutorAgent" in review
    assert "generate_mols" in review
    assert "created_at" not in review
    assert "status" not in review


def _run(agent, ctx):
    async def collect():
        return [event async for event in agent._run_async_impl(ctx)]

    return asyncio.run(collect())


def test_experiment_review_runs_headless_on_non_final_plan(monkeypatch):
    from CoScientist.config import get_settings

    monkeypatch.setattr(get_settings().web, "hitl_enabled", False)

    async def fake_run(self, ctx) -> AsyncGenerator[Event, None]:
        plan = '{"schema_version": "experiment-plan/1.0", "plan_id": "P1"}'
        ctx.session.state["experiment_plan"] = plan
        yield _non_final_plan_event(self.name, plan)

    monkeypatch.setattr(LlmAgent, "_run_async_impl", fake_run)
    seen: list[str] = []

    async def fake_review(self, ctx, output_text):
        seen.append(str(output_text))
        return HITLResponse(action=HITLAction.APPROVE, approved=True, instructions="")

    monkeypatch.setattr(ExperimentReviewSessionAgent, "_review_decision", fake_review)

    agent = ExperimentReviewSessionAgent(
        name="ExperimentPlannerAgent",
        model="stub-model",
        review_kind="plan",
        output_key="experiment_plan",
    )
    _run(agent, _FakeContext())
    assert seen
    assert "experiment-plan/1.0" in seen[0]


def test_generic_session_agent_skips_review_when_hitl_off(monkeypatch):
    from CoScientist.config import get_settings

    monkeypatch.setattr(get_settings().web, "hitl_enabled", False)

    async def fake_run(self, ctx) -> AsyncGenerator[Event, None]:
        yield _non_final_plan_event(self.name, "registered plan")

    monkeypatch.setattr(LlmAgent, "_run_async_impl", fake_run)
    seen: list[str] = []

    async def fake_review(self, ctx, output_text):
        seen.append(str(output_text))
        return HITLResponse(action=HITLAction.APPROVE, approved=True, instructions="")

    monkeypatch.setattr(SessionAgent, "_review_decision", fake_review)
    handler = ExperimentReviewSessionAgent(
        name="ExperimentPlannerAgent",
        model="stub-model",
        review_kind="plan",
    ).hitl_handler
    agent = SessionAgent(name="PlannerAgent", model="stub-model", hitl_handler=handler)
    _run(agent, _FakeContext())
    assert seen == []
