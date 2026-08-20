"""Regression tests for the planner's durable session result."""

from types import SimpleNamespace

from google.genai import types

from CoScientist.hitl.session_agent import SessionAgent


def test_planner_creates_a_fallback_plan_when_model_skips_create_plan(monkeypatch):
    """A model response without a tool call must not leave the run un-routable."""
    created_plans = []

    def create_plan(tasks, tool_context):
        created_plans.append(tasks)
        tool_context.state["active_tasks"] = [{"id": "TASK-1", **tasks[0]}]
        return {"result": "success"}

    monkeypatch.setattr(
        "CoScientist.hitl.session_agent.task_tracker_instance.create_plan",
        create_plan,
    )
    agent = SessionAgent(
        name="PlannerAgent",
        model="openai/test",
        result_state_key="active_tasks",
        output_key="planner_roadmap",
    )
    user_message = types.Content(role="user", parts=[types.Part(text="Find a catalyst")])
    ctx = SimpleNamespace(
        session=SimpleNamespace(
            state={},
            events=[SimpleNamespace(content=user_message)],
        )
    )
    event = SimpleNamespace(
        content=types.Content(role="model", parts=[types.Part(text="I will help.")]),
    )

    agent._apply_state_result(ctx, event)

    assert len(created_plans) == 1
    assert created_plans[0][0]["assignee"] == "ResearchAgent"
    assert "Find a catalyst" in created_plans[0][0]["description"]
    assert ctx.session.state["active_tasks"][0]["id"] == "TASK-1"
    assert "TASK-1" in ctx.session.state["planner_roadmap"]
