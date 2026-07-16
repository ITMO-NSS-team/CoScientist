from types import SimpleNamespace

from CoScientist.agents.callbacks.tool_callbacks import before_get_task
from CoScientist.tools.task_tracker import TaskTrackerToolset


def _context(agent_name="OrchestratorAgent"):
    return SimpleNamespace(state={}, agent_name=agent_name)


def _plan(title, assignee="ResearchAgent"):
    return [{
        "title": title,
        "description": f"Investigate {title}",
        "assignee": assignee,
    }]


def test_task_plans_are_isolated_by_context():
    tracker = TaskTrackerToolset()
    first = _context()
    second = _context()

    tracker.create_plan(_plan("alpha"), first)
    tracker.create_plan(_plan("beta"), second)

    assert first.state["active_tasks"][0]["title"] == "alpha"
    assert second.state["active_tasks"][0]["title"] == "beta"

    tracker.update_task_status("TASK-1", "DONE", first, notes="complete")
    assert first.state["active_tasks"][0]["status"] == "DONE"
    assert second.state["active_tasks"][0]["status"] == "TODO"


def test_get_active_tasks_reads_only_current_session_state():
    tracker = TaskTrackerToolset()
    context = _context("ResearchAgent")
    tracker.create_plan(_plan("session-only"), context)

    result = tracker.get_active_tasks(context)

    assert [task["title"] for task in result["tasks"]] == ["session-only"]
    assert result["tasks"][0]["description"] == "Investigate session-only"


def test_before_get_task_initializes_list_without_overwriting_plan():
    empty = _context()
    before_get_task(empty)
    assert empty.state["active_tasks"] == []

    existing = _context()
    tasks = [{"id": "TASK-1", "title": "keep me"}]
    existing.state["active_tasks"] = tasks
    before_get_task(existing)
    assert existing.state["active_tasks"] is tasks
