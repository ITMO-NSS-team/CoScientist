"""Task Tracker infrastructure and tools for managing and tracking tasks."""
from google.adk.tools.tool_context import ToolContext  
from fastapi import status
from typing import Any, Dict, List, Optional
import json
import os
from datetime import datetime

from google.adk.tools import BaseTool, FunctionTool
from google.adk.tools.base_toolset import BaseToolset
from google.adk.agents.readonly_context import ReadonlyContext

class TaskTrackerToolset(BaseToolset):
    """Toolset for tracking tasks across the MAS."""

    def __init__(self, prefix: str = None, storage_path: str = "task_tracker_data.json"):
        super().__init__(tool_name_prefix=prefix)
        self.storage_path = storage_path
        self.tasks: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        """Load tasks from disk."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self.tasks = json.load(f)
            except json.JSONDecodeError:
                self.tasks = []

    def _save(self):
        """Save tasks to disk."""
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self.tasks, f, indent=2, ensure_ascii=False)

    async def get_tools(self, readonly_context: Optional[ReadonlyContext] = None) -> List[BaseTool]:
        return [
            #FunctionTool(self.create_task),
            FunctionTool(self.update_task_status),
            FunctionTool(self.get_active_tasks),
        ]

    async def close(self) -> None:
        pass

    def create_plan(self, tasks: List[Dict[str, Any]], tool_context: ToolContext) -> Dict[str, Any]:  

        """Replace ALL tasks with a new plan provided by the planner agent.

        Each task in the list should have:
          - title (str)
          - description (str)
          - assignee (str)
          - parent_id (str or None, optional): ID of a parent task to build hierarchy.
          - notes (str, optional)
        If 'id' is omitted, an auto-generated ID like TASK-<n> will be assigned.
        """
        if not isinstance(tasks, list):
            return {"result": "error", "message": "'tasks' must be a list of task definitions."}

        for i, t in enumerate(tasks):
            if not isinstance(t, dict):
                return {"result": "error", "message": f"Task at index {i} must be a dictionary."}
            if "title" not in t:
                return {"result": "error", "message": f"Task at index {i} missing 'title'."}
            if "description" not in t:
                return {"result": "error", "message": f"Task at index {i} missing 'description'."}
            if "assignee" not in t:
                return {"result": "error", "message": f"Task at index {i} missing 'assignee'."}

        processed_tasks = []
        coder_task = None

        for t in tasks:
            if t.get("assignee") == "OrchestratorAgent":
                continue
            
            # Merge coder tasks
            if t.get("assignee") == "CoderAgent":
                if coder_task is None:
                    coder_task = {
                        "id": t.get("id"),
                        "title": t.get("title"),
                        "description": t.get("description"),
                        "assignee": "CoderAgent",
                        "parent_id": t.get("parent_id", None),
                        "notes": t.get("notes", "")
                    }
                else:
                    coder_task["title"] += f" - {t.get('title')}"
                    coder_task["description"] += f" - {t.get('description')}"
                    
                    current_note = t.get("notes", "")
                    if current_note:
                        if coder_task["notes"]:
                            coder_task["notes"] += f" - {current_note}"
                        else:
                            coder_task["notes"] = current_note
            else:
                processed_tasks.append(t)

        if coder_task:
            processed_tasks.insert(0, coder_task)

        new_tasks = []
        for t in processed_tasks:
            task = {
                "id": t.get("id") if t.get("id") else f"TASK-{len(new_tasks) + 1}",
                "title": t.get("title"),
                "description": t.get("description"),
                "assignee": t.get("assignee"),
                "status": "TODO",
                "parent_id": t.get("parent_id", None),
                "notes": t.get("notes", ""),
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
            new_tasks.append(task)

        ids = {task["id"] for task in new_tasks}
        for task in new_tasks:
            if task["parent_id"] and task["parent_id"] not in ids:
                return {
                    "result": "error",
                    "message": f"parent_id '{task['parent_id']}' not found in the plan."
                }

        tool_context.state["active_tasks"] = new_tasks  

        self.tasks = new_tasks
        self._save()
        return {
            "result": "success",
            "message": f"Plan created with {len(self.tasks)} tasks."
        }


    def create_task(self, title: str, description: str, assignee: Optional[str] = None) -> Dict[str, Any]:
        """Create a new task in the task tracker.
        
        Args:
            title: The title of the task.
            description: Detailed description of what needs to be done.
            assignee: The agent or sub-system responsible for this task.
            
        Returns:
            A dictionary with the task ID and current state.
        """
        task_id = f"TASK-{len(self.tasks) + 1}"
        task = {
            "id": task_id,
            "title": title,
            "description": description,
            "status": "TODO",
            "assignee": assignee or "unassigned",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "notes": "",
            "parent_id": None
        }
        self.tasks.append(task)
        self._save()
        return {"result": "success", "task": task}

    def update_task_status(self, task_id: str, status: str, notes: Optional[str] = None) -> Dict[str, Any]:
        """Use this tool REGULARLY to provide clear progress updates. Never forget to update the task status. 
        
        Args:
            task_id: The ID of the task to update (e.g., TASK-1).
            status: The new status (IN_PROGRESS, DONE, FAILED).
            notes: Optional notes or results from the task execution.
            
        Returns:
            A dictionary indicating success or failure.
        """
        for task in self.tasks:
            if task["id"] == task_id:
                task["status"] = status
                task["updated_at"] = datetime.now().isoformat()
                if notes:
                    task["notes"] += f"\n[{datetime.now().isoformat()}] {notes}"
                self._save()
                return {"result": "success", "task": task}
        return {"result": "error", "message": f"Task {task_id} not found."}

    def get_active_tasks(self, **kwargs: Any) -> Dict[str, Any]:
        """Get a list of all tracked tasks.
            Returns:
                A dictionary containing the matching tasks.
        """
        self._load()
        readonly_context: Optional[ReadonlyContext] = kwargs.get("readonly_context")

        current_agent = None
        if readonly_context:
            current_agent = getattr(readonly_context, "agent_name", None)

        cleaned_tasks = []
        for task in self.tasks:
            cleaned_task = {k: v for k, v in task.items() if k not in ("created_at", "updated_at")}
            if task.get("assignee") != current_agent:
                cleaned_task.pop("description", None)
                
            if task.get("assignee") != current_agent and current_agent != "OrchestratorAgent":
                cleaned_task.pop("notes", None)

            cleaned_tasks.append(cleaned_task)
            
        return {"tasks": cleaned_tasks}

# Global instance to share state across tools
task_tracker_instance = TaskTrackerToolset()

def get_task_tracker_tools() -> list:
    tools = [
        #FunctionTool(task_tracker_instance.create_task),
        FunctionTool(task_tracker_instance.update_task_status),
        FunctionTool(task_tracker_instance.get_active_tasks)
    ]
    return tools

def create_plan_tool():
    return FunctionTool(task_tracker_instance.create_plan)
