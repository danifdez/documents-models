"""list_tasks: enumerate the user's tasks."""

from typing import Any, Dict

from agents.tool_base import Tool, ToolContext, register
from common.chat.http import http_json


def _execute(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    pid = args.get("projectId")
    path = f"/user-tasks/project/{int(pid)}" if isinstance(pid, int) and pid > 0 else "/user-tasks"
    data = http_json("GET", path)
    if not isinstance(data, list):
        return {"tasks": []}
    status_filter = str(args.get("status") or "").strip().lower() or None
    tasks = []
    for t in data:
        if not isinstance(t, dict) or not t.get("id"):
            continue
        if status_filter and t.get("status") != status_filter:
            continue
        tasks.append({
            "id": t["id"],
            "title": t.get("title") or "",
            "status": t.get("status"),
            "projectId": (t.get("project") or {}).get("id"),
        })
    return {"tasks": tasks[:50]}


def _summarize(result: Dict[str, Any]):
    n = len(result.get("tasks") or [])
    return f"{n} tasks", None


register(Tool(
    schema={
        "type": "function",
        "function": {
            "name": "list_tasks",
            "description": (
                "List the user's tasks (id, title, status, projectId). Use "
                "it when the user asks 'what tasks do I have?', 'what's "
                "pending', or as a preliminary step to find a taskId for "
                "update_task / delete_task. Pass status='pending' or "
                "'completed' to filter; pass projectId to narrow by project.\n\n"
                "Pairs well with: update_task (mark done, rename), "
                "delete_task (with confirmation card)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "projectId": {
                        "type": "integer",
                        "description": "Project ID. Omit to list all.",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pending", "completed"],
                        "description": "Filter by status. By default all are returned.",
                    },
                },
            },
        },
    },
    execute=_execute,
    summarize=_summarize,
))
