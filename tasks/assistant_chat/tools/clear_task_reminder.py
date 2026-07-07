"""clear_task_reminder: remove the reminder from a task."""

from typing import Any, Dict

from .base import Tool, ToolContext, register
from common.chat.http import http_json_with_status
from common.chat.user_tasks import resolve_user_task


def _execute(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    task_id, err = resolve_user_task(args)
    if err is not None:
        return err
    status, body = http_json_with_status(
        "PATCH", f"/user-tasks/{task_id}", {"reminderAt": None},
    )
    if status == 200 and isinstance(body, dict) and "id" in body:
        return {
            "ok": True,
            "task": {
                "id": body["id"],
                "title": body.get("title"),
                "reminderAt": None,
            },
        }
    if status == 404:
        return {"error": "task_not_found", "taskId": task_id}
    return {"error": "http_error", "status": status, "detail": body}


def _summarize(result: Dict[str, Any]):
    if result.get("ok"):
        task = result.get("task") or {}
        return f"Reminder cleared: {task.get('title') or ''}", None
    return None


register(Tool(
    schema={
        "type": "function",
        "function": {
            "name": "clear_task_reminder",
            "description": (
                "Remove the reminder from a task (sets reminderAt=null). The "
                "task stays pending. Identify by taskId or titleQuery. Use "
                "this when the user explicitly says they no longer want the "
                "reminder, NOT when they finished the task — for the latter "
                "use update_task with status='completed', which already "
                "clears the reminder automatically."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "taskId": {"type": "integer"},
                    "titleQuery": {"type": "string"},
                },
            },
        },
    },
    execute=_execute,
    summarize=_summarize,
))
