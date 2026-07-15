"""delete_task: delete a task, behind a confirmation card."""

from typing import Any, Dict

from lib.framework.tool import Tool, ToolContext, register
from lib.backend.http import http_json, post_tool_event
from lib.backend.user_tasks import resolve_user_task


def _execute(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    task_id, err = resolve_user_task(args)
    if err is not None:
        return err
    # Always GET to obtain a human-readable title for the confirmation card —
    # the resolver may have skipped the network round-trip on explicit taskId.
    task = http_json("GET", f"/user-tasks/{task_id}")
    if not isinstance(task, dict) or "id" not in task:
        return {"error": "not_found", "taskId": task_id}
    title = task.get("title") or f"#{task_id}"

    if isinstance(ctx.job_id, int) and isinstance(ctx.owner_id, int):
        post_tool_event(
            ctx.owner_segment, ctx.owner_id, ctx.job_id, "delete_task", title,
            status="pending_confirmation",
            summary=f"Pending: delete {title}",
            kind="task_delete",
            payload={"taskId": task_id, "title": title},
            confirm_label="Confirm delete",
            cancel_label="Cancel",
        )

    return {
        "ok": True,
        "pendingConfirmation": True,
        "taskId": task_id,
        "title": title,
    }


register(Tool(
    schema={
        "type": "function",
        "function": {
            "name": "delete_task",
            "description": (
                "Delete a task entirely. Identify it by taskId (preferred) or "
                "titleQuery (approximate match). ALWAYS shows a confirmation "
                "card to the user before deletion. Use update_task with "
                "status='completed' instead if the user only wants to mark "
                "the task as done.\n\n"
                "Pairs well with: list_tasks to confirm the candidate before "
                "deleting."
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
))
