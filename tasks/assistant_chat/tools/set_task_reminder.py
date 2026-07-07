"""set_task_reminder: set or update a one-shot reminder on a task."""

from datetime import datetime
from typing import Any, Dict

from .base import Tool, ToolContext, register
from common.chat.http import http_json_with_status
from common.chat.user_tasks import resolve_user_task


def _execute(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    task_id, err = resolve_user_task(args)
    if err is not None:
        return err
    remind_at = args.get("remindAt")
    if not isinstance(remind_at, str) or not remind_at.strip():
        return {"error": "missing_remind_at"}
    try:
        parsed = datetime.fromisoformat(remind_at.replace("Z", "+00:00"))
    except ValueError:
        return {"error": "invalid_remind_at"}
    if parsed.tzinfo is None:
        return {
            "error": "remind_at_naive",
            "hint": "Include timezone offset (Z or ±HH:MM).",
        }
    # Reject past timestamps in the tool — the backend would accept them but
    # the scheduler would immediately classify them as lost, which is rarely
    # what the user wants.
    if parsed < datetime.now(parsed.tzinfo):
        return {"error": "remind_at_in_past", "hint": "Pick a future timestamp."}

    status, body = http_json_with_status(
        "PATCH", f"/user-tasks/{task_id}", {"reminderAt": remind_at},
    )
    if status == 200 and isinstance(body, dict) and "id" in body:
        return {
            "ok": True,
            "task": {
                "id": body["id"],
                "title": body.get("title"),
                "reminderAt": body.get("reminderAt"),
            },
        }
    if status == 404:
        return {"error": "task_not_found", "taskId": task_id}
    return {"error": "http_error", "status": status, "detail": body}


def _summarize(result: Dict[str, Any]):
    if result.get("ok"):
        task = result.get("task") or {}
        return (
            f"Reminder set: {task.get('title') or ''} at {task.get('reminderAt') or ''}",
            None,
        )
    return None


register(Tool(
    schema={
        "type": "function",
        "function": {
            "name": "set_task_reminder",
            "description": (
                "Set or update a one-shot reminder on an existing task. "
                "Identify by taskId (preferred) or titleQuery (approximate "
                "match). The reminder fires once at remindAt; the task stays "
                "pending until the user marks it done. remindAt must be ISO "
                "8601 with timezone offset; do not send naive datetimes.\n\n"
                "If the user describes a RECURRING reminder ('remind me every "
                "3 days to water the plants'), DO NOT use this tool — create "
                "a recurring calendar event with trackCompletion=true via "
                "create_calendar_event instead. Tasks are single-shot only.\n\n"
                "Pairs well with: list_tasks when the user references the "
                "task by name and ambiguity is possible."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "taskId": {"type": "integer"},
                    "titleQuery": {"type": "string"},
                    "remindAt": {
                        "type": "string",
                        "description": "ISO 8601 with TZ offset, e.g. 2026-05-23T09:00:00+02:00.",
                    },
                },
                "required": ["remindAt"],
            },
        },
    },
    execute=_execute,
    summarize=_summarize,
))
