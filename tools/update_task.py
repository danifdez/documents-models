"""update_task: update an existing task."""

from typing import Any, Dict

from agents.tool_base import Tool, ToolContext, register
from common.chat.http import http_json_with_status
from common.chat.user_tasks import resolve_user_task


def _execute(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    task_id, err = resolve_user_task(args)
    if err is not None:
        return err
    # Build the PATCH body iterating only the fields explicitly present —
    # distinguish "not sent" (skip) from "sent as null" (clear).
    payload: Dict[str, Any] = {}
    if "status" in args:
        status = args.get("status")
        if status not in ("pending", "completed"):
            return {"error": "invalid_status",
                    "hint": "status must be 'pending' or 'completed'."}
        payload["status"] = status
    if "title" in args:
        title = str(args.get("title") or "").strip()
        if not title:
            return {"error": "title_required"}
        payload["title"] = title[:200]
    if "description" in args:
        desc = args.get("description")
        payload["description"] = None if desc is None else str(desc)
    if "projectId" in args:
        pid = args.get("projectId")
        if pid is not None and not isinstance(pid, int):
            return {"error": "invalid_projectId"}
        payload["projectId"] = pid
    if not payload:
        return {"error": "nothing_to_update"}
    status_code, body = http_json_with_status(
        "PATCH", f"/user-tasks/{task_id}", payload,
    )
    if status_code >= 400 or not isinstance(body, dict) or "id" not in body:
        detail = body.get("message") if isinstance(body, dict) else None
        return {"error": "could not update task",
                "detail": detail, "status": status_code}
    return {
        "ok": True,
        "task": {
            "id": body["id"],
            "title": body.get("title"),
            "status": body.get("status"),
        },
        "changed": list(payload.keys()),
    }


def _summarize(result: Dict[str, Any]):
    if isinstance(result.get("task"), dict):
        task = result["task"]
        changed = result.get("changed") or []
        if "status" in changed:
            verb = "Done" if task.get("status") == "completed" else "Re-opened"
            summary = f"{verb}: {task.get('title') or ''}"
        elif "title" in changed:
            summary = f"Renamed: {task.get('title') or ''}"
        elif "description" in changed:
            summary = f"Description updated: {task.get('title') or ''}"
        elif "projectId" in changed:
            summary = f"Project changed: {task.get('title') or ''}"
        else:
            summary = f"Updated: {task.get('title') or ''}"
        return summary, {"kind": "task", "id": task.get("id"), "title": task.get("title")}
    return None


register(Tool(
    schema={
        "type": "function",
        "function": {
            "name": "update_task",
            "description": (
                "Update an existing task: mark done, re-open, rename, change "
                "description, move to/clear a project. Identify it by taskId "
                "(preferred, from a previous list_tasks) or titleQuery "
                "(approximate match). Send only the fields that change. To "
                "clear a description or project, send null. If titleQuery is "
                "ambiguous you receive candidates — ask the user and retry "
                "with taskId.\n\n"
                "Pairs well with: list_tasks when you do not have the taskId."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "taskId": {"type": "integer"},
                    "titleQuery": {
                        "type": "string",
                        "description": "Approximate title match if taskId unknown.",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pending", "completed"],
                    },
                    "title": {"type": "string"},
                    "description": {"type": ["string", "null"]},
                    "projectId": {"type": ["integer", "null"]},
                },
            },
        },
    },
    execute=_execute,
    summarize=_summarize,
))
