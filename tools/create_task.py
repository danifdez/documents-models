"""create_task: create a pending to-do task (no date)."""

from typing import Any, Dict

from lib.framework.tool import Tool, ToolContext, register
from lib.backend.http import http_json


def _execute(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    title = str(args.get("title") or "").strip()
    if not title:
        return {"error": "title required"}
    payload: Dict[str, Any] = {"title": title[:200]}
    desc = str(args.get("description") or "").strip()
    if desc:
        payload["description"] = desc
    if isinstance(args.get("projectId"), int) and args["projectId"] > 0:
        payload["projectId"] = args["projectId"]
    task = http_json("POST", "/user-tasks", payload)
    if not isinstance(task, dict) or "id" not in task:
        return {"error": "could not create task"}
    return {
        "ok": True,
        "task": {"id": task["id"], "title": task.get("title") or title},
    }


def _summarize(result: Dict[str, Any]):
    if isinstance(result.get("task"), dict):
        task = result["task"]
        return (
            f"Task: {task.get('title') or ''}",
            {"kind": "task", "id": task.get("id"), "title": task.get("title")},
        )
    return None


register(Tool(
    schema={
        "type": "function",
        "function": {
            "name": "create_task",
            "description": (
                "Create a pending to-do task (no date). Use it when the user "
                "wants to remember to do something without a specific time "
                "('remind me to call the bank', 'add task buy bread', "
                "'I have to do Z'). For anything with a date or time, use "
                "create_calendar_event instead — that's the only path that "
                "schedules alarms.\n\n"
                "Pairs well with: list_projects to resolve a projectId."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "What needs to be done, in concise imperative form.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Additional details if the user provided them. Optional.",
                    },
                    "projectId": {
                        "type": "integer",
                        "description": (
                            "DO NOT set this field unless you have called list_projects "
                            "earlier in this conversation AND the user explicitly named "
                            "one of the projects from that result. Never guess a value. "
                            "If no project is mentioned, OMIT this parameter entirely."
                        ),
                    },
                },
                "required": ["title"],
            },
        },
    },
    execute=_execute,
    summarize=_summarize,
))
