"""list_projects: enumerate the user's projects."""

from typing import Any, Dict

from .base import Tool, ToolContext, register
from common.chat.http import http_json


def _execute(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    data = http_json("GET", "/projects")
    if not isinstance(data, list):
        return {"projects": []}
    projects = [
        {"id": p.get("id"), "name": p.get("name")}
        for p in data
        if isinstance(p, dict) and p.get("id")
    ]
    return {"projects": projects[:30]}


def _summarize(result: Dict[str, Any]):
    n = len(result.get("projects") or [])
    return f"{n} projects", None


register(Tool(
    schema={
        "type": "function",
        "function": {
            "name": "list_projects",
            "description": (
                "List the user's projects with id and name. Use it when the "
                "user directly asks 'what projects do I have?', or as a "
                "preliminary step when you need a projectId to attach a "
                "note/task/event.\n\n"
                "Pairs well with: create_note, create_task, "
                "create_calendar_event (the resulting ids feed their "
                "projectId argument)."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    execute=_execute,
    summarize=_summarize,
))
