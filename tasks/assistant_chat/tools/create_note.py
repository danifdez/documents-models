"""create_note: create a workspace note."""

from typing import Any, Dict

from .base import Tool, ToolContext, register
from common.chat.http import http_json


def _execute(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    title = str(args.get("title") or "").strip()
    body = str(args.get("body") or "").strip()
    if not title:
        return {"error": "title required"}
    payload: Dict[str, Any] = {"title": title[:200], "content": body}
    if isinstance(args.get("projectId"), int) and args["projectId"] > 0:
        payload["projectId"] = args["projectId"]
    note = http_json("POST", "/notes", payload)
    if not isinstance(note, dict) or "id" not in note:
        return {"error": "could not create note"}
    return {
        "ok": True,
        "note": {"id": note["id"], "title": note.get("title") or title},
    }


def _summarize(result: Dict[str, Any]):
    if isinstance(result.get("note"), dict):
        note = result["note"]
        return (
            f"Note: {note.get('title') or ''}",
            {"kind": "note", "id": note.get("id"), "title": note.get("title")},
        )
    return None


register(Tool(
    schema={
        "type": "function",
        "function": {
            "name": "create_note",
            "description": (
                "Create a workspace note with a title and body. Use it when "
                "the user explicitly asks to jot something down as a note "
                "('jot down a note about X', 'create a note titled Y'). "
                "DO NOT use for pending tasks (use create_task) or for files "
                "in the assistant folder (use folder_write).\n\n"
                "Pairs well with: list_projects when you need a projectId to "
                "attach the note to a project."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short title for the note (max ~80 chars).",
                    },
                    "body": {
                        "type": "string",
                        "description": "Note body. May use markdown.",
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
                "required": ["title", "body"],
            },
        },
    },
    execute=_execute,
    summarize=_summarize,
))
