"""delete_calendar_event: delete a calendar event entirely."""

from typing import Any, Dict

from agents.tool_base import Tool, ToolContext, register
from common.chat.http import http_json_with_status
from common.chat.calendars import resolve_calendar_event


def _execute(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    event_id, err = resolve_calendar_event(args)
    if err is not None:
        return err
    status, body = http_json_with_status("DELETE", f"/calendar-events/{event_id}")
    if status >= 400:
        detail = body.get("message") if isinstance(body, dict) else None
        return {"deleted": False, "error": "could not delete event", "detail": detail, "status": status}
    if isinstance(body, dict) and body.get("deleted") is False:
        return {"deleted": False, "error": "not_found", "eventId": event_id}
    return {"deleted": True, "eventId": event_id}


def _summarize(result: Dict[str, Any]):
    if result.get("deleted") is True:
        return f"Deleted event #{result.get('eventId')}", None
    return None


register(Tool(
    schema={
        "type": "function",
        "function": {
            "name": "delete_calendar_event",
            "description": (
                "Delete a calendar event entirely (stops all future "
                "occurrences and alarms). Identify by eventId (preferred) or "
                "'match' (approximate title). There is no per-occurrence "
                "override — to skip a single occurrence, edit the event.\n\n"
                "Pairs well with: search_workspace when you do not have the "
                "eventId."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "eventId": {"type": "integer"},
                    "match": {"type": "string"},
                },
            },
        },
    },
    execute=_execute,
    summarize=_summarize,
))
