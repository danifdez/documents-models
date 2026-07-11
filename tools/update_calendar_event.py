"""update_calendar_event: update an existing calendar event."""

from typing import Any, Dict

from agents.tool_base import Tool, ToolContext, register
from common.chat.http import http_json_with_status
from common.chat.calendars import build_calendar_payload, resolve_calendar_event


def _execute(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    event_id, err = resolve_calendar_event(args)
    if err is not None:
        return err
    # Strip resolution keys before mapping.
    payload_args = {k: v for k, v in args.items() if k not in ("eventId", "match")}
    if not payload_args:
        return {"error": "nothing_to_update"}
    payload, build_err = build_calendar_payload(payload_args)
    if build_err:
        return {"error": build_err}
    status, body = http_json_with_status("PATCH", f"/calendar-events/{event_id}", payload)
    if status >= 400 or not isinstance(body, dict) or "id" not in body:
        detail = body.get("message") if isinstance(body, dict) else None
        return {"error": "could not update event", "detail": detail, "status": status}
    return {
        "ok": True,
        "event": {
            "id": body["id"],
            "title": body.get("title"),
            "startDate": body.get("startDate"),
            "recurrenceRule": body.get("recurrenceRule"),
            "alarm": body.get("alarm"),
        },
    }


def _summarize(result: Dict[str, Any]):
    if isinstance(result.get("event"), dict):
        ev = result["event"]
        return (
            f"Updated: {ev.get('title') or ''}",
            {"kind": "calendarEvent", "id": ev.get("id"), "title": ev.get("title")},
        )
    return None


register(Tool(
    schema={
        "type": "function",
        "function": {
            "name": "update_calendar_event",
            "description": (
                "Update an existing calendar event. Identify it by eventId "
                "(preferred) or 'match' (approximate title). Any field is "
                "optional; send only those that change. To clear a field "
                "send null (alarm: null to remove, recurrenceRule: null to "
                "stop the recurrence). Cannot edit a single occurrence — "
                "only the whole event.\n\n"
                "Pairs well with: search_workspace when you do not have the "
                "eventId."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "eventId": {"type": "integer"},
                    "match": {
                        "type": "string",
                        "description": "Approximate title match if eventId unknown.",
                    },
                    "title": {"type": "string"},
                    "startAt": {"type": "string"},
                    "endAt": {"type": "string"},
                    "recurrenceRule": {"type": ["string", "null"]},
                    "alarm": {"type": ["object", "null"]},
                    "projectId": {"type": ["integer", "null"]},
                },
            },
        },
    },
    execute=_execute,
    summarize=_summarize,
))
