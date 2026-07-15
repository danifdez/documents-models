"""mark_event_occurrence_done: mark a single occurrence of a trackable event."""

import urllib.parse
from datetime import datetime
from typing import Any, Dict

from lib.framework.tool import Tool, ToolContext, register
from lib.backend.http import http_json_with_status
from lib.backend.calendars import resolve_calendar_event


def _execute(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    event_id, err = resolve_calendar_event(args)
    if err is not None:
        return err
    occurrence_date = args.get("occurrenceDate")
    if not isinstance(occurrence_date, str) or not occurrence_date.strip():
        return {"error": "missing_occurrence_date"}
    try:
        parsed = datetime.fromisoformat(occurrence_date.replace("Z", "+00:00"))
    except ValueError:
        return {"error": "invalid_occurrence_date"}
    if parsed.tzinfo is None:
        return {
            "error": "occurrence_date_naive",
            "hint": "Include timezone offset (Z or ±HH:MM).",
        }
    encoded = urllib.parse.quote(occurrence_date, safe="")
    status, body = http_json_with_status(
        "POST", f"/calendar-events/{event_id}/occurrences/{encoded}/complete"
    )
    if status == 204:
        return {"ok": True, "eventId": event_id, "occurrenceDate": occurrence_date}
    if status == 400:
        detail = ""
        code = ""
        if isinstance(body, dict):
            code = str(body.get("error") or "").strip()
            detail = str(body.get("message") or body.get("error") or "").strip()
        if code == "event_not_trackable" or "trackable" in detail.lower():
            return {
                "error": "event_not_trackable",
                "eventId": event_id,
                "hint": "Enable 'I want to mark it as done' on the event first.",
            }
        return {"error": "bad_request", "detail": detail or body}
    if status == 404:
        return {"error": "event_not_found", "eventId": event_id}
    return {"error": "http_error", "status": status, "detail": body}


def _summarize(result: Dict[str, Any]):
    if result.get("ok"):
        return f"Done: occurrence {result.get('occurrenceDate')}", None
    return None


register(Tool(
    schema={
        "type": "function",
        "function": {
            "name": "mark_event_occurrence_done",
            "description": (
                "Mark a single occurrence of a trackable calendar event as "
                "done. Identify the event by eventId (preferred) or 'match' "
                "(approximate title substring). The event MUST have "
                "trackCompletion=true; if not, the tool returns "
                "event_not_trackable and you should tell the user to enable "
                "it from the event editor first. Provide occurrenceDate as "
                "the ISO 8601 timestamp of the specific occurrence (with "
                "timezone offset, e.g. 2026-05-22T20:00:00Z or "
                "2026-05-22T22:00:00+02:00). For one-shot events the "
                "occurrenceDate equals the event's startDate.\n\n"
                "Pairs well with: search_workspace when the user references "
                "the event by name and the eventId is unknown.\n\n"
                "Idempotent: marking the same occurrence twice is a no-op."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "eventId": {"type": "integer"},
                    "match": {
                        "type": "string",
                        "description": "Approximate title match if eventId unknown.",
                    },
                    "occurrenceDate": {
                        "type": "string",
                        "description": (
                            "ISO 8601 with timezone offset. Examples: "
                            "2026-05-22T20:00:00Z, 2026-05-22T22:00:00+02:00."
                        ),
                    },
                },
                "required": ["occurrenceDate"],
            },
        },
    },
    execute=_execute,
    summarize=_summarize,
))
