"""create_calendar_event: create a one-shot or recurring calendar event."""

from typing import Any, Dict

from lib.framework.tool import Tool, ToolContext, register
from lib.backend.http import http_json_with_status
from lib.backend.calendars import build_calendar_payload


def _execute(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    if not str(args.get("title") or "").strip():
        return {"error": "title required"}
    if not str(args.get("startAt") or "").strip():
        return {"error": "startAt required"}
    payload, err = build_calendar_payload(args)
    if err:
        return {"error": err}
    status, body = http_json_with_status("POST", "/calendar-events", payload)
    if status >= 400 or not isinstance(body, dict) or "id" not in body:
        detail = body.get("message") if isinstance(body, dict) else None
        return {"error": "could not create event", "detail": detail, "status": status}
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
            f"Event: {ev.get('title') or ''}",
            {"kind": "calendarEvent", "id": ev.get("id"), "title": ev.get("title")},
        )
    return None


register(Tool(
    schema={
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": (
                "Create a calendar event (one-shot or recurring), optionally "
                "with an alarm. Use it for anything with a date or time: "
                "appointments, recurring reminders, one-off alerts. A time-only "
                "reference ('at 10pm') is still an event — resolve it to today "
                "(or the next occurrence) using the current date in the system "
                "prompt. Only when there is NO date and NO time does the user "
                "want a task — use create_task. Include alarm only when the user "
                "asks to be reminded/alerted/notified.\n\n"
                "Pairs well with: list_projects to resolve a projectId.\n\n"
                "RRULE examples (RFC 5545):\n"
                "  'every day for 7 days at 10pm' -> FREQ=DAILY;COUNT=7;BYHOUR=22;BYMINUTE=0\n"
                "  'every 3 days' -> FREQ=DAILY;INTERVAL=3\n"
                "  'every monday' -> FREQ=WEEKLY;BYDAY=MO\n"
                "  'first friday of every month' -> FREQ=MONTHLY;BYDAY=1FR\n"
                "  'every year on may 18' -> FREQ=YEARLY;BYMONTH=5;BYMONTHDAY=18\n"
                "Omit recurrenceRule for one-shot events."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short, concrete. Stored as-is — not translated.",
                    },
                    "startAt": {
                        "type": "string",
                        "description": "ISO 8601 with TZ offset, e.g. 2026-05-20T22:00:00+02:00.",
                    },
                    "endAt": {
                        "type": "string",
                        "description": "ISO 8601. Omit if it's a point in time, not a span.",
                    },
                    "recurrenceRule": {
                        "type": "string",
                        "description": "Valid RRULE without 'RRULE:' prefix. Omit for one-shot.",
                    },
                    "alarm": {
                        "type": "object",
                        "description": "Optional reminder. Omit if the user did not ask to be alerted.",
                        "properties": {
                            "offsetMinutes": {
                                "type": "integer",
                                "description": "0=at start; negative=before. Default 0 if alarm is present without explicit offset.",
                            },
                            "label": {"type": "string"},
                        },
                        "required": ["offsetMinutes"],
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
                "required": ["title", "startAt"],
            },
        },
    },
    execute=_execute,
    summarize=_summarize,
))
