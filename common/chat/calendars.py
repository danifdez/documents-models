"""Shared calendar helpers used by the create/update/delete/mark-done tools."""

from typing import Any, Dict, List, Optional, Tuple

from .http import http_json


def normalize_alarm(value: Any) -> Optional[Dict[str, Any]]:
    """Coerce an LLM-provided alarm into the AlarmDescriptor shape, or return
    None to signal absence. Returns the sentinel string 'invalid' if it cannot
    be made valid."""
    if value is None:
        return None
    if not isinstance(value, dict):
        return "invalid"  # type: ignore[return-value]
    offset = value.get("offsetMinutes")
    if isinstance(offset, bool) or not isinstance(offset, int):
        return "invalid"  # type: ignore[return-value]
    if offset < -10080 or offset > 0:
        return "invalid"  # type: ignore[return-value]
    out: Dict[str, Any] = {"offsetMinutes": offset}
    label = value.get("label")
    if isinstance(label, str) and label.strip():
        out["label"] = label.strip()[:100]
    return out


def build_calendar_payload(args: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Map LLM tool args → backend create/update payload. Returns (payload, err)."""
    payload: Dict[str, Any] = {}
    if "title" in args:
        title = str(args.get("title") or "").strip()
        if not title:
            return None, "title required"
        payload["title"] = title[:200]
    if "startAt" in args:
        start = str(args.get("startAt") or "").strip()
        if not start:
            return None, "startAt required"
        payload["startDate"] = start
    if "endAt" in args and args.get("endAt") is not None:
        payload["endDate"] = str(args.get("endAt"))
    if "recurrenceRule" in args:
        rr = args.get("recurrenceRule")
        if rr is None:
            payload["recurrenceRule"] = None
        else:
            rr_s = str(rr).strip()
            if not rr_s:
                payload["recurrenceRule"] = None
            elif not rr_s.startswith("FREQ="):
                return None, "recurrenceRule must start with 'FREQ='"
            else:
                payload["recurrenceRule"] = rr_s
    if "alarm" in args:
        normalized = normalize_alarm(args.get("alarm"))
        if normalized == "invalid":
            return None, "alarm.offsetMinutes must be integer in [-10080, 0]"
        payload["alarm"] = normalized
    if isinstance(args.get("projectId"), int) and args["projectId"] > 0:
        payload["projectId"] = args["projectId"]
    return payload, None


def resolve_calendar_event(args: Dict[str, Any]) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
    """Resolve a target event id. Returns (eventId, errorPayload). If eventId is
    set, errorPayload is None. If errorPayload is set, the caller forwards it to
    the model."""
    explicit = args.get("eventId")
    if isinstance(explicit, int):
        return explicit, None
    match = str(args.get("match") or "").strip()
    if not match:
        return None, {"error": "missing_identifier", "hint": "Provide eventId or match."}
    needle = match.lower()
    events = http_json("GET", "/calendar-events")
    if not isinstance(events, list):
        return None, {"error": "lookup_failed"}
    candidates: List[Dict[str, Any]] = []
    for e in events:
        if not isinstance(e, dict):
            continue
        title = str(e.get("title") or "")
        if needle in title.lower():
            candidates.append({
                "id": e.get("id"),
                "title": title,
                "startDate": e.get("startDate"),
            })
    if not candidates:
        return None, {"error": "not_found", "match": match}
    if len(candidates) > 1:
        return None, {
            "error": "ambiguous",
            "candidates": candidates[:10],
            "hint": "Ask the user which event, then retry with eventId.",
        }
    return candidates[0]["id"], None
