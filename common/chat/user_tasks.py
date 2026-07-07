"""Shared user-task helpers used by the update/delete/reminder tools."""

from typing import Any, Dict, List, Optional, Tuple

from .http import http_json


def resolve_user_task(args: Dict[str, Any]) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
    """Resolve a target task id. Explicit taskId wins; otherwise titleQuery
    does a case-insensitive substring search over GET /user-tasks. Returns
    (taskId, errorPayload) — errorPayload is forwarded to the model on
    missing identifier, lookup failure, not_found, or ambiguous match."""
    explicit = args.get("taskId")
    if isinstance(explicit, int):
        return explicit, None
    query = str(args.get("titleQuery") or "").strip()
    if not query:
        return None, {"error": "missing_identifier", "hint": "Provide taskId or titleQuery."}
    needle = query.lower()
    tasks = http_json("GET", "/user-tasks")
    if not isinstance(tasks, list):
        return None, {"error": "lookup_failed"}
    candidates: List[Dict[str, Any]] = []
    for t in tasks:
        if not isinstance(t, dict):
            continue
        title = str(t.get("title") or "")
        if needle in title.lower():
            candidates.append({
                "id": t.get("id"),
                "title": title,
                "status": t.get("status"),
            })
    if not candidates:
        return None, {"error": "not_found", "titleQuery": query}
    if len(candidates) > 1:
        return None, {
            "error": "ambiguous",
            "candidates": candidates[:10],
            "hint": "Ask the user which task, then retry with taskId.",
        }
    return candidates[0]["id"], None
