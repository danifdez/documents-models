"""list_notes: enumerate the user's notes."""

from typing import Any, Dict

from .base import Tool, ToolContext, register
from common.chat.http import http_json


def _execute(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    pid = args.get("projectId")
    path = f"/notes/project/{int(pid)}" if isinstance(pid, int) and pid > 0 else "/notes"
    data = http_json("GET", path)
    if not isinstance(data, list):
        return {"notes": []}
    notes = []
    for n in data[:30]:
        if not isinstance(n, dict) or not n.get("id"):
            continue
        # Trim content to keep prompt size manageable. If the user asks the
        # assistant about a specific note, it can fetch its full body via a
        # follow-up tool — for now a short preview is enough to reason about.
        body = (n.get("content") or "").strip()
        preview = body[:160] + ("…" if len(body) > 160 else "")
        notes.append({
            "id": n["id"],
            "title": n.get("title") or "",
            "preview": preview,
            "projectId": (n.get("project") or {}).get("id"),
        })
    return {"notes": notes}


def _summarize(result: Dict[str, Any]):
    n = len(result.get("notes") or [])
    return f"{n} notes", None


register(Tool(
    schema={
        "type": "function",
        "function": {
            "name": "list_notes",
            "description": (
                "List the user's notes (id, title, short preview). Use it "
                "when the user wants the enumeration itself ('what notes do "
                "I have?', 'show my notes in project X'). For content-based "
                "discovery use search_workspace (one-shot) or "
                "workspace_research (multi-source). The preview is enough to "
                "identify a note — you do not need get_resource_content for "
                "a note.\n\n"
                "Pairs well with: list_projects to filter by a project."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "projectId": {
                        "type": "integer",
                        "description": "Project ID. Omit to list all.",
                    },
                },
            },
        },
    },
    execute=_execute,
    summarize=_summarize,
    hidden_from_main=True,
))
