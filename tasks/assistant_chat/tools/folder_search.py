"""folder_search: semantic search inside the assistant's working folder."""

import urllib.parse
from typing import Any, Dict

from .base import Tool, ToolContext, register
from common.chat.http import http_json_with_status


def _execute(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    if not isinstance(ctx.owner_id, int):
        return {"error": "internal", "message": "missing owner context"}

    query = str(args.get("query") or "").strip()
    if len(query) < 3:
        return {"error": "query_too_short"}

    limit = args.get("limit") if isinstance(args.get("limit"), int) else 10
    encoded = urllib.parse.quote(query, safe="")
    path = (
        f"/{ctx.owner_segment}/{ctx.owner_id}/indexed-files/search"
        f"?query={encoded}&limit={limit}"
    )
    status, body = http_json_with_status("GET", path)

    if status == 200 and isinstance(body, dict):
        hits = body.get("hits") or []
        return {"ok": True, "hits": hits, "query": query}
    if status == 409 and isinstance(body, dict) and body.get("error") == "no_folder_configured":
        return {"error": "no_folder_configured"}
    if status == 400 and isinstance(body, dict):
        return {"error": body.get("error") or "bad_request"}
    # Backend failure or vector store unavailable — empty hits with a note so
    # the model can distinguish "nothing found" from "couldn't search".
    return {
        "ok": True,
        "hits": [],
        "query": query,
        "note": "search temporarily degraded or no matches",
    }


def _summarize(result: Dict[str, Any]):
    if result.get("ok"):
        hits = result.get("hits") or []
        q = result.get("query") or ""
        if hits:
            return f"{len(hits)} file(s) for «{q}»", None
        return f"No matches for «{q}»", None
    return None


register(Tool(
    schema={
        "type": "function",
        "function": {
            "name": "folder_search",
            "description": (
                "Semantic search inside the assistant's working folder. Use "
                "it when the user describes a file by content/topic ('my "
                "files', 'in my folder', 'the doc about X') and does not "
                "give an exact filename. Returns hits with indexedFileId, "
                "filename and a snippet. DO NOT confuse with "
                "search_workspace, which covers notes/tasks/resources of "
                "the wider workspace. For multi-step folder operations "
                "(search → read → edit), consider folder_assistant.\n\n"
                "Pairs well with: folder_read (read a hit's full content), "
                "folder_write with overwrite=true (modify a hit), "
                "folder_delete (delete a hit)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language query describing what to find.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max hits to return. Default 10.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    execute=_execute,
    summarize=_summarize,
    agent_allowed=True,
    folder_scoped=True,
))
