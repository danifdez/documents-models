"""get_resource_content: read the full text of a single indexed resource."""

from typing import Any, Dict

from .base import Tool, ToolContext, register
from common.chat.http import http_json


def _execute(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    rid = args.get("resourceId")
    if not isinstance(rid, int):
        return {"error": "integer resourceId required"}
    data = http_json("GET", f"/resources/{rid}/content")
    if not isinstance(data, dict):
        return {"error": "resource not found"}
    content = data.get("content")
    if not isinstance(content, str):
        return {"resourceId": rid, "content": None, "note": "no extracted content"}
    # Cap the content to keep the prompt manageable. The model can ask for
    # more (paged read) in a follow-up if we ever expose it.
    MAX_CONTENT_CHARS = 6000
    truncated = len(content) > MAX_CONTENT_CHARS
    return {
        "resourceId": rid,
        "content": content[:MAX_CONTENT_CHARS],
        "truncated": truncated,
    }


def _summarize(result: Dict[str, Any]):
    n = len(result.get("content") or "")
    return f"{n} chars read" + (" (truncated)" if result.get("truncated") else ""), None


register(Tool(
    schema={
        "type": "function",
        "function": {
            "name": "get_resource_content",
            "description": (
                "Read the full text of a single indexed resource by id. Use "
                "it when you already have a resourceId (from a prior "
                "search_workspace) and need its content to answer. If you "
                "need to read or summarise several resources, prefer "
                "workspace_research — it does the chaining internally and "
                "returns one compact summary instead of inflating your "
                "context with raw text.\n\n"
                "Pairs well with: search_workspace (which yields the "
                "resourceId you pass in here)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "resourceId": {
                        "type": "integer",
                        "description": "Resource ID (returned by search_workspace when collection='resources').",
                    },
                },
                "required": ["resourceId"],
            },
        },
    },
    execute=_execute,
    summarize=_summarize,
    hidden_from_main=True,
))
