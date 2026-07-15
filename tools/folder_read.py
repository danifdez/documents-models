"""folder_read: read the full content of a file in the working folder."""

import urllib.parse
from typing import Any, Dict

from lib.framework.tool import Tool, ToolContext, register
from lib.backend.http import http_json_with_status


def _execute(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    if not isinstance(ctx.owner_id, int):
        return {"error": "internal", "message": "missing owner context"}

    indexed_file_id = args.get("indexedFileId")
    filename_raw = args.get("filename")

    if isinstance(indexed_file_id, int):
        path = f"/{ctx.owner_segment}/{ctx.owner_id}/indexed-files/{indexed_file_id}/content"
    elif isinstance(filename_raw, str) and filename_raw.strip():
        # urllib.parse.quote keeps slashes by default; force quoting all chars
        # so the filename arrives intact even when it contains spaces or '#'.
        encoded = urllib.parse.quote(filename_raw.strip(), safe="")
        path = f"/{ctx.owner_segment}/{ctx.owner_id}/indexed-files/by-filename?filename={encoded}"
    else:
        return {"error": "bad_request", "message": "indexedFileId or filename required"}

    status, body = http_json_with_status("GET", path)

    if status == 200 and isinstance(body, dict):
        return body
    if status in (404, 409, 422) and isinstance(body, dict):
        return body
    if status == 202 and isinstance(body, dict):
        return body
    return {"error": "internal", "status": status}


def _summarize(result: Dict[str, Any]):
    if result.get("ok"):
        fn = result.get("filename") or ""
        summary = f"Read: {fn}"
        if result.get("derivedFromExtraction"):
            summary += " (extracted)"
        return summary, None
    return None


register(Tool(
    schema={
        "type": "function",
        "function": {
            "name": "folder_read",
            "description": (
                "Read the full content of a file in the assistant's working "
                "folder. Identify it by indexedFileId (preferred, when you "
                "saw it in a previous folder_search or folder_write call) or "
                "by filename. For non-text files (PDF, etc.) you receive the "
                "extracted text; derivedFromExtraction is set so you know. "
                "Use it for a single read-and-answer. If the goal is "
                "read-and-modify, prefer folder_assistant — it absorbs the "
                "file content internally so it does not flood your context.\n\n"
                "Pairs well with: folder_search (yields the indexedFileId), "
                "folder_write with overwrite=true (modify after reading)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "indexedFileId": {
                        "type": "integer",
                        "description": "Numeric id of the file (preferred).",
                    },
                    "filename": {
                        "type": "string",
                        "description": "Relative filename, e.g. 'shopping-list.md'.",
                    },
                },
            },
        },
    },
    execute=_execute,
    summarize=_summarize,
))
