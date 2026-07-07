"""folder_delete: delete a file from the assistant's working folder."""

from typing import Any, Dict

from .base import Tool, ToolContext, register
from common.chat.http import post_tool_event
from common.chat.folder import resolve_folder_target


def _execute(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    if not isinstance(ctx.owner_id, int):
        return {"error": "internal", "message": "missing owner context"}

    target = resolve_folder_target(args, ctx.owner_segment, ctx.owner_id)
    if not target.get("ok"):
        return target

    indexed_file_id = target.get("indexedFileId")
    filename = target.get("filename") or ""

    if isinstance(ctx.job_id, int):
        post_tool_event(
            ctx.owner_segment, ctx.owner_id, ctx.job_id, "folder_delete", filename,
            status="pending_confirmation",
            summary=f"Pending: delete {filename}",
            kind="folder_delete",
            payload={"indexedFileId": indexed_file_id, "filename": filename},
            confirm_label="Confirm delete",
            cancel_label="Cancel",
        )

    return {
        "ok": True,
        "pendingConfirmation": True,
        "indexedFileId": indexed_file_id,
        "filename": filename,
    }


register(Tool(
    schema={
        "type": "function",
        "function": {
            "name": "folder_delete",
            "description": (
                "Delete a file from the assistant's working folder. Identify "
                "it by indexedFileId (preferred) or filename. ALWAYS shows a "
                "confirmation card to the user: the deletion happens only "
                "after they confirm. If the filename is ambiguous you'll "
                "receive a list of candidates and must ask the user. Use it "
                "directly for one-shot deletes. For chained operations "
                "(locate-by-content then delete), consider folder_assistant.\n\n"
                "Pairs well with: folder_search to find the file when the "
                "user does not give an exact filename."
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
    agent_allowed=True,
    folder_scoped=True,
))
