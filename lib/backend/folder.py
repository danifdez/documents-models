"""Shared working-folder helper used by the folder_* tools."""

import urllib.parse
from typing import Any, Dict

from .http import http_json_with_status


def resolve_folder_target(
    args: Dict[str, Any], owner_segment: str, owner_id: int,
) -> Dict[str, Any]:
    """Resolve a `folder_*` tool argument set to a concrete IndexedFile, using
    the same flow as folder_read (by id, then by exact filename, then by
    basename). Returns `{ok: True, indexedFileId, filename}` or an error dict.
    """
    indexed_file_id = args.get("indexedFileId")
    filename_raw = args.get("filename")

    if isinstance(indexed_file_id, int):
        path = f"/{owner_segment}/{owner_id}/indexed-files/{indexed_file_id}/content"
    elif isinstance(filename_raw, str) and filename_raw.strip():
        encoded = urllib.parse.quote(filename_raw.strip(), safe="")
        path = f"/{owner_segment}/{owner_id}/indexed-files/by-filename?filename={encoded}"
    else:
        return {"error": "bad_request", "message": "indexedFileId or filename required"}

    status, body = http_json_with_status("GET", path)
    if status == 200 and isinstance(body, dict) and body.get("ok"):
        return {
            "ok": True,
            "indexedFileId": body.get("indexedFileId"),
            "filename": body.get("filename"),
        }
    if status == 202 and isinstance(body, dict):
        return {
            "ok": True,
            "indexedFileId": body.get("indexedFileId"),
            "filename": body.get("filename"),
        }
    if status in (404, 409, 422) and isinstance(body, dict):
        return body
    return {"error": "internal", "status": status}
