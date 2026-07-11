"""folder_write: create or overwrite a file in the working folder."""

import base64
from typing import Any, Dict

from agents.tool_base import Tool, ToolContext, register
from common.chat.http import http_json_with_status, post_tool_event
from .file_writers import (
    ConversionError,
    UnsupportedExtension,
    normalize_and_categorize,
    to_bytes as _writer_to_bytes,
)


def _execute(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    if not isinstance(ctx.owner_id, int):
        return {"error": "internal", "message": "missing owner context"}

    filename_raw = str(args.get("filename") or "")
    content = args.get("content")
    if not isinstance(content, str):
        return {"error": "internal", "message": "content must be a string"}

    # Categorize the target so we know whether the content needs conversion
    # (markdown → pdf/docx/odt, csv → xlsx) or is written verbatim.
    try:
        filename, category = normalize_and_categorize(filename_raw)
    except UnsupportedExtension as e:
        return {
            "error": "unsupported_extension",
            "filename": filename_raw,
            "extension": e.ext,
            "hint": (
                "Supported extensions: text files (.md, .txt, .csv, .json, "
                ".yaml, code…), documents (.pdf, .docx, .odt) where content "
                "must be markdown, and spreadsheets (.xlsx) where content "
                "must be CSV."
            ),
        }

    # Convert content to bytes according to category. For text we keep the
    # string and let the backend write UTF-8; for binary we send base64.
    body_payload: Dict[str, Any] = {"filename": filename}
    if category == "text":
        body_payload["content"] = content
    else:
        try:
            data = _writer_to_bytes(content, filename, category)
        except ConversionError as e:
            hint = {
                "pandoc_not_available": "Install pypandoc-binary in the worker.",
                "typst_not_available": "Install the typst pip package in the worker.",
                "openpyxl_not_available": "Install openpyxl in the worker.",
                "conversion_failed": (
                    "Check that the markdown is well-formed. For .pdf try "
                    "simpler markdown; for .xlsx ensure the content is valid "
                    "CSV with consistent column counts."
                ),
                "csv_parse_error": "The content was not valid CSV.",
            }.get(e.reason, "")
            return {
                "error": e.reason,
                "filename": filename,
                "detail": e.detail[:200] if e.detail else "",
                "hint": hint,
            }
        body_payload["contentBase64"] = base64.b64encode(data).decode("ascii")

    overwrite_requested = bool(args.get("overwrite"))

    # Always try create first. If the file does not exist, this is a clean
    # create with no destructive side-effect, regardless of `overwrite`.
    status, body = http_json_with_status(
        "POST",
        f"/{ctx.owner_segment}/{ctx.owner_id}/indexed-files",
        body_payload,
    )

    if status == 201 and isinstance(body, dict):
        return {
            "ok": True,
            "indexedFileId": body.get("id"),
            "filename": body.get("filename") or filename,
            "category": category,
        }

    if status == 409 and isinstance(body, dict) and body.get("error") == "file_exists":
        if not overwrite_requested:
            return {"error": "file_exists", "filename": filename}
        # Overwrite intended: hand the action to the user via a confirmation
        # card. The frontend's folder_overwrite handler reposts with overwrite.
        # For binary categories we ship contentBase64 in the payload so the
        # frontend doesn't have to re-run pandoc/openpyxl.
        if isinstance(ctx.job_id, int):
            payload: Dict[str, Any] = {"filename": filename}
            if "content" in body_payload:
                payload["content"] = body_payload["content"]
            else:
                payload["contentBase64"] = body_payload["contentBase64"]
            post_tool_event(
                ctx.owner_segment, ctx.owner_id, ctx.job_id, "folder_write", filename,
                status="pending_confirmation",
                summary=f"Pending: overwrite {filename}",
                kind="folder_overwrite",
                payload=payload,
                confirm_label="Confirm overwrite",
                cancel_label="Cancel",
            )
        return {"ok": True, "pendingConfirmation": True, "filename": filename}

    if status == 409 and isinstance(body, dict):
        return {"error": body.get("error") or "conflict", "filename": filename}
    if status == 400 and isinstance(body, dict):
        err = body.get("error") or body.get("message") or "bad_request"
        return {"error": err, "filename": filename}
    if status == 403 and isinstance(body, dict):
        return {"error": body.get("error") or "forbidden", "filename": filename}
    return {"error": "internal", "filename": filename, "status": status}


def _summarize(result: Dict[str, Any]):
    if result.get("ok"):
        fn = result.get("filename") or ""
        return (
            f"Wrote: {fn}",
            {"kind": "indexedFile", "id": result.get("indexedFileId"), "title": fn},
        )
    return None


register(Tool(
    schema={
        "type": "function",
        "function": {
            "name": "folder_write",
            "description": (
                "Create or overwrite a file in the assistant's working folder. "
                "Use it directly when the new content is already in hand — "
                "either provided by the user verbatim, or composed by you "
                "from the conversation. For tasks that require reading the "
                "existing file or searching for it first, prefer "
                "folder_assistant.\n\n"
                "Pairs well with: folder_search to verify a file exists "
                "before creating, folder_read to inspect the existing "
                "content before overwriting.\n\n"
                "The content you must emit depends on the filename's extension:\n"
                "\n"
                "  TEXT (.md, .markdown, .txt, .csv, .tsv, .json, .xml, .yaml, "
                ".yml, .toml, .ini, .log, .html, .htm, .svg, .py, .js, .ts, "
                ".tsx, .jsx, .sh, .bash, .sql, .css, .scss, .less, .go, .rs, "
                ".rb, .java, .c, .cpp, .h, .hpp, .cs, .php, .r, .kt, .swift, "
                ".dockerfile)\n"
                "    -> `content` is written verbatim as UTF-8.\n"
                "\n"
                "  DOCUMENT (.pdf, .docx, .odt)\n"
                "    -> `content` MUST be MARKDOWN source. The worker renders "
                "it to the target format with pandoc. Supports headings, "
                "lists, tables, code blocks, links, blockquotes. Embedded "
                "images are NOT supported in this version.\n"
                "\n"
                "  SPREADSHEET (.xlsx)\n"
                "    -> `content` MUST be CSV (comma-separated; first row = "
                "column headers). The worker converts via openpyxl. Numbers "
                "and floats are auto-detected; leading-zero strings stay as "
                "strings (phone numbers, zip codes are safe).\n"
                "\n"
                "If `filename` has no extension, `.md` is assumed.\n"
                "\n"
                "If the file already exists and overwrite=false you'll receive "
                "`file_exists` and must ask the user before retrying with "
                "overwrite=true. With overwrite=true on an existing file the "
                "user sees a confirmation card; the change is NOT applied "
                "until they confirm. Unknown extensions return "
                "`unsupported_extension`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": (
                            "Relative filename including extension, e.g. "
                            "'shopping-list.md', 'report.pdf', 'data.xlsx', "
                            "'script.py'. Subfolders allowed (e.g. "
                            "'notes/2025-01-15.md')."
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": (
                            "Content shaped according to the extension (see "
                            "tool description): UTF-8 text, markdown source "
                            "for .pdf/.docx/.odt, or CSV for .xlsx."
                        ),
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": (
                            "Set to true ONLY after the user has explicitly "
                            "agreed to overwrite an existing file. Default false."
                        ),
                    },
                },
                "required": ["filename", "content"],
            },
        },
    },
    execute=_execute,
    summarize=_summarize,
))
