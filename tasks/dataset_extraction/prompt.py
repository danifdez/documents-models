"""Prompt template for dataset.extract-row.

PROMPT_VERSION is persisted with every CellAnchor produced. Bump it
manually whenever the prompt body changes
"""

from typing import List, Optional


PROMPT_VERSION = "v1-2026-05"


def _format_field(field: dict) -> str:
    lines = [
        f"  - key: \"{field['key']}\"",
        f"    name: \"{field.get('name', field['key'])}\"",
        f"    type: \"{field.get('type', 'text')}\"",
        f"    description: \"{field.get('description', '')}\"",
    ]
    if field.get("type") == "select" and field.get("options"):
        opts = ", ".join(f"\"{o}\"" for o in field["options"])
        lines.append(f"    options: [{opts}]")
    return "\n".join(lines)


def build_prompt(
    fields_to_extract: List[dict],
    document_text: str,
    source_title: str,
    *,
    is_audio: bool = False,
) -> str:
    field_block = "\n".join(_format_field(f) for f in fields_to_extract)

    audio_clause = (
        "This document is an audio transcript. Page numbers are not applicable; "
        "always set _page to null.\n\n"
        if is_audio
        else ""
    )

    return (
        "You are extracting structured data from a single document. For each "
        "field below, output an object with three keys:\n"
        "  - \"value\": the value found in the document, or null if the document "
        "does not contain it.\n"
        "  - \"_quote\": the verbatim sentence or short passage from the document "
        "that supports the value (max 280 characters). Empty string if value is null.\n"
        "  - \"_page\": the page number if you can determine it from the text, or "
        "null otherwise.\n\n"
        "HARD RULES:\n"
        "- Do NOT use general knowledge outside the document.\n"
        "- If the document does not explicitly or by clear inference contain the "
        "value, set value to null AND _quote to \"\" AND _page to null.\n"
        "- Never invent quotes. The _quote must be a verbatim substring of the "
        "document content shown below.\n\n"
        f"{audio_clause}"
        "Fields to extract:\n"
        f"{field_block}\n\n"
        f"Document title: {source_title}\n"
        "Document content:\n"
        "---\n"
        f"{document_text}\n"
        "---\n\n"
        "Return a single JSON object with one key per field above (in the same "
        "order). Adhere strictly to the provided grammar."
    )
