"""Prompt template for dataset.extract-row.

PROMPT_VERSION is persisted with every CellAnchor produced. Bump it
manually whenever the prompt body (prompts/extract_row.md) changes.
"""

import os
from typing import List, Optional

from services.prompts import load_prompt


PROMPT_VERSION = "v1-2026-05"

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")
_EXTRACT_ROW_PROMPT = load_prompt(_PROMPTS_DIR, "extract_row.md")


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

    return _EXTRACT_ROW_PROMPT.format(
        audio_clause=audio_clause,
        field_block=field_block,
        source_title=source_title,
        document_text=document_text,
    )
