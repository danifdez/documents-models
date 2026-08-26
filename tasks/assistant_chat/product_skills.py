import hashlib
import json
from typing import Any, Dict, List


WORKSPACE_DOCUMENT_SKILL_ID = "workspace-document-workflow"
WORKSPACE_DOCUMENT_SKILL_VERSION = "workspace-document-workflow/1"
WORKSPACE_DOCUMENT_SKILL_TITLE = "Workspace document workflow"
WORKSPACE_DOCUMENT_SKILL_DESCRIPTION = (
    "Discover, inspect, create, edit, or remove documents in the optional "
    "workspace folder."
)
WORKSPACE_DOCUMENT_SKILL_INSTRUCTIONS = """Use the configured workspace folder as an optional source of context and as the place where requested file changes are applied.

Discover before acting: list files when the relevant path is unknown, search indexed content to locate evidence, and read concrete files before modifying them. Preserve the existing format and unrelated content when editing. A document may be text or binary; use UTF-8 content for text formats and base64 bytes for binary formats.

Writing or deleting is allowed only when the user's explicit request requires that effect. A skill never grants a tool, permission, confirmation, path, or broader data access. If a required workspace tool is absent, explain the limitation instead of inventing an effect."""


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


_PRODUCT_SKILLS = {
    (WORKSPACE_DOCUMENT_SKILL_ID, WORKSPACE_DOCUMENT_SKILL_VERSION): {
        "title": WORKSPACE_DOCUMENT_SKILL_TITLE,
        "description": WORKSPACE_DOCUMENT_SKILL_DESCRIPTION,
        "contentHash": _canonical_hash(WORKSPACE_DOCUMENT_SKILL_INSTRUCTIONS),
        "instructions": WORKSPACE_DOCUMENT_SKILL_INSTRUCTIONS,
    }
}


def resolve_active_skill_instructions(value: Any) -> List[str]:
    if not isinstance(value, dict):
        raise ValueError("Missing active capability set")
    skills = value.get("skills")
    if not isinstance(skills, list) or len(skills) > 4:
        raise ValueError("Invalid active skill selection")

    resolved = []
    seen = set()
    for selection in skills:
        if not isinstance(selection, dict):
            raise ValueError("Invalid active skill selection")
        identity = (selection.get("skillId"), selection.get("version"))
        definition: Dict[str, str] | None = _PRODUCT_SKILLS.get(identity)
        if (
            definition is None
            or identity in seen
            or selection.get("title") != definition["title"]
            or selection.get("description") != definition["description"]
            or selection.get("contentHash") != definition["contentHash"]
            or selection.get("activationReason") != "objective_match"
            or set(selection) != {
                "skillId",
                "version",
                "title",
                "description",
                "contentHash",
                "activationReason",
            }
        ):
            raise ValueError("Unsupported active skill selection")
        seen.add(identity)
        resolved.append(definition["instructions"])
    return resolved
