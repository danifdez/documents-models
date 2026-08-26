import json
from typing import Any, Dict


ACTIVE_CONTEXT_ARTIFACT_ROLE = "active_context"
ACTIVE_CONTEXT_SCHEMA = "active-context/1"
CHAT_TASK_TYPES = {"assistant-chat", "agent-chat"}


def effective_payload_from_active_context(
    task_type: str,
    assignment_payload: Dict[str, Any],
    artifacts: Dict[str, bytes],
) -> Dict[str, Any]:
    if task_type not in CHAT_TASK_TYPES:
        return dict(assignment_payload)
    body = artifacts.get(ACTIVE_CONTEXT_ARTIFACT_ROLE)
    if not isinstance(body, bytes):
        raise ValueError("Missing active context artifact")
    try:
        snapshot = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Invalid active context artifact") from error
    if not isinstance(snapshot, dict):
        raise ValueError("Invalid active context artifact")
    if snapshot.get("schemaVersion") != ACTIVE_CONTEXT_SCHEMA:
        raise ValueError("Unsupported active context artifact")
    effective_payload = snapshot.get("effectivePayload")
    if not isinstance(effective_payload, dict):
        raise ValueError("Active context payload must be an object")
    return dict(effective_payload)
