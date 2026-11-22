import logging
from typing import Any, Dict

from common.execution_registry import TASK_HANDLERS
from lib.execution.active_context import effective_payload_from_active_context
from lib.execution.outcome import InferenceOutcome
from lib.execution.inference_capture import capture_assignment, safe_model
from lib.execution.output_artifact import HandlerOutput, prepare_output_artifacts
from lib.execution.step_result import step_result_base
from lib.llm.config import get_task_config
from lib.llm.prompts import prompt_package_fingerprint
from utils.task_dispatch import call_handler, ensure_task_handler

from lib.telemetry import assignment_context

logger = logging.getLogger(__name__)

_JSON_VALUE_SCHEMA_ID = "https://documents.local/harness/v1/schemas/common.schema.json#/$defs/jsonValue"


def _inference_outcome(value):
    if isinstance(value, InferenceOutcome):
        return value.value
    # Plain handler values use the JSON value definition from the v1 result contract.
    return {
        "kind": "structured_result",
        "schemaId": _JSON_VALUE_SCHEMA_ID,
        "value": value,
    }


@assignment_context
@capture_assignment
def execute_assignment(
    assignment: Dict[str, Any],
    artifacts: Dict[str, bytes] | None = None,
    output_artifacts: list[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    work = assignment.get("work") or {}
    task_type = work.get("taskType")
    step_kind = assignment.get("stepKind")
    base = step_result_base(assignment)
    if not isinstance(task_type, str) or not ensure_task_handler(task_type):
        return _failed_result(
            base, task_type, "CAPABILITY_UNAVAILABLE", f"No handler for {task_type}",
        )

    handler = TASK_HANDLERS[task_type]
    payload = work.get("payload") or {}
    if not isinstance(payload, dict):
        return _failed_result(
            base, task_type, "INVALID_ASSIGNMENT", "work.payload must be an object",
        )
    try:
        handler_payload = effective_payload_from_active_context(
            task_type,
            payload,
            artifacts or {},
        )
        handler_payload["_task_type"] = task_type
        handler_payload["_input_artifacts"] = artifacts or {}
        value = call_handler(handler, handler_payload)
        produced_artifacts = []
        if isinstance(value, HandlerOutput):
            produced_artifacts, artifact_refs = prepare_output_artifacts(
                value.artifacts
            )
            value = value.value
        else:
            artifact_refs = []
        if output_artifacts is not None:
            output_artifacts.extend(produced_artifacts)
        success_base = {**base, "artifactRefs": artifact_refs}
        if step_kind == "inference":
            outcome = _inference_outcome(value)
            return {
                **success_base,
                "status": "succeeded",
                "output": {
                    "kind": "inference",
                    "outcome": outcome,
                },
                **_inference_metadata(task_type),
                "error": None,
            }
        return {
            **success_base,
            "status": "succeeded",
            "output": {"kind": step_kind, "value": value},
            "error": None,
        }
    except Exception as error:
        logger.exception("Step %s failed", assignment.get("stepId"))
        return _failed_result(base, task_type, "STEP_EXECUTION_FAILED", str(error))


def _failed_result(base, task_type, code, message):
    failed = {
        **base,
        "status": "failed",
        "error": {"code": code, "message": message, "retryable": False},
    }
    if base["stepKind"] == "inference":
        failed.update({
            "output": {"kind": "inference", "outcome": {"kind": "failed"}},
            **_inference_metadata(task_type if isinstance(task_type, str) else ""),
        })
    return failed


def _inference_metadata(task_type: str) -> Dict[str, Any]:
    try:
        config = get_task_config(task_type)
    except (ValueError, OSError):
        # A failed configuration must still produce a durable failed result.
        config = {}
    inference = {
        "requestedModel": safe_model(config.get("model")) or "unreported",
        "effectiveModel": "unreported",
        "effectivePromptPackages": [prompt_package_fingerprint()],
        "finishReason": "unknown",
        "inferenceMs": 0,
        "cacheOutcome": "unknown",
        "warnings": ["token_usage_unavailable"],
    }
    return {
        "usage": {
            "promptTokens": None,
            "completionTokens": None,
            "totalTokens": None,
        },
        "inference": inference,
    }
