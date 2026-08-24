import logging
from typing import Any, Dict

from common.execution_registry import TASK_HANDLERS
from utils.task_dispatch import call_handler, ensure_task_handler

logger = logging.getLogger(__name__)


def execute_assignment(
    assignment: Dict[str, Any], artifacts: Dict[str, bytes] | None = None
) -> Dict[str, Any]:
    work = assignment.get("work") or {}
    task_type = work.get("taskType")
    step_kind = assignment.get("stepKind")
    base = {
        "schemaVersion": "step-result/1",
        "executionId": assignment["executionId"],
        "stepId": assignment["stepId"],
        "operationId": assignment["operationId"],
        "attemptId": assignment["attemptId"],
        "stepKind": step_kind,
        "artifactRefs": [],
    }
    if not isinstance(task_type, str) or not ensure_task_handler(task_type):
        return {
            **base,
            "status": "failed",
            "error": {
                "code": "CAPABILITY_UNAVAILABLE",
                "message": f"No handler for {task_type}",
                "retryable": False,
            },
        }

    handler = TASK_HANDLERS[task_type]
    payload = work.get("payload") or {}
    if not isinstance(payload, dict):
        return {
            **base,
            "status": "failed",
            "error": {
                "code": "INVALID_ASSIGNMENT",
                "message": "work.payload must be an object",
                "retryable": False,
            },
        }
    try:
        handler_payload = dict(payload)
        handler_payload["_input_artifacts"] = artifacts or {}
        value = call_handler(handler, handler_payload)
        return {
            **base,
            "status": "succeeded",
            "output": {"kind": step_kind, "value": value},
            "error": None,
        }
    except Exception as error:
        logger.exception("Step %s failed", assignment.get("stepId"))
        return {
            **base,
            "status": "failed",
            "error": {
                "code": "STEP_EXECUTION_FAILED",
                "message": str(error),
                "retryable": False,
            },
        }
