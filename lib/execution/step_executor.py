import hashlib
import json
import logging
import re
import time
from typing import Any, Dict

from common.execution_registry import TASK_HANDLERS
from lib.execution.outcome import InferenceOutcome
from lib.llm.config import get_task_config
from lib.llm.prompts import prompt_package_fingerprint
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
    started_at = time.monotonic()
    try:
        handler_payload = dict(payload)
        handler_payload["_task_type"] = task_type
        handler_payload["_input_artifacts"] = artifacts or {}
        value = call_handler(handler, handler_payload)
        if step_kind == "inference":
            outcome = (
                value.value
                if isinstance(value, InferenceOutcome)
                else {
                    "kind": "structured_result",
                    "schemaId": f"{task_type}-output/1",
                    "value": value,
                }
            )
            return {
                **base,
                "status": "succeeded",
                "output": {
                    "kind": "inference",
                    "outcome": outcome,
                },
                **_inference_metadata(
                    task_type,
                    started_at,
                    "tool_calls"
                    if outcome.get("kind") == "tool_requests"
                    else "completed",
                ),
                "error": None,
            }
        return {
            **base,
            "status": "succeeded",
            "output": {"kind": step_kind, "value": value},
            "error": None,
        }
    except Exception as error:
        logger.exception("Step %s failed", assignment.get("stepId"))
        failed = {
            **base,
            "status": "failed",
            "error": {
                "code": "STEP_EXECUTION_FAILED",
                "message": str(error),
                "retryable": False,
            },
        }
        if step_kind == "inference":
            failed.update(
                {
                    "output": {
                        "kind": "inference",
                        "outcome": {"kind": "failed"},
                    },
                    **_inference_metadata(
                        task_type,
                        started_at,
                        "error",
                    ),
                }
            )
        return failed


def _inference_metadata(
    task_type: str,
    started_at: float,
    finish_reason: str,
) -> Dict[str, Any]:
    config = get_task_config(task_type)
    inference = {
        "effectiveModel": str(config.get("model") or "unreported"),
        "effectivePromptPackages": [prompt_package_fingerprint()],
        "finishReason": finish_reason,
        "inferenceMs": max(0, int((time.monotonic() - started_at) * 1000)),
        "cacheOutcome": "unknown",
        "warnings": ["token_usage_unavailable"],
    }
    if config.get("lora_path"):
        adapter = _adapter_fingerprint(config)
        if adapter:
            inference["effectiveAdapter"] = adapter
    else:
        inference["effectiveAdapter"] = None
    return {
        "usage": {
            "promptTokens": None,
            "completionTokens": None,
            "totalTokens": None,
        },
        "inference": inference,
    }


def _adapter_fingerprint(config: Dict[str, Any]) -> str | None:
    sha256 = str(config.get("_deployment_sha256") or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", sha256):
        return None
    try:
        scale = float(config.get("lora_scale", 1.0))
    except (TypeError, ValueError):
        return None
    identity = {"available": True, "scale": scale, "sha256": sha256}
    encoded = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()
