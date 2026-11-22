from lib.execution.code_identity import code_fingerprint
from lib.execution.runtime_identity import runtime_fingerprint


def step_result_base(assignment: dict) -> dict:
    return {
        "schemaVersion": "step-result/1",
        "executionId": assignment["executionId"],
        "stepId": assignment["stepId"],
        "operationId": assignment["operationId"],
        "attemptId": assignment["attemptId"],
        "stepKind": assignment.get("stepKind"),
        "codeFingerprint": code_fingerprint(),
        "runtimeFingerprint": runtime_fingerprint(),
        "artifactRefs": [],
    }


def cancelled_inference_fields() -> dict:
    return {
        "output": {
            "kind": "inference",
            "outcome": {"kind": "cancelled", "reason": "Execution cancelled"},
        },
        "usage": {
            "promptTokens": None,
            "completionTokens": None,
            "totalTokens": None,
        },
        "inference": {
            "effectiveModel": "unreported",
            "effectiveAdapter": None,
            "effectivePromptPackages": ["unreported"],
            "finishReason": "unknown",
            "inferenceMs": 0,
            "cacheOutcome": "unknown",
            "warnings": ["inference_metadata_unavailable_after_cancellation"],
        },
    }


def result_for_delivery(result: dict) -> dict:
    """Supply fields absent from cancelled inference records written by older workers."""
    if result.get("stepKind") == "inference" and result.get("status") == "cancelled":
        return {**cancelled_inference_fields(), **result}
    return result
