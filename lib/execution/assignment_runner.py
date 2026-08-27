import logging
from datetime import datetime, timezone
from threading import Event
from typing import Callable

from lib.execution.code_identity import code_fingerprint
from lib.execution.handler_process_pool import (
    HandlerPreempted,
    HandlerProcessFailed,
    HandlerProcessPool,
)
from lib.execution.protocol_client import (
    ExecutionProtocolClient,
    ProtocolTransportError,
)
from lib.execution.result_outbox import ResultOutbox
from lib.execution.runtime_identity import runtime_fingerprint

logger = logging.getLogger(__name__)

HandlerExecutor = Callable[
    [dict, dict[str, bytes], Event], tuple[dict, list[dict]]
]


def run_assignment(
    client: ExecutionProtocolClient,
    outbox: ResultOutbox,
    assignment: dict,
    cancellation: Event,
    handler_executor: HandlerExecutor | None = None,
) -> None:
    attempt_id = assignment["attemptId"]
    control = client.read_control(attempt_id)
    if control.get("cancelled"):
        cancellation.set()

    if cancellation.is_set():
        outbox.store(_cancelled_result(assignment))
        return

    policy_error = _assignment_policy_error(assignment)
    if policy_error is not None:
        outbox.store(_policy_rejected_result(assignment, policy_error))
        return

    artifacts = {}
    for ref in assignment.get("inputArtifactRefs", []):
        if cancellation.is_set():
            outbox.store(_cancelled_result(assignment))
            return
        try:
            artifacts[ref["role"]] = client.download_artifact(
                attempt_id, ref["artifactId"]
            )
        except ProtocolTransportError:
            if cancellation.is_set() or client.read_control(attempt_id).get(
                "cancelled"
            ):
                cancellation.set()
                outbox.store(_cancelled_result(assignment))
                return
            raise
    if cancellation.is_set():
        outbox.store(_cancelled_result(assignment))
        return

    try:
        if handler_executor is None:
            result, output_artifacts = _execute_in_temporary_process(
                assignment,
                artifacts,
                cancellation,
            )
        else:
            result, output_artifacts = handler_executor(
                assignment,
                artifacts,
                cancellation,
            )
    except HandlerPreempted:
        outbox.store(_cancelled_result(assignment))
        return
    except HandlerProcessFailed as error:
        logger.error(
            "Handler process failed for attempt %s: %s",
            attempt_id,
            error,
        )
        return
    try:
        if client.read_control(attempt_id).get("cancelled"):
            cancellation.set()
    except ProtocolTransportError as error:
        logger.warning(
            "Could not refresh final control for attempt %s: %s",
            attempt_id,
            error,
        )
    if cancellation.is_set():
        result = _cancelled_result(assignment)
        output_artifacts = []
    outbox.store(result, output_artifacts)


def _execute_in_temporary_process(
    assignment: dict,
    artifacts: dict[str, bytes],
    cancellation: Event,
) -> tuple[dict, list[dict]]:
    pool = HandlerProcessPool(1)
    try:
        return pool.execute(assignment, artifacts, cancellation)
    finally:
        pool.close()


def _cancelled_result(assignment: dict) -> dict:
    return {
        "schemaVersion": "step-result/1",
        "executionId": assignment["executionId"],
        "stepId": assignment["stepId"],
        "operationId": assignment["operationId"],
        "attemptId": assignment["attemptId"],
        "stepKind": assignment["stepKind"],
        "status": "cancelled",
        "codeFingerprint": code_fingerprint(),
        "runtimeFingerprint": runtime_fingerprint(),
        "artifactRefs": [],
        "error": None,
    }


def _assignment_policy_error(assignment: dict) -> tuple[str, str] | None:
    for ref in assignment.get("inputArtifactRefs", []):
        policy = ref.get("dataPolicy") if isinstance(ref, dict) else None
        if not isinstance(policy, dict):
            return (
                "ARTIFACT_POLICY_MISSING",
                "Input artifact policy is required",
            )
        if "execution" not in policy.get("allowedPurposes", []):
            return (
                "ARTIFACT_PURPOSE_DENIED",
                "Input artifact is not authorized for execution",
            )
        if "documents-models" not in policy.get("allowedDestinations", []):
            return (
                "ARTIFACT_DESTINATION_DENIED",
                "Input artifact is not authorized for Models",
            )
        if policy.get("classification") == "secret":
            return (
                "SECRET_INPUT_REJECTED",
                "Secret content cannot be provided to Models",
            )
        if policy.get("classification") not in {
            "public",
            "workspace",
            "personal",
            "sensitive",
        }:
            return (
                "ARTIFACT_CLASSIFICATION_INVALID",
                "Input artifact classification is invalid",
            )
        if policy.get("retentionClass") not in {
            "operational",
            "diagnostic",
            "evaluation",
        }:
            return (
                "ARTIFACT_RETENTION_INVALID",
                "Input artifact retention policy is invalid",
            )
        if not isinstance(policy.get("sourceRefs"), list):
            return (
                "ARTIFACT_PROVENANCE_MISSING",
                "Input artifact provenance is required",
            )
        expires_at = policy.get("expiresAt")
        if expires_at is None:
            continue
        if not isinstance(expires_at, str):
            return (
                "ARTIFACT_EXPIRY_INVALID",
                "Input artifact expiry is invalid",
            )
        try:
            expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        except ValueError:
            return (
                "ARTIFACT_EXPIRY_INVALID",
                "Input artifact expiry is invalid",
            )
        if expiry.tzinfo is None:
            return (
                "ARTIFACT_EXPIRY_INVALID",
                "Input artifact expiry must include a timezone",
            )
        if expiry <= datetime.now(timezone.utc):
            return (
                "ARTIFACT_EXPIRED",
                "Input artifact has expired",
            )
    return None


def _policy_rejected_result(
    assignment: dict,
    policy_error: tuple[str, str],
) -> dict:
    code, message = policy_error
    result = {
        "schemaVersion": "step-result/1",
        "executionId": assignment["executionId"],
        "stepId": assignment["stepId"],
        "operationId": assignment["operationId"],
        "attemptId": assignment["attemptId"],
        "stepKind": assignment["stepKind"],
        "status": "failed",
        "codeFingerprint": code_fingerprint(),
        "runtimeFingerprint": runtime_fingerprint(),
        "artifactRefs": [],
        "error": {
            "code": code,
            "message": message,
            "retryable": False,
        },
    }
    if assignment["stepKind"] == "inference":
        result.update(
            {
                "output": {
                    "kind": "inference",
                    "outcome": {"kind": "failed"},
                },
                "usage": {
                    "promptTokens": None,
                    "completionTokens": None,
                    "totalTokens": None,
                },
                "inference": {
                    "effectiveModel": "not_executed",
                    "effectiveAdapter": None,
                    "effectivePromptPackages": ["not_executed"],
                    "finishReason": "policy_rejected",
                    "inferenceMs": 0,
                    "cacheOutcome": "bypass",
                    "warnings": [code],
                },
            }
        )
    return result
