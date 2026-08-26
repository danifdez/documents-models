import logging
from threading import Event

from lib.execution.code_identity import code_fingerprint
from lib.execution.protocol_client import (
    ExecutionProtocolClient,
    ProtocolTransportError,
)
from lib.execution.result_outbox import ResultOutbox
from lib.execution.runtime_identity import runtime_fingerprint
from lib.execution.step_executor import execute_assignment

logger = logging.getLogger(__name__)


def run_assignment(
    client: ExecutionProtocolClient,
    outbox: ResultOutbox,
    assignment: dict,
    cancellation: Event,
) -> None:
    attempt_id = assignment["attemptId"]
    control = client.read_control(attempt_id)
    if control.get("cancelled"):
        cancellation.set()

    if cancellation.is_set():
        outbox.store(_cancelled_result(assignment))
        return

    artifacts = {
        ref["role"]: client.download_artifact(
            attempt_id, ref["artifactId"]
        )
        for ref in assignment.get("inputArtifactRefs", [])
    }
    if cancellation.is_set():
        outbox.store(_cancelled_result(assignment))
        return

    output_artifacts = []
    result = execute_assignment(
        assignment,
        artifacts,
        output_artifacts=output_artifacts,
    )
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
