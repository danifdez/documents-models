import logging
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
