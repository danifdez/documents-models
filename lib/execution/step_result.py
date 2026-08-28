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
