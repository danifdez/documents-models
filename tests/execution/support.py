import copy
from typing import Any, Callable, Dict, Optional

from lib.execution.emitter import ExecutionEmitter


class RecordingIngestClient:
    def __init__(
        self,
        responder: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
    ):
        self.responder = responder
        self.requests = []
        self.sent_events = []
        self.sent_artifacts = []
        self.budget_states = {}

    def post(self, root_execution_id, suffix, body):
        copied = copy.deepcopy(body)
        self.requests.append((suffix, copied))
        if suffix == "events":
            self.sent_events.extend(copied["events"])
        elif suffix == "artifacts":
            self.sent_artifacts.extend(copied["artifacts"])
        if self.responder:
            return self.responder(suffix, copied)
        if suffix == "progress/grants":
            requested = copied["requestedPolicy"]
            effective = copy.deepcopy(requested)
            normal = int(effective.get("normal", 0))
            normal_soft = int(effective.get("normalInferenceSoftLimit", 0))
            effective["normalInferenceSoftLimit"] = (
                0 if normal <= 1 or normal_soft <= 0
                else min(normal_soft, normal - 1)
            )
            tool_calls = int(effective.get("toolCalls", 0))
            tool_soft = int(effective.get("toolCallSoftLimit", 0))
            effective["toolCallSoftLimit"] = (
                0 if tool_calls <= 1 or tool_soft <= 0
                else min(tool_soft, tool_calls - 1)
            )
            grant_id = "00000000-0000-4000-8000-000000000011"
            self.budget_states[grant_id] = {
                "normal": {
                    "granted": effective.get("normal", 0),
                    "reserved": 0,
                    "consumed": 0,
                    "available": effective.get("normal", 0),
                    "softLimit": effective.get(
                        "normalInferenceSoftLimit", 0
                    ),
                    "softLimitReached": False,
                    "softLimitWarningPending": False,
                },
                "tool": {
                    "granted": effective.get("toolCalls", 0),
                    "reserved": 0,
                    "consumed": 0,
                    "available": effective.get("toolCalls", 0),
                    "softLimit": effective.get("toolCallSoftLimit", 0),
                    "softLimitReached": False,
                },
            }
            return {
                "eventId": "00000000-0000-4000-8000-000000000010",
                "grant": {
                    "version": "1",
                    "grantId": grant_id,
                    "executionId": copied["executionId"],
                    "turnId": copied.get("turnId"),
                    "loopId": copied["loopId"],
                    "executionAttemptId": copied["executionAttemptId"],
                    "profileId": "documents_chat_v1",
                    "policyVersion": "1",
                    "requestedPolicy": requested,
                    "effectivePolicy": effective,
                    "grantedAt": "2026-08-20T10:00:00Z",
                },
                "budgetState": copy.deepcopy(self.budget_states[grant_id]),
            }
        if suffix == "progress/reservations":
            state = self.budget_states.get(copied["grantId"], {
                "normal": {
                    "granted": 0,
                    "reserved": 0,
                    "consumed": 0,
                    "available": 0,
                    "softLimit": 0,
                    "softLimitReached": False,
                    "softLimitWarningPending": False,
                },
                "tool": {
                    "granted": 0,
                    "reserved": 0,
                    "consumed": 0,
                    "available": 0,
                    "softLimit": 0,
                    "softLimitReached": False,
                },
            })
            signal = None
            bucket = copied["bucket"]
            usage = state.get(bucket)
            if usage is not None:
                usage["reserved"] += 1
                usage["available"] = max(0, usage["available"] - 1)
                if (
                    usage.get("softLimit", 0) > 0
                    and not usage.get("softLimitReached", False)
                    and usage["reserved"] + usage["consumed"]
                    >= usage["softLimit"]
                ):
                    usage["softLimitReached"] = True
                    if bucket == "normal":
                        usage["softLimitWarningPending"] = True
                    signal = {
                        "version": "1",
                        "grantId": copied["grantId"],
                        "operationKind": copied["operationKind"],
                        "bucket": bucket,
                        "softLimit": usage["softLimit"],
                        "hardLimit": usage["granted"],
                        "committed": usage["reserved"] + usage["consumed"],
                        "available": usage["available"],
                        "triggeringOperationId": copied["operationId"],
                        "executionAttemptId": copied["executionAttemptId"],
                        "decidedAt": "2026-08-20T10:00:01Z",
                    }
            return {
                "granted": True,
                "eventId": "00000000-0000-4000-8000-000000000012",
                "reservation": {
                    "version": "1",
                    "reservationId": "00000000-0000-4000-8000-000000000013",
                    "grantId": copied["grantId"],
                    "operationId": copied["operationId"],
                    "executionAttemptId": copied["executionAttemptId"],
                    "operationKind": copied["operationKind"],
                    "bucket": copied["bucket"],
                    **(
                        {"toolCallId": copied["toolCallId"]}
                        if copied.get("toolCallId") else {}
                    ),
                    "phase": copied["phase"],
                    "round": copied["round"],
                    "name": copied["name"],
                    "status": "reserved",
                    "decidedAt": "2026-08-20T10:00:01Z",
                },
                "budgetState": copy.deepcopy(state),
                **({"softLimitSignal": signal} if signal else {}),
            }
        items = copied.get("events", copied.get("artifacts", []))
        return {"accepted": len(items), "duplicates": 0}


class RecordingExecutionEmitter(ExecutionEmitter):
    def __init__(self, context, responder=None):
        self.recording_client = RecordingIngestClient(responder)
        super().__init__(context, ingest_client=self.recording_client)

    @property
    def sent_events(self):
        return self.recording_client.sent_events

    @property
    def sent_artifacts(self):
        return self.recording_client.sent_artifacts
