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
            grant_id = "00000000-0000-4000-8000-000000000011"
            self.budget_states[grant_id] = {
                "tool": {
                    "granted": requested.get("toolCalls", 0),
                    "reserved": 0,
                    "consumed": 0,
                    "available": requested.get("toolCalls", 0),
                    "softLimit": requested.get("toolCallSoftLimit", 0),
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
                    "effectivePolicy": requested,
                    "grantedAt": "2026-08-20T10:00:00Z",
                },
                "budgetState": copy.deepcopy(self.budget_states[grant_id]),
            }
        if suffix == "progress/reservations":
            state = self.budget_states.get(copied["grantId"], {"tool": {
                "granted": 0,
                "reserved": 0,
                "consumed": 0,
                "available": 0,
                "softLimit": 0,
                "softLimitReached": False,
            }})
            signal = None
            if copied["operationKind"] == "tool_call":
                tool = state["tool"]
                tool["reserved"] += 1
                tool["available"] = max(0, tool["available"] - 1)
                if (
                    tool["softLimit"] > 0
                    and not tool["softLimitReached"]
                    and tool["reserved"] + tool["consumed"] >= tool["softLimit"]
                ):
                    tool["softLimitReached"] = True
                    signal = {
                        "version": "1",
                        "grantId": copied["grantId"],
                        "operationKind": "tool_call",
                        "bucket": "tool",
                        "softLimit": tool["softLimit"],
                        "hardLimit": tool["granted"],
                        "committed": tool["reserved"] + tool["consumed"],
                        "available": tool["available"],
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
