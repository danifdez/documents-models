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
            return {
                "eventId": "00000000-0000-4000-8000-000000000010",
                "grant": {
                    "version": "1",
                    "grantId": "00000000-0000-4000-8000-000000000011",
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
            }
        if suffix == "progress/reservations":
            return {
                "granted": True,
                "eventId": "00000000-0000-4000-8000-000000000012",
                "reservation": {
                    "version": "1",
                    "reservationId": "00000000-0000-4000-8000-000000000013",
                    "grantId": copied["grantId"],
                    "operationId": copied["operationId"],
                    "executionAttemptId": copied["executionAttemptId"],
                    "bucket": copied["bucket"],
                    "phase": copied["phase"],
                    "round": copied["round"],
                    "name": copied["name"],
                    "status": "reserved",
                    "decidedAt": "2026-08-20T10:00:01Z",
                },
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
