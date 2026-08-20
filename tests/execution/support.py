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
