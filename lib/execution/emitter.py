import base64
import contextvars
import hashlib
import json
import logging
import os
import re
import socket
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_ACTIVE_EMITTER = contextvars.ContextVar("execution_emitter", default=None)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_SECRET_VALUE_RE = re.compile(
    r"(?i)\b(access[_-]?token|api[_-]?key|auth[_-]?token|authorization|cookie|"
    r"id[_-]?token|password|refresh[_-]?token|session[_-]?token|token)"
    r"\s*[:=]\s*([^\s,;]+)"
)
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]+=*")
_FORBIDDEN_KEYS = {
    "accesstoken",
    "apikey",
    "authtoken",
    "authorization",
    "chainofthought",
    "cookie",
    "credential",
    "idtoken",
    "password",
    "refreshtoken",
    "secret",
    "sessiontoken",
    "token",
    "thoughts",
}
_MAX_ARTIFACT_BYTES = 1024 * 1024
CONTRACT_SET_HASH = "sha256:34107e825299d24137d2b5add08c42a6dd5b77b4c18b991e19e7ec32c8888d76"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _hash_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _redact_text(value: str) -> str:
    without_thinking = _THINK_RE.sub("", value)
    without_bearer = _BEARER_RE.sub("Bearer [REDACTED]", without_thinking)
    return _SECRET_VALUE_RE.sub(lambda match: f"{match.group(1)}=[REDACTED]", without_bearer)


def _safe_value(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, list):
        return [_safe_value(child) for child in value]
    if isinstance(value, dict):
        safe: Dict[str, Any] = {}
        for key, child in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
            if normalized in _FORBIDDEN_KEYS:
                safe[str(key)] = "[REDACTED]"
            elif normalized in {"reasoning", "reasoningcontent"}:
                continue
            else:
                safe[str(key)] = _safe_value(child)
        return safe
    if isinstance(value, float):
        return int(round(value))
    if value is None or isinstance(value, (bool, int)):
        return value
    return _redact_text(str(value))


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        _safe_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass
class OperationHandle:
    operation_id: str
    attempt_id: str
    kind: str
    name: str
    started_event_id: Optional[str]
    started_monotonic: float
    tool_call_id: Optional[str] = None


class ExecutionEmitter:
    def __init__(self, context: Optional[Dict[str, Any]]):
        self.context = context if isinstance(context, dict) else None
        self.token = os.environ.get("EXECUTION_INGEST_TOKEN", "")
        self.backend_url = os.environ.get("BACKEND_URL", "http://localhost:3000").rstrip("/")
        try:
            self.http_timeout = max(
                0.05,
                float(os.environ.get("EXECUTION_HTTP_TIMEOUT_SECONDS", "2")),
            )
        except ValueError:
            self.http_timeout = 2
        if self.context and not self.token:
            raise RuntimeError("EXECUTION_INGEST_TOKEN is required")
        self.producer_sequence = 0
        self.attempted_events = 0
        self.accepted_events = 0
        self.artifact_bytes = 0
        self.instrumentation_ms = 0
        self.errors = []
        self.pending_artifacts = []
        self.pending_events = []
        self.flushed = False
        self.last_event_id = self.context.get("causedByEventId") if self.context else None
        worker_id = os.environ.get(
            "WORKER_ID",
            f"{socket.gethostname()}-{os.getpid()}",
        )
        attempt_id = self.context.get("attemptId") if self.context else None
        execution_id = self.context.get("executionId") if self.context else None
        self.instance_id = f"{worker_id}:{attempt_id or execution_id or 'standalone'}"

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]):
        return cls((payload or {}).get("execution"))

    def summary(self) -> Dict[str, Any]:
        return {
            "attemptedEvents": self.attempted_events,
            "acceptedEvents": self.accepted_events,
            "artifactBytes": self.artifact_bytes,
            "instrumentationMs": self.instrumentation_ms,
            "errors": list(dict.fromkeys(self.errors)),
        }

    def attach_summary(self, result: Dict[str, Any]) -> Dict[str, Any]:
        value = dict(result)
        if self.context:
            value["executionTelemetry"] = self.summary()
        return value

    def start_inference(self, name: str, request: Any) -> Optional[OperationHandle]:
        artifact_id = self.record_artifact("materialized_prompt", request, "application/json")
        return self.start_operation(
            "inference",
            name,
            artifact_refs=[artifact_id] if artifact_id else [],
            input_artifact_id=artifact_id,
        )

    def finish_inference(
        self,
        handle: OperationHandle,
        response: Any,
        *,
        outcome: str,
        status: str = "succeeded",
        error: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        safe_response = _safe_value(response)
        artifact_id = self.record_artifact("model_response", safe_response, "application/json")
        result = {"responseArtifactId": artifact_id} if artifact_id else {}
        return self.finish_operation(
            handle,
            status=status,
            result=result,
            error=error,
            outcome=outcome,
            metrics=metrics,
            artifact_refs=[artifact_id] if artifact_id else [],
        )

    def start_tool(
        self,
        name: str,
        arguments: Any,
        provider_tool_call_id: Optional[str],
    ) -> Optional[OperationHandle]:
        tool_call_id = str(uuid.uuid4())
        handle = self.start_operation(
            "tool_call",
            name,
            extra_payload={
                "inputSummary": _safe_value(arguments),
                "providerToolCallId": provider_tool_call_id,
            },
            tool_call_id=tool_call_id,
        )
        return handle

    def observe_tool_result(self, handle: OperationHandle, result: Any) -> Optional[str]:
        artifact_id = self.record_artifact("tool_result", result, "application/json")
        if not artifact_id:
            return None
        source_id = str(uuid.uuid4())
        artifact_hash = _hash_bytes(_json_bytes(result))
        return self.emit(
            "source.observed",
            "source.observed/1",
            {
                "sourceId": source_id,
                "kind": "tool_output",
                "originComponent": "documents-models",
                "observedAt": _utc_now(),
                "contentHash": artifact_hash,
                "snapshotArtifactId": artifact_id,
                "trustLevel": "tool_observation",
                "dataClassification": "workspace",
            },
            actor={"type": "tool"},
            operation_id=handle.operation_id,
            attempt_id=handle.attempt_id,
            tool_call_id=handle.tool_call_id,
            source_id=source_id,
            caused_by=handle.started_event_id,
            artifact_refs=[artifact_id],
        )

    def finish_tool(
        self,
        handle: OperationHandle,
        result: Any,
        source_event_id: Optional[str],
        error: Optional[str] = None,
    ) -> Optional[str]:
        return self.finish_operation(
            handle,
            status="failed" if error else "succeeded",
            result=_safe_value(result),
            error=error,
            caused_by=source_event_id or handle.started_event_id,
        )

    def record_final_message(self, reply: str) -> Optional[str]:
        artifact_id = self.record_artifact("model_response", reply, "text/plain")
        return self.emit(
            "message.recorded",
            "message.recorded/1",
            {
                "messageKind": "final_response",
                "role": "assistant",
                "contentPreview": _redact_text(reply)[:512],
                "contentArtifactId": artifact_id,
                "format": "text",
            },
            actor={"type": "model"},
            artifact_refs=[artifact_id] if artifact_id else [],
        )

    def flush_evidence(self) -> None:
        self.flush()

    def start_operation(
        self,
        kind: str,
        name: str,
        *,
        artifact_refs=None,
        input_artifact_id: Optional[str] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
        tool_call_id: Optional[str] = None,
    ) -> Optional[OperationHandle]:
        operation_id = str(uuid.uuid4())
        attempt_id = str(uuid.uuid4())
        payload = {"operationKind": kind, "status": "dispatched", "name": name}
        if input_artifact_id:
            payload["inputArtifactId"] = input_artifact_id
        payload.update(extra_payload or {})
        started_event_id = self.emit(
            "operation.started",
            "operation.started/1",
            payload,
            actor={"type": "tool" if kind == "tool_call" else "model"},
            operation_id=operation_id,
            attempt_id=attempt_id,
            tool_call_id=tool_call_id,
            artifact_refs=artifact_refs or [],
        )
        return OperationHandle(
            operation_id=operation_id,
            attempt_id=attempt_id,
            kind=kind,
            name=name,
            started_event_id=started_event_id,
            started_monotonic=time.monotonic(),
            tool_call_id=tool_call_id,
        )

    def finish_operation(
        self,
        handle: OperationHandle,
        *,
        status: str,
        result: Any,
        error: Optional[str],
        outcome: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        caused_by: Optional[str] = None,
        artifact_refs=None,
    ) -> Optional[str]:
        measured = int(round((time.monotonic() - handle.started_monotonic) * 1000))
        normalized_metrics = {
            "durationMs": measured,
            "timeToFirstTokenMs": "unknown",
            "promptTokens": "unknown",
            "generatedTokens": "unknown",
        }
        for key, value in (metrics or {}).items():
            if key in normalized_metrics and (value == "unknown" or isinstance(value, int)):
                normalized_metrics[key] = value
        payload: Dict[str, Any] = {
            "operationKind": handle.kind,
            "status": status,
            "result": _safe_value(result),
            "error": (
                {"code": "OPERATION_FAILED", "message": _redact_text(error)}
                if error else None
            ),
            "metrics": normalized_metrics,
        }
        if outcome:
            payload["outcome"] = outcome
        return self.emit(
            "operation.finished",
            "operation.finished/1",
            payload,
            actor={"type": "tool" if handle.kind == "tool_call" else "model"},
            operation_id=handle.operation_id,
            attempt_id=handle.attempt_id,
            tool_call_id=handle.tool_call_id,
            caused_by=caused_by or handle.started_event_id,
            artifact_refs=artifact_refs or [],
        )

    def record_artifact(self, kind: str, value: Any, media_type: str) -> Optional[str]:
        if not self.context:
            return None
        safe_value = _redact_text(value) if isinstance(value, str) else _safe_value(value)
        body = safe_value.encode("utf-8") if isinstance(safe_value, str) else _json_bytes(safe_value)
        if len(body) > _MAX_ARTIFACT_BYTES:
            self.errors.append(f"artifact_omitted:size:{kind}:{len(body)}")
            return None
        artifact_id = str(uuid.uuid4())
        manifest = {
            "artifactId": artifact_id,
            "kind": kind,
            "contentHash": _hash_bytes(body),
            "size": len(body),
            "mediaType": media_type,
            "encoding": "identity",
            "dataClassification": "workspace",
            "redaction": {"applied": safe_value != value},
            "retentionClass": "evaluation",
            "inputSourceIds": [],
            "bodyBase64": base64.b64encode(body).decode("ascii"),
        }
        self.artifact_bytes += len(body)
        self.pending_artifacts.append(manifest)
        return artifact_id

    def emit(
        self,
        event_type: str,
        payload_schema: str,
        payload: Dict[str, Any],
        *,
        actor: Dict[str, Any],
        operation_id: Optional[str] = None,
        attempt_id: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        source_id: Optional[str] = None,
        caused_by: Optional[str] = None,
        artifact_refs=None,
    ) -> Optional[str]:
        if not self.context:
            return None
        self.producer_sequence += 1
        event_id = str(uuid.uuid4())
        event = {
            "eventId": event_id,
            "rootExecutionId": self.context["rootExecutionId"],
            "executionId": self.context["executionId"],
            "turnId": self.context.get("turnId"),
            "producerSequence": self.producer_sequence,
            "eventType": event_type,
            "producer": {
                "component": "documents-models",
                "instanceId": self.instance_id,
                "version": os.environ.get("MODELS_REVISION", "development"),
            },
            "actor": actor,
            "operationId": operation_id,
            "attemptId": attempt_id,
            "toolCallId": tool_call_id,
            "sourceId": source_id,
            "causedByEventId": caused_by or self.last_event_id,
            "occurredAt": _utc_now(),
            "payloadSchema": payload_schema,
            "payload": _safe_value(payload),
            "artifactRefs": [item for item in (artifact_refs or []) if item],
            "security": {
                "dataClassification": "workspace",
                "purpose": "evaluation",
                "allowedDestinations": ["documents", "ai-train"],
                "redactionApplied": True,
            },
        }
        event = {key: value for key, value in event.items() if value is not None}
        self.attempted_events += 1
        self.pending_events.append(event)
        self.last_event_id = event_id
        return event_id

    def flush(self) -> None:
        if self.flushed or not self.context:
            return
        self.flushed = True
        failed_artifacts = set()
        for start in range(0, len(self.pending_artifacts), 100):
            batch = self.pending_artifacts[start:start + 100]
            started = time.monotonic()
            response = self._post("artifacts", {"artifacts": batch})
            self.instrumentation_ms += int(round((time.monotonic() - started) * 1000))
            if response is None:
                failed_artifacts.update(item["artifactId"] for item in batch)
                self.errors.append(f"artifact_batch:{start // 100 + 1}")

        events = [self._without_failed_artifacts(event, failed_artifacts)
                  for event in self.pending_events]
        for start in range(0, len(events), 200):
            batch = events[start:start + 200]
            started = time.monotonic()
            response = self._post("events", {"events": batch})
            self.instrumentation_ms += int(round((time.monotonic() - started) * 1000))
            if response is None:
                self.errors.append(f"event_batch:{start // 200 + 1}")
                continue
            self.accepted_events += int(response.get("accepted", 0))
            self.accepted_events += int(response.get("duplicates", 0))

    def _without_failed_artifacts(self, event: Dict[str, Any], failed: set) -> Dict[str, Any]:
        if not failed:
            return event

        def scrub(value):
            if isinstance(value, list):
                return [
                    scrub(child) for child in value
                    if not (isinstance(child, str) and child in failed)
                ]
            if isinstance(value, dict):
                return {
                    key: scrub(child)
                    for key, child in value.items()
                    if not (
                        key.lower().endswith("artifactid")
                        and isinstance(child, str)
                        and child in failed
                    )
                }
            return value

        scrubbed = scrub(event)
        scrubbed["artifactRefs"] = [
            artifact_id for artifact_id in event.get("artifactRefs", [])
            if artifact_id not in failed
        ]
        return scrubbed

    def _post(self, suffix: str, body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.context:
            return None
        root_execution_id = self.context["rootExecutionId"]
        url = f"{self.backend_url}/executions/internal/{root_execution_id}/{suffix}"
        request = urllib.request.Request(
            url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-Execution-Ingest-Token": self.token,
            },
        )
        with urllib.request.urlopen(request, timeout=self.http_timeout) as response:
            raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def activate_emitter(emitter: ExecutionEmitter):
    return _ACTIVE_EMITTER.set(emitter)


def get_active_emitter() -> Optional[ExecutionEmitter]:
    return _ACTIVE_EMITTER.get()


def reset_emitter(token) -> None:
    _ACTIVE_EMITTER.reset(token)
