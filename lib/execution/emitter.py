import base64
import contextvars
import hashlib
import json
import logging
import os
import re
import socket
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .ingest_client import ExecutionIngestClient

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
CONTRACT_SET_HASH = "sha256:13dbef9a0c6bbe1f902f796c2c31572819157e3f1a32aa32f491d10dab5bd6e2"


class InferenceBudgetDenied(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class ToolBudgetDenied(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


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
    def __init__(
        self,
        context: Optional[Dict[str, Any]],
        *,
        ingest_client=None,
    ):
        self.context = context if isinstance(context, dict) else None
        self.token = os.environ.get("EXECUTION_INGEST_TOKEN", "")
        if self.context and not self.token:
            raise RuntimeError("EXECUTION_INGEST_TOKEN is required")
        self._ingest_client = ingest_client or ExecutionIngestClient.from_environment(
            self.token
        )
        self.producer_sequence = 0
        self.attempted_events = 0
        self.accepted_events = 0
        self.artifact_bytes = 0
        self.instrumentation_ms = 0
        self.errors = []
        self.pending_artifacts = []
        self.pending_events = []
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

    def start_inference(
        self,
        name: str,
        request: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[OperationHandle]:
        operation_id = str(uuid.uuid4())
        attempt_id = str(uuid.uuid4())
        budget_payload: Dict[str, Any] = {}
        trace = metadata or {}
        if (
            self.context
            and trace.get("loopKind") == "top_level"
            and trace.get("budgetGrantId")
        ):
            reservation = self.reserve_operation_budget(
                grant_id=str(trace["budgetGrantId"]),
                loop_id=str(trace["loopId"]),
                operation_id=operation_id,
                operation_kind="inference",
                bucket=self._budget_bucket(str(trace.get("phase") or "")),
                phase=str(trace.get("phase") or ""),
                round=int(trace.get("round") or 1),
                name=name,
            )
            budget_payload = {
                "budgetGrantId": reservation["grantId"],
                "budgetReservationId": reservation["reservationId"],
                "budgetBucket": reservation["bucket"],
                "executionAttemptId": reservation["executionAttemptId"],
            }
        artifact_id = self.record_artifact("materialized_prompt", request, "application/json")
        return self.start_operation(
            "inference",
            name,
            artifact_refs=[artifact_id] if artifact_id else [],
            input_artifact_id=artifact_id,
            extra_payload={**_safe_value(trace), **budget_payload},
            operation_id=operation_id,
            attempt_id=attempt_id,
        )

    def request_progress_grant(self, request: Dict[str, Any]) -> Dict[str, Any]:
        response = self._post("progress/grants", request)
        grant = (response or {}).get("grant")
        event_id = (response or {}).get("eventId")
        if not isinstance(grant, dict) or not grant.get("grantId") or not event_id:
            raise RuntimeError("Required progress grant failed")
        self.last_event_id = str(event_id)
        return grant

    def reserve_operation_budget(
        self,
        *,
        grant_id: str,
        loop_id: str,
        operation_id: str,
        operation_kind: str,
        bucket: str,
        phase: str,
        round: int,
        name: str,
        tool_call_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.context or not self.context.get("attemptId"):
            raise RuntimeError("Execution attempt is required for budget reservation")
        self.flush_evidence()
        try:
            response = self._post("progress/reservations", {
                "executionId": self.context["executionId"],
                "loopId": loop_id,
                "grantId": grant_id,
                "operationId": operation_id,
                "operationKind": operation_kind,
                "bucket": bucket,
                "phase": phase,
                "round": round,
                "name": name,
                "executionAttemptId": self.context["attemptId"],
                **({"toolCallId": tool_call_id} if tool_call_id else {}),
            })
        except Exception as error:
            raise InferenceBudgetDenied("budget_reservation_failed") from error
        reservation = (response or {}).get("reservation")
        event_id = (response or {}).get("eventId")
        if not isinstance(reservation, dict) or not event_id:
            raise RuntimeError("Required inference budget reservation failed")
        self.last_event_id = str(event_id)
        if not response.get("granted"):
            reason = str(reservation.get("reason") or "budget_reservation_failed")
            if operation_kind == "tool_call":
                raise ToolBudgetDenied(reason)
            raise InferenceBudgetDenied(reason)
        return reservation

    @staticmethod
    def _budget_bucket(phase: str) -> str:
        if phase == "output_repair":
            return "repair"
        if phase == "forced_finalization":
            return "closing"
        return "normal"

    def record_progress_policy(self, policy: Dict[str, Any]) -> Optional[str]:
        previous_event_id = self.last_event_id
        event_id = self.emit(
            "progress.reported",
            "progress.reported/1",
            {
                "message": "Effective progress policy recorded",
                "kind": "policy_snapshot",
                "policy": policy,
            },
            actor={"type": "worker"},
        )
        self.last_event_id = previous_event_id
        return event_id

    def finish_inference(
        self,
        handle: OperationHandle,
        response: Any,
        *,
        outcome: str,
        status: str = "succeeded",
        error: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        raw_response: Any = None,
        reason: Optional[str] = None,
    ) -> Optional[str]:
        safe_response = _safe_value(response)
        response_artifact_id = self.record_artifact(
            "model_response", safe_response, "application/json"
        )
        raw_response_artifact_id = (
            self.record_artifact(
                "model_response_raw", raw_response, "application/json"
            )
            if raw_response is not None else None
        )
        result = {}
        artifact_refs = []
        if response_artifact_id:
            result["responseArtifactId"] = response_artifact_id
            artifact_refs.append(response_artifact_id)
        if raw_response_artifact_id:
            result["rawResponseArtifactId"] = raw_response_artifact_id
            artifact_refs.append(raw_response_artifact_id)
        return self.finish_operation(
            handle,
            status=status,
            result=result,
            error=error,
            outcome=outcome,
            reason=reason,
            metrics=metrics,
            artifact_refs=artifact_refs,
        )

    def start_tool(
        self,
        name: str,
        arguments: Any,
        provider_tool_call_id: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[OperationHandle]:
        tool_call_id = str(uuid.uuid4())
        operation_id = str(uuid.uuid4())
        attempt_id = str(uuid.uuid4())
        trace = _safe_value(metadata or {})
        budget_payload: Dict[str, Any] = {}
        if (
            self.context
            and trace.get("loopKind") == "top_level"
            and trace.get("budgetGrantId")
        ):
            reservation = self.reserve_operation_budget(
                grant_id=str(trace["budgetGrantId"]),
                loop_id=str(trace["loopId"]),
                operation_id=operation_id,
                operation_kind="tool_call",
                bucket="tool",
                phase=str(trace.get("phase") or ""),
                round=int(trace.get("round") or 1),
                name=name,
                tool_call_id=tool_call_id,
            )
            budget_payload = {
                "budgetGrantId": reservation["grantId"],
                "budgetReservationId": reservation["reservationId"],
                "budgetBucket": reservation["bucket"],
                "executionAttemptId": reservation["executionAttemptId"],
            }
        handle = self.start_operation(
            "tool_call",
            name,
            extra_payload={
                "inputSummary": _safe_value(arguments),
                "providerToolCallId": provider_tool_call_id,
                **trace,
                **budget_payload,
            },
            tool_call_id=tool_call_id,
            operation_id=operation_id,
            attempt_id=attempt_id,
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
        operation_id: Optional[str] = None,
        attempt_id: Optional[str] = None,
    ) -> Optional[OperationHandle]:
        operation_id = operation_id or str(uuid.uuid4())
        attempt_id = attempt_id or str(uuid.uuid4())
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
        reason: Optional[str] = None,
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
        if reason:
            payload["reason"] = reason
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
        if not self.context:
            return
        while self.pending_artifacts:
            batch = self.pending_artifacts[:100]
            started = time.monotonic()
            response = self._post("artifacts", {"artifacts": batch})
            self.instrumentation_ms += int(round((time.monotonic() - started) * 1000))
            if response is None:
                self.errors.append("artifact_batch")
                raise RuntimeError("Required execution artifact ingestion failed")
            del self.pending_artifacts[:len(batch)]

        while self.pending_events:
            batch = self.pending_events[:200]
            started = time.monotonic()
            response = self._post("events", {"events": batch})
            self.instrumentation_ms += int(round((time.monotonic() - started) * 1000))
            if response is None:
                self.errors.append("event_batch")
                raise RuntimeError("Required execution event ingestion failed")
            self.accepted_events += int(response.get("accepted", 0))
            self.accepted_events += int(response.get("duplicates", 0))
            del self.pending_events[:len(batch)]

    def _post(self, suffix: str, body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.context:
            return None
        return self._ingest_client.post(
            self.context["rootExecutionId"],
            suffix,
            body,
        )

def activate_emitter(emitter: ExecutionEmitter):
    return _ACTIVE_EMITTER.set(emitter)


def get_active_emitter() -> Optional[ExecutionEmitter]:
    return _ACTIVE_EMITTER.get()


def reset_emitter(token) -> None:
    _ACTIVE_EMITTER.reset(token)
