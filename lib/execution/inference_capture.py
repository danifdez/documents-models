"""Opt-in inference exchanges carried by the existing attempt artifact protocol."""

import base64
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
import re
import time
from urllib.parse import urlsplit

from lib.execution.output_artifact import json_output_artifact, prepare_output_artifacts
from lib.execution.step_result import step_result_base, cancelled_inference_fields

_CAPTURE = ContextVar("inference_capture", default=None)
MAX_BODY_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 4 * 1024 * 1024
MAX_CALLS = 4
KIND = "inference_exchange"


class CaptureError(RuntimeError):
    """A safe diagnostic: never includes engine bodies, URLs or credentials."""


def current_capture():
    return _CAPTURE.get()


def _now():
    return datetime.now(timezone.utc).isoformat()


class Capture:
    def __init__(self, assignment):
        self.identity = {key: assignment[key] for key in
                         ("executionId", "stepId", "operationId", "attemptId")}
        self.calls = []
        self.size = 0

    def begin(self, url, data, stream):
        parts = urlsplit(url)
        if stream or parts.path not in {"/completion", "/v1/chat/completions"} or parts.query or parts.fragment or parts.username or parts.password:
            raise CaptureError("Unsupported endpoint or streaming for inference capture")
        if len(self.calls) >= MAX_CALLS or len(data) > MAX_BODY_BYTES or self.size + len(data) > MAX_TOTAL_BYTES:
            raise CaptureError("Inference capture request limit exceeded")
        self.size += len(data)
        call = {"schemaVersion": "inference-exchange/1", **self.identity,
                "callIndex": len(self.calls) + 1, "startedAt": _now(),
                "status": "started", "request": {"method": "POST", "path": parts.path,
                "mediaType": "application/json", "bodyBase64": base64.b64encode(data).decode("ascii")},
                "response": None}
        self.calls.append(call)
        return Exchange(self, call)

    def artifacts(self):
        return [json_output_artifact(KIND, KIND, value) for value in self.calls]


class Exchange:
    def __init__(self, capture, value):
        self.capture, self.value = capture, value
        self.started = time.monotonic()

    def finish(self, status):
        self.value.update(status=status, finishedAt=_now(),
                          durationMs=max(0, int((time.monotonic() - self.started) * 1000)))

    def read(self, response, status):
        limit = min(MAX_BODY_BYTES, MAX_TOTAL_BYTES - self.capture.size)
        raw = response.read(limit + 1)
        if len(raw) > limit:
            self.value["response"] = {"status": status, "bodyBase64": None, "omissionReason": "size_limit"}
            self.finish("response_too_large")
            raise CaptureError("Inference capture response limit exceeded")
        self.capture.size += len(raw)
        self.value["response"] = {"status": status, "bodyBase64": base64.b64encode(raw).decode("ascii")}
        self.finish("completed" if 200 <= status < 300 else "http_error")
        return raw


def capture_assignment(function):
    @wraps(function)
    def wrapped(assignment, artifacts=None, output_artifacts=None):
        work = assignment.get("work") or {}
        grant = work.get("inferenceCapture")
        if grant is None:
            # Explicitly clear an outer context, including on reused handler processes.
            token = _CAPTURE.set(None)
            try:
                return function(assignment, artifacts, output_artifacts)
            finally:
                _CAPTURE.reset(token)
        valid = (assignment.get("stepKind") == "inference"
                 and isinstance(grant, dict) and grant.get("schemaVersion") == "inference-capture/1"
                 and grant.get("purpose") == "evaluation"
                 and isinstance(grant.get("authorizedBy"), str)
                 and grant["authorizedBy"] not in {"standalone", "system"}
                 and re.fullmatch(r"[A-Za-z0-9_-]{1,200}", grant["authorizedBy"])
                 and output_artifacts is not None)
        if not valid:
            result = {**step_result_base(assignment), "status": "failed",
                      "error": {"code": "INFERENCE_CAPTURE_DENIED", "message": "Inference capture authorization or artifact delivery is unavailable", "retryable": False}}
            if assignment.get("stepKind") == "inference":
                result.update(cancelled_inference_fields())
                result["output"]["outcome"] = {"kind": "failed"}
                result["inference"]["finishReason"] = "error"
            return result
        capture = Capture(assignment)
        token = _CAPTURE.set(capture)
        try:
            result = function(assignment, artifacts, output_artifacts)
        finally:
            _CAPTURE.reset(token)
        uploads, refs = prepare_output_artifacts(capture.artifacts())
        result["artifactRefs"].extend(refs)
        output_artifacts.extend(uploads)
        return result
    return wrapped
