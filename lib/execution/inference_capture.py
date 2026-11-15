"""Body-free engine metadata and opt-in canonical attempt artifacts."""

import base64
from contextvars import ContextVar
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
import re
import math
import time
from urllib.parse import urlsplit

from common.canonical_json import canonical_sha256
from lib.execution.engine_configuration import generation_defaults

from lib.execution.output_artifact import json_output_artifact, prepare_output_artifacts
from lib.execution.step_result import step_result_base, cancelled_inference_fields

_CAPTURE = ContextVar("inference_capture", default=None)
MAX_BODY_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 4 * 1024 * 1024
MAX_CALLS = 4
MAX_METADATA_CALLS = 128
KIND = "inference_exchange"


class CaptureError(RuntimeError):
    """A safe diagnostic: never includes engine bodies, URLs or credentials."""


def current_capture():
    context = _CAPTURE.get()
    return context if context is not None and context.enabled else None


def _now():
    return datetime.now(timezone.utc).isoformat()


def safe_model(value):
    if not isinstance(value, str) or '://' in value:
        return None
    # Engines may report an absolute model path. Keep only its public label.
    value = value.replace('\\', '/').rsplit('/', 1)[-1]
    return value if re.fullmatch(r'[A-Za-z0-9_.:-]{1,120}', value) else None


def _counter(value):
    return value if type(value) is int and 0 <= value <= 1_000_000 else None


def engine_properties(data):
    """Allowlisted server declarations; paths and template content never leave here."""
    data = data if isinstance(data, dict) else {}
    properties = {}
    model = safe_model(data.get('model_path') or data.get('model_alias'))
    if model:
        properties['modelLabel'] = model
    build = data.get('build_info')
    match = re.fullmatch(r'b([0-9]{1,9})-([0-9a-f]{7,40})(-dirty)?', build) if isinstance(build, str) else None
    if match:
        properties.update(buildNumber=int(match[1]), buildCommit=match[2])
        if match[3]:
            properties['buildDirty'] = True
    defaults = data.get('default_generation_settings')
    defaults = defaults if isinstance(defaults, dict) else {}
    for name, value, minimum in (('contextSize', defaults.get('n_ctx'), 1),
                                  ('slots', data.get('total_slots'), 1),
                                  ('modelFileType', data.get('model_ftype'), 0)):
        if type(value) is int and minimum <= value <= 2**31 - 1:
            properties[name] = value
    for source, name in (('chat_template', 'chatTemplateHash'), ('chat_template_tool_use', 'toolTemplateHash')):
        template = data.get(source)
        if isinstance(template, str) and 0 < len(template) <= 256 * 1024:
            try:
                properties[name] = canonical_sha256(template)
            except UnicodeError:
                pass
    generation = generation_defaults(data)
    if not properties and generation is None:
        return None
    result = {'source': 'server_props', 'properties': properties}
    if generation is not None:
        result['generationDefaults'] = generation
    return result


def response_metadata(response):
    """Project only bounded, body-free engine fields for results and telemetry."""
    response = response if isinstance(response, dict) else {}
    usage = response.get('usage')
    usage = usage if isinstance(usage, dict) else {}
    prompt = _counter(usage.get('prompt_tokens'))
    if prompt is None:
        prompt = _counter(response.get('tokens_evaluated'))
    completion = _counter(usage.get('completion_tokens'))
    if completion is None:
        completion = _counter(response.get('tokens_predicted'))
    total = _counter(usage.get('total_tokens'))
    if total is None and prompt is not None and completion is not None:
        total = _counter(prompt + completion)
    choices = response.get('choices')
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    reason = choice.get('finish_reason')
    if not isinstance(reason, str) or reason not in {'stop', 'length', 'tool_calls', 'function_call', 'content_filter'}:
        stop_type = response.get('stop_type')
        reason = {'eos': 'stop', 'word': 'stop', 'limit': 'length'}.get(stop_type) if isinstance(stop_type, str) else None
    details = usage.get('prompt_tokens_details')
    cached = _counter(details.get('cached_tokens')) if isinstance(details, dict) else None
    if cached is None:
        cached = _counter(response.get('tokens_cached'))
    timing = response.get('timings')
    engine_ms = None
    if isinstance(timing, dict):
        values = [timing.get(key) for key in ('prompt_ms', 'predicted_ms')]
        if all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1_000_000_000 for value in values):
            engine_ms = int(sum(values))
    return {'effectiveModel': safe_model(response.get('model')), 'finishReason': reason,
            'cacheOutcome': None if cached is None else ('hit' if cached > 0 else 'miss'),
            'engineMs': engine_ms,
            'usage': {'promptTokens': prompt, 'completionTokens': completion, 'totalTokens': total}}


def update_observation(observation, metadata):
    if observation is None:
        return
    for key in ('effectiveModel', 'finishReason', 'cacheOutcome', 'engineMs'):
        value = metadata[key]
        if value is not None:
            if key == 'effectiveModel' and observation[key] not in (None, value):
                observation['mixedModel'] = True
            observation[key] = value
    # Stream usage is cumulative: replace values rather than adding chunks.
    for key, value in metadata['usage'].items():
        if value is not None:
            observation['usage'][key] = value


@contextmanager
def observe_inference(payload, identity=None):
    context = _CAPTURE.get()
    observation = None
    if context is not None:
        if len(context.observations) < MAX_METADATA_CALLS:
            observation = {**response_metadata(None), 'requestedModel': safe_model(payload.get('model')),
                           'callIndex': len(context.observations) + 1, 'status': 'completed',
                           '_identity': identity, '_observedAt': _now()}
            context.observations.append(observation)
        else:
            context.metadata_truncated = True
    started = time.monotonic()
    try:
        yield observation
    except BaseException:
        if observation is not None:
            observation['status'] = 'incomplete'
        raise
    finally:
        if observation is not None:
            observation['inferenceMs'] = max(0, int((time.monotonic() - started) * 1000))


class Capture:
    def __init__(self, assignment, enabled=True):
        self.enabled = enabled
        self.observations = []
        self.metadata_truncated = False
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

    def metadata(self):
        calls = self.observations
        warnings = []
        usage = {key: sum(call['usage'][key] for call in calls)
                 if calls and not self.metadata_truncated and all(call['usage'][key] is not None for call in calls) else None
                 for key in ('promptTokens', 'completionTokens', 'totalTokens')}
        if any(value is None for value in usage.values()):
            warnings.append('token_usage_unavailable')
        models = {call['effectiveModel'] for call in calls if call['effectiveModel'] is not None}
        model_known = not self.metadata_truncated and bool(calls) and len(models) == 1 and all(call['effectiveModel'] and not call.get('mixedModel') for call in calls)
        if not model_known:
            warnings.append('served_model_unavailable_or_mixed')
        reasons = {call['finishReason'] for call in calls}
        reason = next(iter(reasons)) if len(reasons) == 1 and None not in reasons else 'unknown'
        if reason == 'unknown':
            warnings.append('engine_finish_reason_unavailable_or_mixed')
        caches = {call['cacheOutcome'] for call in calls}
        cache = next(iter(caches)) if len(caches) == 1 and None not in caches else 'unknown'
        requested = {call['requestedModel'] for call in calls if call['requestedModel']}
        if any(call['status'] != 'completed' for call in calls):
            warnings.append('inference_response_incomplete')
        if not calls:
            warnings.append('inference_time_unavailable')
        if self.metadata_truncated:
            warnings.append('inference_metadata_truncated')
            reason, cache = 'unknown', 'unknown'
        metadata = {'effectiveModel': next(iter(models)) if model_known else 'unreported',
                    'finishReason': reason, 'inferenceMs': sum(call['inferenceMs'] for call in calls) if not self.metadata_truncated else 0,
                    'engineMs': sum(call['engineMs'] for call in calls) if calls and not self.metadata_truncated and all(call['engineMs'] is not None for call in calls) else None,
                    'cacheOutcome': cache, 'warnings': warnings,
                    'calls': [{key: value for key, value in call.items() if not key.startswith('_')} for call in calls]}
        if len(requested) == 1:
            metadata['requestedModel'] = next(iter(requested))
        identities = [call.get('_identity') for call in calls]
        if calls and any(not identity or identity['engine'] is None for identity in identities):
            warnings.append('engine_properties_unavailable')
        if calls and not self.metadata_truncated and all(identity is not None for identity in identities):
            adapters = [identity['adapter'] for identity in identities]
            fingerprints = {canonical_sha256({'available': True, 'sha256': adapter['sha256'], 'scale': adapter['scale']})
                            if adapter is not None else None for adapter in adapters}
            if len(fingerprints) == 1:
                metadata['effectiveAdapter'] = next(iter(fingerprints))
            else:
                warnings.append('adapter_identity_unavailable_or_mixed')
        return {'usage': usage, 'inference': metadata}

    def artifacts(self):
        artifacts = [json_output_artifact(KIND, KIND, value) for value in self.calls]
        if any(call.get('_identity') and (call['_identity']['engine'] or call['_identity']['adapter']) for call in self.observations):
            evidence = {'schemaVersion': 'inference-runtime/1', **self.identity,
                        'complete': not self.metadata_truncated,
                        'calls': [{'callIndex': call['callIndex'], 'observedAt': call['_observedAt'],
                                   'engine': (call.get('_identity') or {}).get('engine'),
                                   'adapter': (call.get('_identity') or {}).get('adapter')} for call in self.observations]}
            artifacts.append(json_output_artifact('inference_runtime', 'inference_runtime', evidence))
        return artifacts


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
        if grant is None and assignment.get('stepKind') != 'inference':
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
        if grant is not None and not valid:
            result = {**step_result_base(assignment), "status": "failed",
                      "error": {"code": "INFERENCE_CAPTURE_DENIED", "message": "Inference capture authorization or artifact delivery is unavailable", "retryable": False}}
            if assignment.get("stepKind") == "inference":
                result.update(cancelled_inference_fields())
                result["output"]["outcome"] = {"kind": "failed"}
                result["inference"]["finishReason"] = "unknown"
            return result
        capture = Capture(assignment, enabled=grant is not None)
        token = _CAPTURE.set(capture)
        try:
            result = function(assignment, artifacts, output_artifacts)
        finally:
            _CAPTURE.reset(token)
        if result.get('stepKind') == 'inference':
            metadata = capture.metadata()
            result['usage'] = metadata['usage']
            result.setdefault('inference', {}).update(metadata['inference'])
        artifacts_to_upload = capture.artifacts()
        if artifacts_to_upload and output_artifacts is None:
            # Direct in-process callers have no artifact delivery channel.
            result.setdefault('inference', {}).setdefault('warnings', []).append('runtime_evidence_delivery_unavailable')
        else:
            uploads, refs = prepare_output_artifacts(artifacts_to_upload)
            result["artifactRefs"].extend(refs)
            if uploads:
                output_artifacts.extend(uploads)
        return result
    return wrapped
