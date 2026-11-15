"""Opt-in, metadata-only OpenTelemetry. No product bodies or exception messages."""
import atexit
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import logging
import os
import re
from urllib.parse import urlsplit
from lib.execution.inference_capture import observe_inference, response_metadata, update_observation

_provider = None
_initialized = False
_execution_ids = ContextVar("documents_execution_ids", default={})
_parent = re.compile(r"00-(?!0{32}-)[0-9a-f]{32}-(?!0{16}-)[0-9a-f]{16}-0[01]\Z")


def safe_trace_context(value):
    parent = value.get("traceparent") if isinstance(value, dict) else None
    return {"traceparent": parent} if isinstance(parent, str) and _parent.fullmatch(parent) else {}


def safe_endpoint(value):
    try:
        url = urlsplit(value)
        return len(value) <= 2048 and not any(ord(c) <= 32 or ord(c) >= 127 for c in value) and "\\" not in value and bool(url.hostname) and not (url.username or url.password or url.query or url.fragment) and (
            url.scheme == "https" or (url.scheme == "http" and url.hostname in {"127.0.0.1", "::1"})
        )
    except ValueError:
        return False


def initialize():
    try:
        return _initialize()
    finally:
        # Spawned handlers initialize before launching ffmpeg or model tools.
        # The exporter retains its private copy, including TLS settings; subprocesses
        # need no OTLP endpoint or credentials.
        for key in list(os.environ):
            if key.startswith("OTEL_EXPORTER_OTLP"):
                os.environ.pop(key, None)


def _initialize():
    global _provider, _initialized
    if _initialized:
        return _provider
    _initialized = True
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    base = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if os.environ.get("OTEL_SDK_DISABLED") == "true" or not (endpoint or base):
        return None
    endpoint = endpoint or base.rstrip("/") + "/v1/traces"
    protocol = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf"))
    if not safe_endpoint(endpoint) or protocol != "http/protobuf":
        logging.getLogger(__name__).warning("OpenTelemetry disabled: invalid endpoint or protocol")
        return None
    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider, SpanLimits
        from opentelemetry.sdk.trace.sampling import ParentBased, ALWAYS_ON
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        _provider = TracerProvider(
            resource=Resource({"service.name": "documents-models"}),
            sampler=ParentBased(ALWAYS_ON),
            span_limits=SpanLimits(max_attributes=24, max_attribute_length=160, max_events=0, max_links=0),
            shutdown_on_exit=False,
        )
        _provider.add_span_processor(BatchSpanProcessor(
            OTLPSpanExporter(endpoint=endpoint, timeout=3), max_queue_size=512,
            max_export_batch_size=64, schedule_delay_millis=1000, export_timeout_millis=3000,
        ))
        atexit.register(shutdown)
    except Exception:
        # Configuration/dependency failures must not break product execution or
        # expose endpoint credentials in a diagnostic exception.
        logging.getLogger(__name__).warning("OpenTelemetry could not be initialized")
        _provider = None
    return _provider


def shutdown():
    if _provider:
        _provider.shutdown()


def flush():
    if _provider:
        _provider.force_flush(timeout_millis=3000)


@contextmanager
def span(name, carrier=None, attributes=None, kind="internal"):
    provider = initialize()
    if provider is None:
        yield None
        return
    from opentelemetry.trace import Status, StatusCode, SpanKind
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    from opentelemetry.context import Context
    parent = None if carrier is None else TraceContextTextMapPropagator().extract(safe_trace_context(carrier), context=Context())
    with provider.get_tracer("documents-models").start_as_current_span(
        name, context=parent, attributes=attributes or {},
        kind={"internal": SpanKind.INTERNAL, "client": SpanKind.CLIENT, "consumer": SpanKind.CONSUMER}[kind],
        record_exception=False, set_status_on_exception=False,
    ) as current:
        try:
            yield current
        except BaseException:
            current.set_status(Status(StatusCode.ERROR))
            raise


def assignment_context(function):
    """Restore the canonical attempt context without creating a second tracker."""
    @wraps(function)
    def wrapped(assignment, *args, **kwargs):
        if initialize() is None:
            return function(assignment, *args, **kwargs)
        from opentelemetry.context import Context, attach, detach
        from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
        carrier = safe_trace_context((assignment.get("work") or {}).get("traceContext"))
        token = attach(TraceContextTextMapPropagator().extract(carrier, context=Context()))
        ids = {}
        for key in ("executionId", "stepId", "operationId", "attemptId"):
            value = assignment.get(key)
            if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F-]{36}", value):
                ids["execution." + key] = value
        ids_token = _execution_ids.set(ids)
        try:
            return function(assignment, *args, **kwargs)
        finally:
            _execution_ids.reset(ids_token)
            detach(token)
    return wrapped


@contextmanager
def inference_span(payload, operation="chat"):
    attrs = {**_execution_ids.get(), "gen_ai.operation.name": operation, "gen_ai.provider.name": "llama.cpp"}
    model = payload.get("model")
    if isinstance(model, str) and re.fullmatch(r"[\w.:-]{1,120}", model, flags=re.ASCII):
        attrs["gen_ai.request.model"] = model
    for key, attr in (("temperature", "temperature"), ("top_p", "top_p"), ("max_tokens", "max_tokens"), ("n_predict", "max_tokens")):
        value = payload.get(key)
        if type(value) in (int, float) and -1 <= value <= 1_000_000:
            attrs["gen_ai.request." + attr] = value
    with span("gen_ai." + operation, attributes=attrs, kind="client") as current:
        yield current


def inference_response(current, response, observation=None):
    metadata = response_metadata(response)
    update_observation(observation, metadata)
    if current is None:
        return
    for key, attr in (("promptTokens", "input_tokens"), ("completionTokens", "output_tokens")):
        value = metadata['usage'][key]
        if value is not None:
            current.set_attribute("gen_ai.usage." + attr, value)
    if metadata['effectiveModel']:
        current.set_attribute("gen_ai.response.model", metadata['effectiveModel'])


def trace_post(function):
    @wraps(function)
    def wrapped(url, payload, stream=False, identity=None):
        # Streaming spans must be owned by the consumer, until the final chunk.
        if stream:
            return function(url, payload, stream=True)
        operation = "chat" if "messages" in payload else "text_completion"
        with inference_span(payload, operation) as current, observe_inference(payload, identity) as observation:
            response = function(url, payload, stream=False)
            inference_response(current, response, observation)
            return response
    return wrapped
