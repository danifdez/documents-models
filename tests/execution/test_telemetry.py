import unittest
from unittest.mock import patch

from lib import telemetry
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


class TelemetryBoundaryTest(unittest.TestCase):
    def setUp(self):
        self.exporter = InMemorySpanExporter()
        self.provider = TracerProvider(resource=Resource({"service.name": "test"}), shutdown_on_exit=False)
        self.provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.old = telemetry._provider, telemetry._initialized
        telemetry._provider, telemetry._initialized = self.provider, True

    def tearDown(self):
        self.provider.shutdown()
        telemetry._provider, telemetry._initialized = self.old

    def test_only_w3c_identifiers_cross_boundary(self):
        parent = "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"
        self.assertEqual(telemetry.safe_trace_context({"traceparent": parent, "baggage": "secret"}), {"traceparent": parent})
        for value in (None, {}, {"traceparent": parent + "\n"}, {"traceparent": "00-" + "0" * 32 + "-0123456789abcdef-01"}):
            self.assertEqual(telemetry.safe_trace_context(value), {})

    def test_remote_endpoints_require_tls(self):
        for value in ("https://collector.example/v1/traces", "http://127.0.0.1:4318/v1/traces"):
            self.assertTrue(telemetry.safe_endpoint(value))
        for value in ("http://collector.example", "https://user:secret@host", "https://host?token=secret", "https://host/#secret", "file:///tmp/traces", "https://host/\n"):
            self.assertFalse(telemetry.safe_endpoint(value))

    def test_assignment_and_inference_share_trace_without_content(self):
        parent = "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"
        private = "PRIVATE_DOCUMENT_PASSWORD_AUTHORIZATION_PROMPT_RESPONSE"

        @telemetry.assignment_context
        def operation(assignment):
            with telemetry.inference_span({"messages": [{"content": private}], "temperature": 0.2}) as current:
                telemetry.inference_response(current, {"model": "model-id", "choices": [{"message": {"content": private}}], "usage": {"prompt_tokens": 12, "completion_tokens": 8}})
            return {"status": "succeeded", "result": private}

        result = operation({"work": {"traceContext": {"traceparent": parent, "baggage": private}}})
        self.assertEqual(result["result"], private)
        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 1, "The backend ledger already owns attempt lifecycle")
        inference = spans[0]
        self.assertEqual(inference.kind.name, "CLIENT")
        self.assertEqual(inference.context.trace_id, int(parent[3:35], 16))
        self.assertEqual(inference.parent.span_id, int(parent[36:52], 16))
        self.assertEqual(inference.attributes["gen_ai.usage.input_tokens"], 12)
        self.assertNotIn(private, inference.to_json())

    def test_assignment_context_is_restored_after_failure(self):
        from opentelemetry import trace
        before = trace.get_current_span().get_span_context()
        @telemetry.assignment_context
        def operation(_assignment):
            raise RuntimeError("private")
        with self.assertRaises(RuntimeError):
            operation({"work": {"traceContext": {"traceparent": "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"}}})
        self.assertEqual(trace.get_current_span().get_span_context(), before)
        self.assertEqual(self.exporter.get_finished_spans(), ())

    def test_exception_message_is_never_recorded(self):
        error = RuntimeError("private password or document content")
        trace_span = telemetry.span("test")
        with self.assertRaisesRegex(RuntimeError, "private"):
            with trace_span:
                raise error
        span = self.exporter.get_finished_spans()[0]
        self.assertEqual(span.status.status_code.name, "ERROR")
        self.assertNotIn("private", span.to_json())

    def test_disabled_has_no_exporter(self):
        telemetry._provider, telemetry._initialized = None, False
        with patch.dict("os.environ", {"OTEL_SDK_DISABLED": "true", "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "https://collector.example/v1/traces"}):
            with telemetry.span("disabled") as current:
                self.assertIsNone(current)


if __name__ == "__main__":
    unittest.main()
