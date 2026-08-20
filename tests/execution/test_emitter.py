import base64
import importlib.util
import json
import os
import unittest
from unittest.mock import patch
from pathlib import Path

from lib.execution.emitter import CONTRACT_SET_HASH, ExecutionEmitter, _safe_value


CONTEXT = {
    "rootExecutionId": "00000000-0000-4000-8000-000000000002",
    "executionId": "00000000-0000-4000-8000-000000000002",
    "turnId": "00000000-0000-4000-8000-000000000003",
    "attemptId": "00000000-0000-4000-8000-000000000004",
    "causedByEventId": "00000000-0000-4000-8000-000000000005",
}


class ExecutionEmitterTest(unittest.TestCase):
    def emitter(self):
        environment = {"EXECUTION_INGEST_TOKEN": "test-token", "WORKER_ID": "worker-test"}
        with patch.dict(os.environ, environment, clear=False):
            return ExecutionEmitter(CONTEXT)

    def test_redacts_camel_case_secrets_and_private_reasoning(self):
        safe = _safe_value(
            {
                "accessToken": "secret-value",
                "refresh-token": "refresh-value",
                "reasoning_content": "private chain",
                "nested": "Authorization: Bearer abc.def",
                "stringSecret": "accessToken=camel-secret",
            }
        )

        self.assertEqual(safe["accessToken"], "[REDACTED]")
        self.assertEqual(safe["refresh-token"], "[REDACTED]")
        self.assertNotIn("reasoning_content", safe)
        self.assertNotIn("abc.def", safe["nested"])
        self.assertNotIn("camel-secret", safe["stringSecret"])

    def test_artifact_body_and_manifest_are_redacted_before_transport(self):
        emitter = self.emitter()
        captured = []
        emitter._post = lambda suffix, body: captured.append((suffix, body)) or {"accepted": 1}

        emitter.record_artifact(
            "tool_result",
            {"cookie": "session=secret", "answer": "visible"},
            "application/json",
        )

        emitter.flush()
        suffix, request = captured[0]
        artifact = request["artifacts"][0]
        body = json.loads(base64.b64decode(artifact["bodyBase64"]))
        self.assertEqual(suffix, "artifacts")
        self.assertEqual(body, {"answer": "visible", "cookie": "[REDACTED]"})
        self.assertTrue(artifact["redaction"]["applied"])

    def test_buffered_events_preserve_causality_before_transport(self):
        emitter = self.emitter()
        requests = []
        emitter._post = lambda suffix, body: requests.append((suffix, body)) or {
            "accepted": len(body.get("events", [])), "duplicates": 0
        }
        handle = emitter.start_operation("browser_action", "click")
        emitter.finish_operation(handle, status="failed", result={}, error="not found")
        emitter.flush()

        start, finish = requests[0][1]["events"]
        self.assertEqual(handle.started_event_id, start["eventId"])
        self.assertEqual(start["causedByEventId"], CONTEXT["causedByEventId"])
        self.assertEqual(finish["causedByEventId"], start["eventId"])

    def test_worker_evidence_does_not_finalize_the_execution(self):
        emitter = self.emitter()
        requests = []
        emitter._post = lambda suffix, body: requests.append((suffix, body)) or {
            "accepted": len(body.get("events", [])), "duplicates": 0
        }

        emitter.record_final_message("done")
        emitter.flush_evidence()

        events = next(body["events"] for suffix, body in requests if suffix == "events")
        self.assertEqual([event["eventType"] for event in events], ["message.recorded"])

    def test_producer_instance_is_scoped_to_the_execution_attempt(self):
        emitter = self.emitter()

        self.assertIn(CONTEXT["attemptId"], emitter.instance_id)

    def test_forced_finalization_metadata_is_recorded_on_the_operation(self):
        emitter = self.emitter()

        emitter.start_inference(
            "forced_finalization",
            {"messages": []},
            {
                "phase": "forced_finalization",
                "reason": "step_budget_exhausted",
            },
        )

        event = emitter.pending_events[-1]
        self.assertEqual(event["eventType"], "operation.started")
        self.assertEqual(event["payload"]["name"], "forced_finalization")
        self.assertEqual(event["payload"]["phase"], "forced_finalization")
        self.assertEqual(event["payload"]["reason"], "step_budget_exhausted")

    def test_oversized_artifact_is_declared_missing_without_transport(self):
        emitter = self.emitter()
        emitter._post = lambda *_: self.fail("oversized artifact must not be sent")

        artifact_id = emitter.record_artifact("model_response", "x" * (1024 * 1024 + 1), "text/plain")

        self.assertIsNone(artifact_id)
        self.assertIn("artifact_omitted:size:model_response:1048577", emitter.summary()["errors"])

    def test_failed_artifact_transport_removes_dangling_event_references(self):
        emitter = self.emitter()
        requests = []

        def post(suffix, body):
            requests.append((suffix, body))
            return None if suffix == "artifacts" else {"accepted": len(body["events"])}

        emitter._post = post
        artifact_id = emitter.record_artifact("model_response", "reply", "text/plain")
        emitter.emit(
            "message.recorded",
            "message.recorded/1",
            {"messageKind": "final_response", "role": "assistant", "contentArtifactId": artifact_id},
            actor={"type": "model"},
            artifact_refs=[artifact_id],
        )
        emitter.flush()

        event = requests[1][1]["events"][0]
        self.assertNotIn("contentArtifactId", event["payload"])
        self.assertEqual(event["artifactRefs"], [])
        self.assertEqual(emitter.artifact_bytes, len("reply"))
        self.assertIn("artifact_batch:1", emitter.summary()["errors"])

    def test_ingest_token_is_required(self):
        with patch.dict(os.environ, {"EXECUTION_INGEST_TOKEN": ""}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "EXECUTION_INGEST_TOKEN"):
                ExecutionEmitter(CONTEXT)

    def test_required_instrumentation_failures_propagate(self):
        emitter = self.emitter()
        cyclic = {}
        cyclic["self"] = cyclic
        with self.assertRaises(RecursionError):
            emitter.record_artifact("tool_result", cyclic, "application/json")

    def test_contract_copy_is_pinned_to_the_shared_manifest(self):
        root = Path(__file__).resolve().parents[2] / "contracts" / "execution" / "v1"
        spec = importlib.util.spec_from_file_location("models_execution_contract", root / "validate.py")
        contract = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(contract)

        self.assertEqual(contract.verify_schema_manifest(), CONTRACT_SET_HASH)

    def test_posts_to_the_execution_endpoint_with_the_renamed_header(self):
        emitter = self.emitter()
        captured = {}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b'{"accepted":1}'

        def open_request(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return Response()

        with patch(
            "lib.execution.emitter.urllib.request.urlopen",
            side_effect=open_request,
        ):
            response = emitter._post("events", {"events": []})

        request = captured["request"]
        self.assertEqual(
            request.full_url,
            "http://localhost:3000/executions/internal/"
            f"{CONTEXT['rootExecutionId']}/events",
        )
        self.assertEqual(
            request.get_header("X-execution-ingest-token"),
            "test-token",
        )
        self.assertEqual(response, {"accepted": 1})


if __name__ == "__main__":
    unittest.main()
