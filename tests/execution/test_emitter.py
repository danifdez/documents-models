import base64
import importlib.util
import json
import os
import unittest
from unittest.mock import Mock, patch
from pathlib import Path

from lib.execution.emitter import (
    CONTRACT_SET_HASH,
    ExecutionEmitter,
    InferenceBudgetDenied,
    _safe_value,
)
from lib.execution.ingest_client import ExecutionIngestClient
from lib.execution.progress import ProgressLoopContext
from tests.execution.support import RecordingIngestClient


CONTEXT = {
    "rootExecutionId": "00000000-0000-4000-8000-000000000002",
    "executionId": "00000000-0000-4000-8000-000000000002",
    "turnId": "00000000-0000-4000-8000-000000000003",
    "attemptId": "00000000-0000-4000-8000-000000000004",
    "causedByEventId": "00000000-0000-4000-8000-000000000005",
}


class ExecutionEmitterTest(unittest.TestCase):
    def emitter(self, ingest_client=None):
        environment = {"EXECUTION_INGEST_TOKEN": "test-token", "WORKER_ID": "worker-test"}
        with patch.dict(os.environ, environment, clear=False):
            return ExecutionEmitter(CONTEXT, ingest_client=ingest_client)

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
        client = RecordingIngestClient()
        emitter = self.emitter(client)

        emitter.record_artifact(
            "tool_result",
            {"cookie": "session=secret", "answer": "visible"},
            "application/json",
        )

        emitter.flush()
        suffix, request = client.requests[0]
        artifact = request["artifacts"][0]
        body = json.loads(base64.b64decode(artifact["bodyBase64"]))
        self.assertEqual(suffix, "artifacts")
        self.assertEqual(body, {"answer": "visible", "cookie": "[REDACTED]"})
        self.assertTrue(artifact["redaction"]["applied"])

    def test_buffered_events_preserve_causality_before_transport(self):
        client = RecordingIngestClient()
        emitter = self.emitter(client)
        handle = emitter.start_operation("browser_action", "click")
        emitter.finish_operation(handle, status="failed", result={}, error="not found")
        emitter.flush()

        start, finish = client.requests[0][1]["events"]
        self.assertEqual(handle.started_event_id, start["eventId"])
        self.assertEqual(start["causedByEventId"], CONTEXT["causedByEventId"])
        self.assertEqual(finish["causedByEventId"], start["eventId"])

    def test_worker_evidence_does_not_finalize_the_execution(self):
        client = RecordingIngestClient()
        emitter = self.emitter(client)

        emitter.record_final_message("done")
        emitter.flush_evidence()

        events = client.sent_events
        self.assertEqual([event["eventType"] for event in events], ["message.recorded"])

    def test_producer_instance_is_scoped_to_the_execution_attempt(self):
        emitter = self.emitter()

        self.assertIn(CONTEXT["attemptId"], emitter.instance_id)

    def test_ingest_retry_reuses_the_same_idempotent_request(self):
        first_connection = Mock()
        first_connection.request.side_effect = TimeoutError("timeout")
        second_connection = Mock()
        response = Mock(status=200)
        response.read.return_value = b'{"granted":true}'
        second_connection.getresponse.return_value = response
        client = ExecutionIngestClient("http://backend:3000", "token", 1)
        body = {"operationId": "operation-1"}

        with patch(
            "lib.execution.ingest_client.http.client.HTTPConnection",
            side_effect=[first_connection, second_connection],
        ):
            result = client.post("execution-1", "progress/reservations", body)

        self.assertEqual(result, {"granted": True})
        self.assertEqual(
            first_connection.request.call_args.kwargs["body"],
            second_connection.request.call_args.kwargs["body"],
        )

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

    def test_progress_loop_keeps_policy_and_operation_identity_together(self):
        client = RecordingIngestClient()
        emitter = self.emitter(client)

        progress = ProgressLoopContext.start(
            emitter,
            agent_name="assistant",
            loop_kind="top_level",
            max_rounds=3,
            max_output_repairs=1,
            forced_finalization_available=True,
            max_tokens_per_inference=256,
        )
        trace = progress.trace(
            round=2,
            phase="output_repair",
            extra={"reason": "empty_model_response"},
        )

        policy = next(
            event["payload"]["policy"]
            for event in client.sent_events
            if event["payload"].get("kind") == "policy_snapshot"
        )
        self.assertEqual(trace["loopId"], policy["loopId"])
        self.assertEqual(trace["agentName"], policy["agentName"])
        self.assertEqual(trace["maxRounds"], policy["maxRounds"])
        self.assertEqual(trace["round"], 2)
        self.assertEqual(trace["phase"], "output_repair")
        self.assertEqual(trace["reason"], "empty_model_response")
        self.assertEqual(
            trace["budgetGrantId"],
            "00000000-0000-4000-8000-000000000011",
        )

    def test_progress_loop_uses_the_effective_backend_policy(self):
        fallback = RecordingIngestClient()

        def respond(suffix, body):
            response = fallback.post(CONTEXT["rootExecutionId"], suffix, body)
            if suffix == "progress/grants":
                response["grant"]["effectivePolicy"] = {
                    "normal": 1,
                    "repair": 0,
                    "closing": 0,
                    "maxTokensPerInference": 64,
                }
            return response

        emitter = self.emitter(RecordingIngestClient(respond))
        progress = ProgressLoopContext.start(
            emitter,
            agent_name="assistant",
            loop_kind="top_level",
            max_rounds=3,
            max_output_repairs=1,
            forced_finalization_available=True,
            max_tokens_per_inference=256,
        )

        self.assertEqual(progress.max_rounds, 1)
        self.assertEqual(progress.max_output_repairs, 0)
        self.assertFalse(progress.forced_finalization_available)
        self.assertEqual(progress.max_tokens_per_inference, 64)

    def test_reserved_inference_records_budget_references_on_its_start(self):
        client = RecordingIngestClient()
        emitter = self.emitter(client)
        progress = ProgressLoopContext.start(
            emitter,
            agent_name="assistant",
            loop_kind="top_level",
            max_rounds=1,
            max_output_repairs=0,
            forced_finalization_available=False,
            max_tokens_per_inference=64,
        )

        handle = emitter.start_inference(
            "direct_response",
            {"messages": []},
            progress.trace(round=1, phase="direct_response"),
        )
        emitter.flush_evidence()

        start = next(
            event
            for event in client.sent_events
            if event["eventType"] == "operation.started"
        )
        self.assertEqual(start["operationId"], handle.operation_id)
        self.assertEqual(start["payload"]["budgetGrantId"], progress.grant_id)
        self.assertEqual(
            start["payload"]["budgetReservationId"],
            "00000000-0000-4000-8000-000000000013",
        )
        self.assertEqual(start["payload"]["budgetBucket"], "normal")
        self.assertLess(
            next(i for i, request in enumerate(client.requests)
                 if request[0] == "progress/reservations"),
            next(i for i, request in enumerate(client.requests)
                 if request[0] == "artifacts"),
        )

    def test_tool_operation_does_not_inherit_inference_budget_grant(self):
        client = RecordingIngestClient()
        emitter = self.emitter(client)
        progress = ProgressLoopContext.start(
            emitter,
            agent_name="assistant",
            loop_kind="top_level",
            max_rounds=2,
            max_output_repairs=1,
            forced_finalization_available=True,
            max_tokens_per_inference=64,
        )

        emitter.start_tool(
            "lookup_value",
            {"key": "alpha"},
            "provider-call-1",
            progress.trace(round=1, phase="agent_loop"),
        )
        emitter.flush_evidence()

        start = next(
            event
            for event in client.sent_events
            if event["eventType"] == "operation.started"
        )
        self.assertEqual(start["payload"]["operationKind"], "tool_call")
        self.assertEqual(start["payload"]["loopId"], progress.loop_id)
        self.assertNotIn("budgetGrantId", start["payload"])
        self.assertNotIn("budgetReservationId", start["payload"])
        self.assertNotIn("budgetBucket", start["payload"])

    def test_denied_budget_stops_before_prompt_artifact_and_operation_start(self):
        fallback = RecordingIngestClient()

        def respond(suffix, body):
            if suffix == "progress/reservations":
                return {
                    "granted": False,
                    "eventId": "00000000-0000-4000-8000-000000000020",
                    "reservation": {
                        "version": "1",
                        "reservationId": "00000000-0000-4000-8000-000000000021",
                        "grantId": body["grantId"],
                        "operationId": body["operationId"],
                        "executionAttemptId": body["executionAttemptId"],
                        "bucket": body["bucket"],
                        "phase": body["phase"],
                        "round": body["round"],
                        "name": body["name"],
                        "status": "denied",
                        "reason": "budget_hard_limit_reached",
                        "decidedAt": "2026-08-20T10:00:02Z",
                    },
                }
            return fallback.post(CONTEXT["rootExecutionId"], suffix, body)

        client = RecordingIngestClient(respond)
        emitter = self.emitter(client)
        progress = ProgressLoopContext.start(
            emitter,
            agent_name="assistant",
            loop_kind="top_level",
            max_rounds=3,
            max_output_repairs=1,
            forced_finalization_available=True,
            max_tokens_per_inference=256,
        )

        with self.assertRaisesRegex(
            InferenceBudgetDenied, "budget_hard_limit_reached"
        ):
            emitter.start_inference(
                "chat_with_tools",
                {"messages": [{"role": "user", "content": "hello"}]},
                progress.trace(round=1, phase="agent_loop"),
            )

        self.assertEqual(emitter.pending_artifacts, [])
        self.assertFalse(any(
            event["eventType"] == "operation.started"
            for event in emitter.pending_events
        ))

    def test_reservation_protocol_error_is_typed_before_dispatch(self):
        fallback = RecordingIngestClient()

        def respond(suffix, body):
            if suffix == "progress/reservations":
                raise RuntimeError("HTTP 409 stale attempt")
            return fallback.post(CONTEXT["rootExecutionId"], suffix, body)

        emitter = self.emitter(RecordingIngestClient(respond))
        progress = ProgressLoopContext.start(
            emitter,
            agent_name="assistant",
            loop_kind="top_level",
            max_rounds=1,
            max_output_repairs=0,
            forced_finalization_available=False,
            max_tokens_per_inference=64,
        )

        with self.assertRaisesRegex(
            InferenceBudgetDenied, "budget_reservation_failed"
        ):
            emitter.start_inference(
                "direct_response",
                {"messages": []},
                progress.trace(round=1, phase="direct_response"),
            )

        self.assertEqual(emitter.pending_artifacts, [])
        self.assertFalse(any(
            event["eventType"] == "operation.started"
            for event in emitter.pending_events
        ))

    def test_oversized_artifact_is_declared_missing_without_transport(self):
        client = RecordingIngestClient(
            lambda *_: self.fail("oversized artifact must not be sent")
        )
        emitter = self.emitter(client)

        artifact_id = emitter.record_artifact("model_response", "x" * (1024 * 1024 + 1), "text/plain")

        self.assertIsNone(artifact_id)
        self.assertIn("artifact_omitted:size:model_response:1048577", emitter.summary()["errors"])

    def test_failed_required_artifact_transport_stops_before_events(self):
        def respond(suffix, body):
            return None if suffix == "artifacts" else {"accepted": len(body["events"])}

        client = RecordingIngestClient(respond)
        emitter = self.emitter(client)
        artifact_id = emitter.record_artifact("model_response", "reply", "text/plain")
        emitter.emit(
            "message.recorded",
            "message.recorded/1",
            {"messageKind": "final_response", "role": "assistant", "contentArtifactId": artifact_id},
            actor={"type": "model"},
            artifact_refs=[artifact_id],
        )
        with self.assertRaisesRegex(RuntimeError, "artifact ingestion"):
            emitter.flush()

        self.assertEqual([suffix for suffix, _body in client.requests], ["artifacts"])
        self.assertEqual(len(emitter.pending_artifacts), 1)
        self.assertEqual(len(emitter.pending_events), 1)
        self.assertEqual(emitter.artifact_bytes, len("reply"))
        self.assertIn("artifact_batch", emitter.summary()["errors"])

    def test_multiple_drains_send_only_new_evidence(self):
        client = RecordingIngestClient()
        emitter = self.emitter(client)

        first = emitter.start_operation("inference", "first")
        emitter.flush()
        emitter.finish_operation(
            first,
            status="succeeded",
            result={},
            error=None,
            outcome="final_text",
        )
        emitter.flush()
        emitter.flush()

        events = [
            event
            for suffix, body in client.requests
            if suffix == "events"
            for event in body["events"]
        ]
        self.assertEqual(
            [event["eventType"] for event in events],
            ["operation.started", "operation.finished"],
        )
        self.assertEqual(emitter.pending_events, [])

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
        captured = {}

        class Response:
            status = 200

            def read(self):
                return b'{"accepted":1}'

        class Connection:
            def request(self, method, path, body, headers):
                captured.update({
                    "method": method,
                    "path": path,
                    "body": body,
                    "headers": headers,
                })

            def getresponse(self):
                return Response()

        client = ExecutionIngestClient("http://localhost:3000", "test-token", 2)
        client._connection = Connection()
        response = client.post(CONTEXT["rootExecutionId"], "events", {"events": []})

        self.assertEqual(captured["method"], "POST")
        self.assertEqual(
            captured["path"],
            f"/executions/internal/{CONTEXT['rootExecutionId']}/events",
        )
        self.assertEqual(
            captured["headers"]["X-Execution-Ingest-Token"],
            "test-token",
        )
        self.assertEqual(response, {"accepted": 1})

    def test_reconnects_once_with_the_same_idempotent_payload(self):
        connections = []

        class Response:
            status = 200

            def read(self):
                return b'{"accepted":1}'

        class Connection:
            def __init__(self, *_args, **_kwargs):
                self.requests = []
                connections.append(self)

            def request(self, method, path, body, headers):
                self.requests.append((method, path, body, headers))
                if len(connections) == 1:
                    raise ConnectionResetError("stale keep-alive")

            def getresponse(self):
                return Response()

            def close(self):
                pass

        with patch(
            "lib.execution.ingest_client.http.client.HTTPConnection",
            Connection,
        ):
            client = ExecutionIngestClient(
                "http://localhost:3000",
                "test-token",
                2,
            )
            response = client.post(
                CONTEXT["rootExecutionId"],
                "events",
                {"events": []},
            )

        self.assertEqual(response, {"accepted": 1})
        self.assertEqual(len(connections), 2)
        self.assertEqual(connections[0].requests, connections[1].requests)


if __name__ == "__main__":
    unittest.main()
