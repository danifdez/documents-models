import base64
import hashlib
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
    ToolBudgetDenied,
    ToolLoopGuardBlocked,
    _safe_value,
    canonical_tool_input_fingerprint,
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

    def test_tool_fingerprint_is_canonical_and_input_sensitive(self):
        first = canonical_tool_input_fingerprint(
            "folder_read",
            {"nested": {"b": 2, "a": 1}, "items": [2, 1]},
        )
        reordered = canonical_tool_input_fingerprint(
            "folder_read",
            {"items": [2, 1], "nested": {"a": 1, "b": 2}},
        )

        self.assertEqual(first, reordered)
        self.assertNotEqual(
            first,
            canonical_tool_input_fingerprint(
                "folder_read",
                {"nested": {"a": 1, "b": 2}, "items": [1, 2]},
            ),
        )
        self.assertNotEqual(
            first,
            canonical_tool_input_fingerprint(
                "file_read",
                {"nested": {"a": 1, "b": 2}, "items": [2, 1]},
            ),
        )
        self.assertIsNone(canonical_tool_input_fingerprint("folder_read", []))
        self.assertIsNone(
            canonical_tool_input_fingerprint("folder_read", {"value": float("nan")})
        )
        self.assertEqual(
            canonical_tool_input_fingerprint(
                "búsqueda",
                {"ż": 1, "A": {"ä": True, "z": None}},
            ),
            canonical_tool_input_fingerprint(
                "búsqueda",
                {"A": {"z": None, "ä": True}, "ż": 1},
            ),
        )

    def test_redaction_is_idempotent_inside_serialized_tool_content(self):
        safe = _safe_value(
            '{"raw": "accessToken=[REDACTED]", "answer": "visible"}'
        )

        self.assertEqual(
            safe,
            '{"raw": "accessToken=[REDACTED]", "answer": "visible"}',
        )

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

    def test_materialized_prompt_preserves_finite_sampling_parameters(self):
        client = RecordingIngestClient()
        emitter = self.emitter(client)

        emitter.record_artifact(
            "materialized_prompt",
            {"min_p": 0.05, "temperature": 0.7},
            "application/json",
        )
        emitter.flush()

        artifact = client.requests[0][1]["artifacts"][0]
        body = json.loads(base64.b64decode(artifact["bodyBase64"]))
        self.assertEqual(body, {"min_p": 0.05, "temperature": 0.7})

    def test_tool_result_source_hash_matches_the_redacted_artifact(self):
        client = RecordingIngestClient()
        emitter = self.emitter(client)
        handle = emitter.start_tool("read_fixture", {}, "call-1")

        source_event_id = emitter.observe_tool_result(
            handle,
            {"answer": "visible", "accessToken": "must-not-leak"},
        )
        emitter.finish_tool(
            handle,
            {"answer": "visible"},
            source_event_id,
            result_summary="Fixture ready",
            result_summary_kind="leaf_tool",
        )
        emitter.flush()

        artifact_request = next(
            body for suffix, body in client.requests if suffix == "artifacts"
        )
        artifact = artifact_request["artifacts"][0]
        artifact_body = base64.b64decode(artifact["bodyBase64"])
        source_event = next(
            event for event in client.sent_events
            if event["eventType"] == "source.observed"
        )

        self.assertEqual(
            source_event["payload"]["contentHash"],
            "sha256:" + hashlib.sha256(artifact_body).hexdigest(),
        )
        self.assertNotIn(b"must-not-leak", artifact_body)

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

    def test_runtime_template_message_is_attributed_to_the_system(self):
        client = RecordingIngestClient()
        emitter = self.emitter(client)

        emitter.record_final_message(
            "Completed work",
            generation_source="runtime_template",
        )
        emitter.flush_evidence()

        event = client.sent_events[0]
        self.assertEqual(event["actor"], {"type": "system"})
        self.assertEqual(
            event["payload"]["generationSource"],
            "runtime_template",
        )

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

    def test_ingest_http_error_closes_the_connection(self):
        connection = Mock()
        response = Mock(status=409)
        response.read.return_value = b'{"message":"stale attempt"}'
        connection.getresponse.return_value = response
        client = ExecutionIngestClient("http://backend:3000", "token", 1)
        client._connection = connection

        with self.assertRaisesRegex(RuntimeError, "HTTP 409"):
            client.post("execution-1", "progress/reservations", {})

        connection.close.assert_called_once_with()
        self.assertIsNone(client._connection)

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
                    "toolCalls": 1,
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
        self.assertEqual(progress.normal_inference_soft_limit, 0)
        self.assertEqual(progress.max_output_repairs, 0)
        self.assertFalse(progress.forced_finalization_available)
        self.assertEqual(progress.max_tokens_per_inference, 64)
        self.assertEqual(progress.max_tool_calls, 1)

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

    def test_normal_inference_soft_limit_warns_before_the_triggering_dispatch(self):
        client = RecordingIngestClient()
        emitter = self.emitter(client)
        progress = ProgressLoopContext.start(
            emitter,
            agent_name="assistant",
            loop_kind="top_level",
            max_rounds=3,
            normal_inference_soft_limit=2,
            max_output_repairs=0,
            forced_finalization_available=True,
            max_tokens_per_inference=64,
        )
        messages = [{"role": "user", "content": "question"}]
        first_request = {"messages": list(messages)}
        second_request = {"messages": [
            *messages,
            {"role": "system", "content": "Tool budget is low: 1 remains."},
        ]}

        first = emitter.start_inference(
            "chat_with_tools",
            first_request,
            progress.trace(round=1, phase="agent_loop"),
        )
        second = emitter.start_inference(
            "chat_with_tools",
            second_request,
            progress.trace(round=2, phase="agent_loop"),
        )
        emitter.flush_evidence()

        self.assertFalse(first.soft_limit_signal)
        self.assertTrue(second.soft_limit_signal)
        self.assertEqual(first_request["messages"], messages)
        self.assertEqual(len(second_request["messages"]), 3)
        self.assertTrue(
            second_request["messages"][-2]["content"].startswith(
                "Tool budget is low"
            )
        )
        self.assertIn(
            "1 of 3 calls remain",
            second_request["messages"][-1]["content"],
        )
        starts = [
            event for event in client.sent_events
            if event["eventType"] == "operation.started"
        ]
        self.assertNotIn(
            "budgetSoftLimitWarningApplied",
            starts[0]["payload"],
        )
        self.assertTrue(
            starts[1]["payload"]["budgetSoftLimitWarningApplied"]
        )
        artifact = client.sent_artifacts[-1]
        body = json.loads(base64.b64decode(artifact["bodyBase64"]))
        self.assertEqual(body["messages"], second_request["messages"])

        closing_request = {"messages": list(messages)}
        emitter.start_inference(
            "forced_finalization",
            closing_request,
            progress.trace(round=3, phase="forced_finalization"),
        )
        emitter.flush_evidence()
        self.assertEqual(closing_request["messages"], messages)
        closing_start = [
            event for event in client.sent_events
            if event["eventType"] == "operation.started"
        ][-1]
        self.assertNotIn(
            "budgetSoftLimitWarningApplied",
            closing_start["payload"],
        )

    def test_repeat_warning_is_materialized_between_budget_warnings(self):
        fallback = RecordingIngestClient()

        def respond(suffix, body):
            response = fallback.post(CONTEXT["rootExecutionId"], suffix, body)
            if suffix == "progress/reservations" and body["bucket"] == "normal":
                response["guardState"] = {
                    "detections": 1,
                    "warningIssued": True,
                    "warningPending": True,
                }
                response["loopGuardSignal"] = {
                    "guardKind": "immediate_exact_tool_repeat",
                }
            return response

        client = RecordingIngestClient(respond)
        emitter = self.emitter(client)
        progress = ProgressLoopContext.start(
            emitter,
            agent_name="assistant",
            loop_kind="top_level",
            max_rounds=2,
            normal_inference_soft_limit=1,
            max_output_repairs=0,
            forced_finalization_available=True,
            max_tokens_per_inference=64,
            exact_tool_repeat_warning=True,
        )
        request = {"messages": [
            {"role": "user", "content": "question"},
            {"role": "system", "content": "Tool budget is low."},
        ]}

        handle = emitter.start_inference(
            "chat_with_tools",
            request,
            progress.trace(round=1, phase="agent_loop"),
        )
        emitter.flush_evidence()

        self.assertEqual(len(request["messages"]), 4)
        self.assertEqual(request["messages"][-2]["role"], "system")
        self.assertIn("exact tool call was repeated", request["messages"][-2]["content"])
        self.assertIn("1 of 2 calls remain", request["messages"][-1]["content"])
        self.assertTrue(handle.guard_state["warningPending"])
        self.assertTrue(handle.loop_guard_signal)
        start = next(
            event for event in client.sent_events
            if event["eventType"] == "operation.started"
        )
        self.assertTrue(start["payload"]["loopGuardWarningApplied"])
        self.assertTrue(start["payload"]["budgetSoftLimitWarningApplied"])

    def test_repeat_warning_is_not_materialized_for_repair_or_closing(self):
        fallback = RecordingIngestClient()

        def respond(suffix, body):
            response = fallback.post(CONTEXT["rootExecutionId"], suffix, body)
            if suffix == "progress/reservations":
                response["guardState"] = {
                    "detections": 1,
                    "warningIssued": True,
                    "warningPending": True,
                }
            return response

        client = RecordingIngestClient(respond)
        emitter = self.emitter(client)
        progress = ProgressLoopContext.start(
            emitter,
            agent_name="assistant",
            loop_kind="top_level",
            max_rounds=1,
            max_output_repairs=1,
            forced_finalization_available=True,
            max_tokens_per_inference=64,
            exact_tool_repeat_warning=True,
        )

        for phase, name in (
            ("output_repair", "output_repair"),
            ("forced_finalization", "forced_finalization"),
        ):
            request = {"messages": [{"role": "user", "content": "question"}]}
            emitter.start_inference(
                name,
                request,
                progress.trace(round=1, phase=phase),
            )
            self.assertEqual(len(request["messages"]), 1)

        emitter.flush_evidence()
        starts = [
            event for event in client.sent_events
            if event["eventType"] == "operation.started"
            and event["payload"]["operationKind"] == "inference"
        ]
        self.assertEqual(len(starts), 2)
        self.assertTrue(all(
            "loopGuardWarningApplied" not in event["payload"]
            for event in starts
        ))

    def test_tool_reservation_exposes_the_authoritative_soft_limit_state(self):
        client = RecordingIngestClient()
        emitter = self.emitter(client)
        progress = ProgressLoopContext.start(
            emitter,
            agent_name="assistant",
            loop_kind="top_level",
            max_rounds=2,
            max_output_repairs=0,
            forced_finalization_available=True,
            max_tokens_per_inference=64,
            max_tool_calls=2,
            tool_call_soft_limit=1,
        )

        handle = emitter.start_tool(
            "folder_read",
            {"path": "fixture.txt"},
            "provider-call-1",
            progress.trace(round=1, phase="agent_loop"),
        )
        progress.observe_tool_budget(handle)
        messages = progress.messages_for_inference([
            {"role": "user", "content": "question"},
        ])

        self.assertTrue(handle.soft_limit_signal)
        self.assertTrue(progress.tool_soft_limit_reached)
        self.assertEqual(progress.tool_budget_available, 1)
        self.assertIn("1 of 2 calls remain", messages[-1]["content"])
        self.assertIs(
            progress.messages_for_inference(messages),
            messages,
        )

    def test_tool_operation_reserves_budget_before_its_start(self):
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
            max_tool_calls=1,
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
        self.assertEqual(start["payload"]["budgetGrantId"], progress.grant_id)
        self.assertEqual(start["payload"]["budgetBucket"], "tool")
        reservation_request = next(
            body for suffix, body in client.requests
            if suffix == "progress/reservations"
        )
        self.assertEqual(reservation_request["operationKind"], "tool_call")
        self.assertEqual(reservation_request["toolCallId"], start["toolCallId"])
        self.assertEqual(reservation_request["operationId"], start["operationId"])

    def test_comparable_tool_persists_the_same_fingerprint_on_reserve_and_start(self):
        client = RecordingIngestClient()
        emitter = self.emitter(client)
        progress = ProgressLoopContext.start(
            emitter,
            agent_name="assistant",
            loop_kind="top_level",
            max_rounds=2,
            max_output_repairs=0,
            forced_finalization_available=True,
            max_tokens_per_inference=64,
            max_tool_calls=2,
            exact_tool_repeat_warning=True,
        )

        emitter.start_tool(
            "folder_read",
            {"path": "fixture.txt", "depth": 1},
            "provider-call-1",
            progress.trace(round=1, phase="agent_loop"),
            repeat_comparable=True,
        )
        emitter.flush_evidence()

        reservation = next(
            body for suffix, body in client.requests
            if suffix == "progress/reservations"
        )
        start = next(
            event for event in client.sent_events
            if event["eventType"] == "operation.started"
        )
        self.assertEqual(
            reservation["operationFingerprint"],
            start["payload"]["operationFingerprint"],
        )
        self.assertEqual(
            reservation["operationFingerprintVersion"],
            "canonical_tool_input_v1",
        )
        self.assertEqual(
            start["payload"]["operationFingerprintVersion"],
            "canonical_tool_input_v1",
        )

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

    def test_denied_tool_budget_stops_before_operation_start(self):
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
                        "operationKind": "tool_call",
                        "bucket": "tool",
                        "toolCallId": body["toolCallId"],
                        "phase": body["phase"],
                        "round": body["round"],
                        "name": body["name"],
                        "status": "denied",
                        "reason": "tool_budget_hard_limit_reached",
                        "decidedAt": "2026-08-20T10:00:02Z",
                    },
                }
            return fallback.post(CONTEXT["rootExecutionId"], suffix, body)

        emitter = self.emitter(RecordingIngestClient(respond))
        progress = ProgressLoopContext.start(
            emitter,
            agent_name="assistant",
            loop_kind="top_level",
            max_rounds=1,
            max_output_repairs=0,
            forced_finalization_available=True,
            max_tokens_per_inference=64,
            max_tool_calls=1,
        )

        with self.assertRaisesRegex(
            ToolBudgetDenied, "tool_budget_hard_limit_reached"
        ):
            emitter.start_tool(
                "folder_read",
                {"path": "fixture.txt"},
                "provider-call-1",
                progress.trace(round=1, phase="agent_loop"),
            )

        self.assertFalse(any(
            event["eventType"] == "operation.started"
            for event in emitter.pending_events
        ))

    def test_loop_guard_block_is_typed_before_operation_start(self):
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
                        "operationKind": "tool_call",
                        "bucket": "tool",
                        "toolCallId": body["toolCallId"],
                        "phase": body["phase"],
                        "round": body["round"],
                        "name": body["name"],
                        "status": "denied",
                        "reason": "immediate_exact_tool_repeat_blocked",
                        "decidedAt": "2026-08-21T10:00:02Z",
                    },
                }
            return fallback.post(CONTEXT["rootExecutionId"], suffix, body)

        emitter = self.emitter(RecordingIngestClient(respond))
        progress = ProgressLoopContext.start(
            emitter,
            agent_name="assistant",
            loop_kind="top_level",
            max_rounds=1,
            max_output_repairs=0,
            forced_finalization_available=True,
            max_tokens_per_inference=64,
            max_tool_calls=1,
            exact_tool_repeat_warning=True,
            exact_tool_repeat_block_after_warning=True,
        )

        with self.assertRaisesRegex(
            ToolLoopGuardBlocked, "immediate_exact_tool_repeat_blocked"
        ):
            emitter.start_tool(
                "folder_read",
                {"path": "fixture.txt"},
                "provider-call-1",
                progress.trace(round=1, phase="agent_loop"),
                repeat_comparable=True,
            )

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
