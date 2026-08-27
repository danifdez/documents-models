import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import ANY, Mock

from lib.execution.assignment_runner import run_assignment
from lib.execution.handler_process_pool import (
    HandlerProcessFailed,
    HandlerProcessPool,
)
from lib.execution.protocol_client import (
    ProtocolRejectionError,
    ProtocolTransportError,
)
from lib.execution.result_outbox import ResultOutbox
from lib.execution.worker_runtime import WorkerRuntime


FIRST_ATTEMPT_ID = "018f1d8a-54d7-7d63-a1ee-5e9a6adca704"
SECOND_ATTEMPT_ID = "018f1d8a-54d7-7d63-a1ee-5e9a6adca705"


def _artifact_policy(**overrides):
    return {
        "classification": "workspace",
        "allowedPurposes": ["execution"],
        "allowedDestinations": ["documents-models"],
        "retentionClass": "operational",
        "expiresAt": "2999-01-01T00:00:00Z",
        "sourceRefs": [],
        **overrides,
    }


def _blocking_process_executor(
    assignment,
    _artifacts,
    output_artifacts=None,
):
    Path(assignment["markerPath"]).write_text(
        str(os.getpid()), encoding="utf-8"
    )
    while True:
        time.sleep(0.1)


class WorkerRuntimeTest(unittest.TestCase):
    def test_stores_a_cancelled_result_without_running_the_handler(self):
        client = Mock()
        client.read_control.return_value = {"cancelled": True}
        outbox = Mock(spec=ResultOutbox)
        assignment = {
            "executionId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca701",
            "stepId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca702",
            "operationId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca703",
            "attemptId": FIRST_ATTEMPT_ID,
            "stepKind": "service",
        }

        run_assignment(client, outbox, assignment, threading.Event())

        outbox.store.assert_called_once_with(
            {
                "schemaVersion": "step-result/1",
                "executionId": assignment["executionId"],
                "stepId": assignment["stepId"],
                "operationId": assignment["operationId"],
                "attemptId": FIRST_ATTEMPT_ID,
                "stepKind": "service",
                "status": "cancelled",
                "codeFingerprint": ANY,
                "runtimeFingerprint": ANY,
                "artifactRefs": [],
                "error": None,
            }
        )
        client.download_artifact.assert_not_called()

    def test_stops_downloading_artifacts_when_cancellation_is_observed(self):
        cancellation = threading.Event()
        client = Mock()
        client.read_control.return_value = {"cancelled": False}
        client.download_artifact.side_effect = lambda *_args: (
            cancellation.set() or b"first"
        )
        outbox = Mock(spec=ResultOutbox)
        assignment = {
            "executionId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca701",
            "stepId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca702",
            "operationId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca703",
            "attemptId": FIRST_ATTEMPT_ID,
            "stepKind": "service",
            "inputArtifactRefs": [
                {
                    "role": "first",
                    "artifactId": "artifact-1",
                    "dataPolicy": _artifact_policy(),
                },
                {
                    "role": "second",
                    "artifactId": "artifact-2",
                    "dataPolicy": _artifact_policy(),
                },
            ],
        }

        run_assignment(client, outbox, assignment, cancellation)

        client.download_artifact.assert_called_once_with(
            FIRST_ATTEMPT_ID, "artifact-1"
        )
        self.assertEqual(outbox.store.call_args.args[0]["status"], "cancelled")

    def test_confirms_cancellation_after_an_artifact_download_is_rejected(self):
        client = Mock()
        client.read_control.side_effect = [
            {"cancelled": False},
            {"cancelled": True},
        ]
        client.download_artifact.side_effect = ProtocolTransportError(
            "artifact_not_authorized"
        )
        outbox = Mock(spec=ResultOutbox)
        assignment = {
            "executionId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca701",
            "stepId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca702",
            "operationId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca703",
            "attemptId": FIRST_ATTEMPT_ID,
            "stepKind": "service",
            "inputArtifactRefs": [
                {
                    "role": "source",
                    "artifactId": "artifact-1",
                    "dataPolicy": _artifact_policy(),
                }
            ],
        }

        run_assignment(client, outbox, assignment, threading.Event())

        self.assertEqual(outbox.store.call_args.args[0]["status"], "cancelled")

    def test_preserves_an_active_artifact_download_failure_for_retry(self):
        client = Mock()
        client.read_control.return_value = {"cancelled": False}
        client.download_artifact.side_effect = ProtocolTransportError(
            "offline"
        )
        outbox = Mock(spec=ResultOutbox)
        assignment = {
            "executionId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca701",
            "stepId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca702",
            "operationId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca703",
            "attemptId": FIRST_ATTEMPT_ID,
            "stepKind": "service",
            "inputArtifactRefs": [
                {
                    "role": "source",
                    "artifactId": "artifact-1",
                    "dataPolicy": _artifact_policy(),
                }
            ],
        }

        with self.assertRaisesRegex(ProtocolTransportError, "offline"):
            run_assignment(client, outbox, assignment, threading.Event())

        outbox.store.assert_not_called()

    def test_rejects_an_expired_artifact_before_downloading_or_running(self):
        client = Mock()
        client.read_control.return_value = {"cancelled": False}
        outbox = Mock(spec=ResultOutbox)
        handler = Mock()
        assignment = {
            "executionId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca701",
            "stepId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca702",
            "operationId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca703",
            "attemptId": FIRST_ATTEMPT_ID,
            "stepKind": "service",
            "inputArtifactRefs": [
                {
                    "role": "source",
                    "artifactId": "artifact-1",
                    "dataPolicy": _artifact_policy(
                        expiresAt="2000-01-01T00:00:00Z"
                    ),
                }
            ],
        }

        run_assignment(
            client,
            outbox,
            assignment,
            threading.Event(),
            handler_executor=handler,
        )

        client.download_artifact.assert_not_called()
        handler.assert_not_called()
        result = outbox.store.call_args.args[0]
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["code"], "ARTIFACT_EXPIRED")

    def test_rejects_an_unauthorized_destination_before_any_download(self):
        client = Mock()
        client.read_control.return_value = {"cancelled": False}
        outbox = Mock(spec=ResultOutbox)
        handler = Mock()
        assignment = {
            "executionId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca701",
            "stepId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca702",
            "operationId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca703",
            "attemptId": FIRST_ATTEMPT_ID,
            "stepKind": "service",
            "inputArtifactRefs": [
                {
                    "role": "source",
                    "artifactId": "artifact-1",
                    "dataPolicy": _artifact_policy(
                        allowedDestinations=["ia-browser"]
                    ),
                }
            ],
        }

        run_assignment(
            client,
            outbox,
            assignment,
            threading.Event(),
            handler_executor=handler,
        )

        client.download_artifact.assert_not_called()
        handler.assert_not_called()
        result = outbox.store.call_args.args[0]
        self.assertEqual(result["error"]["code"], "ARTIFACT_DESTINATION_DENIED")

    def test_rejects_secret_input_without_starting_inference(self):
        client = Mock()
        client.read_control.return_value = {"cancelled": False}
        outbox = Mock(spec=ResultOutbox)
        handler = Mock()
        assignment = {
            "executionId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca701",
            "stepId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca702",
            "operationId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca703",
            "attemptId": FIRST_ATTEMPT_ID,
            "stepKind": "inference",
            "inputArtifactRefs": [
                {
                    "role": "source",
                    "artifactId": "artifact-1",
                    "dataPolicy": _artifact_policy(classification="secret"),
                }
            ],
        }

        run_assignment(
            client,
            outbox,
            assignment,
            threading.Event(),
            handler_executor=handler,
        )

        client.download_artifact.assert_not_called()
        handler.assert_not_called()
        result = outbox.store.call_args.args[0]
        self.assertEqual(result["error"]["code"], "SECRET_INPUT_REJECTED")
        self.assertEqual(result["inference"]["effectiveModel"], "not_executed")
        self.assertEqual(result["usage"]["totalTokens"], None)

    def test_leaves_a_crashed_handler_attempt_to_lease_recovery(self):
        client = Mock()
        client.read_control.return_value = {"cancelled": False}
        outbox = Mock(spec=ResultOutbox)
        assignment = {
            "executionId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca701",
            "stepId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca702",
            "operationId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca703",
            "attemptId": FIRST_ATTEMPT_ID,
            "stepKind": "service",
        }

        def crashed_handler(*_args):
            raise HandlerProcessFailed("process exited")

        run_assignment(
            client,
            outbox,
            assignment,
            threading.Event(),
            handler_executor=crashed_handler,
        )

        outbox.store.assert_not_called()

    def test_runs_two_assignments_in_independent_slots(self):
        both_running = threading.Event()
        release = threading.Event()
        started = []
        started_lock = threading.Lock()

        def runner(_client, _outbox, assignment, _cancellation):
            with started_lock:
                started.append(assignment["attemptId"])
                if len(started) == 2:
                    both_running.set()
            release.wait(timeout=2)

        client = Mock()
        runtime = WorkerRuntime(client, Mock(), 2, 30_000, runner=runner)
        try:
            runtime.submit({"attemptId": FIRST_ATTEMPT_ID})
            runtime.submit({"attemptId": SECOND_ATTEMPT_ID})

            self.assertTrue(both_running.wait(timeout=1))
            self.assertEqual(runtime.available_slots, 0)
            self.assertEqual(
                runtime.active_attempt_ids,
                {FIRST_ATTEMPT_ID, SECOND_ATTEMPT_ID},
            )
            self.assertEqual(client.start.call_count, 2)
        finally:
            release.set()
            runtime.close()

    def test_renews_each_active_lease_and_propagates_cancellation(self):
        release = threading.Event()

        def runner(_client, _outbox, _assignment, cancellation):
            release.wait(timeout=2)

        client = Mock()
        client.renew_lease.return_value = {"cancelled": True}
        runtime = WorkerRuntime(client, Mock(), 1, 30_000, runner=runner)
        try:
            runtime.submit({"attemptId": FIRST_ATTEMPT_ID})
            active = runtime.active[FIRST_ATTEMPT_ID]
            runtime.maintain_leases(active.next_lease_renewal)

            self.assertTrue(active.cancellation.is_set())
            client.renew_lease.assert_called_once_with(
                FIRST_ATTEMPT_ID, 30_000
            )
        finally:
            release.set()
            runtime.close()

    def test_retries_a_transient_lease_failure_without_losing_the_slot(self):
        release = threading.Event()
        client = Mock()
        client.renew_lease.side_effect = ProtocolTransportError("offline")
        runtime = WorkerRuntime(
            client,
            Mock(),
            1,
            30_000,
            runner=lambda *_args: release.wait(timeout=2),
        )
        try:
            runtime.submit({"attemptId": FIRST_ATTEMPT_ID})
            active = runtime.active[FIRST_ATTEMPT_ID]
            renewal_at = active.next_lease_renewal
            runtime.maintain_leases(renewal_at)

            self.assertEqual(runtime.available_slots, 0)
            self.assertEqual(active.next_lease_renewal, renewal_at + 1)
        finally:
            release.set()
            runtime.close()

    def test_polls_control_and_preempts_between_lease_renewals(self):
        release = threading.Event()
        client = Mock()
        client.read_control.return_value = {"cancelled": True}
        runtime = WorkerRuntime(
            client,
            Mock(),
            1,
            30_000,
            runner=lambda *_args: release.wait(timeout=2),
        )
        try:
            runtime.submit({"attemptId": FIRST_ATTEMPT_ID})
            active = runtime.active[FIRST_ATTEMPT_ID]

            runtime.maintain_leases(active.next_control_poll)

            self.assertTrue(active.cancellation.is_set())
            client.read_control.assert_called_once_with(FIRST_ATTEMPT_ID)
            client.renew_lease.assert_not_called()
        finally:
            release.set()
            runtime.close()

    def test_preempts_after_a_terminal_lease_rejection(self):
        release = threading.Event()
        client = Mock()
        client.renew_lease.side_effect = ProtocolRejectionError(
            "/models-work/attempts/attempt/lease",
            409,
            "lease_expired",
            '{"message":"lease_expired"}',
        )
        runtime = WorkerRuntime(
            client,
            Mock(),
            1,
            30_000,
            runner=lambda *_args: release.wait(timeout=2),
        )
        try:
            runtime.submit({"attemptId": FIRST_ATTEMPT_ID})
            active = runtime.active[FIRST_ATTEMPT_ID]
            active.next_control_poll = active.next_lease_renewal + 1

            runtime.maintain_leases(active.next_lease_renewal)

            self.assertTrue(active.cancellation.is_set())
            client.renew_lease.assert_called_once()
        finally:
            release.set()
            runtime.close()

    def test_preempts_when_the_local_lease_window_elapses(self):
        release = threading.Event()
        client = Mock()
        runtime = WorkerRuntime(
            client,
            Mock(),
            1,
            30_000,
            runner=lambda *_args: release.wait(timeout=2),
        )
        try:
            runtime.submit({"attemptId": FIRST_ATTEMPT_ID})
            active = runtime.active[FIRST_ATTEMPT_ID]

            runtime.maintain_leases(active.lease_expires_at)

            self.assertTrue(active.cancellation.is_set())
            client.read_control.assert_not_called()
            client.renew_lease.assert_not_called()
        finally:
            release.set()
            runtime.close()

    def test_preempts_the_running_process_and_stores_cancellation(self):
        client = Mock()
        client.read_control.return_value = {"cancelled": False}
        outbox = Mock(spec=ResultOutbox)
        pool = HandlerProcessPool(1, executor=_blocking_process_executor)
        runtime = WorkerRuntime(
            client,
            outbox,
            1,
            30_000,
            handler_pool=pool,
        )
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "handler.pid"
            assignment = {
                "executionId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca701",
                "stepId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca702",
                "operationId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca703",
                "attemptId": FIRST_ATTEMPT_ID,
                "stepKind": "service",
                "markerPath": str(marker),
            }
            try:
                runtime.submit(assignment)
                deadline = time.monotonic() + 3
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(marker.exists())
                active = runtime.active[FIRST_ATTEMPT_ID]
                client.read_control.return_value = {"cancelled": True}

                runtime.maintain_leases(active.next_control_poll)
                active.future.result(timeout=3)
                runtime.collect_completed()

                stored = outbox.store.call_args.args[0]
                self.assertEqual(stored["status"], "cancelled")
                self.assertEqual(runtime.available_slots, 1)
            finally:
                runtime.close()

    def test_shutdown_preempts_a_running_handler_process(self):
        client = Mock()
        client.read_control.return_value = {"cancelled": False}
        outbox = Mock(spec=ResultOutbox)
        pool = HandlerProcessPool(1, executor=_blocking_process_executor)
        runtime = WorkerRuntime(
            client,
            outbox,
            1,
            30_000,
            handler_pool=pool,
        )
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "handler.pid"
            assignment = {
                "executionId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca701",
                "stepId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca702",
                "operationId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca703",
                "attemptId": FIRST_ATTEMPT_ID,
                "stepKind": "service",
                "markerPath": str(marker),
            }
            runtime.submit(assignment)
            deadline = time.monotonic() + 3
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(marker.exists())

            started = time.monotonic()
            runtime.close()

            self.assertLess(time.monotonic() - started, 3)
            self.assertEqual(outbox.store.call_args.args[0]["status"], "cancelled")


if __name__ == "__main__":
    unittest.main()
