import threading
import unittest
from unittest.mock import ANY, Mock

from lib.execution.assignment_runner import run_assignment
from lib.execution.protocol_client import ProtocolTransportError
from lib.execution.result_outbox import ResultOutbox
from lib.execution.worker_runtime import WorkerRuntime


FIRST_ATTEMPT_ID = "018f1d8a-54d7-7d63-a1ee-5e9a6adca704"
SECOND_ATTEMPT_ID = "018f1d8a-54d7-7d63-a1ee-5e9a6adca705"


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


if __name__ == "__main__":
    unittest.main()
