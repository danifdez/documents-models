import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from common.execution_registry import TASK_HANDLERS
from lib.execution.protocol_client import ExecutionProtocolClient
from lib.execution.step_executor import execute_assignment
import executions


class _Response:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return json.dumps(self.value).encode("utf-8")


class StepProtocolTest(unittest.TestCase):
    def test_registers_once_and_uses_worker_credential(self):
        requests = []

        def urlopen(request, timeout):
            requests.append((request, timeout))
            return _Response({"credential": "worker-secret"})

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"MODELS_ENROLLMENT_TOKEN": "enrollment"}, clear=False
        ), patch(
            "lib.execution.protocol_client.worker_data_dir",
            return_value=directory,
        ), patch(
            "urllib.request.urlopen", side_effect=urlopen
        ):
            client = ExecutionProtocolClient()
            client.ensure_registered(["detect-language"], {"runtime": "test"})
            request = requests[0][0]
            self.assertEqual(
                request.full_url, "http://localhost:3000/models-work/register"
            )
            self.assertEqual(
                request.headers["X-models-enrollment-token"], "enrollment"
            )
            self.assertEqual(
                (Path(directory) / ".worker_credential").read_text(
                    encoding="utf-8"
                ),
                "worker-secret",
            )

    def test_executes_a_deterministic_assignment_without_database_context(self):
        task_type = "protocol-test-task"
        TASK_HANDLERS[task_type] = lambda payload: {
            "echo": payload["value"],
            "artifact": payload["_input_artifacts"]["source"].decode(
                "utf-8"
            ),
        }
        assignment = {
            "executionId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca701",
            "stepId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca702",
            "operationId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca703",
            "attemptId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca704",
            "stepKind": "service",
            "work": {"taskType": task_type, "payload": {"value": "ok"}},
        }
        try:
            with patch(
                "lib.execution.step_executor.ensure_task_handler",
                return_value=True,
            ):
                result = execute_assignment(
                    assignment, {"source": b"artifact body"}
                )
        finally:
            TASK_HANDLERS.pop(task_type, None)

        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(
            result["output"],
            {
                "kind": "service",
                "value": {"echo": "ok", "artifact": "artifact body"},
            },
        )
        self.assertIsNone(result["error"])

    def test_wraps_inference_output_in_canonical_outcome(self):
        task_type = "protocol-inference-test"
        TASK_HANDLERS[task_type] = lambda _payload: {"response": "summary"}
        assignment = {
            "executionId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca701",
            "stepId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca702",
            "operationId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca703",
            "attemptId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca704",
            "stepKind": "inference",
            "work": {"taskType": task_type, "payload": {}},
        }
        try:
            with patch(
                "lib.execution.step_executor.ensure_task_handler",
                return_value=True,
            ), patch(
                "lib.execution.step_executor.get_task_config",
                return_value={"model": "test-model"},
            ):
                result = execute_assignment(assignment)
        finally:
            TASK_HANDLERS.pop(task_type, None)

        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(
            result["output"],
            {
                "kind": "inference",
                "outcome": {
                    "kind": "structured_result",
                    "schemaId": f"{task_type}-output/1",
                    "value": {"response": "summary"},
                },
            },
        )
        self.assertEqual(result["inference"]["effectiveModel"], "test-model")
        self.assertEqual(result["usage"]["totalTokens"], None)

    def test_keeps_a_result_until_backend_acknowledges_it(self):
        result = {"attemptId": "attempt-1", "status": "succeeded"}

        class Client:
            def submit_result(self, submitted):
                self.submitted = submitted
                return {"code": "received"}

        client = Client()
        with tempfile.TemporaryDirectory() as directory, patch(
            "executions.worker_data_dir", return_value=directory
        ):
            executions._store_pending(result)
            pending = Path(directory) / ".pending_step_result.json"
            self.assertTrue(pending.exists())
            executions._deliver_pending(client)
            self.assertEqual(client.submitted, result)
            self.assertFalse(pending.exists())


if __name__ == "__main__":
    unittest.main()
