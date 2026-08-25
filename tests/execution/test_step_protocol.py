import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from common.execution_registry import TASK_HANDLERS
from lib.execution.code_identity import code_fingerprint
from lib.execution.protocol_client import ExecutionProtocolClient
from lib.execution.step_executor import _inference_metadata, execute_assignment
from lib.execution.outcome import InferenceOutcome
from lib.execution.runtime_identity import runtime_fingerprint
from lib.llm.config import active_deployments
from lib.llm.prompts import prompt_package_fingerprint
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
    def test_worker_advertises_migrated_inference_capabilities(self):
        self.assertIn("assistant-chat", executions.CAPABILITIES)
        self.assertIn("agent-chat", executions.CAPABILITIES)
        self.assertIn("document-extraction", executions.CAPABILITIES)
        self.assertIn("dataset.propose-columns", executions.CAPABILITIES)
        self.assertIn("distribution", executions.CAPABILITIES)
        self.assertIn("query", executions.CAPABILITIES)
        self.assertIn("transcribe", executions.CAPABILITIES)
        self.assertIn("translate", executions.CAPABILITIES)

    def test_worker_filters_tasks_by_effective_requirements(self):
        with patch(
            "worker.capabilities.get_worker_config",
            return_value={"disable_llm": True, "disable_embeddings": True},
        ):
            self.assertEqual(
                executions.effective_task_capabilities(),
                [
                    "detect-language",
                    "document-extraction",
                    "distribution",
                    "correlation",
                    "correlation-matrix",
                    "group-by",
                    "time-series",
                    "outliers",
                    "pivot-table",
                    "summary",
                    "query",
                    "chart",
                    "transcribe",
                    "translate",
                ],
            )

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
        self.assertEqual(result["codeFingerprint"], code_fingerprint())
        self.assertEqual(result["runtimeFingerprint"], runtime_fingerprint())

    def test_runtime_fingerprint_changes_with_the_dependency_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = root / "requirements.txt"
            lock.write_text("package==1.0.0\n", encoding="utf-8")
            first = runtime_fingerprint(root)
            lock.write_text("package==2.0.0\n", encoding="utf-8")
            second = runtime_fingerprint(root)

        self.assertRegex(first, r"^sha256:[0-9a-f]{64}$")
        self.assertNotEqual(first, second)

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
        self.assertIsNone(result["inference"]["effectiveAdapter"])
        self.assertEqual(
            result["inference"]["effectivePromptPackages"],
            [prompt_package_fingerprint()],
        )
        self.assertEqual(result["usage"]["totalTokens"], None)

    def test_fingerprints_managed_prompt_resources_deterministically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt = root / "tasks" / "summarize" / "prompt.md"
            prompt.parent.mkdir(parents=True)
            prompt.write_text("Summarize this.\n", encoding="utf-8")
            prompt.chmod(0o644)
            ignored = root / "tasks" / "summarize" / "notes.txt"
            ignored.write_text("not a prompt", encoding="utf-8")
            expected = hashlib.sha256(
                b"tasks/summarize/prompt.md\0"
                b"0644\0"
                b"Summarize this.\n\0"
            ).hexdigest()

            with patch("lib.llm.prompts._PROJECT_DIR", directory):
                actual = prompt_package_fingerprint()

        self.assertEqual(actual, expected)

    def test_reports_the_deployed_adapter_without_exposing_its_path(self):
        task_type = "protocol-adapter-test"
        TASK_HANDLERS[task_type] = lambda _payload: {"response": "summary"}
        assignment = {
            "executionId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca701",
            "stepId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca702",
            "operationId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca703",
            "attemptId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca704",
            "stepKind": "inference",
            "work": {"taskType": task_type, "payload": {}},
        }
        identity = {
            "available": True,
            "scale": 0.75,
            "sha256": "a" * 64,
        }
        expected = "sha256:" + hashlib.sha256(
            json.dumps(
                identity,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        try:
            with patch(
                "lib.execution.step_executor.ensure_task_handler",
                return_value=True,
            ), patch(
                "lib.execution.step_executor.get_task_config",
                return_value={
                    "model": "test-model",
                    "lora_path": "/private/adapter.gguf",
                    "lora_scale": 0.75,
                    "_deployment_sha256": "a" * 64,
                },
            ):
                result = execute_assignment(assignment)
        finally:
            TASK_HANDLERS.pop(task_type, None)

        self.assertEqual(result["inference"]["effectiveAdapter"], expected)
        self.assertNotIn("/private/adapter.gguf", json.dumps(result))

    def test_preserves_the_verified_deployment_checksum_in_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deployments.json"
            path.write_text(json.dumps({"tasks": {"summarize": {
                "enabled": True,
                "path": "/models/adapter.gguf",
                "scale": 0.5,
                "sha256": "b" * 64,
            }}}), encoding="utf-8")
            with patch.dict(
                "os.environ", {"MODELS_DEPLOYMENTS_PATH": str(path)}
            ):
                deployments = active_deployments()

        self.assertEqual(deployments["summarize"]["sha256"], "b" * 64)

    def test_does_not_claim_an_identity_for_an_unverified_adapter(self):
        with patch(
            "lib.execution.step_executor.get_task_config",
            return_value={
                "model": "test-model",
                "lora_path": "/private/unverified.gguf",
            },
        ):
            metadata = _inference_metadata("summarize", 0, "completed")

        self.assertNotIn("effectiveAdapter", metadata["inference"])
        self.assertEqual(
            metadata["inference"]["effectivePromptPackages"],
            [prompt_package_fingerprint()],
        )

    def test_preserves_an_explicit_canonical_inference_outcome(self):
        task_type = "protocol-chat-test"
        TASK_HANDLERS[task_type] = lambda _payload: InferenceOutcome(
            {"kind": "final_text", "text": "Ready"}
        )
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

        self.assertEqual(
            result["output"]["outcome"],
            {"kind": "final_text", "text": "Ready"},
        )

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
