import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from lib.execution.step_executor import execute_assignment


class ArtifactAssignmentTest(unittest.TestCase):
    def assignment(self, task_type, step_kind, payload):
        return {
            "executionId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca701",
            "stepId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca702",
            "operationId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca703",
            "attemptId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca704",
            "stepKind": step_kind,
            "work": {"taskType": task_type, "payload": payload},
        }

    def test_extracts_a_document_from_its_assignment_artifact(self):
        result = execute_assignment(
            self.assignment(
                "document-extraction",
                "service",
                {"extension": ".txt"},
            ),
            {"document": b"Hello\nWorld"},
        )

        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(
            result["output"],
            {
                "kind": "service",
                "value": {"content": "<p>Hello</p><p>World</p>"},
            },
        )

    def test_rejects_document_extraction_without_its_artifact(self):
        result = execute_assignment(
            self.assignment(
                "document-extraction",
                "service",
                {"extension": ".txt"},
            )
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["code"], "STEP_EXECUTION_FAILED")

    def test_rejects_mhtml_without_an_html_document(self):
        result = execute_assignment(
            self.assignment(
                "document-extraction",
                "service",
                {"extension": ".mhtml"},
            ),
            {"document": b"Content-Type: text/plain\n\nNo HTML"},
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["code"], "STEP_EXECUTION_FAILED")

    @patch("lib.llm.config.get_task_config")
    @patch("tasks.transcribe.transcribe._get_model")
    def test_transcribes_the_assignment_media_artifact(
        self, get_model, get_task_config
    ):
        get_task_config.return_value = {
            "model": "test",
            "beam_size": 1,
        }
        model = Mock()
        model.transcribe.return_value = (
            [SimpleNamespace(text=" Hello "), SimpleNamespace(text="world")],
            SimpleNamespace(
                language="en",
                language_probability=0.99,
                duration=1.25,
            ),
        )
        get_model.return_value = model

        result = execute_assignment(
            self.assignment(
                "transcribe",
                "inference",
                {"extension": ".wav"},
            ),
            {"media": b"fake-wave"},
        )

        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(
            result["output"]["outcome"]["value"],
            {
                "transcript": "Hello world",
                "language": "en",
                "language_probability": 0.99,
                "duration": 1.25,
            },
        )


if __name__ == "__main__":
    unittest.main()
