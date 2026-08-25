import unittest
from unittest.mock import Mock, patch

from lib.execution.step_executor import execute_assignment


class DatasetProposeAssignmentTest(unittest.TestCase):
    def assignment(self, payload):
        return {
            "executionId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca701",
            "stepId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca702",
            "operationId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca703",
            "attemptId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca704",
            "stepKind": "inference",
            "work": {
                "taskType": "dataset.propose-columns",
                "payload": payload,
            },
        }

    @patch("tasks.dataset_extraction.propose_columns.get_llm_params")
    @patch("tasks.dataset_extraction.propose_columns.get_task_config")
    @patch("tasks.dataset_extraction.propose_columns.get_llm_service")
    def test_returns_a_canonical_structured_result(
        self, get_llm_service, get_task_config, get_llm_params
    ):
        get_task_config.return_value = {
            "model": "test-model",
            "max_tokens": 100,
        }
        get_llm_params.return_value = {}
        llm = Mock()
        llm.generate.return_value = (
            '[{"key":"published_at","name":"Published at",'
            '"type":"date","description":"Publication date"}]'
        )
        get_llm_service.return_value = llm

        result = execute_assignment(
            self.assignment(
                {
                    "resources": [
                        {"id": 7, "title": "Article", "excerpt": "Text"}
                    ]
                }
            )
        )

        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(
            result["output"]["outcome"]["value"],
            {
                "columns": [
                    {
                        "key": "published_at",
                        "name": "Published at",
                        "type": "date",
                        "description": "Publication date",
                        "required": False,
                    }
                ]
            },
        )

    def test_rejects_an_assignment_without_resources(self):
        result = execute_assignment(self.assignment({"resources": []}))

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["code"], "STEP_EXECUTION_FAILED")


if __name__ == "__main__":
    unittest.main()
