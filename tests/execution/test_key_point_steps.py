import unittest
from unittest.mock import Mock, patch

from common.execution_registry import TASK_HANDLERS
from lib.execution.step_executor import execute_assignment
from tasks.key_points.key_points import key_point_map, key_point_reduce


class KeyPointStepTest(unittest.TestCase):
    def test_registers_only_durable_key_point_capabilities(self):
        self.assertIn("key-point-map", TASK_HANDLERS)
        self.assertIn("key-point-reduce", TASK_HANDLERS)
        self.assertNotIn("key-point", TASK_HANDLERS)

    @patch(
        "tasks.key_points.key_points._extract_candidates",
        return_value=["Backend coordinates every durable execution"],
    )
    def test_map_extracts_one_self_contained_chunk(self, extract_candidates):
        result = key_point_map(
            {
                "content": "Backend coordinates durable executions.",
                "targetLanguage": "en",
            }
        )

        self.assertEqual(
            result,
            {"key_points": ["Backend coordinates every durable execution"]},
        )
        extract_candidates.assert_called_once()

    @patch("tasks.key_points.key_points.get_llm_service")
    def test_reduce_selects_points_with_one_inference(self, get_llm_service):
        llm = Mock()
        llm.ask.return_value = (
            "1. Backend persists the workflow state\n"
            "2. Workers execute bounded assignments"
        )
        get_llm_service.return_value = llm

        result = key_point_reduce(
            {
                "targetLanguage": "en",
                "partials": [
                    ["Backend persists the workflow state"],
                    [
                        "Workers execute bounded assignments",
                        "backend persists the workflow state",
                    ],
                ],
            }
        )

        self.assertEqual(
            result,
            {
                "key_points": [
                    "Backend persists the workflow state",
                    "Workers execute bounded assignments",
                ]
            },
        )
        llm.ask.assert_called_once()

    def test_map_failure_becomes_a_failed_step_result(self):
        assignment = {
            "executionId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca701",
            "stepId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca702",
            "operationId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca703",
            "attemptId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca704",
            "stepKind": "inference",
            "work": {
                "taskType": "key-point-map",
                "payload": {"content": "", "targetLanguage": "en"},
            },
        }

        result = execute_assignment(assignment)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["code"], "STEP_EXECUTION_FAILED")
        self.assertEqual(result["output"]["outcome"], {"kind": "failed"})

    @patch("tasks.key_points.key_points.get_llm_service")
    def test_reduce_executes_as_inference_assignment(self, get_llm_service):
        llm = Mock()
        llm.ask.return_value = "Backend stores durable workflow state"
        get_llm_service.return_value = llm
        assignment = {
            "executionId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca701",
            "stepId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca702",
            "operationId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca703",
            "attemptId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca704",
            "stepKind": "inference",
            "work": {
                "taskType": "key-point-reduce",
                "payload": {
                    "targetLanguage": "en",
                    "partials": [["Backend stores durable workflow state"]],
                },
            },
        }

        result = execute_assignment(assignment)

        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["output"]["kind"], "inference")
        self.assertEqual(
            result["output"]["outcome"]["value"],
            {"key_points": ["Backend stores durable workflow state"]},
        )


if __name__ == "__main__":
    unittest.main()
