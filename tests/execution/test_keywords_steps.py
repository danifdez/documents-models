import unittest
from unittest.mock import patch

from common.execution_registry import TASK_HANDLERS
from lib.execution.step_executor import execute_assignment
from tasks.keywords.keywords import keywords_map, keywords_reduce


class KeywordsStepTest(unittest.TestCase):
    def test_registers_only_durable_keywords_capabilities(self):
        self.assertIn("keywords-map", TASK_HANDLERS)
        self.assertIn("keywords-reduce", TASK_HANDLERS)
        self.assertNotIn("keywords", TASK_HANDLERS)

    @patch(
        "tasks.keywords.keywords._extract_candidates",
        return_value=["durable workflows", "PostgreSQL"],
    )
    def test_map_extracts_one_self_contained_chunk(self, extract_candidates):
        result = keywords_map(
            {"content": "Durable workflows use PostgreSQL.", "targetLanguage": "en"}
        )

        self.assertEqual(
            result,
            {"keywords": ["durable workflows", "PostgreSQL"]},
        )
        extract_candidates.assert_called_once()

    def test_reduce_ranks_frequency_then_first_appearance(self):
        result = keywords_reduce(
            {
                "partials": [
                    ["Durable workflows", "PostgreSQL", "Durable workflows"],
                    ["postgresql", "execution evidence"],
                    ["PostgreSQL", "retries"],
                ]
            }
        )

        self.assertEqual(
            result,
            {
                "keywords": [
                    "PostgreSQL",
                    "Durable workflows",
                    "execution evidence",
                    "retries",
                ]
            },
        )

    def test_map_failure_becomes_a_failed_step_result(self):
        assignment = {
            "executionId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca701",
            "stepId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca702",
            "operationId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca703",
            "attemptId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca704",
            "stepKind": "inference",
            "work": {
                "taskType": "keywords-map",
                "payload": {"content": "", "targetLanguage": "en"},
            },
        }

        result = execute_assignment(assignment)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["code"], "STEP_EXECUTION_FAILED")
        self.assertEqual(result["output"]["outcome"], {"kind": "failed"})

    def test_reduce_rejects_an_empty_map_result(self):
        with self.assertRaisesRegex(ValueError, "string arrays"):
            keywords_reduce({"partials": [["PostgreSQL"], []]})

    def test_reduce_executes_as_code_assignment(self):
        assignment = {
            "executionId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca701",
            "stepId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca702",
            "operationId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca703",
            "attemptId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca704",
            "stepKind": "code",
            "work": {
                "taskType": "keywords-reduce",
                "payload": {"partials": [["PostgreSQL"], ["durable workflows"]]},
            },
        }

        result = execute_assignment(assignment)

        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["output"]["kind"], "code")
        self.assertEqual(
            result["output"]["value"],
            {"keywords": ["PostgreSQL", "durable workflows"]},
        )


if __name__ == "__main__":
    unittest.main()
