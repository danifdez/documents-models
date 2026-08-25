import unittest
from unittest.mock import patch

from common.execution_registry import TASK_HANDLERS
from lib.execution.step_executor import execute_assignment
from tasks.entities.entities import (
    entity_extraction_map,
    entity_extraction_reduce,
)


class EntityExtractionStepTest(unittest.TestCase):
    def test_registers_only_durable_entity_extraction_capabilities(self):
        self.assertIn("entity-extraction-map", TASK_HANDLERS)
        self.assertIn("entity-extraction-reduce", TASK_HANDLERS)
        self.assertNotIn("entity-extraction", TASK_HANDLERS)

    @patch(
        "tasks.entities.entities._extract_entities",
        return_value=[{"word": "Ada", "entity": "PERSON"}],
    )
    def test_map_extracts_one_self_contained_chunk(self, extract_entities):
        result = entity_extraction_map({"content": "Ada visited Paris."})

        self.assertEqual(
            result,
            {"entities": [{"word": "Ada", "entity": "PERSON"}]},
        )
        extract_entities.assert_called_once()

    def test_reduce_deduplicates_in_declared_map_order(self):
        result = entity_extraction_reduce(
            {
                "partials": [
                    [{"word": "Ada", "entity": "PERSON"}],
                    [
                        {"word": "ada", "entity": "ORG"},
                        {"word": "Paris", "entity": "GPE"},
                    ],
                ]
            }
        )

        self.assertEqual(
            result,
            {
                "entities": [
                    {"word": "Ada", "entity": "PERSON"},
                    {"word": "Paris", "entity": "GPE"},
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
                "taskType": "entity-extraction-map",
                "payload": {"content": ""},
            },
        }

        result = execute_assignment(assignment)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["code"], "STEP_EXECUTION_FAILED")
        self.assertEqual(result["output"]["outcome"], {"kind": "failed"})

    def test_reduce_executes_as_replayable_code_assignment(self):
        assignment = {
            "executionId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca701",
            "stepId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca702",
            "operationId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca703",
            "attemptId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca704",
            "stepKind": "code",
            "work": {
                "taskType": "entity-extraction-reduce",
                "payload": {
                    "partials": [
                        [{"word": "Ada", "entity": "PERSON"}],
                        [{"word": "Paris", "entity": "GPE"}],
                    ]
                },
            },
        }

        result = execute_assignment(assignment)

        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["output"]["kind"], "code")
        self.assertEqual(
            result["output"]["value"]["entities"],
            [
                {"word": "Ada", "entity": "PERSON"},
                {"word": "Paris", "entity": "GPE"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
