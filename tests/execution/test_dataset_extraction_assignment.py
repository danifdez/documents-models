import unittest

from lib.execution.step_executor import execute_assignment
from tasks.dataset_extraction.handler import extract_dataset_row_reduce


class DatasetExtractionAssignmentTest(unittest.TestCase):
    def assignment(self, payload):
        return {
            "executionId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca701",
            "stepId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca702",
            "operationId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca703",
            "attemptId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca704",
            "stepKind": "inference",
            "work": {"taskType": "dataset.extract-row", "payload": payload},
        }

    def test_returns_a_failed_step_result_for_a_resource_without_content(self):
        result = execute_assignment(
            self.assignment({"documentText": "   "})
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["code"], "STEP_EXECUTION_FAILED")
        self.assertEqual(result["output"]["outcome"], {"kind": "failed"})

    def test_returns_a_structured_result_without_a_domain_error_field(self):
        result = execute_assignment(
            self.assignment(
                {
                    "documentText": "Readable content",
                    "schema": [{"key": "name", "description": ""}],
                    "model": "test-model",
                }
            )
        )

        self.assertEqual(result["status"], "succeeded")
        value = result["output"]["outcome"]["value"]
        self.assertEqual(value["data"], {})
        self.assertEqual(value["cellMetadata"], {})
        self.assertNotIn("error", value)

    def test_reduce_prefers_grounded_values_and_preserves_source_metadata(self):
        result = extract_dataset_row_reduce(
            {
                "final": True,
                "partials": [
                    [
                        {
                            "chunkIndex": 0,
                            "data": {"name": "Wrong", "year": None},
                            "cellMetadata": {"name": {"quote": ""}},
                            "model": "test-model",
                            "promptVersion": "test-prompt",
                        }
                    ],
                    [
                        {
                            "chunkIndex": 1,
                            "data": {"name": "Ada", "year": 1843},
                            "cellMetadata": {
                                "name": {"quote": "Ada Lovelace"},
                                "year": {"quote": "In 1843"},
                            },
                            "model": "test-model",
                            "promptVersion": "test-prompt",
                        }
                    ],
                ],
            }
        )

        self.assertEqual(result["data"], {"name": "Ada", "year": 1843})
        self.assertEqual(
            result["cellMetadata"]["name"]["quote"], "Ada Lovelace"
        )
        self.assertEqual(result["cellMetadata"]["year"]["quote"], "In 1843")
        self.assertEqual(result["model"], "test-model")

    def test_intermediate_reduce_keeps_one_bounded_candidate(self):
        result = extract_dataset_row_reduce(
            {
                "final": False,
                "partials": [
                    [
                        {
                            "chunkIndex": 0,
                            "data": {"name": "Ada"},
                            "cellMetadata": {"name": {"quote": "Ada"}},
                            "model": "test-model",
                            "promptVersion": "test-prompt",
                        }
                    ],
                    [
                        {
                            "chunkIndex": 1,
                            "data": {"year": 1843},
                            "cellMetadata": {"year": {"quote": "1843"}},
                            "model": "test-model",
                            "promptVersion": "test-prompt",
                        }
                    ],
                ],
            }
        )

        self.assertEqual(len(result["candidates"]), 1)
        self.assertEqual(
            result["candidates"][0]["data"],
            {"name": "Ada", "year": 1843},
        )


if __name__ == "__main__":
    unittest.main()
