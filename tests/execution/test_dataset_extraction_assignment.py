import unittest

from lib.execution.step_executor import execute_assignment


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


if __name__ == "__main__":
    unittest.main()
