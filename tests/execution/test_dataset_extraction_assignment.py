import unittest
from unittest.mock import patch

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

    @patch("tasks.dataset_extraction.handler.get_llm_service")
    def test_source_pagination_constrains_generation_and_persisted_anchor(self, get_llm):
        completion = '{"name":{"value":"Ada","_quote":"Ada","_page":3}}'
        payload = {
            "documentText": "Ada",
            "resourceId": 1,
            "schema": [{"key": "name", "description": "Person name"}],
            "model": "test-model",
        }
        generate = get_llm.return_value.generate

        for source_metadata, page_rule, expected_page, retry in (
            ({}, "(integer | null-lit)", 3, False),
            ({"sourcePagination": "unpaginated"}, "null-lit", None, False),
            ({"sourcePagination": "unpaginated"}, "null-lit", None, True),
        ):
            with self.subTest(source_metadata=source_metadata, retry=retry):
                generate.reset_mock()
                generate.side_effect = ["not JSON", completion] if retry else [completion]
                result = execute_assignment(self.assignment({**payload, **source_metadata}))
                self.assertEqual(result["status"], "succeeded")
                value = result["output"]["outcome"]["value"]
                self.assertEqual(value["data"]["name"], "Ada")
                self.assertEqual(value["cellMetadata"]["name"]["quote"], "Ada")
                self.assertEqual(value["cellMetadata"]["name"]["page"], expected_page)
                prompt, = generate.call_args_list[0].args
                self.assertEqual("This document has no page numbers." in prompt, bool(source_metadata))
                grammar = generate.call_args_list[0].kwargs["grammar"]
                page_field = next(line for line in grammar.splitlines() if "_page" in line)
                self.assertIn(f"ws {page_rule} ws", page_field)
                if retry:
                    self.assertNotIn("grammar", generate.call_args_list[1].kwargs)

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
