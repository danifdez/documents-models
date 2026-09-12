import unittest
from unittest.mock import patch

from common.execution_registry import TASK_HANDLERS
from lib.execution.step_executor import execute_assignment
from tasks.entities.entities import (
    _extract_entities,
    _text_windows,
    entity_extraction_map,
    entity_extraction_reduce,
)


class EntityExtractionStepTest(unittest.TestCase):
    def test_long_inputs_are_split_into_overlapping_token_windows(self):
        class Tokenizer:
            model_max_length = 6

            def num_special_tokens_to_add(self, pair=False):
                return 2

            def __call__(self, content, **kwargs):
                return {"offset_mapping": [(i, i + 1) for i in range(len(content))]}

        self.assertEqual(
            _text_windows("abcdefghij", Tokenizer(), stride=1),
            [(0, "abcd"), (3, "defg"), (6, "ghij")],
        )

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

    @patch("tasks.entities.entities._get_entity_pipeline")
    def test_xlm_roberta_output_is_mapped_to_the_public_contract(self, get_pipeline):
        classifier = get_pipeline.return_value
        classifier.return_value = [
            {
                "entity_group": "PER",
                "word": "Ada Lovelace",
                "start": 0,
                "end": 12,
            },
            {
                "entity_group": "LOC",
                "word": "Paris",
                "start": 21,
                "end": 26,
            },
            {
                "entity_group": "DATE",
                "word": "1843",
                "start": 30,
                "end": 34,
            },
        ]

        result = _extract_entities(
            "Ada Lovelace visited Paris in 1843.",
            {"stride": 32, "max_entities": 10},
        )

        self.assertEqual(
            result,
            [
                {"word": "Ada Lovelace", "entity": "PERSON"},
                {"word": "Paris", "entity": "LOC"},
            ],
        )
        classifier.assert_called_once_with("Ada Lovelace visited Paris in 1843.")

    @patch("tasks.entities.entities._get_entity_pipeline")
    def test_xlm_roberta_respects_ignored_types_and_entity_limit(self, get_pipeline):
        get_pipeline.return_value.return_value = [
            {"entity_group": "ORG", "word": "ACME"},
            {"entity_group": "PER", "word": "Ada"},
            {"entity_group": "LOC", "word": "Paris"},
        ]

        result = _extract_entities(
            "ACME hired Ada in Paris.",
            {
                "ignored_entity_types": ["org"],
                "max_entities": 1,
            },
        )

        self.assertEqual(result, [{"word": "Ada", "entity": "PERSON"}])

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
