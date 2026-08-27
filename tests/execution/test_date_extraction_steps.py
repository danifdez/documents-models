import json
import unittest
from unittest.mock import Mock, patch

from common.execution_registry import TASK_HANDLERS
from lib.execution.step_executor import execute_assignment
from tasks.dates.dates import date_extraction_map, date_extraction_reduce


def _entry(raw, date, offset, precision="day"):
    return {
        "rawExpression": raw,
        "date": date,
        "endDate": None,
        "precision": precision,
        "charOffset": offset,
        "contextSnippet": raw,
        "unresolvedReason": None,
    }


class DateExtractionStepTest(unittest.TestCase):
    def test_registers_only_durable_date_capabilities(self):
        self.assertIn("date-extraction-map", TASK_HANDLERS)
        self.assertIn("date-extraction-reduce", TASK_HANDLERS)
        self.assertNotIn("date-extraction", TASK_HANDLERS)

    @patch("tasks.dates.dates.get_llm_params", return_value={})
    @patch("tasks.dates.dates.get_llm_service")
    def test_map_detects_with_one_inference_and_resolves_in_code(
        self, get_llm_service, _get_llm_params
    ):
        llm = Mock()
        llm.chat.return_value = json.dumps(
            ["20 de julio de 1969", "149,90 euros"], ensure_ascii=False
        )
        get_llm_service.return_value = llm

        result = date_extraction_map(
            {
                "content": "El alunizaje ocurrió el 20 de julio de 1969. Costó 149,90 euros.",
                "language": "es",
                "anchorDate": None,
                "charOffset": 12,
            }
        )

        self.assertEqual(len(result["dates"]), 1)
        self.assertEqual(result["dates"][0]["date"], "1969-07-20")
        self.assertEqual(result["dates"][0]["precision"], "day")
        self.assertGreaterEqual(result["dates"][0]["charOffset"], 12)
        self.assertNotIn("resolver", result["dates"][0])
        self.assertNotIn("isRelative", result["dates"][0])
        llm.chat.assert_called_once()

    @patch("tasks.dates.dates.get_llm_params", return_value={})
    @patch("tasks.dates.dates.get_llm_service")
    def test_map_keeps_a_missing_anchor_explicit(
        self, get_llm_service, _get_llm_params
    ):
        llm = Mock()
        llm.chat.return_value = '["hace 3 días"]'
        get_llm_service.return_value = llm

        result = date_extraction_map(
            {"content": "La visita fue hace 3 días.", "language": "es"}
        )

        self.assertIsNone(result["dates"][0]["date"])
        self.assertEqual(
            result["dates"][0]["unresolvedReason"], "missing_anchor"
        )
        llm.chat.assert_called_once()

    @patch("tasks.dates.dates.get_llm_params", return_value={})
    @patch("tasks.dates.dates.get_llm_service")
    def test_map_resolves_an_italian_date_with_a_leading_article(
        self, get_llm_service, _get_llm_params
    ):
        llm = Mock()
        llm.chat.return_value = '["il 20 luglio 1969"]'
        get_llm_service.return_value = llm

        result = date_extraction_map(
            {
                "content": "Il modulo lunare atterrò il 20 luglio 1969.",
                "language": "it",
            }
        )

        self.assertEqual(result["dates"][0]["date"], "1969-07-20")
        self.assertEqual(result["dates"][0]["precision"], "day")
        self.assertIsNone(result["dates"][0]["unresolvedReason"])

    def test_reduce_merges_empty_maps_and_sorts_dates(self):
        result = date_extraction_reduce(
            {
                "partials": [
                    [_entry("2024", "2024-08-25", 80, "year")],
                    [],
                    [_entry("20 July 1969", "1969-07-20", 10)],
                ]
            }
        )

        self.assertEqual(
            [entry["date"] for entry in result["dates"]],
            ["1969-07-20", "2024-08-25"],
        )

    def test_map_failure_becomes_a_failed_inference_result(self):
        assignment = {
            "executionId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca701",
            "stepId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca702",
            "operationId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca703",
            "attemptId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca704",
            "stepKind": "inference",
            "work": {
                "taskType": "date-extraction-map",
                "payload": {"content": ""},
            },
        }

        result = execute_assignment(assignment)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["output"]["outcome"], {"kind": "failed"})

    def test_reduce_executes_as_code_assignment(self):
        assignment = {
            "executionId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca701",
            "stepId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca702",
            "operationId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca703",
            "attemptId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca704",
            "stepKind": "code",
            "work": {
                "taskType": "date-extraction-reduce",
                "payload": {"partials": []},
            },
        }

        result = execute_assignment(assignment)

        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["output"], {"kind": "code", "value": {"dates": []}})


if __name__ == "__main__":
    unittest.main()
