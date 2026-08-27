import unittest
from unittest.mock import patch

from lib.execution.step_executor import execute_assignment


class TranslateAssignmentTest(unittest.TestCase):
    def assignment(self, task_type, payload, step_kind="inference"):
        return {
            "executionId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca701",
            "stepId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca702",
            "operationId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca703",
            "attemptId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca704",
            "stepKind": step_kind,
            "work": {"taskType": task_type, "payload": payload},
        }

    @patch("tasks.translate.translate._get_translation_pipeline")
    def test_map_executes_one_bounded_translation_batch(self, get_pipeline):
        pipeline = get_pipeline.return_value
        pipeline.return_value = [
            {"translation_text": "Hola"},
            {"translation_text": "Mundo"},
        ]
        units = [
            {
                "itemIndex": 0,
                "pieceIndex": 0,
                "text": "Hello",
                "originalText": "Hello world",
                "path": "p",
            },
            {
                "itemIndex": 0,
                "pieceIndex": 1,
                "text": "world",
                "originalText": "Hello world",
                "path": "p",
            },
        ]

        result = execute_assignment(
            self.assignment(
                "translate-map",
                {
                    "sourceLanguage": "en",
                    "targetLanguage": "es",
                    "units": units,
                },
            )
        )

        self.assertEqual(result["status"], "succeeded")
        pipeline.assert_called_once_with(["Hello", "world"])
        translations = result["output"]["outcome"]["value"]["translations"]
        self.assertEqual(
            [item["translationText"] for item in translations],
            ["Hola", "Mundo"],
        )

    def test_final_reduce_restores_target_language_order(self):
        translations = [
            {
                "targetLanguage": "fr",
                "itemIndex": 0,
                "pieceIndex": 0,
                "translationText": "Entité",
                "originalText": "Entity",
            },
            {
                "targetLanguage": "es",
                "itemIndex": 0,
                "pieceIndex": 0,
                "translationText": "Entidad",
                "originalText": "Entity",
            },
        ]

        result = execute_assignment(
            self.assignment(
                "translate-reduce",
                {
                    "partials": [translations],
                    "final": True,
                    "responseMode": "targets",
                    "itemCount": 1,
                    "targetLanguages": ["es", "fr"],
                },
                "code",
            )
        )

        self.assertEqual(result["status"], "succeeded")
        response = result["output"]["value"]["response"]
        self.assertEqual(
            [item["translation_text"] for item in response],
            ["Entidad", "Entité"],
        )

    def test_rejects_removed_monolithic_translation_task(self):
        result = execute_assignment(
            self.assignment(
                "translate",
                {
                    "sourceLanguage": "en",
                    "targetLanguage": "es",
                    "texts": ["Hello"],
                },
            )
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["code"], "CAPABILITY_UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
