import unittest
from unittest.mock import patch

from lib.execution.step_executor import execute_assignment


class TranslateAssignmentTest(unittest.TestCase):
    def assignment(self, payload):
        return {
            "executionId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca701",
            "stepId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca702",
            "operationId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca703",
            "attemptId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca704",
            "stepKind": "inference",
            "work": {"taskType": "translate", "payload": payload},
        }

    @patch("lib.llm.config.get_task_config")
    @patch("tasks.translate.translate._get_translation_pipeline")
    def test_executes_canonical_translation_assignment(
        self, get_pipeline, get_task_config
    ):
        get_task_config.return_value = {
            "chunk_size": 32,
            "max_words_per_item": 400,
        }
        get_pipeline.return_value = lambda texts: [
            {"translation_text": f"translated:{text}"} for text in texts
        ]

        result = execute_assignment(
            self.assignment(
                {
                    "sourceLanguage": "en",
                    "targetLanguage": "es",
                    "texts": [{"text": "Hello", "path": "p"}],
                }
            )
        )

        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(
            result["output"]["outcome"]["value"],
            {
                "response": [
                    {
                        "translation_text": "translated:Hello",
                        "original_text": "Hello",
                        "path": "p",
                    }
                ]
            },
        )

    def test_rejects_removed_texts_alias(self):
        result = execute_assignment(
            self.assignment(
                {
                    "sourceLanguage": "en",
                    "targetLanguage": "es",
                    "textsToTranslate": ["Hello"],
                }
            )
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["code"], "STEP_EXECUTION_FAILED")


if __name__ == "__main__":
    unittest.main()
