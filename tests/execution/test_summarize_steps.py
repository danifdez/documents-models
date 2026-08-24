import unittest
from unittest.mock import patch

from common.execution_registry import TASK_HANDLERS
from tasks.summarize_map.summarize_map import summarize_map
from tasks.summarize_reduce.summarize_reduce import summarize_reduce


class SummarizeStepTest(unittest.TestCase):
    def test_registers_only_durable_summarize_capabilities(self):
        self.assertIn("summarize-map", TASK_HANDLERS)
        self.assertIn("summarize-reduce", TASK_HANDLERS)
        self.assertNotIn("summarize", TASK_HANDLERS)

    @patch(
        "tasks.summarize_map.summarize_map._summarize_chunk",
        return_value="partial",
    )
    def test_map_summarizes_one_self_contained_chunk(self, summarize_chunk):
        result = summarize_map(
            {"content": "source chunk", "targetLanguage": "en"}
        )

        self.assertEqual(result, {"response": "partial"})
        summarize_chunk.assert_called_once()

    @patch(
        "tasks.summarize_reduce.summarize_reduce._merge_summaries",
        return_value="merged",
    )
    def test_reduce_merges_materialized_partials(self, merge_summaries):
        result = summarize_reduce(
            {"partials": ["first", "second"], "targetLanguage": "en"}
        )

        self.assertEqual(result, {"response": "merged"})
        merge_summaries.assert_called_once()

    def test_reduce_rejects_missing_partials(self):
        with self.assertRaisesRegex(ValueError, "requires string partials"):
            summarize_reduce({"partials": []})


if __name__ == "__main__":
    unittest.main()
