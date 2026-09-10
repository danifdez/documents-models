import unittest
from unittest.mock import ANY, patch

from common.execution_registry import TASK_HANDLERS
from tasks.summarize import summarize as summarize_task
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
            {
                "partials": ["first", "second"],
                "targetLanguage": "en",
            }
        )

        self.assertEqual(result, {"response": "merged"})
        merge_summaries.assert_called_once_with(
            ["first", "second"],
            "en",
            ANY,
            final=True,
        )

    @patch(
        "tasks.summarize_reduce.summarize_reduce._merge_summaries",
        return_value="merged",
    )
    def test_reduce_marks_intermediate_stages(self, merge_summaries):
        summarize_reduce({"partials": ["first"], "final": False})

        self.assertFalse(merge_summaries.call_args.kwargs["final"])

    def test_reduce_rejects_missing_partials(self):
        with self.assertRaisesRegex(ValueError, "requires string partials"):
            summarize_reduce({"partials": []})

    @patch("tasks.summarize.summarize.get_llm_params", return_value={})
    @patch("tasks.summarize.summarize.get_llm_service")
    def test_map_keeps_headroom_and_uses_a_deterministic_seed(
        self, get_llm_service, _get_llm_params,
    ):
        llm = get_llm_service.return_value
        llm.chat.return_value = "summary"

        result = summarize_task._summarize_chunk(
            "source", "en", {"input_char_budget": 1000}
        )

        self.assertEqual(result, "summary")
        messages = llm.chat.call_args.args[0]
        self.assertIn("under 250 tokens", messages[1]["content"])
        self.assertEqual(llm.chat.call_args.kwargs, {
            "max_tokens": 400,
            "temperature": 0.0,
            "seed": 0,
        })

    @patch("tasks.summarize.summarize.get_llm_params", return_value={})
    @patch("tasks.summarize.summarize.get_llm_service")
    def test_reduce_keeps_headroom_and_uses_a_deterministic_seed(
        self, get_llm_service, _get_llm_params,
    ):
        llm = get_llm_service.return_value
        llm.chat.return_value = "summary"

        result = summarize_task._merge_summaries(
            ["partial"], "en", {"input_char_budget": 1000}
        )

        self.assertEqual(result, "summary")
        messages = llm.chat.call_args.args[0]
        self.assertIn("under 1200 tokens", messages[1]["content"])
        self.assertEqual(llm.chat.call_args.kwargs, {
            "max_tokens": 1800,
            "temperature": 0.0,
            "seed": 0,
        })

    @patch("tasks.summarize.summarize.get_llm_params", return_value={})
    @patch("tasks.summarize.summarize.get_llm_service")
    def test_intermediate_reduce_compresses_before_the_final_merge(
        self, get_llm_service, _get_llm_params,
    ):
        llm = get_llm_service.return_value
        llm.chat.return_value = "summary"

        result = summarize_task._merge_summaries(
            ["first", "second", "third"],
            "en",
            {"input_char_budget": 1000},
            final=False,
        )

        self.assertEqual(result, "summary")
        messages = llm.chat.call_args.args[0]
        self.assertIn("under 400 tokens", messages[1]["content"])
        self.assertEqual(llm.chat.call_args.kwargs, {
            "max_tokens": 800,
            "temperature": 0.0,
            "seed": 0,
        })

    def test_output_target_must_leave_generation_headroom(self):
        with self.assertRaisesRegex(ValueError, "chunk_target_tokens"):
            summarize_task._output_limits(
                {
                    "chunk_target_tokens": 400,
                    "chunk_max_tokens": 400,
                },
                target_key="chunk_target_tokens",
                max_key="chunk_max_tokens",
                target_default=250,
                max_default=400,
            )


if __name__ == "__main__":
    unittest.main()
