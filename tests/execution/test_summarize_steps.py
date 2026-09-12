import json
import unittest
from unittest.mock import ANY, patch

from common.execution_registry import TASK_HANDLERS
from lib.llm.config import get_task_config
from tasks.summarize import summarize as summarize_task
from tasks.summarize_compose.summarize_compose import summarize_compose
from tasks.summarize_finalize.summarize_finalize import summarize_finalize
from tasks.summarize_map.summarize_map import summarize_map
from tasks.summarize_reduce.summarize_reduce import summarize_reduce


class SummarizeStepTest(unittest.TestCase):
    def test_runtime_configuration_matches_hierarchical_summary_contract(self):
        map_config = get_task_config("summarize-map")
        reduce_config = get_task_config("summarize-reduce")
        compose_config = get_task_config("summarize-compose")

        self.assertEqual(map_config["model"], "Qwen3.8-27B-Q4_K_M.gguf")
        self.assertEqual(map_config["output_word_limit_floor"], 80)
        self.assertEqual(map_config["output_word_limit_ceiling"], 220)
        self.assertEqual(map_config["output_words_per_information_unit"], 3)
        self.assertEqual(map_config["output_max_tokens"], 1050)
        self.assertEqual(map_config["output_tokens_per_word"], 4)
        self.assertEqual(reduce_config["type"], "utility")
        self.assertEqual(reduce_config["capabilities"], [])
        self.assertEqual(compose_config["type"], "llm")
        self.assertEqual(compose_config["model"], "Qwen3.8-27B-Q4_K_M.gguf")
        self.assertEqual(compose_config["capabilities"], ["llm"])
        self.assertEqual(compose_config["input_max_sections"], 16)
        self.assertEqual(compose_config["output_word_limit_ceiling"], 400)
        self.assertEqual(compose_config["output_input_word_ratio"], 0.55)
        self.assertEqual(compose_config["output_max_paragraphs"], 4)
        self.assertEqual(compose_config["output_max_tokens"], 1600)
        self.assertEqual(compose_config["output_tokens_per_word"], 4)
        finalize_config = get_task_config("summarize-finalize")
        self.assertEqual(finalize_config["type"], "utility")
        self.assertEqual(finalize_config["capabilities"], [])

    def test_registers_only_durable_summarize_capabilities(self):
        self.assertIn("summarize-map", TASK_HANDLERS)
        self.assertIn("summarize-reduce", TASK_HANDLERS)
        self.assertIn("summarize-compose", TASK_HANDLERS)
        self.assertIn("summarize-finalize", TASK_HANDLERS)
        self.assertNotIn("summarize", TASK_HANDLERS)

    @patch(
        "tasks.summarize_map.summarize_map._summarize_chunk",
        return_value="One cohesive section summary.",
    )
    def test_map_writes_one_section_summary(self, summarize_chunk):
        result = summarize_map(
            {"content": "source chunk", "targetLanguage": "en"}
        )

        self.assertEqual(result, {"summary": "One cohesive section summary."})
        summarize_chunk.assert_called_once_with("source chunk", "en", ANY)

    def test_reduce_flattens_section_summaries_without_rewriting_them(self):
        result = summarize_reduce({
            "partials": ["First summary.", ["Second summary."]],
        })

        self.assertEqual(
            result,
            {"summaries": ["First summary.", "Second summary."]},
        )

    @patch(
        "tasks.summarize_compose.summarize_compose._write_summary",
        return_value="One cohesive global summary.",
    )
    def test_compose_writes_one_global_summary(self, write_summary):
        result = summarize_compose({
            "partials": ["First section.", "Second section."],
            "targetLanguage": "en",
        })

        self.assertEqual(result, {"response": "One cohesive global summary."})
        write_summary.assert_called_once_with(
            ["First section.", "Second section."], "en", ANY,
        )

    def test_steps_reject_missing_or_invalid_section_summaries(self):
        with self.assertRaisesRegex(ValueError, "requires summary partials"):
            summarize_reduce({"partials": []})
        with self.assertRaisesRegex(ValueError, "requires section summaries"):
            summarize_compose({"partials": []})
        with self.assertRaisesRegex(ValueError, "invalid section summary"):
            summarize_compose({"partials": [["unfinished summary"]]})

    @patch("tasks.summarize.summarize.get_llm_params", return_value={})
    @patch("tasks.summarize.summarize.get_llm_service")
    def test_map_returns_a_constrained_cohesive_summary_with_a_dynamic_ceiling(
        self, get_llm_service, _get_llm_params,
    ):
        llm = get_llm_service.return_value
        llm.chat.return_value = json.dumps({
            "summary": "The causes and proposal form one connected argument.",
        })

        result = summarize_task._summarize_chunk(
            "First cause. Second consequence; final proposal.",
            "en",
            {"input_char_budget": 1000},
        )

        self.assertEqual(
            result,
            "The causes and proposal form one connected argument.",
        )
        system_prompt = llm.chat.call_args.args[0][0]["content"]
        user_prompt = llm.chat.call_args.args[0][1]["content"]
        self.assertIn("not an inventory", system_prompt)
        self.assertIn("essential relationships", system_prompt)
        self.assertIn("no more than 80 words", user_prompt)
        self.assertIn("safety ceiling, not a target", user_prompt)
        self.assertIn("do not return a list", user_prompt)
        self.assertEqual(llm.chat.call_args.kwargs["max_tokens"], 416)
        schema = llm.chat.call_args.kwargs["response_format"]["schema"]
        self.assertEqual(set(schema["properties"]), {"summary"})
        self.assertEqual(schema["properties"]["summary"]["type"], "string")
        self.assertEqual(llm.chat.call_args.kwargs["temperature"], 0.0)
        self.assertEqual(llm.chat.call_args.kwargs["seed"], 0)

    def test_dense_sections_receive_a_larger_word_ceiling_up_to_the_cap(self):
        cfg = {
            "output_word_limit_floor": 60,
            "output_word_limit_ceiling": 110,
            "output_words_per_information_unit": 3,
        }

        self.assertEqual(summarize_task._dynamic_section_word_limit(1, cfg), 60)
        self.assertEqual(summarize_task._dynamic_section_word_limit(20, cfg), 92)
        self.assertEqual(summarize_task._dynamic_section_word_limit(100, cfg), 110)

    def test_dense_outputs_receive_more_token_headroom_up_to_the_cap(self):
        sparse = summarize_task._dynamic_max_tokens(
            1, {}, min_key="min", max_key="max", per_unit_key="per",
            min_default=200, max_default=1000, per_unit_default=30,
        )
        dense = summarize_task._dynamic_max_tokens(
            20, {}, min_key="min", max_key="max", per_unit_key="per",
            min_default=200, max_default=1000, per_unit_default=30,
        )
        capped = summarize_task._dynamic_max_tokens(
            100, {}, min_key="min", max_key="max", per_unit_key="per",
            min_default=200, max_default=1000, per_unit_default=30,
        )

        self.assertGreater(dense, sparse)
        self.assertEqual(capped, 1000)

    def test_combining_sections_preserves_source_order_and_relationships(self):
        partials = [
            "The problem leads to the first response.",
            ["That response creates a later consequence."],
        ]

        self.assertEqual(
            summarize_task._combine_section_summaries(partials),
            [
                "The problem leads to the first response.",
                "That response creates a later consequence.",
            ],
        )

    @patch("tasks.summarize.summarize.get_llm_params", return_value={})
    @patch("tasks.summarize.summarize.get_llm_service")
    def test_global_writer_synthesizes_sections_into_adaptive_paragraphs(
        self, get_llm_service, _get_llm_params,
    ):
        llm = get_llm_service.return_value
        llm.chat.return_value = json.dumps({
            "paragraphs": [
                "The problem and its causes establish the central argument.",
                "The proposed response follows from that argument.",
            ],
        })
        summaries = [
            "The problem has several causes and creates an urgent consequence.",
            "The author therefore proposes a response and explains its purpose.",
        ]

        result = summarize_task._write_summary(
            summaries,
            "en",
            {
                "input_max_sections": 3,
                "input_max_chars": 1000,
                "output_word_limit_floor": 30,
                "output_word_limit_ceiling": 60,
                "output_input_word_ratio": 0.5,
                "output_min_paragraphs": 1,
                "output_max_paragraphs": 3,
                "output_min_tokens": 200,
                "output_max_tokens": 800,
                "output_tokens_per_word": 3,
            },
        )

        self.assertEqual(
            result,
            "The problem and its causes establish the central argument.\n\n"
            "The proposed response follows from that argument.",
        )
        prompt = llm.chat.call_args.args[0][1]["content"]
        self.assertIn("between 1 and 3 complete paragraphs", prompt)
        self.assertIn("no more than 30 words", prompt)
        self.assertIn("not a target", prompt)
        self.assertIn("Do not write one paragraph per section", prompt)
        self.assertIn("what the whole text is about", prompt)
        self.assertIn('"The problem has several causes', prompt)
        self.assertEqual(llm.chat.call_args.kwargs["max_tokens"], 200)
        paragraphs = llm.chat.call_args.kwargs["response_format"]["schema"][
            "properties"
        ]["paragraphs"]
        self.assertEqual(paragraphs["minItems"], 1)
        self.assertEqual(paragraphs["maxItems"], 3)
        self.assertEqual(paragraphs["items"]["type"], "string")

    def test_global_writer_accepts_only_complete_paragraphs_within_the_range(self):
        with self.assertRaisesRegex(ValueError, "invalid paragraphs"):
            summarize_task._parse_summary(
                json.dumps({"paragraphs": ["One.", "Two."]}), 1, 1,
            )
        with self.assertRaisesRegex(ValueError, "invalid paragraphs"):
            summarize_task._parse_summary(
                json.dumps({"paragraphs": [["Sentence-shaped item."]]}), 1, 2,
            )

    def test_finalize_joins_sections_only_at_the_terminal_step(self):
        partials = [["First paragraph."], ["Second paragraph."]]

        self.assertEqual(
            summarize_finalize({"partials": partials, "final": False}),
            {"responses": ["First paragraph.", "Second paragraph."]},
        )
        self.assertEqual(
            summarize_finalize({"partials": partials, "final": True}),
            {"response": "First paragraph.\n\nSecond paragraph."},
        )

    def test_finalize_rejects_empty_sections(self):
        with self.assertRaisesRegex(ValueError, "no summary sections"):
            summarize_finalize({"partials": [[]], "final": True})

    def test_global_word_ceiling_tracks_partial_summary_size_and_is_capped(self):
        cfg = {
            "output_word_limit_floor": 60,
            "output_word_limit_ceiling": 110,
            "output_input_word_ratio": 0.5,
        }
        short = ["word " * 20 + "end."]
        medium = ["word " * 199 + "end."]
        long = ["word " * 299 + "end."]

        self.assertEqual(summarize_task._dynamic_output_word_limit(short, cfg), 60)
        self.assertEqual(summarize_task._dynamic_output_word_limit(medium, cfg), 100)
        self.assertEqual(summarize_task._dynamic_output_word_limit(long, cfg), 110)

    def test_global_writer_rejects_sections_that_cannot_fit(self):
        with self.assertRaisesRegex(ValueError, "input exceeds"):
            summarize_task._write_summary(
                ["First summary.", "Second summary.", "Third summary."],
                "en",
                {"input_max_sections": 2, "input_max_chars": 1000},
            )

    def test_section_parser_fails_closed_and_rejects_cut_text(self):
        with self.assertRaisesRegex(ValueError, "invalid section JSON"):
            summarize_task._parse_section_summary("not-json")
        with self.assertRaisesRegex(ValueError, "invalid section summary"):
            summarize_task._parse_section_summary(json.dumps({
                "summary": "Sentence cut at the lim",
            }))


if __name__ == "__main__":
    unittest.main()
