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
    def test_runtime_configuration_matches_summarization_capacity_contract(self):
        map_config = get_task_config("summarize-map")
        reduce_config = get_task_config("summarize-reduce")
        compose_config = get_task_config("summarize-compose")

        self.assertEqual(map_config["model"], "Qwen3.8-27B-Q4_K_M.gguf")
        self.assertEqual(map_config["output_min_ideas"], 0)
        self.assertEqual(map_config["output_max_ideas"], 16)
        self.assertEqual(map_config["information_units_per_idea"], 5)
        self.assertEqual(map_config["idea_target_chars"], 150)
        self.assertEqual(map_config["idea_max_chars"], 210)
        self.assertEqual(map_config["output_max_tokens"], 1150)
        self.assertEqual(map_config["output_tokens_per_idea"], 64)
        self.assertEqual(reduce_config["type"], "utility")
        self.assertEqual(reduce_config["capabilities"], [])
        self.assertEqual(compose_config["type"], "llm")
        self.assertEqual(compose_config["model"], "Qwen3.8-27B-Q4_K_M.gguf")
        self.assertEqual(compose_config["capabilities"], ["llm"])
        self.assertEqual(compose_config["input_max_ideas"], 64)
        self.assertEqual(compose_config["output_max_words"], 400)
        self.assertEqual(compose_config["output_words_per_idea"], 12)
        self.assertEqual(compose_config["output_max_sentences_per_paragraph"], 6)
        self.assertEqual(compose_config["input_ideas_per_paragraph"], 12)
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
        "tasks.summarize_map.summarize_map._extract_ideas",
        return_value=["first idea", "second idea"],
    )
    def test_map_extracts_an_idea_inventory(self, extract_ideas):
        result = summarize_map(
            {"content": "source chunk", "targetLanguage": "en"}
        )

        self.assertEqual(result, {"ideas": ["first idea", "second idea"]})
        extract_ideas.assert_called_once()

    @patch(
        "tasks.summarize_reduce.summarize_reduce._combine_idea_lists",
        return_value=["first", "second"],
    )
    def test_reduce_always_keeps_a_structured_inventory(self, combine_ideas):
        result = summarize_reduce(
            {
                "partials": [["first"], ["second"]],
                "targetLanguage": "en",
            }
        )

        self.assertEqual(result, {"ideas": ["first", "second"]})
        combine_ideas.assert_called_once_with([["first"], ["second"]])

    @patch(
        "tasks.summarize_compose.summarize_compose._write_summary",
        return_value="First cohesive summary.",
    )
    @patch(
        "tasks.summarize_compose.summarize_compose._combine_idea_lists",
        return_value=["first", "second"],
    )
    def test_compose_writes_one_global_summary(
        self, combine_ideas, write_summary,
    ):
        result = summarize_compose(
            {
                "partials": [["first", "second"]],
                "targetLanguage": "en",
            }
        )

        self.assertEqual(result, {"response": "First cohesive summary."})
        combine_ideas.assert_called_once_with([["first", "second"]])
        write_summary.assert_called_once_with(["first", "second"], "en", ANY)

    def test_reduce_rejects_missing_partials(self):
        with self.assertRaisesRegex(ValueError, "requires idea partials"):
            summarize_reduce({"partials": []})

    def test_compose_rejects_missing_inventories(self):
        with self.assertRaisesRegex(ValueError, "requires idea inventories"):
            summarize_compose({"partials": []})

    def test_empty_fragment_inventory_flows_through_reduce_and_compose(self):
        self.assertEqual(summarize_reduce({"partials": [[]]}), {"ideas": []})
        with self.assertRaisesRegex(ValueError, "no material ideas"):
            summarize_compose({"partials": [[]]})

    @patch("tasks.summarize.summarize.get_llm_params", return_value={})
    @patch("tasks.summarize.summarize.get_llm_service")
    def test_map_returns_constrained_ideas_with_a_dynamic_maximum(
        self, get_llm_service, _get_llm_params,
    ):
        llm = get_llm_service.return_value
        llm.chat.return_value = json.dumps({
            "material_idea_count": 1,
            "ideas": ["An idea."],
        })

        result = summarize_task._extract_ideas(
            "First fact. Second fact; third fact.",
            "en",
            {"input_char_budget": 1000},
        )

        self.assertEqual(result, ["An idea."])
        self.assertIn(
            "complete proposition",
            llm.chat.call_args.args[0][0]["content"],
        )
        self.assertIn("central theses", llm.chat.call_args.args[0][0]["content"])
        self.assertIn(
            "between zero and 1",
            llm.chat.call_args.args[0][1]["content"],
        )
        self.assertIn("in source order", llm.chat.call_args.args[0][0]["content"])
        self.assertIn("safety ceiling, never a target", llm.chat.call_args.args[0][0]["content"])
        self.assertEqual(llm.chat.call_args.kwargs["max_tokens"], 384)
        schema = llm.chat.call_args.kwargs["response_format"]["schema"]
        ideas_schema = schema["properties"]["ideas"]
        self.assertEqual(next(iter(schema["properties"])), "material_idea_count")
        self.assertEqual(ideas_schema["maxItems"], 1)
        self.assertEqual(ideas_schema["minItems"], 0)
        self.assertEqual(ideas_schema["items"]["maxLength"], 240)
        self.assertIn("Aim for at most 180 characters", llm.chat.call_args.args[0][1]["content"])
        self.assertIn("never cut a word or sentence", llm.chat.call_args.args[0][1]["content"])
        self.assertEqual(llm.chat.call_args.kwargs["temperature"], 0.0)
        self.assertEqual(llm.chat.call_args.kwargs["seed"], 0)

        wider_schema = summarize_task._ideas_response_format(4, 240)
        wider_ideas_schema = wider_schema["schema"]["properties"]["ideas"]
        self.assertEqual(wider_ideas_schema["minItems"], 0)
        self.assertEqual(wider_ideas_schema["maxItems"], 4)

    def test_dense_chunks_allow_more_ideas_up_to_the_configured_cap(self):
        sparse = summarize_task._dynamic_idea_limit(
            1,
            {},
            min_key="min",
            max_key="max",
            units_per_idea_key="per",
            min_default=4,
            max_default=12,
            units_per_idea_default=4,
        )
        dense = summarize_task._dynamic_idea_limit(
            20,
            {},
            min_key="min",
            max_key="max",
            units_per_idea_key="per",
            min_default=4,
            max_default=12,
            units_per_idea_default=4,
        )
        capped = summarize_task._dynamic_idea_limit(
            100,
            {},
            min_key="min",
            max_key="max",
            units_per_idea_key="per",
            min_default=4,
            max_default=12,
            units_per_idea_default=4,
        )

        self.assertEqual(sparse, 4)
        self.assertEqual(dense, 5)
        self.assertEqual(capped, 12)

    def test_dense_chunks_receive_more_headroom_up_to_the_configured_cap(self):
        sparse = summarize_task._dynamic_max_tokens(
            1,
            {},
            min_key="min",
            max_key="max",
            per_unit_key="per",
            min_default=200,
            max_default=1000,
            per_unit_default=30,
        )
        dense = summarize_task._dynamic_max_tokens(
            20,
            {},
            min_key="min",
            max_key="max",
            per_unit_key="per",
            min_default=200,
            max_default=1000,
            per_unit_default=30,
        )
        capped = summarize_task._dynamic_max_tokens(
            100,
            {},
            min_key="min",
            max_key="max",
            per_unit_key="per",
            min_default=200,
            max_default=1000,
            per_unit_default=30,
        )

        self.assertGreater(dense, sparse)
        self.assertEqual(capped, 1000)

    def test_reduce_prefilter_preserves_source_order_and_removes_duplicates(self):
        partials = [
            ["first-1", "first-2", "Same idea."],
            ["second-1", "second-2", " same   idea "],
        ]

        result = summarize_task._combine_idea_lists(partials)

        self.assertEqual(
            result,
            ["first-1", "first-2", "Same idea.", "second-1", "second-2"],
        )
    @patch("tasks.summarize.summarize.get_llm_params", return_value={})
    @patch("tasks.summarize.summarize.get_llm_service")
    def test_global_writer_plans_paragraphs_from_the_complete_inventory(
        self, get_llm_service, _get_llm_params,
    ):
        llm = get_llm_service.return_value
        llm.chat.return_value = json.dumps({
            "paragraphs": [
                ["First and second form one theme."],
                ["Third completes the argument."],
            ],
        })

        result = summarize_task._write_summary(
            ["First idea.", "Second idea.", "Third idea."],
            "en",
            {
                "input_max_ideas": 3,
                "input_max_chars": 1000,
                "output_min_words": 30,
                "output_max_words": 60,
                "output_words_per_idea": 10,
                "input_ideas_per_paragraph": 2,
                "output_min_paragraphs": 1,
                "output_max_paragraphs": 3,
                "output_max_sentences_per_paragraph": 6,
                "output_min_tokens": 200,
                "output_max_tokens": 800,
                "output_tokens_per_word": 3,
            },
        )

        self.assertEqual(
            result,
            "First and second form one theme.\n\nThird completes the argument.",
        )
        prompt = llm.chat.call_args.args[0][1]["content"]
        self.assertIn("exactly 2 connected thematic paragraphs", prompt)
        self.assertIn("no more than 30 words", prompt)
        self.assertIn("silently organize their distinct meanings into 2 themes", prompt)
        self.assertIn("Synthesize overlapping or closely related propositions", prompt)
        self.assertIn("Preserve every distinct claim", prompt)
        self.assertIn("at most 6 complete sentences", prompt)
        self.assertIn('"First idea."', prompt)
        self.assertEqual(llm.chat.call_args.kwargs["max_tokens"], 200)
        properties = llm.chat.call_args.kwargs["response_format"]["schema"][
            "properties"
        ]
        paragraphs = properties["paragraphs"]
        self.assertEqual(paragraphs["minItems"], 2)
        self.assertEqual(paragraphs["maxItems"], 2)
        self.assertEqual(paragraphs["items"]["type"], "array")
        self.assertEqual(paragraphs["items"]["minItems"], 1)
        self.assertEqual(paragraphs["items"]["maxItems"], 6)
        self.assertIn("approximately 15 words", prompt)
        self.assertIn("no more than 3 words per sentence", prompt)

    def test_global_writer_rejects_the_wrong_paragraph_count(self):
        raw = json.dumps({
            "paragraphs": [["Only one paragraph."]],
        })

        with self.assertRaisesRegex(ValueError, "invalid paragraphs"):
            summarize_task._parse_summary(raw, 2)

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

    def test_finalize_rejects_missing_sections(self):
        with self.assertRaisesRegex(ValueError, "requires summary partials"):
            summarize_finalize({"partials": []})

    def test_summary_word_maximum_grows_with_information_and_is_capped(self):
        cfg = {
            "output_min_words": 60,
            "output_max_words": 110,
            "output_words_per_idea": 7,
        }

        self.assertEqual(summarize_task._dynamic_output_word_limit(3, cfg), 60)
        self.assertEqual(summarize_task._dynamic_output_word_limit(15, cfg), 105)
        self.assertEqual(summarize_task._dynamic_output_word_limit(30, cfg), 110)

    def test_global_writer_rejects_an_inventory_that_cannot_fit(self):
        with self.assertRaisesRegex(ValueError, "input exceeds"):
            summarize_task._write_summary(
                ["First idea", "Second idea", "Third idea"],
                "en",
                {"input_max_ideas": 2, "input_max_chars": 1000},
            )

    def test_paragraph_count_grows_but_stays_bounded(self):
        cfg = {
            "output_min_paragraphs": 1,
            "output_max_paragraphs": 4,
            "input_ideas_per_paragraph": 12,
        }

        self.assertEqual(summarize_task._dynamic_paragraph_count(5, cfg), 1)
        self.assertEqual(summarize_task._dynamic_paragraph_count(32, cfg), 3)
        self.assertEqual(summarize_task._dynamic_paragraph_count(64, cfg), 4)

    def test_invalid_idea_json_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "invalid idea JSON"):
            summarize_task._parse_ideas("not-json")

    def test_idea_parser_enforces_the_generation_contract(self):
        with self.assertRaisesRegex(ValueError, "too many ideas"):
            summarize_task._parse_ideas(
                json.dumps({
                    "material_idea_count": 2,
                    "ideas": ["one", "two"],
                }),
                max_ideas=1,
            )
        with self.assertRaisesRegex(ValueError, "oversized idea"):
            summarize_task._parse_ideas(
                json.dumps({
                    "material_idea_count": 1,
                    "ideas": ["complete idea"],
                }),
                max_idea_chars=3,
            )

    def test_idea_parser_accepts_zero_and_rejects_inconsistent_counts(self):
        self.assertEqual(
            summarize_task._parse_ideas(json.dumps({
                "material_idea_count": 0,
                "ideas": [],
            })),
            [],
        )
        with self.assertRaisesRegex(ValueError, "invalid ideas"):
            summarize_task._parse_ideas(json.dumps({
                "material_idea_count": 2,
                "ideas": ["one"],
            }))
        with self.assertRaisesRegex(ValueError, "duplicate ideas"):
            summarize_task._parse_ideas(json.dumps({
                "material_idea_count": 2,
                "ideas": ["one", "one"],
            }))

    def test_idea_parser_rejects_a_cut_sentence(self):
        with self.assertRaisesRegex(ValueError, "incomplete idea"):
            summarize_task._parse_ideas(json.dumps({
                "material_idea_count": 1,
                "ideas": ["Sentence cut at the lim"],
            }))
