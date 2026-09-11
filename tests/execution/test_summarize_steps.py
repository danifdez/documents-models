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
        finalize_config = get_task_config("summarize-finalize")

        self.assertEqual(map_config["output_max_ideas"], 12)
        self.assertEqual(map_config["information_units_per_idea"], 4)
        self.assertEqual(map_config["output_max_tokens"], 1400)
        self.assertEqual(reduce_config["type"], "utility")
        self.assertEqual(reduce_config["capabilities"], [])
        self.assertEqual(compose_config["type"], "llm")
        self.assertEqual(compose_config["capabilities"], ["llm"])
        self.assertEqual(compose_config["input_max_ideas"], 36)
        self.assertEqual(compose_config["final_ideas_per_paragraph"], 4)
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
        "tasks.summarize_compose.summarize_compose._compose_summary",
        return_value="Natural summary.",
    )
    @patch(
        "tasks.summarize_compose.summarize_compose._combine_idea_lists",
        return_value=["first", "second"],
    )
    def test_compose_turns_the_complete_inventory_into_prose(
        self, combine_ideas, compose_summary,
    ):
        result = summarize_compose(
            {
                "partials": [["first", "second"]],
                "targetLanguage": "en",
            }
        )

        self.assertEqual(result, {"responses": ["Natural summary."]})
        combine_ideas.assert_called_once_with([["first", "second"]])
        compose_summary.assert_called_once_with(["first", "second"], "en", ANY)

    def test_reduce_rejects_missing_partials(self):
        with self.assertRaisesRegex(ValueError, "requires idea partials"):
            summarize_reduce({"partials": []})

    def test_compose_rejects_missing_partials(self):
        with self.assertRaisesRegex(ValueError, "requires idea partials"):
            summarize_compose({"partials": []})

    def test_finalize_joins_sections_only_at_the_terminal_step(self):
        partials = [
            ["First paragraph.\n\nSecond paragraph."],
            ["Third paragraph."],
        ]

        self.assertEqual(
            summarize_finalize({"partials": partials, "final": False}),
            {
                "responses": [
                    "First paragraph.\n\nSecond paragraph.",
                    "Third paragraph.",
                ]
            },
        )
        self.assertEqual(
            summarize_finalize({"partials": partials, "final": True}),
            {
                "response": (
                    "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
                )
            },
        )

    def test_finalize_rejects_missing_partials(self):
        with self.assertRaisesRegex(ValueError, "requires summary partials"):
            summarize_finalize({"partials": []})

    @patch("tasks.summarize.summarize.get_llm_params", return_value={})
    @patch("tasks.summarize.summarize.get_llm_service")
    def test_map_returns_constrained_ideas_with_a_dynamic_maximum(
        self, get_llm_service, _get_llm_params,
    ):
        llm = get_llm_service.return_value
        llm.chat.return_value = json.dumps({"ideas": ["An idea."]})

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
        self.assertIn("central theses", llm.chat.call_args.args[0][1]["content"])
        self.assertIn(
            "at most 4 distinct, complete theses",
            llm.chat.call_args.args[0][1]["content"],
        )
        self.assertIn("ranked by importance", llm.chat.call_args.args[0][1]["content"])
        self.assertEqual(llm.chat.call_args.kwargs["max_tokens"], 480)
        schema = llm.chat.call_args.kwargs["response_format"]["schema"]
        ideas_schema = schema["properties"]["ideas"]
        self.assertEqual(ideas_schema["maxItems"], 4)
        self.assertEqual(ideas_schema["items"]["maxLength"], 240)
        self.assertEqual(llm.chat.call_args.kwargs["temperature"], 0.0)
        self.assertEqual(llm.chat.call_args.kwargs["seed"], 0)

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

    def test_reduce_prefilter_preserves_order_and_removes_exact_duplicates(self):
        partials = [
            ["first-1", "first-2", "Same idea."],
            ["second-1", "second-2", " same   idea "],
        ]

        result = summarize_task._combine_idea_lists(partials)

        self.assertEqual(
            result,
            ["first-1", "first-2", "Same idea.", "second-1", "second-2"],
        )

    def test_final_reduce_writes_every_idea_as_complete_prose(self):
        result = summarize_task._write_summary(
            ["First idea", "Second idea.", "Third idea?"],
            {"final_ideas_per_paragraph": 2},
        )

        self.assertEqual(
            result,
            "First idea. Second idea.\n\nThird idea?",
        )

    def test_final_composition_requires_every_idea_exactly_once(self):
        valid = json.dumps({
            "paragraphs": [
                {"text": "Combined prose.", "idea_ids": ["idea-1", "idea-2"]},
                {"text": "Closing prose.", "idea_ids": ["idea-3"]},
            ]
        })

        self.assertEqual(
            summarize_task._parse_composition(
                valid,
                ["idea-1", "idea-2", "idea-3"],
                max_paragraph_chars=200,
                max_output_chars=200,
            ),
            ["Combined prose.", "Closing prose."],
        )
        for invalid in (
            {"paragraphs": [{"text": "Missing.", "idea_ids": ["idea-1"]}]},
            {
                "paragraphs": [
                    {"text": "Duplicate.", "idea_ids": ["idea-1", "idea-1"]},
                    {"text": "Other.", "idea_ids": ["idea-2", "idea-3"]},
                ]
            },
        ):
            with self.assertRaisesRegex(ValueError, "cover every idea exactly once"):
                summarize_task._parse_composition(
                    json.dumps(invalid),
                    ["idea-1", "idea-2", "idea-3"],
                    max_paragraph_chars=200,
                    max_output_chars=200,
                )

        with self.assertRaisesRegex(ValueError, "expanded beyond"):
            summarize_task._parse_composition(
                json.dumps({
                    "paragraphs": [{
                        "text": "This prose is longer than the allowed inventory.",
                        "idea_ids": ["idea-1"],
                    }]
                }),
                ["idea-1"],
                max_paragraph_chars=200,
                max_output_chars=20,
            )

    @patch("tasks.summarize.summarize.get_llm_params", return_value={})
    @patch("tasks.summarize.summarize.get_llm_service")
    def test_final_composition_rewrites_bounded_batches(
        self, get_llm_service, _get_llm_params,
    ):
        llm = get_llm_service.return_value
        llm.chat.side_effect = [
            json.dumps({
                "paragraphs": [{
                    "text": "First and second form one paragraph.",
                    "idea_ids": ["idea-1", "idea-2"],
                }]
            }),
            json.dumps({
                "paragraphs": [{
                    "text": "Third closes the summary.",
                    "idea_ids": ["idea-1"],
                }]
            }),
        ]

        result = summarize_task._compose_summary(
            ["First idea.", "Second idea.", "Third idea."],
            "en",
            {
                "input_max_ideas": 2,
                "input_max_chars": 1000,
                "paragraph_max_chars": 200,
                "output_min_tokens": 200,
                "output_max_tokens": 800,
                "output_tokens_per_idea": 100,
            },
        )

        self.assertEqual(
            result,
            "First and second form one paragraph.\n\nThird closes the summary.",
        )
        self.assertEqual(llm.chat.call_count, 2)
        first_call = llm.chat.call_args_list[0]
        self.assertIn("every item", first_call.args[0][0]["content"])
        self.assertIn('"id":"idea-2"', first_call.args[0][1]["content"])
        self.assertEqual(first_call.kwargs["max_tokens"], 296)
        schema = first_call.kwargs["response_format"]["schema"]
        self.assertEqual(
            schema["properties"]["paragraphs"]["items"]["properties"]
            ["idea_ids"]["items"]["enum"],
            ["idea-1", "idea-2"],
        )

    @patch("tasks.summarize.summarize.get_llm_params", return_value={})
    @patch("tasks.summarize.summarize.get_llm_service")
    def test_invalid_composition_falls_back_without_losing_the_batch(
        self, get_llm_service, _get_llm_params,
    ):
        get_llm_service.return_value.chat.return_value = json.dumps({
            "paragraphs": [{"text": "Only one.", "idea_ids": ["idea-1"]}]
        })

        result = summarize_task._compose_summary(
            ["First idea", "Second idea"],
            "en",
            {
                "input_max_ideas": 2,
                "input_max_chars": 1000,
                "paragraph_max_chars": 200,
                "output_min_tokens": 200,
                "output_max_tokens": 800,
                "output_tokens_per_idea": 100,
                "final_ideas_per_paragraph": 2,
            },
        )

        self.assertEqual(result, "First idea. Second idea.")

    def test_invalid_idea_json_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "invalid idea JSON"):
            summarize_task._parse_ideas("not-json")

    def test_idea_parser_enforces_the_generation_contract(self):
        with self.assertRaisesRegex(ValueError, "too many ideas"):
            summarize_task._parse_ideas(
                json.dumps({"ideas": ["one", "two"]}),
                max_ideas=1,
            )
        with self.assertRaisesRegex(ValueError, "oversized idea"):
            summarize_task._parse_ideas(
                json.dumps({"ideas": ["too long"]}),
                max_idea_chars=3,
            )


if __name__ == "__main__":
    unittest.main()
