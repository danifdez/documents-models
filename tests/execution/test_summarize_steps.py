import json
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
        "tasks.summarize_reduce.summarize_reduce._write_summary",
        return_value="merged",
    )
    def test_final_reduce_turns_all_materialized_ideas_into_prose(self, write_summary):
        result = summarize_reduce(
            {
                "partials": [["first"], ["second"]],
                "targetLanguage": "en",
                "final": True,
            }
        )

        self.assertEqual(result, {"response": "merged"})
        write_summary.assert_called_once_with(["first", "second"], "en", ANY)

    @patch(
        "tasks.summarize_reduce.summarize_reduce._merge_idea_lists",
        return_value=["first", "second"],
    )
    def test_intermediate_reduce_keeps_a_structured_inventory(self, merge_ideas):
        result = summarize_reduce(
            {"partials": [["first"], ["second"]], "final": False}
        )

        self.assertEqual(result, {"ideas": ["first", "second"]})
        merge_ideas.assert_called_once_with(
            [["first"], ["second"]], "en", ANY,
        )

    def test_reduce_rejects_missing_partials(self):
        with self.assertRaisesRegex(ValueError, "requires idea partials"):
            summarize_reduce({"partials": []})

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
            "at most 4 complete theses",
            llm.chat.call_args.args[0][1]["content"],
        )
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

    @patch("tasks.summarize.summarize.get_llm_params", return_value={})
    @patch("tasks.summarize.summarize.get_llm_service")
    def test_intermediate_reduce_synthesizes_ideas_under_a_dynamic_ceiling(
        self, get_llm_service, _get_llm_params,
    ):
        llm = get_llm_service.return_value
        llm.chat.return_value = json.dumps({"ideas": ["one", "two"]})

        result = summarize_task._merge_idea_lists(
            [["one"], ["one paraphrased", "two"]],
            "en",
            {},
        )

        self.assertEqual(result, ["one", "two"])
        self.assertIn(
            "smaller synthesis",
            llm.chat.call_args.args[0][0]["content"],
        )
        self.assertIn(
            "at most 3 central theses",
            llm.chat.call_args.args[0][1]["content"],
        )
        self.assertEqual(llm.chat.call_args.kwargs["max_tokens"], 512)
        schema = llm.chat.call_args.kwargs["response_format"]["schema"]
        ideas_schema = schema["properties"]["ideas"]
        self.assertEqual(ideas_schema["maxItems"], 3)
        self.assertEqual(ideas_schema["items"]["maxLength"], 240)

    @patch("tasks.summarize.summarize.get_llm_params", return_value={})
    @patch("tasks.summarize.summarize.get_llm_service")
    def test_large_intermediate_reduce_must_shrink_its_candidate_inventory(
        self, get_llm_service, _get_llm_params,
    ):
        llm = get_llm_service.return_value
        llm.chat.return_value = json.dumps({"ideas": ["combined thesis"]})
        candidates = [f"candidate {index}" for index in range(42)]

        result = summarize_task._merge_idea_lists([candidates], "en", {})

        self.assertEqual(result, ["combined thesis"])
        prompt = llm.chat.call_args.args[0][1]["content"]
        self.assertIn("at most 20 central theses", prompt)
        self.assertEqual(llm.chat.call_args.kwargs["max_tokens"], 1696)
        schema = llm.chat.call_args.kwargs["response_format"]["schema"]
        self.assertEqual(schema["properties"]["ideas"]["maxItems"], 20)

    @patch("tasks.summarize.summarize.get_llm_params", return_value={})
    @patch("tasks.summarize.summarize.get_llm_service")
    def test_final_reduce_writes_prose_from_every_idea(
        self, get_llm_service, _get_llm_params,
    ):
        llm = get_llm_service.return_value
        llm.chat.return_value = "summary"

        result = summarize_task._write_summary(["one", "two"], "en", {})

        self.assertEqual(result, "summary")
        self.assertIn(
            "do not mechanically restate",
            llm.chat.call_args.args[0][0]["content"],
        )
        self.assertIn("complete inventory", llm.chat.call_args.args[0][1]["content"])
        self.assertNotIn("response_format", llm.chat.call_args.kwargs)
        self.assertEqual(llm.chat.call_args.kwargs["max_tokens"], 600)

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
