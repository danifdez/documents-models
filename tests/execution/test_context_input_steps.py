import hashlib
import unittest
from unittest.mock import Mock, patch

from tasks.context_input_map.context_input_map import context_input_map
from tasks.context_input_reduce.context_input_reduce import context_input_reduce


class ContextInputStepsTest(unittest.TestCase):
    @patch("tasks.context_input_map.context_input_map.get_llm_service")
    def test_map_validates_the_frozen_chunk_and_returns_its_digest(self, get_llm):
        content = "bounded source chunk"
        get_llm.return_value.chat.return_value = "faithful chunk digest"

        result = context_input_map(
            {
                "chunkIndex": 0,
                "content": content,
                "contentHash": "sha256:"
                + hashlib.sha256(content.encode("utf-8")).hexdigest(),
            }
        )

        self.assertEqual(result, {"digest": "faithful chunk digest"})
        get_llm.return_value.chat.assert_called_once()

    def test_map_rejects_content_that_does_not_match_the_plan_hash(self):
        with self.assertRaisesRegex(ValueError, "Invalid context input chunk"):
            context_input_map(
                {
                    "chunkIndex": 0,
                    "content": "tampered",
                    "contentHash": "sha256:" + "0" * 64,
                }
            )

    @patch("tasks.context_input_reduce.context_input_reduce.get_llm_service")
    def test_reduce_preserves_partial_order(self, get_llm):
        llm = Mock()
        llm.chat.return_value = "merged digest"
        get_llm.return_value = llm

        result = context_input_reduce({"partials": ["first", "second"]})

        self.assertEqual(result, {"digest": "merged digest"})
        messages = llm.chat.call_args.args[0]
        self.assertLess(
            messages[1]["content"].index("first"),
            messages[1]["content"].index("second"),
        )


if __name__ == "__main__":
    unittest.main()
