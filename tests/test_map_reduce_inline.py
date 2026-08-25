import unittest
from unittest.mock import Mock

from lib.llm.map_reduce import (
    InlineListMapReduceSpec,
    run_inline_list_map_reduce,
)


class InlineMapReduceTest(unittest.TestCase):
    def test_returns_empty_list_without_calling_leaf(self):
        leaf = Mock()
        spec = InlineListMapReduceSpec(
            leaf_fn=leaf,
            reduce_fn=Mock(),
            chunks_fn=lambda _payload, _cfg, _is_chunk: [],
            result_key="items",
        )

        result = run_inline_list_map_reduce(
            {"content": ""},
            spec=spec,
            cfg={},
        )

        self.assertEqual(result, {"items": []})
        leaf.assert_not_called()

    def test_applies_leaf_payload_extras_in_chunk_order_before_reducing(self):
        leaf_payloads = []
        reduce_payloads = []

        def leaf(chunk, payload, _cfg):
            leaf_payloads.append(payload)
            return [{"chunk": chunk, "offset": payload["_chunk_offset"]}]

        def reduce(partials, payload, _cfg):
            reduce_payloads.append(payload)
            return [item for partial in partials for item in partial]

        spec = InlineListMapReduceSpec(
            leaf_fn=leaf,
            reduce_fn=reduce,
            chunks_fn=lambda _payload, _cfg, _is_chunk: ["first", "second"],
            result_key="items",
            leaf_payload_extras_fn=lambda _chunks, _payload, _cfg: [
                {"_chunk_offset": 0},
                {"_chunk_offset": 10},
            ],
        )

        result = run_inline_list_map_reduce(
            {"content": "source", "language": "en"},
            spec=spec,
            cfg={},
        )

        self.assertEqual(
            result,
            {
                "items": [
                    {"chunk": "first", "offset": 0},
                    {"chunk": "second", "offset": 10},
                ]
            },
        )
        self.assertEqual(
            [payload["_chunk_offset"] for payload in leaf_payloads],
            [0, 10],
        )
        self.assertEqual(reduce_payloads[0]["_chunks"], ["first", "second"])
        self.assertEqual(reduce_payloads[0]["language"], "en")

    def test_flattens_leaf_lists_for_a_prechunked_payload(self):
        reduce = Mock()
        seen_is_chunk = []

        def chunks(_payload, _cfg, is_chunk):
            seen_is_chunk.append(is_chunk)
            return ["first", "second"]

        spec = InlineListMapReduceSpec(
            leaf_fn=lambda chunk, _payload, _cfg: [chunk],
            reduce_fn=reduce,
            chunks_fn=chunks,
            result_key="items",
        )

        result = run_inline_list_map_reduce(
            {"content": "source", "_chunk_idx": 3},
            spec=spec,
            cfg={},
        )

        self.assertEqual(result, {"items": ["first", "second"]})
        self.assertEqual(seen_is_chunk, [True])
        reduce.assert_not_called()

if __name__ == "__main__":
    unittest.main()
