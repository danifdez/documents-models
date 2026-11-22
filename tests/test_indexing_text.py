import unittest

from services.text import clean_html_text, semantic_chunk_text


class IndexingTextTests(unittest.TestCase):
    def test_nested_html_follows_source_order_without_duplicate_text(self):
        body = (
            "<div>Start<p>One <b>nested</b> paragraph</p>Then finish<br>here</div>"
            "<table><tr><th>Weight</th><td>25 grams</td></tr></table>"
            "<script>ignored()</script><style>ignored</style>"
        )
        self.assertEqual(clean_html_text(body), [
            "Start", "One nested paragraph", "Then finish here", "Weight", "25 grams",
        ])

    def test_long_plain_text_preserves_words_with_bounded_overlap(self):
        words = [f"word{index}" for index in range(18)]
        chunks = semantic_chunk_text([" ".join(words)], 5, 7, 2)
        observed = []
        for chunk in chunks:
            tokens = chunk.split()
            self.assertLessEqual(len(tokens), 7)
            if observed:
                self.assertEqual(observed[-2:], tokens[:2])
            observed.extend(tokens[2:] if observed else tokens)
        self.assertEqual(observed, words)

    def test_overlap_leaves_short_passages_and_respects_the_target(self):
        chunks = semantic_chunk_text(["first", "one two three four", "five six"], 4, 6, 2)

        self.assertEqual(chunks, ["first", "one two three four", "three four five six"])
        with self.assertRaisesRegex(ValueError, "Invalid chunk word limits"):
            semantic_chunk_text(["content"], 4, 6, 3)

    def test_rejects_uncooked_string_instead_of_splitting_into_characters(self):
        with self.assertRaisesRegex(ValueError, "sequence of text blocks"):
            semantic_chunk_text("not a block list", 5, 7, 2)
