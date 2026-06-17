import unittest

from ingest.split import extract_concepts, simple_chunk


class SplitTests(unittest.TestCase):
    def test_simple_chunk_respects_max_chars_for_short_paragraphs(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = simple_chunk(text, max_chars=30)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(chunk.strip() for chunk in chunks))
        self.assertTrue(all(len(chunk) <= 30 for chunk in chunks))

    def test_extract_concepts_filters_common_stop_words(self):
        concepts = extract_concepts(
            "The graph graph retrieval system uses FAISS and graph concepts.",
            top_k=5,
        )

        self.assertIn("graph", concepts)
        self.assertIn("retrieval", concepts)
        self.assertNotIn("the", concepts)


if __name__ == "__main__":
    unittest.main()
