import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.evaluate_retrieval import load_golden_queries, ranked_doc_ids_from_passages
from ingest.split import simple_chunk


class RetrievalEvaluationDatasetTests(unittest.TestCase):
    def test_golden_queries_reference_existing_markdown_documents(self):
        queries = load_golden_queries(ROOT / "evaluation" / "golden_queries.json")
        valid_doc_ids = {path.stem for path in (ROOT / "data" / "docs").glob("*.md")}

        self.assertGreaterEqual(len(valid_doc_ids), 18)
        self.assertGreaterEqual(len(queries), 30)

        for query in queries:
            with self.subTest(query_id=query["id"]):
                self.assertTrue(query["question"].strip())
                self.assertTrue(query.get("notes", "").strip())
                self.assertTrue(set(query["relevant_doc_ids"]).issubset(valid_doc_ids))

    def test_golden_query_ids_are_unique(self):
        queries = load_golden_queries(ROOT / "evaluation" / "golden_queries.json")
        query_ids = [query["id"] for query in queries]

        self.assertEqual(len(query_ids), len(set(query_ids)))

    def test_golden_queries_include_multi_document_cases(self):
        queries = load_golden_queries(ROOT / "evaluation" / "golden_queries.json")
        multi_doc_queries = [query for query in queries if len(query["relevant_doc_ids"]) > 1]

        self.assertGreaterEqual(len(multi_doc_queries), 20)

    def test_demo_corpus_has_enough_chunks_for_graph_rag_demo(self):
        docs_dir = ROOT / "data" / "docs"
        docs = sorted(docs_dir.glob("*.md"))
        chunk_count = 0

        for path in docs:
            text = path.read_text(encoding="utf-8")
            self.assertGreaterEqual(len(text), 300, msg=f"{path.name} is too short")
            chunk_count += len(simple_chunk(text))

        self.assertGreaterEqual(len(docs), 18)
        self.assertGreaterEqual(chunk_count, 28)

    def test_golden_queries_file_is_readable_json_list(self):
        path = ROOT / "evaluation" / "golden_queries.json"
        text = path.read_text(encoding="utf-8")
        parsed = json.loads(text)

        self.assertIsInstance(parsed, list)
        self.assertTrue(text.strip().startswith("["))
        self.assertTrue(text.endswith("\n"))

    def test_ranked_doc_ids_from_passages_keeps_first_seen_order(self):
        passages = [
            {"doc_id": "a"},
            {"doc_id": "b"},
            {"doc_id": "a"},
            {"doc_id": "c"},
        ]

        self.assertEqual(ranked_doc_ids_from_passages(passages), ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
