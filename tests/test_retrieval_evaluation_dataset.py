import pickle
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.evaluate_retrieval import load_golden_queries, ranked_doc_ids_from_passages


class RetrievalEvaluationDatasetTests(unittest.TestCase):
    def test_golden_queries_reference_existing_documents(self):
        queries = load_golden_queries(ROOT / "evaluation" / "golden_queries.json")
        with (ROOT / "data" / "index" / "docs.pkl").open("rb") as f:
            docs = pickle.load(f)
        valid_doc_ids = {doc["id"] for doc in docs}

        self.assertGreaterEqual(len(queries), 6)
        for query in queries:
            self.assertTrue(query["question"].strip())
            self.assertTrue(set(query["relevant_doc_ids"]).issubset(valid_doc_ids))

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
