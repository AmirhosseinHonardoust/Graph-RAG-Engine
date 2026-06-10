import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from backend import retriever
except ModuleNotFoundError as exc:  # Allows lightweight checks without optional FAISS/SBERT deps.
    retriever = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class FakeModel:
    def encode(self, texts, normalize_embeddings=True):
        return np.array([[1.0, 0.0]], dtype=np.float32)


class FakeGraph:
    def neighbor_chunks_by_concepts(self, chunk_id, max_neighbors=8):
        return ["chunk_b"] if chunk_id == "chunk_a" else []

    def get_doc_info(self, doc_id):
        return {
            "title": f"Doc {doc_id}",
            "url": f"data/docs/{doc_id}.md",
            "pagerank": 0.2 if doc_id == "doc_a" else 0.1,
        }


class FakeStore:
    def __init__(self):
        self.chunks = [
            {
                "id": "chunk_a",
                "doc_id": "doc_a",
                "doc_title": "Graph Notes",
                "url": "data/docs/graph.md",
                "text": "Graph retrieval text",
                "concepts": ["graph", "retrieval"],
            },
            {
                "id": "chunk_b",
                "doc_id": "doc_b",
                "doc_title": "Streamlit Notes",
                "url": "data/docs/streamlit.md",
                "text": "Streamlit interface text",
                "concepts": ["streamlit"],
            },
        ]
        self.vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        self.model = FakeModel()
        self.graph = FakeGraph()
        self.chunk_by_id = {chunk["id"]: chunk for chunk in self.chunks}
        self.chunk_index_by_id = {chunk["id"]: idx for idx, chunk in enumerate(self.chunks)}


@unittest.skipIf(IMPORT_ERROR is not None, f"Optional retrieval dependency missing: {IMPORT_ERROR}")
class RetrievalTraceTests(unittest.TestCase):
    def test_concept_overlap_is_normalized(self):
        score, count, matches = retriever.concept_overlap_score(
            {"graph", "retrieval", "search"},
            ["graph", "retrieval"],
        )

        self.assertAlmostEqual(score, 2 / 3)
        self.assertEqual(count, 2)
        self.assertEqual(matches, ["graph", "retrieval"])

    def test_build_retrieval_trace_contains_score_components(self):
        store = FakeStore()
        trace = retriever.build_retrieval_trace(
            question_vector=np.array([1.0, 0.0], dtype=np.float32),
            query_terms={"graph", "retrieval"},
            chunk=store.chunks[0],
            chunk_index=0,
            store=store,
        )

        self.assertEqual(trace["chunk_id"], "chunk_a")
        self.assertEqual(trace["concept_overlap"], 1.0)
        self.assertEqual(trace["concept_overlap_count"], 2)
        self.assertEqual(trace["matched_concepts"], ["graph", "retrieval"])
        self.assertIn("embedding_similarity", trace)
        self.assertIn("doc_pagerank", trace)
        self.assertIn("final_score", trace)
        expected = 0.60 * 1.0 + 0.25 * 1.0 + 0.15 * 0.2
        self.assertAlmostEqual(trace["final_score"], expected)

    def test_expand_and_rerank_returns_retrieval_trace(self):
        store = FakeStore()
        with patch.object(retriever, "ann_search", return_value=[("chunk_a", 0.99)]):
            passages = retriever.expand_and_rerank(
                "graph retrieval",
                base_k=1,
                expand_hops=1,
                top_n=2,
                store=store,
            )

        self.assertEqual(passages[0]["id"], "chunk_a")
        self.assertIn("retrieval_trace", passages[0])
        self.assertEqual(passages[0]["retrieval_trace"]["chunk_id"], "chunk_a")
        self.assertIn("final_score", passages[0]["retrieval_trace"])


if __name__ == "__main__":
    unittest.main()
