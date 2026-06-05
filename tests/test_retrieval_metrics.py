import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.metrics import evaluate_query, summarize_evaluations, unique_preserve_order


class RetrievalMetricTests(unittest.TestCase):
    def test_unique_preserve_order_removes_duplicates_without_sorting(self):
        self.assertEqual(
            unique_preserve_order(["a", "b", "a", "c", "b"]),
            ["a", "b", "c"],
        )

    def test_evaluate_query_computes_rank_metrics(self):
        result = evaluate_query(
            query_id="q1",
            question="test query",
            relevant_doc_ids=["doc_c"],
            retrieved_doc_ids=["doc_a", "doc_c", "doc_b"],
            k=3,
        )

        self.assertEqual(result.hit_at_k, 1.0)
        self.assertAlmostEqual(result.precision_at_k, 1 / 3)
        self.assertEqual(result.recall_at_k, 1.0)
        self.assertEqual(result.reciprocal_rank, 0.5)

    def test_summarize_evaluations_averages_query_metrics(self):
        first = evaluate_query(
            query_id="q1",
            question="first",
            relevant_doc_ids=["a"],
            retrieved_doc_ids=["a", "b"],
            k=2,
        )
        second = evaluate_query(
            query_id="q2",
            question="second",
            relevant_doc_ids=["z"],
            retrieved_doc_ids=["a", "b"],
            k=2,
        )

        summary = summarize_evaluations([first, second])

        self.assertEqual(summary["num_queries"], 2.0)
        self.assertEqual(summary["mean_hit_at_k"], 0.5)
        self.assertEqual(summary["mean_recall_at_k"], 0.5)


if __name__ == "__main__":
    unittest.main()
