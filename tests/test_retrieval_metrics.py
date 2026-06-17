import unittest

from evaluation.metrics import (
    evaluate_query,
    ndcg_at_k,
    summarize_evaluations,
    unique_preserve_order,
)


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

    def test_ndcg_at_k_perfect_ranking_is_one(self):
        # All relevant docs at the top in ideal order.
        self.assertAlmostEqual(ndcg_at_k(["a", "b", "c"], {"a", "b"}, 3), 1.0)

    def test_ndcg_at_k_rewards_higher_ranks(self):
        top_rank = ndcg_at_k(["a", "x", "y"], {"a"}, 3)
        low_rank = ndcg_at_k(["x", "y", "a"], {"a"}, 3)
        self.assertGreater(top_rank, low_rank)
        self.assertAlmostEqual(top_rank, 1.0)

    def test_ndcg_at_k_no_relevant_is_zero(self):
        self.assertEqual(ndcg_at_k(["a", "b"], set(), 3), 0.0)

    def test_summary_includes_ndcg(self):
        result = evaluate_query(
            query_id="q1",
            question="test",
            relevant_doc_ids=["doc_a"],
            retrieved_doc_ids=["doc_a", "doc_b"],
            k=3,
        )
        summary = summarize_evaluations([result])
        self.assertIn("mean_ndcg_at_k", summary)
        self.assertAlmostEqual(summary["mean_ndcg_at_k"], 1.0)


if __name__ == "__main__":
    unittest.main()
