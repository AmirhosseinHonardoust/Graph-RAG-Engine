import json
import unittest
from pathlib import Path

from evaluation.evaluate_retrieval import check_against_baseline

ROOT = Path(__file__).resolve().parents[1]


class BaselineGateTests(unittest.TestCase):
    def test_passes_when_metrics_meet_baseline(self):
        summary = {"mean_hit_at_k": 1.0, "mean_recall_at_k": 0.9, "mean_reciprocal_rank": 0.95}
        baseline = {"mean_hit_at_k": 0.95, "mean_recall_at_k": 0.78, "mean_reciprocal_rank": 0.9}
        self.assertEqual(check_against_baseline(summary, baseline), [])

    def test_fails_when_a_metric_drops(self):
        summary = {"mean_recall_at_k": 0.5}
        baseline = {"mean_recall_at_k": 0.78}
        failures = check_against_baseline(summary, baseline)
        self.assertEqual(len(failures), 1)
        self.assertIn("mean_recall_at_k", failures[0])

    def test_ignores_comment_keys(self):
        failures = check_against_baseline(
            {"mean_hit_at_k": 1.0}, {"_comment": "note", "mean_hit_at_k": 0.9}
        )
        self.assertEqual(failures, [])

    def test_committed_baseline_file_is_valid(self):
        baseline = json.loads((ROOT / "evaluation" / "baseline.json").read_text(encoding="utf-8"))
        numeric = {k: v for k, v in baseline.items() if not k.startswith("_")}
        self.assertTrue(numeric)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in numeric.values()))


if __name__ == "__main__":
    unittest.main()
