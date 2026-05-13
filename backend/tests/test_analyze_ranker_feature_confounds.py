import json
import tempfile
import unittest
from pathlib import Path

from ml import analyze_ranker_feature_confounds


class AnalyzeRankerFeatureConfoundsTests(unittest.TestCase):
    def test_analyze_confounds_recommends_excluding_penalty_when_positive_mean_is_higher(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "rows.jsonl"
            rows = [
                _row(1, slow_open=0.8, hook=0.1),
                _row(1, slow_open=0.7, hook=0.2),
                _row(0, slow_open=0.2, hook=0.9),
                _row(0, slow_open=0.1, hook=1.0),
            ]
            dataset.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n", encoding="utf-8")

            report = analyze_ranker_feature_confounds.analyze_confounds(
                dataset_path=dataset,
                feature="slow_open_penalty",
                compare_features=["hook_strength"],
            )

        self.assertEqual(report["recommendation"], "exclude_from_supervised_ranker_training")
        self.assertEqual(report["label_summary"]["1"]["rows"], 2)
        self.assertLess(report["correlations"][0]["correlation"], 0.0)

    def test_feature_correlation_audit_flags_penalty_score_confounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "rows.jsonl"
            rows = [
                _row(1, slow_open=0.1, hook=0.9),
                _row(1, slow_open=0.2, hook=0.8),
                _row(0, slow_open=0.8, hook=0.2),
                _row(0, slow_open=0.9, hook=0.1),
            ]
            dataset.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n", encoding="utf-8")

            audit = analyze_ranker_feature_confounds.analyze_feature_correlation_audit(
                dataset_path=dataset,
                fail_threshold=0.85,
                excluded_feature_names=set(),
            )

        self.assertTrue(audit["failed"])
        pairs = {(item["left"], item["right"]) for item in audit["failing_penalty_score_pairs"]}
        self.assertIn(("hook_strength", "slow_open_penalty"), pairs)

    def test_feature_correlation_audit_ignores_ranker_excluded_confounds_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "rows.jsonl"
            rows = [
                _row(1, slow_open=0.1, hook=0.9),
                _row(1, slow_open=0.2, hook=0.8),
                _row(0, slow_open=0.8, hook=0.2),
                _row(0, slow_open=0.9, hook=0.1),
            ]
            dataset.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n", encoding="utf-8")

            audit = analyze_ranker_feature_confounds.analyze_feature_correlation_audit(
                dataset_path=dataset,
                fail_threshold=0.85,
            )

        self.assertFalse(audit["failed"])
        self.assertIn("slow_open_penalty", audit["excluded_features"])
        self.assertEqual(audit["failing_penalty_score_pairs"], [])


def _row(label: int, *, slow_open: float, hook: float) -> dict:
    return {
        "label": label,
        "features": {
            "slow_open_penalty": slow_open,
            "hook_strength": hook,
        },
    }


if __name__ == "__main__":
    unittest.main()
