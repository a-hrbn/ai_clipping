import csv
import json
import tempfile
import unittest
from pathlib import Path

from ml import diagnose_ranker_holdout


class DiagnoseRankerHoldoutTests(unittest.TestCase):
    def test_diagnose_holdout_forces_rejected_artifact_and_writes_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_dir = root / "artifacts"
            artifact_dir.mkdir()
            dataset = root / "gold_rows.jsonl"
            csv_out = root / "diagnosis.csv"
            artifact = {
                "profile": "gaming",
                "version": 1,
                "model_type": "linear",
                "selected_model": "test_linear",
                "feature_names": ["hook_strength", "payoff_score"],
                "weights": {"hook_strength": 2.0, "payoff_score": 1.0},
                "scales": {
                    "hook_strength": {"mean": 0.0, "std": 1.0},
                    "payoff_score": {"mean": 0.0, "std": 1.0},
                },
                "intercept": -0.2,
                "blend": 0.9,
                "serving_threshold": 0.0,
                "training_stats": {"rows": 4, "positive_rows": 2, "negative_rows": 2},
                "validation": {},
            }
            (artifact_dir / "gaming_ranker.json").write_text(json.dumps(artifact), encoding="utf-8")
            rows = [
                _row("q1", "cand-1", "vod-a", 1, 0.1, 0.1),
                _row("q2", "cand-2", "vod-a", 0, 0.9, 0.7),
                _row("q3", "cand-3", "vod-b", 1, 0.8, 0.8),
                _row("q4", "cand-4", "vod-b", 0, 0.2, 0.1),
                {**_row("q5", "cand-5", "vod-c", 1, 1.0, 1.0), "gold_holdout_split": "train"},
            ]
            dataset.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

            report = diagnose_ranker_holdout.diagnose_holdout(
                dataset_path=dataset,
                profile="gaming",
                artifact_dir=artifact_dir,
                csv_out=csv_out,
                top_groups=2,
            )

            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["selected_rows"], 4)
            self.assertEqual(report["artifact"]["inspection_status"], "rejected")
            self.assertEqual(report["artifact"]["selected_model"], "test_linear")
            self.assertEqual(len(report["worst_groups"]), 2)
            self.assertTrue(csv_out.exists())
            with csv_out.open("r", encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(len(csv_rows), 4)
            self.assertIn("review_url", csv_rows[0])
            self.assertTrue(any(row["top_positive_features"] for row in csv_rows))

    def test_diagnose_holdout_reports_missing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "gold_rows.jsonl"
            csv_out = root / "diagnosis.csv"
            dataset.write_text(json.dumps(_row("q1", "cand-1", "vod-a", 1, 0.5, 0.5)), encoding="utf-8")

            report = diagnose_ranker_holdout.diagnose_holdout(
                dataset_path=dataset,
                profile="gaming",
                artifact_dir=root / "missing-artifacts",
                csv_out=csv_out,
            )

            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["reason"], "artifact_missing")
            self.assertTrue(csv_out.exists())
            with csv_out.open("r", encoding="utf-8", newline="") as handle:
                self.assertEqual(list(csv.DictReader(handle)), [])


def _row(queue_id: str, candidate_id: str, vod_id: str, label: int, hook: float, payoff: float) -> dict:
    return {
        "queue_id": queue_id,
        "job_id": f"job-{vod_id}",
        "candidate_id": candidate_id,
        "vod_id": vod_id,
        "label": label,
        "gold_grade": 3 if label else 0,
        "gold_holdout_split": "holdout",
        "source_url": f"https://www.twitch.tv/videos/{vod_id}",
        "window_start": 65.0,
        "window_end": 90.0,
        "candidate_source": "reaction",
        "content_bucket": "story",
        "features": {
            "candidate_id": candidate_id,
            "duration": 25.0,
            "hook_strength": hook,
            "payoff_score": payoff,
            "source": "reaction",
        },
    }


if __name__ == "__main__":
    unittest.main()
