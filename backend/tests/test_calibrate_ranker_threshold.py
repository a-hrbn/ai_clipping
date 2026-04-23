import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))

from ml import calibrate_ranker_threshold


class CalibrateRankerThresholdTests(unittest.TestCase):
    def test_calibrates_artifact_on_gold_train_rows_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_dir = root / "artifacts"
            artifact_dir.mkdir()
            dataset = root / "gold_rows.jsonl"
            _write_artifact(artifact_dir, serving_threshold=9.0)
            _write_rows(
                dataset,
                [
                    _row("p1", 1, -0.2, "train"),
                    _row("p2", 1, 0.4, "train"),
                    _row("p3", 1, 0.5, "train"),
                    _row("n1", 0, -0.1, "train"),
                    _row("n2", 0, -0.5, "train"),
                    _row("n3", 0, -0.6, "train"),
                    _row("h1", 0, 5.0, "holdout"),
                    _row("h2", 1, -5.0, "holdout"),
                ],
            )

            report = calibrate_ranker_threshold.calibrate_threshold(
                dataset_path=dataset,
                profile="gaming",
                artifact_dir=artifact_dir,
                split_field="gold_holdout_split",
                split_value="train",
                source="gold_eval_train",
                min_rows=4,
                min_positive_rows=2,
                write_artifact=True,
            )

            artifact = json.loads((artifact_dir / "gaming_ranker.json").read_text(encoding="utf-8"))
            health = json.loads((artifact_dir / "gaming_artifact_health.json").read_text(encoding="utf-8"))

        self.assertEqual(report["status"], "updated")
        self.assertEqual(report["calibration_rows"], 6)
        self.assertEqual(report["positive_rows"], 3)
        self.assertNotEqual(report["new_threshold"], 9.0)
        self.assertEqual(artifact["serving_threshold"], report["new_threshold"])
        self.assertEqual(artifact["serving_threshold_calibration"]["source"], "gold_eval_train")
        self.assertEqual(artifact["serving_threshold_calibration"]["split_field"], "gold_holdout_split")
        self.assertEqual(health["serving_threshold"], report["new_threshold"])

    def test_skips_when_filtered_calibration_rows_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_dir = root / "artifacts"
            artifact_dir.mkdir()
            dataset = root / "gold_rows.jsonl"
            _write_artifact(artifact_dir, serving_threshold=0.0)
            _write_rows(dataset, [_row("h1", 1, 0.5, "holdout"), _row("h2", 0, -0.5, "holdout")])

            report = calibrate_ranker_threshold.calibrate_threshold(
                dataset_path=dataset,
                profile="gaming",
                artifact_dir=artifact_dir,
                split_field="gold_holdout_split",
                split_value="train",
                source="gold_eval_train",
                min_rows=1,
                min_positive_rows=1,
                write_artifact=True,
            )

        self.assertEqual(report["status"], "skipped")
        self.assertEqual(report["reason"], "calibration_rows_below_minimum")


def _write_artifact(artifact_dir: Path, *, serving_threshold: float) -> None:
    (artifact_dir / "gaming_ranker.json").write_text(
        json.dumps(
            {
                "version": 3,
                "profile": "gaming",
                "model_type": "linear",
                "feature_names": ["hook_strength"],
                "weights": {"hook_strength": 1.0},
                "scales": {"hook_strength": {"mean": 0.0, "std": 1.0}},
                "training_stats": {
                    "rows": 100,
                    "positive_rows": 40,
                    "negative_rows": 60,
                    "train_rows": 80,
                    "validation_rows": 20,
                },
                "validation": {
                    "accuracy": 0.75,
                    "precision": 0.7,
                    "recall": 0.8,
                },
                "intercept": 0.0,
                "blend": 1.0,
                "serving_threshold": serving_threshold,
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n", encoding="utf-8")


def _row(candidate_id: str, label: int, hook_strength: float, split: str) -> dict:
    return {
        "candidate_id": candidate_id,
        "label": label,
        "features": {
            "candidate_id": candidate_id,
            "hook_strength": hook_strength,
        },
        "gold_holdout_split": split,
        "vod_id": "vod-train" if split == "train" else "vod-holdout",
    }


if __name__ == "__main__":
    unittest.main()
