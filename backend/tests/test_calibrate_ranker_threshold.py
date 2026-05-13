import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))

from ml import calibrate_ranker_threshold, train_ranker


class CalibrateRankerThresholdTests(unittest.TestCase):
    def test_score_lookup_key_prefers_replayed_feature_candidate_id(self) -> None:
        row = {
            "candidate_id": "cross_platform:vod:10.0",
            "features": {"candidate_id": "twitch_vod_123:hype_peak:0:456"},
        }

        self.assertEqual(
            calibrate_ranker_threshold._score_lookup_key(row),
            "twitch_vod_123:hype_peak:0:456",
        )

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
        self.assertIn("filter_gate", artifact)
        self.assertEqual(health["filter_gate"], artifact["filter_gate"])
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

    def test_calibrates_filter_rejected_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_dir = root / "artifacts"
            artifact_dir.mkdir()
            dataset = root / "gold_rows.jsonl"
            _write_artifact(artifact_dir, serving_threshold=9.0, rejected_filter_gate=True)
            _write_rows(
                dataset,
                [
                    _row("p1", 1, 0.7, "train"),
                    _row("p2", 1, 0.6, "train"),
                    _row("p3", 1, 0.5, "train"),
                    _row("n1", 0, -0.1, "train"),
                    _row("n2", 0, -0.2, "train"),
                    _row("n3", 0, -0.3, "train"),
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

        self.assertEqual(report["status"], "updated")
        self.assertEqual(report["artifact_status"], "rejected")
        self.assertEqual(report["artifact_reason"], "validation_filter_recall_below_minimum")
        self.assertGreaterEqual(report["calibration"]["metrics"]["recall"], 0.6)

    def test_recall_precision_mode_maximizes_precision_after_recall_target(self) -> None:
        primary = _reference_set(
            "outcome_holdout",
            labels=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            scores=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.1, 0.45, 0.0, -0.1, -0.2],
        )
        gold = _reference_set(
            "gold_eval",
            labels=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            scores=[0.9, 0.65, 0.55, 0.5, 0.4, 0.3, 0.2, 0.1, 0.35, 0.0, -0.1, -0.2],
        )

        calibration = calibrate_ranker_threshold._calibrate_recall_precision_threshold(
            [primary, gold],
            source="rollout_recall_calibration",
            min_recall=0.85,
            max_selected_rate=0.9,
            target_precision=0.85,
        )

        self.assertEqual(calibration["threshold_policy"], "recall_first_multi_reference")
        self.assertEqual(calibration["selected_threshold_reason"], "max_precision_under_recall_target")
        self.assertEqual(calibration["threshold"], 0.1)
        self.assertGreaterEqual(calibration["metrics"]["precision"], 0.85)
        self.assertGreaterEqual(calibration["reference_metrics"]["outcome_holdout"]["recall"], 0.85)
        self.assertGreaterEqual(calibration["reference_metrics"]["gold_eval"]["recall"], 0.85)

    def test_recall_precision_mode_respects_selected_rate_before_recall_fallback(self) -> None:
        reference = _reference_set(
            "outcome_holdout",
            labels=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            scores=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.65, 0.55, 0.45, 0.35],
        )

        calibration = calibrate_ranker_threshold._calibrate_recall_precision_threshold(
            [reference],
            source="rollout_recall_calibration",
            min_recall=0.85,
            max_selected_rate=0.75,
            target_precision=0.85,
        )

        self.assertEqual(
            calibration["selected_threshold_reason"],
            "best_filter_gate_under_selected_rate_below_recall_target",
        )
        self.assertEqual(calibration["threshold"], 0.4)
        self.assertLessEqual(calibration["metrics"]["selected_rate"], 0.75)
        self.assertLess(calibration["metrics"]["recall"], 0.85)

    def test_recall_precision_mode_rejects_all_open_threshold_when_cap_would_fail(self) -> None:
        reference = _reference_set(
            "outcome_holdout",
            labels=[1, 1, 0, 0],
            scores=[0.1, 0.0, 0.9, 0.8],
        )

        calibration = calibrate_ranker_threshold._calibrate_recall_precision_threshold(
            [reference],
            source="rollout_recall_calibration",
            min_recall=0.5,
            max_selected_rate=0.5,
            target_precision=0.85,
        )

        self.assertLessEqual(calibration["metrics"]["selected_rate"], 0.5)
        self.assertNotEqual(calibration["threshold"], 0.0)
        self.assertNotEqual(
            calibration["selected_threshold_reason"],
            "recall_target_above_selected_rate_rejected",
        )

    def test_write_artifact_preserves_bucket_routing_filter_gate_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_dir = root / "artifacts"
            artifact_dir.mkdir()
            dataset = root / "gold_rows.jsonl"
            _write_artifact(artifact_dir, serving_threshold=9.0, bucket_filter_gate=True)
            _write_rows(
                dataset,
                [
                    _row("p1", 1, 0.7, "train"),
                    _row("p2", 1, 0.6, "train"),
                    _row("p3", 1, 0.5, "train"),
                    _row("n1", 0, -0.1, "train"),
                    _row("n2", 0, -0.2, "train"),
                    _row("n3", 0, -0.3, "train"),
                ],
            )

            calibrate_ranker_threshold.calibrate_threshold(
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

        filter_gate = artifact["filter_gate"]
        self.assertEqual(filter_gate["source"], "gold_eval_train")
        self.assertTrue(filter_gate["passed"])
        self.assertIn("global_failures", filter_gate)
        self.assertEqual(filter_gate["selected_model_per_bucket"]["general"], "fallback_heuristic")
        self.assertEqual(filter_gate["selected_model_per_bucket"]["story"], "trained")
        self.assertEqual(filter_gate["serving_threshold_per_bucket"]["story"], 1.25)
        self.assertEqual(filter_gate["passed_bucket_count"], 3)
        self.assertEqual(health["filter_gate"], filter_gate)


def _write_artifact(
    artifact_dir: Path,
    *,
    serving_threshold: float,
    rejected_filter_gate: bool = False,
    bucket_filter_gate: bool = False,
) -> None:
    payload = {
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
    }
    if rejected_filter_gate:
        payload["filter_gate"] = {
            "passed": False,
            "precision": 0.9,
            "recall": 0.1,
            "f1": 0.18,
            "selected_rate": 0.1,
            "failures": ["filter_recall_below_minimum"],
        }
    if bucket_filter_gate:
        payload["filter_gate"] = {
            "passed": True,
            "precision": 0.9,
            "recall": 0.8,
            "f1": 0.85,
            "selected_rate": 0.3,
            "failures": [],
            "bucket_gates": {
                "general": {"passed": False, "threshold": 2.0},
                "story": {"passed": True, "threshold": 1.25},
            },
            "bucket_gate_policy": {"min_passed_buckets": 3},
            "passed_bucket_count": 3,
            "serving_threshold_per_bucket": {"general": 2.0, "story": 1.25},
            "selected_model_per_bucket": {"general": "fallback_heuristic", "story": "trained"},
            "global_failures": [],
        }
    (artifact_dir / "gaming_ranker.json").write_text(
        json.dumps(payload, ensure_ascii=True),
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


def _reference_set(source: str, *, labels: list[int], scores: list[float]) -> dict:
    return {
        "source": source,
        "rows": len(labels),
        "positive_rows": sum(1 for label in labels if label == 1),
        "negative_rows": sum(1 for label in labels if label == 0),
        "labels": train_ranker.np.array(labels, dtype=int),
        "scores": train_ranker.np.array(scores, dtype=float),
        "groups": [source] * len(labels),
        "split": {"strategy": "test", "unit": "source"},
    }


if __name__ == "__main__":
    unittest.main()
