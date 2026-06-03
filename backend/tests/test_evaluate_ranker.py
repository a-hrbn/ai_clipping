import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))

from ml import evaluate_ranker


class EvaluateRankerTests(unittest.TestCase):
    def test_score_lookup_key_prefers_replayed_feature_candidate_id(self) -> None:
        row = {
            "candidate_id": "cross_platform:vod:10.0",
            "features": {"candidate_id": "twitch_vod_123:hype_peak:0:456"},
        }

        self.assertEqual(evaluate_ranker._score_lookup_key(row), "twitch_vod_123:hype_peak:0:456")

    def test_evaluates_forced_artifact_metrics_when_runtime_rejects_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_dir = root / "artifacts"
            artifact_dir.mkdir()
            dataset = root / "rows.jsonl"
            _write_artifact(artifact_dir)
            _write_rows(
                dataset,
                [
                    _row("p1", 1, 1.0, "vod-a"),
                    _row("n1", 0, -1.0, "vod-a"),
                    _row("p2", 1, 0.8, "vod-b"),
                    _row("n2", 0, -0.8, "vod-b"),
                ],
            )

            report = evaluate_ranker.evaluate_dataset(
                dataset_path=dataset,
                profile="gaming",
                artifact_dir=artifact_dir,
            )

        self.assertEqual(report["artifact"]["status"], "rejected")
        self.assertEqual(report["artifact"]["reason"], "validation_filter_selected_rate_above_maximum")
        self.assertFalse(report["trained_runtime_uses_artifact"])
        self.assertTrue(report["trained_forced_artifact_available"])
        self.assertEqual(report["runtime_fallback_reason"], "validation_filter_selected_rate_above_maximum")
        self.assertIn("trained_runtime_matches_baseline_because_artifact_was_rejected", report["delta_interpretation"])
        self.assertIn("trained_forced_artifact", report["metrics"])
        self.assertIn("forced_artifact_delta", report["metrics"])
        self.assertIn("selected_rate", report["metrics"]["forced_artifact_delta"])
        self.assertIn("viral", report["by_outcome_view_tier"])
        self.assertIn("4", report["by_rank_target"])


def _write_artifact(artifact_dir: Path) -> None:
    payload = {
        "version": 3,
        "profile": "gaming",
        "model_type": "linear",
        "feature_names": ["hook_strength"],
        "weights": {"hook_strength": 5.0},
        "scales": {"hook_strength": {"mean": 0.0, "std": 1.0}},
        "training_stats": {
            "rows": 100,
            "positive_rows": 40,
            "negative_rows": 60,
            "train_rows": 80,
            "validation_rows": 20,
        },
        "validation": {
            "accuracy": 0.4,
            "precision_at_5": 0.8,
            "ndcg_at_5": 0.8,
            "ranking_groups": 2,
            "ndcg_at_5_min": 0.5,
            "ndcg_at_5_p25": 0.6,
        },
        "filter_gate": {
            "precision": 0.55,
            "recall": 0.9,
            "f1": 0.68,
            "selected_rate": 0.95,
        },
        "intercept": 0.0,
        "blend": 1.0,
        "serving_threshold": 0.0,
    }
    (artifact_dir / "gaming_ranker.json").write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n", encoding="utf-8")


def _row(candidate_id: str, label: int, hook_strength: float, vod_id: str) -> dict:
    return {
        "candidate_id": candidate_id,
        "label": label,
        "vod_id": vod_id,
        "outcome_view_tier": "viral" if label else "",
        "rank_target": 4 if label else 0,
        "features": {
            "candidate_id": candidate_id,
            "hook_strength": hook_strength,
        },
    }


if __name__ == "__main__":
    unittest.main()
