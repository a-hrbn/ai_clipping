import json
import tempfile
import unittest
from pathlib import Path

from ml import train_ranker


class _FakeLGBMRanker:
    def __init__(self, *args, **kwargs) -> None:
        self.feature_importances_ = [1.0]

    def fit(self, train_matrix, train_targets, **kwargs) -> None:
        self._positive_mean = max(
            float(train_matrix[idx][0])
            for idx, target in enumerate(train_targets.tolist())
            if int(target) > 0
        )

    def predict(self, matrix):
        return [float(row[0]) - self._positive_mean for row in matrix]


class TrainRankerTests(unittest.TestCase):
    def test_split_rows_creates_train_and_validation_sets(self) -> None:
        rows = [
            {"job_id": "job", "candidate_id": f"c{i}", "label": i % 2, "features": {"x": float(i)}}
            for i in range(10)
        ]
        split = train_ranker._split_rows(rows)
        self.assertTrue(split["train"])
        self.assertTrue(split["validation"])

    def test_split_rows_keeps_creator_groups_together_when_present(self) -> None:
        rows = [
            {"job_id": "j1", "creator_id": "streamer_a", "candidate_id": "a1", "label": 1, "features": {"x": 1.0}},
            {"job_id": "j2", "creator_id": "streamer_a", "candidate_id": "a2", "label": 0, "features": {"x": 2.0}},
            {"job_id": "j3", "creator_id": "streamer_b", "candidate_id": "b1", "label": 1, "features": {"x": 3.0}},
            {"job_id": "j4", "creator_id": "streamer_c", "candidate_id": "c1", "label": 0, "features": {"x": 4.0}},
            {"job_id": "j5", "creator_id": "streamer_c", "candidate_id": "c2", "label": 1, "features": {"x": 5.0}},
        ]

        split = train_ranker._split_rows(rows)
        train_creators = {row["creator_id"] for row in split["train"]}
        validation_creators = {row["creator_id"] for row in split["validation"]}

        self.assertTrue(train_creators)
        self.assertTrue(validation_creators)
        self.assertTrue(train_creators.isdisjoint(validation_creators))

    def test_split_rows_uses_time_holdout_for_single_creator(self) -> None:
        rows = [
            {
                "job_id": f"j{i}",
                "creator_id": "jynxzi",
                "candidate_id": f"c{i}",
                "label": i % 2,
                "publish_timestamp": 1_700_000_000 + i * 60,
                "features": {"x": float(i)},
            }
            for i in range(10)
        ]

        split = train_ranker._split_rows(rows)

        self.assertEqual(split["strategy"], "time_holdout")
        self.assertEqual(split["unit"], "timestamp")
        self.assertLess(
            max(row["publish_timestamp"] for row in split["train"]),
            min(row["publish_timestamp"] for row in split["validation"]),
        )

    def test_split_rows_prefers_vod_group_holdout_when_available(self) -> None:
        rows = [
            {"vod_id": "vod_a", "candidate_id": "a1", "label": 1, "features": {"x": 1.0}},
            {"vod_id": "vod_a", "candidate_id": "a2", "label": 0, "features": {"x": 2.0}},
            {"vod_id": "vod_b", "candidate_id": "b1", "label": 1, "features": {"x": 3.0}},
            {"vod_id": "vod_b", "candidate_id": "b2", "label": 0, "features": {"x": 4.0}},
            {"vod_id": "vod_c", "candidate_id": "c1", "label": 1, "features": {"x": 5.0}},
            {"vod_id": "vod_c", "candidate_id": "c2", "label": 0, "features": {"x": 6.0}},
        ]

        split = train_ranker._split_rows(rows)

        self.assertEqual(split["strategy"], "group_holdout")
        self.assertEqual(split["unit"], "vod_id")
        train_vods = {row["vod_id"] for row in split["train"]}
        validation_vods = {row["vod_id"] for row in split["validation"]}
        self.assertTrue(train_vods.isdisjoint(validation_vods))

    def test_evaluate_returns_metrics(self) -> None:
        rows = [
            {"label": 1, "features": {"score": 2.0}},
            {"label": 0, "features": {"score": -2.0}},
        ]
        metrics = train_ranker._evaluate(
            rows,
            ["score"],
            {"score": 1.0},
            {"score": {"mean": 0.0, "std": 1.0}},
        )
        self.assertGreaterEqual(metrics["accuracy"], 0.5)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)
        self.assertIn("ndcg_at_5", metrics)

    def test_numeric_feature_names_excludes_offline_only_features(self) -> None:
        names = train_ranker._numeric_feature_names({
            "candidate_id": "c1",
            "source": "reaction",
            "hook_strength": 0.8,
            "vod_progress": 0.5,
            "boundary_left_context_sec": 12.0,
            "clip_overlap_count": 2.0,
        })
        self.assertEqual(names, ["hook_strength"])

    def test_select_model_requires_balanced_ranking_and_accuracy(self) -> None:
        selected = train_ranker._select_model([
            {"name": "linear", "artifact_model_type": "linear", "metrics": {"ndcg_at_5": 0.5, "f1": 0.8, "accuracy": 0.8}},
            {"name": "logistic", "artifact_model_type": "logistic", "metrics": {"ndcg_at_5": 0.7, "precision_at_5": 0.8, "f1": 0.6, "accuracy": 0.6}},
            {"name": "gradient_boosting", "artifact_model_type": "gradient_boosting", "metrics": {"ndcg_at_5": 0.9, "precision_at_5": 0.8, "f1": 0.4, "accuracy": 0.55}},
        ])
        self.assertEqual(selected["name"], "logistic")

    def test_pairwise_training_matrix_builds_comparisons_within_group(self) -> None:
        rows = [
            {"job_id": "job-1", "candidate_id": "p1", "label": 1, "features": {"signal": 3.0}},
            {"job_id": "job-1", "candidate_id": "p2", "label": 1, "features": {"signal": 2.0}},
            {"job_id": "job-1", "candidate_id": "n1", "label": 0, "features": {"signal": -1.0}},
            {"job_id": "job-1", "candidate_id": "n2", "label": 0, "features": {"signal": -2.0}},
        ]
        scales = train_ranker._compute_scales(rows, ["signal"])

        matrix, labels, weights = train_ranker._pairwise_training_matrix(rows, ["signal"], scales)

        self.assertEqual(matrix.shape[1], 1)
        self.assertGreaterEqual(len(labels), 4)
        self.assertEqual(set(labels.tolist()), {0, 1})
        self.assertEqual(len(weights), len(labels))

    def test_train_from_rows_writes_model_path_for_gradient_boosting_artifact(self) -> None:
        rows = [
            {
                "job_id": f"j{i}",
                "creator_id": "jynxzi",
                "candidate_id": f"c{i}",
                "label": 1 if i % 2 == 0 else 0,
                "publish_timestamp": 1_700_000_000 + i * 60,
                "features": {"signal": 5.0 if i % 2 == 0 else -5.0},
            }
            for i in range(60)
        ]

        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            rc = train_ranker.train_from_rows(rows, "gaming", artifact_dir)
            self.assertEqual(rc, 0)
            artifact = json.loads((artifact_dir / "gaming_ranker.json").read_text(encoding="utf-8"))
            if artifact["model_type"] == "gradient_boosting":
                self.assertTrue(artifact["model_path"])
                self.assertTrue((artifact_dir / artifact["model_path"]).exists())

    def test_fit_lightgbm_ranker_returns_candidate_when_available(self) -> None:
        rows = [
            {"vod_id": "vod-1", "candidate_id": "a1", "label": 1, "ordinal_label": 4, "features": {"signal": 4.0}},
            {"vod_id": "vod-1", "candidate_id": "a2", "label": 0, "ordinal_label": 0, "features": {"signal": -2.0}},
            {"vod_id": "vod-2", "candidate_id": "b1", "label": 1, "ordinal_label": 3, "features": {"signal": 3.0}},
            {"vod_id": "vod-2", "candidate_id": "b2", "label": 0, "ordinal_label": 0, "features": {"signal": -3.0}},
        ]
        scales = train_ranker._compute_scales(rows, ["signal"])
        original = train_ranker.LGBMRanker
        train_ranker.LGBMRanker = _FakeLGBMRanker
        try:
            candidate = train_ranker._fit_lightgbm_ranker(
                rows[:2] + rows[2:],
                rows,
                ["signal"],
                scales,
                "vod_id",
            )
        finally:
            train_ranker.LGBMRanker = original

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate["artifact_model_type"], "lightgbm_ranker")
        self.assertIn("ndcg_at_5", candidate["metrics"])


if __name__ == "__main__":
    unittest.main()
