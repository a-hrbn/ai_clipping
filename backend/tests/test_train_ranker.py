import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))

from ml import train_ranker


class _FakeLGBMRanker:
    last_init_kwargs = None
    last_fit_kwargs = None

    def __init__(self, *args, **kwargs) -> None:
        type(self).last_init_kwargs = dict(kwargs)
        self.feature_importances_ = [1.0]

    def fit(self, train_matrix, train_targets, **kwargs) -> None:
        type(self).last_fit_kwargs = dict(kwargs)
        self._positive_mean = max(
            float(train_matrix[idx][0])
            for idx, target in enumerate(train_targets.tolist())
            if int(target) > 0
        )

    def predict(self, matrix):
        return [float(row[0]) - self._positive_mean for row in matrix]


class _FakeMlflowSession:
    def __init__(self) -> None:
        self.enabled = True
        self.run_id = "run-123"
        self.logged_metrics: list[tuple[dict, str]] = []
        self.logged_tags: list[dict] = []
        self.logged_artifacts: list[str] = []
        self.register_calls = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def log_params(self, params: dict) -> None:
        return None

    def log_metrics(self, metrics: dict, *, prefix: str = "") -> None:
        self.logged_metrics.append((dict(metrics), prefix))

    def set_tags(self, tags: dict) -> None:
        self.logged_tags.append(dict(tags))

    def log_artifact(self, path: Path, *, artifact_path: str | None = None) -> None:
        del artifact_path
        self.logged_artifacts.append(path.name)

    def register_ranker_bundle(self, **kwargs):
        self.register_calls += 1
        return {
            "enabled": True,
            "model_name": "gaming_ranker",
            "version": "7",
            "stage": "Staging",
            "promotion_gates": {"source_mix_ok": True},
        }


class TrainRankerTests(unittest.TestCase):
    def test_split_rows_creates_train_and_validation_sets(self) -> None:
        rows = [
            {"job_id": "job", "candidate_id": f"c{i}", "label": i % 2, "features": {"x": float(i)}}
            for i in range(10)
        ]
        split = train_ranker._split_rows(rows)
        self.assertTrue(split["train"])
        self.assertTrue(split["validation"])

    def test_split_rows_forces_gold_train_rows_out_of_validation(self) -> None:
        rows = []
        for vod in ("vod-a", "vod-b", "vod-c", "vod-d"):
            rows.append({"vod_id": vod, "candidate_id": f"{vod}-p", "label": 1, "features": {"x": 1.0}})
            rows.append({"vod_id": vod, "candidate_id": f"{vod}-n", "label": 0, "features": {"x": -1.0}})
        rows.extend(
            [
                {
                    "vod_id": "vod-gold",
                    "candidate_id": "gold-p",
                    "label": 1,
                    "supervision_tier": "gold_eval",
                    "gold_holdout_split": "train",
                    "features": {"x": 2.0},
                },
                {
                    "vod_id": "vod-gold",
                    "candidate_id": "gold-n",
                    "label": 0,
                    "label_source": "gold_eval",
                    "gold_holdout_split": "train",
                    "features": {"x": -2.0},
                },
            ]
        )

        split = train_ranker._split_rows(rows)

        validation_ids = {row["candidate_id"] for row in split["validation"]}
        train_ids = {row["candidate_id"] for row in split["train"]}
        self.assertEqual(split["gold_train_rows_forced_to_train"], 2)
        self.assertIn("gold-p", train_ids)
        self.assertIn("gold-n", train_ids)
        self.assertNotIn("gold-p", validation_ids)
        self.assertNotIn("gold-n", validation_ids)

    def test_count_values_counts_non_empty_values(self) -> None:
        rows = [
            {"dataset_source": "twitch_vod_primary"},
            {"dataset_source": "twitch_vod_primary"},
            {"dataset_source": "short_form_supplemental"},
            {"dataset_source": ""},
        ]

        self.assertEqual(
            train_ranker._count_values(rows, "dataset_source"),
            {"short_form_supplemental": 1, "twitch_vod_primary": 2},
        )

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

    def test_split_rows_uses_multiple_mixed_validation_vods(self) -> None:
        rows = [
            {"vod_id": "vod_big", "candidate_id": f"big{i}", "label": 0 if i else 1, "features": {"x": float(i)}}
            for i in range(20)
        ]
        rows.extend([
            {"vod_id": "vod_a", "candidate_id": "a1", "label": 1, "features": {"x": 21.0}},
            {"vod_id": "vod_a", "candidate_id": "a2", "label": 0, "features": {"x": 22.0}},
            {"vod_id": "vod_b", "candidate_id": "b1", "label": 1, "features": {"x": 23.0}},
            {"vod_id": "vod_b", "candidate_id": "b2", "label": 0, "features": {"x": 24.0}},
            {"vod_id": "vod_c", "candidate_id": "c1", "label": 1, "features": {"x": 25.0}},
            {"vod_id": "vod_c", "candidate_id": "c2", "label": 0, "features": {"x": 26.0}},
        ])

        split = train_ranker._split_rows(rows)

        self.assertEqual(split["strategy"], "group_holdout")
        self.assertEqual(split["unit"], "vod_id")
        validation_vods = {row["vod_id"] for row in split["validation"]}
        self.assertGreaterEqual(len(validation_vods), 2)
        self.assertNotEqual(validation_vods, {"vod_big"})

    def test_split_rows_falls_back_when_group_holdout_drops_a_class(self) -> None:
        rows = [
            {"vod_id": "vod_a", "candidate_id": "a1", "label": 1, "features": {"x": 1.0}},
            {"vod_id": "vod_a", "candidate_id": "a2", "label": 1, "features": {"x": 1.5}},
            {"vod_id": "vod_a", "candidate_id": "a3", "label": 1, "features": {"x": 1.7}},
            {"vod_id": "vod_a", "candidate_id": "a4", "label": 1, "features": {"x": 1.9}},
            {"vod_id": "vod_b", "candidate_id": "b1", "label": 0, "features": {"x": -1.0}},
            {"vod_id": "vod_c", "candidate_id": "c1", "label": 0, "features": {"x": -2.0}},
        ]

        split = train_ranker._split_rows(rows)

        self.assertIn("fallback", str(split["strategy"]))
        self.assertIn("warning", split)
        self.assertEqual(split["seed"], train_ranker._SPLIT_RANDOM_SEED)
        self.assertEqual({row["label"] for row in split["train"]}, {0, 1})
        self.assertEqual({row["label"] for row in split["validation"]}, {0, 1})

    def test_stratified_fallback_split_is_seeded_and_deterministic(self) -> None:
        rows = [
            {"job_id": "job", "candidate_id": f"p{i}", "label": 1, "features": {"x": float(i)}}
            for i in range(5)
        ] + [
            {"job_id": "job", "candidate_id": f"n{i}", "label": 0, "features": {"x": float(-i)}}
            for i in range(5)
        ]

        left = train_ranker._stratified_fallback_split(rows, "stratified_fallback", "row")
        right = train_ranker._stratified_fallback_split(rows, "stratified_fallback", "row")

        self.assertEqual(
            [row["candidate_id"] for row in left["train"]],
            [row["candidate_id"] for row in right["train"]],
        )
        self.assertEqual(
            [row["candidate_id"] for row in left["validation"]],
            [row["candidate_id"] for row in right["validation"]],
        )

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

    def test_evaluate_scores_averages_ranking_metrics_per_group(self) -> None:
        labels = train_ranker.np.array([1, 0, 1, 0], dtype=int)
        scores = train_ranker.np.array([0.8, 0.9, 0.7, 0.6], dtype=float)
        groups = ["vod-a", "vod-a", "vod-b", "vod-b"]

        metrics = train_ranker._evaluate_scores(labels, scores, groups=groups)

        self.assertEqual(metrics["ranking_groups"], 2)
        self.assertEqual(metrics["precision_at_3"], 0.5)
        self.assertEqual(metrics["recall_at_3"], 1.0)
        self.assertEqual(metrics["ndcg_at_5"], 0.815)
        self.assertEqual(metrics["ndcg_at_5_min"], 0.631)
        self.assertEqual(metrics["ndcg_at_5_p25"], 0.631)
        self.assertLess(metrics["global_ndcg_at_5"], metrics["ndcg_at_5"])

    def test_evaluate_scores_uses_explicit_classification_threshold(self) -> None:
        labels = train_ranker.np.array([1, 0, 1, 0], dtype=int)
        scores = train_ranker.np.array([-0.2, -0.1, 0.3, -0.4], dtype=float)

        default_metrics = train_ranker._evaluate_scores(labels, scores)
        thresholded_metrics = train_ranker._evaluate_scores(labels, scores, threshold=-0.2)

        self.assertEqual(default_metrics["recall"], 0.5)
        self.assertEqual(thresholded_metrics["classification_threshold"], -0.2)
        self.assertEqual(thresholded_metrics["recall"], 1.0)
        self.assertGreater(thresholded_metrics["f1"], default_metrics["f1"])

    def test_calibrate_serving_threshold_prefers_recall_safe_f1(self) -> None:
        labels = train_ranker.np.array([1, 0, 1, 0], dtype=int)
        scores = train_ranker.np.array([-0.2, -0.1, 0.3, -0.4], dtype=float)

        calibration = train_ranker._calibrate_serving_threshold(labels, scores)

        self.assertEqual(calibration["source"], "validation")
        self.assertEqual(calibration["threshold"], -0.2)
        self.assertEqual(calibration["metrics"]["recall"], 1.0)
        self.assertGreater(calibration["metrics"]["f1"], calibration["default_metrics"]["f1"])

    def test_runtime_blended_scores_apply_serving_blend_cap(self) -> None:
        rows = [
            {"candidate_id": "positive", "label": 1, "features": {"signal": 1.0}},
            {"candidate_id": "negative", "label": 0, "features": {"signal": -1.0}},
        ]
        selected_model = {"artifact_model_type": "linear", "weights": {"signal": 1.0}, "intercept": 0.0}

        with (
            patch.object(train_ranker.Ranker, "_heuristic_scores", return_value=[(10.0, []), (10.0, [])]),
            patch.object(
                train_ranker,
                "_runtime_trained_scores",
                return_value=train_ranker.np.array([-10.0, -10.0], dtype=float),
            ),
        ):
            scores = train_ranker._runtime_blended_scores(
                rows,
                "gaming",
                ["signal"],
                {},
                selected_model,
                1.0,
            )

        self.assertEqual(scores.tolist(), [-3.0, -3.0])

    def test_calibrate_blend_persists_runtime_effective_blend(self) -> None:
        rows = [
            {"candidate_id": "p1", "label": 1, "features": {"signal": 1.0}, "vod_id": "vod-a"},
            {"candidate_id": "n1", "label": 0, "features": {"signal": -1.0}, "vod_id": "vod-a"},
            {"candidate_id": "p2", "label": 1, "features": {"signal": 1.0}, "vod_id": "vod-b"},
            {"candidate_id": "n2", "label": 0, "features": {"signal": -1.0}, "vod_id": "vod-b"},
        ]
        selected_model = {"artifact_model_type": "linear", "weights": {"signal": 1.0}, "intercept": 0.0}

        with (
            patch.object(train_ranker.Ranker, "_heuristic_scores", return_value=[(0.0, []), (10.0, []), (0.0, []), (10.0, [])]),
            patch.object(
                train_ranker,
                "_runtime_trained_scores",
                return_value=train_ranker.np.array([10.0, 0.0, 10.0, 0.0], dtype=float),
            ),
        ):
            calibration = train_ranker._calibrate_blend(
                rows,
                "gaming",
                ["signal"],
                {},
                selected_model,
                "vod_id",
            )

        self.assertEqual(calibration["runtime_max_blend"], 0.65)
        self.assertLessEqual(calibration["best_blend"], calibration["runtime_max_blend"])
        self.assertEqual(calibration["best_blend"], calibration["requested_best_blend"])
        self.assertEqual(calibration["grid_results"]["1.00"]["effective_blend"], 0.65)

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
        ], training_row_count=120)
        self.assertEqual(selected["name"], "logistic")

    def test_model_minimums_require_multiple_ranking_groups_when_reported(self) -> None:
        self.assertFalse(
            train_ranker._model_meets_minimums(
                {"accuracy": 0.9, "ndcg_at_5": 0.9, "precision_at_5": 0.8, "ranking_groups": 1}
            )
        )
        self.assertTrue(
            train_ranker._model_meets_minimums(
                {
                    "accuracy": 0.9,
                    "ndcg_at_5": 0.9,
                    "precision_at_5": 0.8,
                    "ranking_groups": 2,
                    "ndcg_at_5_min": 0.5,
                    "ndcg_at_5_p25": 0.6,
                }
            )
        )

    def test_model_minimums_reject_low_group_ndcg_floor(self) -> None:
        self.assertFalse(
            train_ranker._model_meets_minimums(
                {
                    "accuracy": 0.9,
                    "ndcg_at_5": 0.9,
                    "precision_at_5": 0.8,
                    "ranking_groups": 2,
                    "ndcg_at_5_min": 0.1,
                    "ndcg_at_5_p25": 0.55,
                }
            )
        )

    def test_evaluate_scores_switches_to_ndcg_at_10_when_group_positives_are_below_five(self) -> None:
        labels = train_ranker.np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=int)
        scores = train_ranker.np.array([0.9, 0.8, 0.7, 0.6, 0.2, 0.1, 0.05, 0.0], dtype=float)
        groups = ["vod-a"] * 4 + ["vod-b"] * 4

        metrics = train_ranker._evaluate_scores(labels, scores, groups=groups)

        self.assertEqual(metrics["min_group_positive_rows"], 0)
        self.assertEqual(metrics["ranking_metric"], "mrr")

    def test_evaluate_scores_switches_to_ndcg_at_10_when_group_positives_are_between_five_and_nine(self) -> None:
        labels = train_ranker.np.array(
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            dtype=int,
        )
        scores = train_ranker.np.array(
            [0.99, 0.95, 0.9, 0.85, 0.8, 0.2, 0.1, 0.05, 0.01, 0.0, 0.98, 0.94, 0.89, 0.84, 0.79, 0.19, 0.09, 0.04, 0.02, 0.01],
            dtype=float,
        )
        groups = ["vod-a"] * 10 + ["vod-b"] * 10

        metrics = train_ranker._evaluate_scores(labels, scores, groups=groups)

        self.assertEqual(metrics["min_group_positive_rows"], 5)
        self.assertEqual(metrics["ranking_metric"], "ndcg_at_10")

    def test_train_from_rows_rejects_gold_eval_holdout_rows(self) -> None:
        rows = [
            {
                "job_id": "job-1",
                "candidate_id": "c1",
                "label": 1,
                "supervision_tier": "gold_eval",
                "gold_holdout_split": "holdout",
                "features": {"signal": 1.0},
            },
            {
                "job_id": "job-1",
                "candidate_id": "c2",
                "label": 0,
                "supervision_tier": "gold_eval",
                "gold_holdout_split": "train",
                "features": {"signal": -1.0},
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            rc = train_ranker.train_from_rows(rows, "gaming", Path(tmp))

        self.assertEqual(rc, 1)

    def test_train_from_rows_rejects_gold_eval_rows_without_frozen_split(self) -> None:
        rows = [
            {
                "job_id": "job-1",
                "candidate_id": "c1",
                "label": 1,
                "supervision_tier": "gold_eval",
                "features": {"signal": 1.0},
            },
            {
                "job_id": "job-1",
                "candidate_id": "c2",
                "label": 0,
                "supervision_tier": "gold_eval",
                "features": {"signal": -1.0},
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            rc = train_ranker.train_from_rows(rows, "gaming", Path(tmp))

        self.assertEqual(rc, 1)

    def test_mean_difference_weights_respect_sample_weights(self) -> None:
        rows = [
            {"label": 1, "sample_weight": 5.0, "features": {"signal": 5.0}},
            {"label": 1, "sample_weight": 1.0, "features": {"signal": 1.0}},
            {"label": 0, "sample_weight": 1.0, "features": {"signal": 0.0}},
            {"label": 0, "sample_weight": 1.0, "features": {"signal": -1.0}},
        ]
        scales = {"signal": {"mean": 0.0, "std": 1.0}}

        weights = train_ranker._mean_difference_weights(rows, ["signal"], scales)

        self.assertAlmostEqual(weights["signal"], 4.833, places=3)

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

    def test_select_model_keeps_lightgbm_disabled_below_phase_a_gate(self) -> None:
        selected = train_ranker._select_model(
            [
                {
                    "name": "logistic_regression",
                    "artifact_model_type": "logistic",
                    "metrics": {"ndcg_at_5": 0.81, "precision_at_5": 0.8, "f1": 0.72, "accuracy": 0.71},
                },
                {
                    "name": "lightgbm_ranker",
                    "artifact_model_type": "lightgbm_ranker",
                    "metrics": {"ndcg_at_5": 0.97, "precision_at_5": 0.95, "f1": 0.9, "accuracy": 0.88},
                },
            ],
            training_row_count=train_ranker._LIGHTGBM_PROMOTION_MIN_ROWS - 1,
        )
        self.assertEqual(selected["name"], "logistic_regression")

    def test_select_model_keeps_gradient_boosting_disabled_below_gate(self) -> None:
        selected = train_ranker._select_model(
            [
                {
                    "name": "logistic_regression",
                    "artifact_model_type": "logistic",
                    "metrics": {"ndcg_at_5": 0.81, "precision_at_5": 0.8, "f1": 0.72, "accuracy": 0.71},
                },
                {
                    "name": "gradient_boosting",
                    "artifact_model_type": "gradient_boosting",
                    "metrics": {"ndcg_at_5": 0.97, "precision_at_5": 0.95, "f1": 0.9, "accuracy": 0.88},
                },
            ],
            training_row_count=train_ranker._GRADIENT_BOOSTING_PROMOTION_MIN_ROWS - 1,
        )
        self.assertEqual(selected["name"], "logistic_regression")

    def test_numeric_feature_names_excludes_known_confound_features(self) -> None:
        names = train_ranker._numeric_feature_names(
            {
                "signal": 1.0,
                "slow_open_penalty": 0.8,
                "candidate_id": "c1",
            }
        )

        self.assertEqual(names, ["signal"])

    def test_sample_weight_downweights_twitch_clip_proxy_rows(self) -> None:
        twitch_proxy = {
            "supervision_tier": "twitch_clip_proxy",
            "label_class": "hard_negative",
            "sample_weight": 1.8,
        }
        manual = {
            "supervision_tier": "manual_annotation",
            "label_class": "hard_negative",
            "sample_weight": 1.8,
        }

        self.assertAlmostEqual(train_ranker._sample_weight(twitch_proxy), 0.9)
        self.assertGreater(train_ranker._sample_weight(manual), train_ranker._sample_weight(twitch_proxy))

    def test_select_model_allows_lightgbm_after_phase_a_gate(self) -> None:
        selected = train_ranker._select_model(
            [
                {
                    "name": "logistic_regression",
                    "artifact_model_type": "logistic",
                    "metrics": {"ndcg_at_5": 0.81, "precision_at_5": 0.8, "f1": 0.72, "accuracy": 0.71},
                },
                {
                    "name": "lightgbm_ranker",
                    "artifact_model_type": "lightgbm_ranker",
                    "metrics": {"ndcg_at_5": 0.97, "precision_at_5": 0.95, "f1": 0.9, "accuracy": 0.88},
                },
            ],
            training_row_count=train_ranker._LIGHTGBM_PROMOTION_MIN_ROWS,
        )
        self.assertEqual(selected["name"], "lightgbm_ranker")

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
            artifact_health = json.loads((artifact_dir / "gaming_artifact_health.json").read_text(encoding="utf-8"))
            self.assertEqual(artifact["version"], 4)
            self.assertEqual(artifact["feature_schema_version"], 4)
            self.assertIn("semantic_score_config", artifact)
            self.assertIn("top_feature_importances", artifact)
            self.assertIn("slow_open_penalty", artifact["excluded_feature_names"])
            self.assertIn("model_selection", artifact)
            self.assertEqual(artifact_health["selected_model"], artifact["selected_model"])
            self.assertEqual(artifact_health["semantic_score_config"], artifact["semantic_score_config"])
            self.assertIn("gates", artifact_health)
            if artifact["model_type"] == "gradient_boosting":
                self.assertTrue(artifact["model_path"])
                self.assertTrue((artifact_dir / artifact["model_path"]).exists())

    def test_train_from_rows_records_mlflow_run_metadata_when_tracking_is_enabled(self) -> None:
        rows = [
            {
                "job_id": f"j{i}",
                "creator_id": "jynxzi",
                "candidate_id": f"c{i}",
                "label": 1 if i % 2 == 0 else 0,
                "publish_timestamp": 1_700_000_000 + i * 60,
                "candidate_source": "reaction" if i % 3 else "quote",
                "features": {"signal": 5.0 if i % 2 == 0 else -5.0},
            }
            for i in range(60)
        ]

        fake_session = _FakeMlflowSession()
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(train_ranker.mlflow_tracking, "start_training_session", return_value=fake_session), \
             patch.dict(train_ranker.os.environ, {"MLFLOW_REGISTER_ON_TRAIN": "1"}, clear=False):
            artifact_dir = Path(tmp)
            rc = train_ranker.train_from_rows(rows, "gaming", artifact_dir)
            self.assertEqual(rc, 0)
            artifact = json.loads((artifact_dir / "gaming_ranker.json").read_text(encoding="utf-8"))

        self.assertEqual(artifact["mlflow"]["run_id"], "run-123")
        self.assertEqual(artifact["mlflow_run_id"], "run-123")
        self.assertEqual(artifact["mlflow"]["registered_version"], "7")
        self.assertEqual(fake_session.register_calls, 1)
        metric_keys = {key for payload, _prefix in fake_session.logged_metrics for key in payload}
        self.assertIn("ndcg_at_5", metric_keys)
        self.assertIn("precision_at_5", metric_keys)

    def test_sample_weight_rewards_stronger_outcome_feedback(self) -> None:
        weak = {
            "supervision_tier": "outcome_feedback",
            "label_class": "outcome_positive",
            "retention_3s": 0.4,
            "completion_rate": 0.2,
            "baseline_view_rate": 100.0,
            "observed_view_rate": 95.0,
        }
        strong = {
            "supervision_tier": "outcome_feedback",
            "label_class": "outcome_positive",
            "retention_3s": 0.88,
            "completion_rate": 0.62,
            "baseline_view_rate": 100.0,
            "observed_view_rate": 145.0,
            "publish_platform": "tiktok",
        }

        self.assertGreater(train_ranker._sample_weight(strong), train_ranker._sample_weight(weak))

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

    def test_fit_lightgbm_ranker_uses_early_stopping_and_eval_windows(self) -> None:
        rows = [
            {"vod_id": "vod-1", "candidate_id": "a1", "label": 1, "ordinal_label": 4, "features": {"signal": 4.0}},
            {"vod_id": "vod-1", "candidate_id": "a2", "label": 0, "ordinal_label": 0, "features": {"signal": -2.0}},
            {"vod_id": "vod-2", "candidate_id": "b1", "label": 1, "ordinal_label": 3, "features": {"signal": 3.0}},
            {"vod_id": "vod-2", "candidate_id": "b2", "label": 0, "ordinal_label": 0, "features": {"signal": -3.0}},
        ]
        scales = train_ranker._compute_scales(rows, ["signal"])
        original = train_ranker.LGBMRanker
        original_early_stopping = train_ranker.early_stopping
        train_ranker.LGBMRanker = _FakeLGBMRanker
        train_ranker.early_stopping = lambda rounds, verbose=False: ("early_stopping", rounds, verbose)
        _FakeLGBMRanker.last_fit_kwargs = None
        try:
            candidate = train_ranker._fit_lightgbm_ranker(
                rows,
                rows,
                ["signal"],
                scales,
                "vod_id",
            )
        finally:
            train_ranker.LGBMRanker = original
            train_ranker.early_stopping = original_early_stopping

        self.assertIsNotNone(candidate)
        assert _FakeLGBMRanker.last_fit_kwargs is not None
        self.assertEqual(_FakeLGBMRanker.last_fit_kwargs["eval_at"], [3, 5, 10])
        self.assertEqual(_FakeLGBMRanker.last_fit_kwargs["callbacks"], [("early_stopping", 10, False)])

    def test_train_from_rows_loads_hpo_artifact_for_lightgbm_candidate(self) -> None:
        rows = [
            {
                "vod_id": f"vod-{i // 2}",
                "candidate_id": f"c{i}",
                "label": 1 if i % 2 == 0 else 0,
                "features": {"signal": 6.0 if i % 2 == 0 else -6.0},
            }
            for i in range(60)
        ]
        original = train_ranker.LGBMRanker
        train_ranker.LGBMRanker = _FakeLGBMRanker
        _FakeLGBMRanker.last_init_kwargs = None
        try:
            with tempfile.TemporaryDirectory() as tmp:
                artifact_dir = Path(tmp)
                (artifact_dir / "gaming_ranker_hpo.json").write_text(
                    json.dumps(
                        {
                            "version": 1,
                            "profile": "gaming",
                            "best_value": 0.812,
                            "best_params": {
                                "num_leaves": 31,
                                "learning_rate": 0.05,
                                "min_child_samples": 7,
                                "feature_fraction": 0.8,
                                "bagging_fraction": 0.75,
                                "n_estimators": 120,
                            },
                            "seed": 11,
                            "trials": 8,
                        },
                        ensure_ascii=True,
                    ),
                    encoding="utf-8",
                )
                rc = train_ranker.train_from_rows(rows, "gaming", artifact_dir)
                self.assertEqual(rc, 0)
                artifact = json.loads((artifact_dir / "gaming_ranker.json").read_text(encoding="utf-8"))
        finally:
            train_ranker.LGBMRanker = original

        self.assertIsNotNone(_FakeLGBMRanker.last_init_kwargs)
        assert _FakeLGBMRanker.last_init_kwargs is not None
        self.assertEqual(_FakeLGBMRanker.last_init_kwargs["num_leaves"], 31)
        self.assertEqual(_FakeLGBMRanker.last_init_kwargs["n_estimators"], 120)
        self.assertTrue(artifact["hpo"]["loaded"])
        self.assertEqual(artifact["hpo"]["seed"], 11)
        self.assertIn("blend_calibration", artifact)
        self.assertIn("serving_threshold", artifact)
        self.assertEqual(artifact["serving_threshold_calibration"]["source"], "validation")


if __name__ == "__main__":
    unittest.main()
