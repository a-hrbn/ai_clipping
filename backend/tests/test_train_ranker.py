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

    def test_prepare_ranker_training_rows_refills_source_cap_to_advanced_model_minimum(self) -> None:
        rows = [
            {
                "job_id": "vod",
                "candidate_id": f"boundary-{idx}",
                "candidate_source": "boundary_perturbation",
                "label": 0,
                "features": {"signal": -1.0},
            }
            for idx in range(590)
        ]
        diverse_sources = [
            "audio_burst",
            "chat_sync",
            "clean_reaction",
            "controversy",
            "dense_window",
            "hot_take",
            "hype_peak",
            "narrative_reveal",
            "quote",
            "tension_build",
        ]
        for idx in range(235):
            rows.append(
                {
                    "job_id": "vod",
                    "candidate_id": f"diverse-{idx}",
                    "candidate_source": diverse_sources[idx % len(diverse_sources)],
                    "label": 1 if idx % 5 == 0 else 0,
                    "features": {"signal": 1.0 if idx % 5 == 0 else -0.5},
                }
            )
        rows.extend(
            {
                "job_id": "vod",
                "candidate_id": f"proxy-{idx}",
                "candidate_source": "twitch_label_proxy",
                "supervision_tier": "twitch_clip_proxy",
                "label": 1,
                "features": {"signal": 0.5},
            }
            for idx in range(10)
        )

        prepared, policy = train_ranker._prepare_ranker_training_rows(rows)

        cap = policy["candidate_source_cap"]
        self.assertEqual(policy["input_rows"], 835)
        self.assertEqual(policy["after_twitch_proxy_exclusion"], 825)
        self.assertEqual(policy["excluded_twitch_clip_proxy_rows"], 10)
        self.assertEqual(len(prepared), 442)
        self.assertEqual(cap["target_min_rows"], train_ranker._ADVANCED_MODEL_PROMOTION_MIN_ROWS)
        self.assertEqual(cap["max_rows_per_source"], 207)
        self.assertEqual(cap["adaptive_refill_rows"], 0)
        self.assertEqual(cap["refilled_by_source"], {})
        self.assertEqual(cap["by_source_after"]["boundary_perturbation"], 207)
        self.assertEqual(cap["removed_by_source"], {"boundary_perturbation": 383})
        self.assertFalse(any(row.get("supervision_tier") == "twitch_clip_proxy" for row in prepared))

    def test_prepare_ranker_training_rows_uses_dynamic_cap_when_background_negatives_exist(self) -> None:
        rows = [
            {
                "job_id": "vod",
                "candidate_id": f"boundary-{idx}",
                "candidate_source": "boundary_perturbation",
                "label": 0,
                "features": {"signal": -1.0},
            }
            for idx in range(700)
        ]
        rows.extend(
            {
                "job_id": "vod",
                "candidate_id": f"background-{idx}",
                "candidate_source": "background_negative",
                "supervision_tier": "background_negative",
                "label_class": "background_negative",
                "label": 0,
                "features": {"signal": -0.7},
            }
            for idx in range(700)
        )
        diverse_sources = [
            "audio_burst",
            "chat_sync",
            "clean_reaction",
            "controversy",
            "dense_window",
            "hot_take",
            "hype_peak",
            "narrative_reveal",
            "quote",
            "reaction",
            "silence_break",
            "story_payoff",
            "tension_build",
            "visual_reveal",
        ]
        for idx in range(280):
            source = diverse_sources[idx % len(diverse_sources)]
            rows.append(
                {
                    "job_id": "vod",
                    "candidate_id": f"diverse-{idx}",
                    "candidate_source": source,
                    "supervision_tier": "twitch_top_viewed_weak_positive" if idx % 7 == 0 else "",
                    "dataset_source": "outcome_top_viewed_train" if idx % 7 == 0 else "twitch_vod_background_negative",
                    "label_class": "weak_positive" if idx % 7 == 0 else "background_negative",
                    "label": 1 if idx % 7 == 0 else 0,
                    "features": {"signal": 1.0 if idx % 7 == 0 else -0.4},
                }
            )

        prepared, policy = train_ranker._prepare_ranker_training_rows(rows)

        cap = policy["candidate_source_cap"]
        self.assertLess(cap["max_share"], train_ranker._MAX_CANDIDATE_SOURCE_SHARE)
        self.assertEqual(cap["target_min_rows"], 1000)
        self.assertGreaterEqual(len(prepared), 1000)
        self.assertIn("boundary_perturbation", cap["refill_excluded_sources"])
        self.assertLessEqual(cap["by_source_after"]["boundary_perturbation"], cap["max_rows_per_source"])
        self.assertGreater(cap["by_source_after"]["background_negative"], cap["by_source_after"]["boundary_perturbation"])

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

    def test_calibrate_serving_threshold_prefers_recall_fallback_when_precision_target_is_unreachable(self) -> None:
        labels = train_ranker.np.array([1, 0, 1, 0], dtype=int)
        scores = train_ranker.np.array([-0.2, -0.1, 0.3, -0.4], dtype=float)

        calibration = train_ranker._calibrate_serving_threshold(labels, scores)

        self.assertEqual(calibration["source"], "validation")
        self.assertEqual(calibration["metric"], "max_recall_under_precision_target")
        self.assertEqual(calibration["threshold_policy"], "balanced_filter")
        self.assertEqual(calibration["selected_threshold_reason"], "best_filter_gate_under_hard_selected_rate")
        self.assertGreaterEqual(calibration["metrics"]["recall"], 0.7)
        self.assertGreaterEqual(calibration["metrics"]["f1"], 0.55)

    def test_calibrate_serving_threshold_maximizes_recall_under_precision_target(self) -> None:
        labels = train_ranker.np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=int)
        scores = train_ranker.np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.65, 0.1, 0.0, -0.1, -0.2], dtype=float)

        calibration = train_ranker._calibrate_serving_threshold(labels, scores)

        self.assertEqual(calibration["metric"], "max_recall_under_precision_target")
        self.assertEqual(calibration["selected_threshold_reason"], "max_recall_under_precision_target")
        self.assertEqual(calibration["target_precision"], 0.55)
        self.assertEqual(calibration["threshold"], 0.6)
        self.assertGreaterEqual(calibration["metrics"]["precision"], 0.55)
        self.assertEqual(calibration["metrics"]["recall"], 0.8)
        self.assertLessEqual(calibration["metrics"]["selected_rate"], 0.5)

    def test_calibrate_serving_threshold_prioritizes_recall_before_f_score_when_constraints_pass(self) -> None:
        labels = train_ranker.np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=int)
        scores = train_ranker.np.array([0.9, 0.8, 0.7, 0.1, 0.15, -0.1, -0.2, -0.3, -0.4, -0.5], dtype=float)

        calibration = train_ranker._calibrate_serving_threshold(labels, scores)

        self.assertEqual(calibration["selected_threshold_reason"], "max_recall_under_precision_target")
        self.assertEqual(calibration["threshold"], 0.0)
        self.assertEqual(calibration["metrics"]["recall"], 1.0)
        self.assertGreaterEqual(calibration["metrics"]["precision"], 0.8)
        self.assertLessEqual(calibration["metrics"]["selected_rate"], 0.6)

    def test_calibrate_serving_threshold_falls_back_to_f_score_when_precision_target_costs_recall(self) -> None:
        labels = train_ranker.np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=int)
        scores = train_ranker.np.array([0.9, 0.8, 0.7, 0.6, 0.1, 0.85, 0.75, -0.1, -0.2, -0.3], dtype=float)

        calibration = train_ranker._calibrate_serving_threshold(labels, scores)

        self.assertEqual(calibration["selected_threshold_reason"], "f_score_fallback_under_gate_precision")
        self.assertGreaterEqual(calibration["metrics"]["recall"], 0.7)
        self.assertGreaterEqual(calibration["metrics"]["f1"], 0.55)
        self.assertGreaterEqual(calibration["metrics"]["precision"], 0.5)
        self.assertLessEqual(calibration["metrics"]["selected_rate"], 0.6)

    def test_calibrate_serving_threshold_respects_hard_selected_rate_cap(self) -> None:
        labels = train_ranker.np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=int)
        scores = train_ranker.np.array([0.9, 0.8, 0.7, -0.3, 0.6, 0.5, 0.4, 0.3], dtype=float)

        calibration = train_ranker._calibrate_serving_threshold(labels, scores)

        self.assertLessEqual(calibration["metrics"]["selected_rate"], train_ranker._FILTER_GATE_MAX_SELECTED_RATE)
        self.assertNotEqual(calibration["selected_threshold_reason"], "f_score_fallback_above_selected_rate")

    def test_filter_gate_from_calibration_records_threshold_decision(self) -> None:
        calibration = {
            "source": "validation",
            "metric": "max_recall_under_precision_target",
            "threshold_policy": "balanced_filter",
            "selected_threshold_reason": "max_recall_under_precision_target",
            "target_precision": 0.8,
            "threshold": -0.1,
            "metrics": {
                "precision": 0.52,
                "recall": 0.7,
                "f1": 0.6,
                "accuracy": 0.56,
                "selected_rate": 0.68,
                "selected_count": 34,
                "rows": 50,
            },
        }

        gate = train_ranker._filter_gate_from_calibration(calibration)

        self.assertTrue(gate["passed"])
        self.assertEqual(gate["threshold"], -0.1)
        self.assertEqual(gate["selected_threshold_reason"], "max_recall_under_precision_target")
        self.assertEqual(gate["thresholds"]["target_precision"], 0.8)
        self.assertEqual(gate["selected_rate"], 0.68)
        self.assertTrue(gate["accuracy_gate_is_diagnostic"])

    def test_filter_gate_treats_accuracy_as_diagnostic(self) -> None:
        gate = train_ranker._filter_gate_from_calibration(
            {
                "source": "validation",
                "metric": "max_recall_under_precision_target",
                "threshold_policy": "balanced_filter",
                "selected_threshold_reason": "best_filter_gate_under_hard_selected_rate",
                "threshold": 0.2,
                "metrics": {
                    "precision": 0.55,
                    "recall": 0.7,
                    "f1": 0.62,
                    "accuracy": 0.42,
                    "selected_rate": 0.72,
                    "selected_count": 36,
                    "rows": 50,
                },
            }
        )

        self.assertTrue(gate["passed"])
        self.assertEqual(gate["accuracy"], 0.42)
        self.assertNotIn("filter_accuracy_floor_below_minimum", gate["failures"])

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

    def test_numeric_feature_names_from_rows_uses_union_not_first_row(self) -> None:
        names = train_ranker._numeric_feature_names_from_rows([
            {"hook_strength": 0.8},
            {
                "chat_ema_crossing_score": 0.4,
                "streamer_shout_score": 0.7,
                "vl_prompt_max_score": 0.6,
                "boundary_variant_score": 0.5,
                "source": "reaction",
            },
        ])

        self.assertIn("hook_strength", names)
        self.assertIn("chat_ema_crossing_score", names)
        self.assertIn("streamer_shout_score", names)
        self.assertIn("vl_prompt_max_score", names)
        self.assertIn("boundary_variant_score", names)
        self.assertNotIn("source", names)

    def test_select_model_requires_balanced_ranking_and_filter_gate(self) -> None:
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
                    "f1": 0.7,
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
                    "f1": 0.7,
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

    def test_numeric_feature_names_keeps_runtime_capability_flags(self) -> None:
        names = train_ranker._numeric_feature_names(
            {
                "has_chat_features": True,
                "has_llm_semantics": False,
                "has_vector_similarity": True,
                "feature_build_version": 6,
            }
        )

        self.assertEqual(names, ["has_chat_features", "has_llm_semantics", "has_vector_similarity"])

    def test_numeric_feature_names_includes_judge_teacher_features(self) -> None:
        names = train_ranker._numeric_feature_names(
            {
                "judge_score": 8.4,
                "judge_confidence": 0.7,
                "judge_scroll_stop_power": 0.9,
                "has_judge_score": True,
                "judge_llm_used": True,
            }
        )

        self.assertIn("judge_score", names)
        self.assertIn("judge_confidence", names)
        self.assertIn("judge_scroll_stop_power", names)
        self.assertIn("has_judge_score", names)
        self.assertIn("judge_llm_used", names)

    def test_feature_selection_caps_training_vector_by_supervised_signal(self) -> None:
        rows = [
            {"label": 1, "features": {"strong": 10.0, "weak": 0.2, "noise": 0.0}},
            {"label": 1, "features": {"strong": 9.0, "weak": 0.1, "noise": 1.0}},
            {"label": 0, "features": {"strong": 0.0, "weak": 0.0, "noise": 0.0}},
            {"label": 0, "features": {"strong": 1.0, "weak": 0.0, "noise": 1.0}},
        ]

        selected, report = train_ranker._select_training_feature_names(
            rows,
            ["noise", "strong", "weak"],
            max_features=2,
        )

        self.assertEqual(len(selected), 2)
        self.assertIn("strong", selected)
        self.assertTrue(report["enabled"])
        self.assertEqual(report["dropped_feature_count"], 1)

    def test_feature_selection_prioritizes_active_judge_teacher_features(self) -> None:
        rows = [
            {"label": 1, "features": {"strong": 10.0, "another": 9.0, "judge_score": 0.7}},
            {"label": 1, "features": {"strong": 9.0, "another": 8.0, "judge_score": 0.6}},
            {"label": 0, "features": {"strong": 0.0, "another": 0.0, "judge_score": 0.5}},
            {"label": 0, "features": {"strong": 1.0, "another": 1.0, "judge_score": 0.4}},
        ]

        selected, report = train_ranker._select_training_feature_names(
            rows,
            ["another", "judge_score", "strong"],
            max_features=2,
        )

        self.assertEqual(len(selected), 2)
        self.assertIn("judge_score", selected)
        self.assertIn("judge_score", report["priority_feature_names"])
        self.assertEqual(report["method"], "supervised_univariate_effect_size_with_judge_teacher_priority")

    def test_apply_sign_constraints_clips_positive_signal_negative_weights(self) -> None:
        weights = train_ranker._apply_sign_constraints(
            {
                "hook_strength": -0.4,
                "payoff_score": 0.2,
                "context_cost": -0.1,
            }
        )

        self.assertEqual(weights["hook_strength"], 0.0)
        self.assertEqual(weights["payoff_score"], 0.2)
        self.assertEqual(weights["context_cost"], -0.1)

    def test_select_model_returns_fallback_when_no_candidate_passes_filter_gate(self) -> None:
        selected = train_ranker._select_model(
            [
                {
                    "name": "rank_only",
                    "artifact_model_type": "linear",
                    "metrics": {
                        "ndcg_at_5": 0.95,
                        "precision_at_5": 0.9,
                        "precision": 0.42,
                        "recall": 1.0,
                        "f1": 0.5,
                        "accuracy": 0.44,
                        "selected_rate": 1.0,
                    },
                }
            ],
            training_row_count=500,
        )

        self.assertEqual(selected["name"], "fallback_heuristic")
        self.assertEqual(selected["selection_blocked_reason"], "no_candidate_passed_filter_gate")

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

    def test_sample_weight_keeps_top_viewed_weak_positive_light(self) -> None:
        row = {
            "supervision_tier": "twitch_top_viewed_weak_positive",
            "label_class": "weak_positive",
            "sample_weight": 0.55,
        }

        self.assertAlmostEqual(train_ranker._sample_weight(row), 0.55)

    def test_pairwise_logistic_can_promote_below_advanced_model_row_floor(self) -> None:
        selected = train_ranker._select_model(
            [
                {
                    "name": "mean_difference_linear",
                    "artifact_model_type": "linear",
                    "metrics": {
                        "precision": 0.6,
                        "recall": 0.7,
                        "f1": 0.65,
                        "selected_rate": 0.5,
                        "ndcg_at_5": 0.7,
                        "precision_at_5": 0.7,
                    },
                },
                {
                    "name": "pairwise_logistic_regression",
                    "artifact_model_type": "pairwise_logistic",
                    "score_mode": "pairwise_median",
                    "metrics": {
                        "precision": 0.92,
                        "recall": 0.52,
                        "f1": 0.66,
                        "selected_rate": 0.35,
                        "ndcg_at_5": 0.9,
                        "precision_at_5": 0.9,
                    },
                },
            ],
            training_row_count=train_ranker._ADVANCED_MODEL_PROMOTION_MIN_ROWS - 1,
        )

        self.assertEqual(selected["name"], "pairwise_logistic_regression")

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

    def test_select_model_prefers_lightgbm_when_close_with_better_recall(self) -> None:
        with patch.dict("os.environ", {"RANKER_PREFER_LIGHTGBM": "1", "RANKER_LIGHTGBM_SELECTION_TOLERANCE": "0.03"}):
            selected = train_ranker._select_model(
                [
                    {
                        "name": "logistic_regression",
                        "artifact_model_type": "logistic",
                        "metrics": {
                            "precision": 0.91,
                            "recall": 0.89,
                            "f1": 0.932,
                            "selected_rate": 0.5,
                            "ndcg_at_5": 0.86,
                            "precision_at_5": 0.88,
                            "accuracy": 0.91,
                        },
                    },
                    {
                        "name": "lightgbm_ranker",
                        "artifact_model_type": "lightgbm_ranker",
                        "metrics": {
                            "precision": 0.9,
                            "recall": 0.93,
                            "f1": 0.928,
                            "selected_rate": 0.52,
                            "ndcg_at_5": 0.86,
                            "precision_at_5": 0.87,
                            "accuracy": 0.9,
                        },
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
            self.assertEqual(artifact["feature_schema_version"], 6)
            self.assertIn("semantic_score_config", artifact)
            self.assertIn("top_feature_importances", artifact)
            self.assertIn("available_numeric_feature_names", artifact)
            self.assertIn("ignored_numeric_feature_names", artifact)
            self.assertGreaterEqual(len(artifact["available_numeric_feature_names"]), len(artifact["feature_names"]))
            self.assertIn("slow_open_penalty", artifact["excluded_feature_names"])
            self.assertIn("model_selection", artifact)
            self.assertIn("filter_gate", artifact)
            self.assertEqual(artifact_health["selected_model"], artifact["selected_model"])
            self.assertEqual(artifact_health["semantic_score_config"], artifact["semantic_score_config"])
            self.assertIn("gates", artifact_health)
            self.assertEqual(artifact_health["filter_gate"], artifact["filter_gate"])
            self.assertIn("validation_score_distribution", artifact)
            self.assertIn("training_score_distribution", artifact)
            self.assertIn("training_score_distribution", artifact_health)
            self.assertIn("validation_score_distribution", artifact_health)
            self.assertIn("score_distribution_drift", artifact_health)
            self.assertIn("feature_names_count", artifact_health)
            self.assertIn("ignored_numeric_feature_names", artifact_health)
            self.assertTrue(artifact_health["score_distribution_drift"]["ks_test_available"])
            self.assertIn("ks_statistic", artifact_health["score_distribution_drift"])
            self.assertIn("psi", artifact_health["score_distribution_drift"])
            self.assertIn("warnings", artifact_health["score_distribution_drift"])
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
