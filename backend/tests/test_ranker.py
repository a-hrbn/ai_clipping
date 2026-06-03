import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import joblib
except ModuleNotFoundError:  # pragma: no cover - optional ML deps in test env
    joblib = None

try:
    from sklearn.ensemble import GradientBoostingClassifier
except ModuleNotFoundError:  # pragma: no cover - optional ML deps in test env
    GradientBoostingClassifier = None

import ml as ml_package
from services import semantic_service
from services import ranker as ranker_module, selector
from services.ranker import Ranker
from services.timeline_types import Candidate, CandidateFeatures


class _FakeLightGBMModel:
    def predict(self, matrix):
        return [float(row[0]) * 2.0 for row in matrix]


class RankerTests(unittest.TestCase):
    def test_ranker_semantic_score_uses_full_zero_to_ten_scale(self) -> None:
        feature = CandidateFeatures(
            candidate_id="semantic",
            quoteability_score=1.0,
            surprise_score=1.0,
            contradiction_score=1.0,
            standalone_score=1.0,
            semantic_density=1.0,
            source="quote",
        )

        score = Ranker(enable_trained_model=False)._heuristic_scores([feature], profile="general")[0][0]

        self.assertAlmostEqual(score, 0.8, places=3)

    def test_ranker_does_not_apply_slow_open_penalty_in_serving_heuristic(self) -> None:
        ranker = Ranker(enable_trained_model=False)
        base = CandidateFeatures(candidate_id="base", payoff_score=5.0, source="reaction")
        slow = CandidateFeatures(
            candidate_id="slow",
            payoff_score=5.0,
            slow_open_penalty=1.0,
            source="reaction",
        )

        base_score, _base_rationale = ranker._heuristic_scores([base], profile="general")[0]
        slow_score, slow_rationale = ranker._heuristic_scores([slow], profile="general")[0]

        self.assertEqual(slow_score, base_score)
        self.assertNotIn("slow_open_risk", slow_rationale)

    def test_ranker_caps_mid_thought_context_dependent_clips(self) -> None:
        feature = CandidateFeatures(
            candidate_id="mid-thought",
            hook_strength=0.9,
            max_hype=9.0,
            arc_score=8.0,
            chat_peak=0.8,
            motion_peak=0.8,
            face_reaction_peak=0.8,
            payoff_score=8.0,
            quoteability_score=0.8,
            surprise_score=0.8,
            standalone_score=0.1,
            semantic_density=0.8,
            context_dependency_score=0.85,
            source="reaction",
        )

        score, rationale = Ranker(enable_trained_model=False)._heuristic_scores([feature], profile="gaming")[0]

        self.assertLessEqual(score, 3.0)
        self.assertIn("standalone_hard_filter", rationale)

    def test_ranker_applies_hook_reranker_stage_after_scoring(self) -> None:
        feature = CandidateFeatures(
            candidate_id="hook",
            hook_strength=0.4,
            payoff_score=3.0,
            max_hype=3.0,
            standalone_score=0.6,
            has_hook_rewrite=True,
            hook_rewrite_score=0.85,
            hook_curiosity_gap_score=0.7,
            source="reaction",
        )

        detail = Ranker(enable_trained_model=False).score_details([feature], profile="general")[0]

        self.assertGreater(detail["score"], detail["heuristic_score"])
        self.assertIn("hook_reranker_boost", detail["rationale"])

    def test_ranker_applies_boundary_variant_set_stage_after_scoring(self) -> None:
        feature = CandidateFeatures(
            candidate_id="boundary",
            hook_strength=0.4,
            payoff_score=3.0,
            max_hype=3.0,
            standalone_score=0.6,
            boundary_variant_score=0.9,
            boundary_variant_margin_score=0.4,
            boundary_variant_count=12,
            source="reaction",
        )

        detail = Ranker(enable_trained_model=False).score_details([feature], profile="general")[0]

        self.assertGreater(detail["score"], detail["heuristic_score"])
        self.assertEqual(detail["boundary_variant_set_score"], 0.9)
        self.assertIn("boundary_variant_set_max", detail["rationale"])

    def test_ranker_rewards_chat_velocity_and_vision_language(self) -> None:
        feature = CandidateFeatures(
            candidate_id="vl-chat",
            hook_strength=0.4,
            payoff_score=3.0,
            max_hype=3.0,
            standalone_score=0.6,
            chat_ema_crossing_score=0.8,
            chat_rate_derivative_score=0.7,
            vl_prompt_max_score=0.75,
            source="reaction",
        )

        _score, rationale = Ranker(enable_trained_model=False)._heuristic_scores([feature], profile="general")[0]

        self.assertIn("chat_velocity_spike", rationale)
        self.assertIn("vision_language_match", rationale)

    def test_ranker_context_primary_stage_beats_context_poor_hype(self) -> None:
        hype_only = CandidateFeatures(
            candidate_id="hype-only",
            max_hype=8.0,
            mean_hype=7.0,
            chat_peak=0.2,
            source="dense_inventory",
        )
        contextual = CandidateFeatures(
            candidate_id="contextual",
            max_hype=1.0,
            mean_hype=1.0,
            hook_strength=0.35,
            payoff_score=2.0,
            standalone_score=0.7,
            setup_present_score=0.82,
            payoff_present_score=0.86,
            context_strength_score=0.78,
            context_completeness_score=0.81,
            context_pattern_confidence=0.75,
            context_pattern_label="fanmail_unboxing",
            source="quote",
        )

        details = Ranker(enable_trained_model=False).score_details([hype_only, contextual], profile="general")
        by_id = {item["candidate_id"]: item for item in details}

        self.assertGreater(by_id["contextual"]["score"], by_id["hype-only"]["score"])
        self.assertIn("context_primary_boost", by_id["contextual"]["rationale"])
        self.assertIn("context_missing_penalty", by_id["hype-only"]["rationale"])

    def test_selector_prefers_stronger_candidate(self) -> None:
        candidates = [
            Candidate(id="strong", start=0.0, end=12.0, source="reaction", anchor_time=2.0),
            Candidate(id="weak", start=20.0, end=32.0, source="quote", anchor_time=25.0),
        ]
        features = [
            CandidateFeatures(
                candidate_id="strong",
                duration=12.0,
                hook_delay_sec=1.0,
                hook_strength=0.9,
                max_hype=6.5,
                mean_hype=4.0,
                arc_score=4.0,
                chat_peak=0.7,
                motion_peak=0.6,
                face_reaction_peak=0.7,
                speaker_turns=2,
                exclamation_count=1,
                laughter_count=1,
                question_open=True,
                context_cost=0.1,
                payoff_score=6.0,
                context_efficiency=0.9,
                quoteability_score=0.6,
                surprise_score=0.8,
                contradiction_score=0.4,
                standalone_score=0.8,
                semantic_density=0.5,
                profile_fit_score=0.8,
                source="reaction",
            ),
            CandidateFeatures(
                candidate_id="weak",
                duration=12.0,
                hook_delay_sec=3.0,
                hook_strength=0.1,
                max_hype=2.0,
                mean_hype=1.0,
                arc_score=1.0,
                chat_peak=0.0,
                motion_peak=0.0,
                face_reaction_peak=0.0,
                speaker_turns=0,
                exclamation_count=0,
                laughter_count=0,
                question_open=False,
                context_cost=0.8,
                payoff_score=1.0,
                context_efficiency=0.2,
                quoteability_score=0.2,
                surprise_score=0.1,
                contradiction_score=0.0,
                standalone_score=0.2,
                semantic_density=0.1,
                profile_fit_score=0.2,
                source="quote",
            ),
        ]

        ranked = selector.select_top_candidates(candidates, features, top_k=2)

        self.assertEqual(ranked[0].candidate.id, "strong")
        self.assertGreater(ranked[0].rank_score, ranked[1].rank_score)

    def test_selector_spreads_early_results_across_time_when_possible(self) -> None:
        candidates = [
            Candidate(id="a", start=0.0, end=12.0, source="reaction", anchor_time=2.0),
            Candidate(id="b", start=10.0, end=22.0, source="reaction", anchor_time=13.0),
            Candidate(id="c", start=60.0, end=72.0, source="quote", anchor_time=63.0),
        ]
        features = [
            CandidateFeatures(candidate_id="a", hook_strength=0.9, max_hype=7.0, payoff_score=7.0, source="reaction"),
            CandidateFeatures(candidate_id="b", hook_strength=0.85, max_hype=6.8, payoff_score=6.9, source="reaction"),
            CandidateFeatures(candidate_id="c", hook_strength=0.7, max_hype=5.5, payoff_score=5.5, source="quote"),
        ]

        ranked = selector.select_top_candidates(candidates, features, top_k=2)

        self.assertEqual([item.candidate.id for item in ranked], ["a", "c"])

    def test_ranker_uses_trained_artifact_when_available(self) -> None:
        feature = CandidateFeatures(
            candidate_id="ml",
            duration=12.0,
            hook_delay_sec=1.0,
            hook_strength=0.2,
            max_hype=1.0,
            mean_hype=1.0,
            arc_score=1.0,
            chat_peak=0.0,
            motion_peak=0.0,
            face_reaction_peak=0.0,
            speaker_turns=0,
            exclamation_count=0,
            laughter_count=0,
            question_open=False,
            context_cost=0.0,
            payoff_score=1.0,
            context_efficiency=0.5,
            quoteability_score=1.0,
            surprise_score=0.0,
            contradiction_score=0.0,
            standalone_score=1.0,
            semantic_density=1.0,
            profile_fit_score=1.0,
            source="quote",
        )
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            (artifact_dir / "gaming_ranker.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "profile": "gaming",
                        "feature_names": ["quoteability_score", "standalone_score", "semantic_density"],
                        "weights": {
                            "quoteability_score": 5.0,
                            "standalone_score": 5.0,
                            "semantic_density": 5.0,
                        },
                        "scales": {
                            "quoteability_score": {"mean": 0.0, "std": 1.0},
                            "standalone_score": {"mean": 0.0, "std": 1.0},
                            "semantic_density": {"mean": 0.0, "std": 1.0},
                        },
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
                    }
                ),
                encoding="utf-8",
            )
            score = Ranker(artifact_dir=artifact_dir, enable_trained_model=True).score([feature], profile="gaming")[0][1]

        self.assertGreater(score, 1.0)

    def test_score_details_exposes_serving_threshold_decision(self) -> None:
        feature = CandidateFeatures(
            candidate_id="ml",
            quoteability_score=1.0,
            source="quote",
        )
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            (artifact_dir / "gaming_ranker.json").write_text(
                json.dumps(
                    {
                        "version": 3,
                        "profile": "gaming",
                        "model_type": "linear",
                        "feature_names": ["quoteability_score"],
                        "weights": {"quoteability_score": 5.0},
                        "scales": {"quoteability_score": {"mean": 0.0, "std": 1.0}},
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
                        "serving_threshold": 2.0,
                    }
                ),
                encoding="utf-8",
            )
            details = Ranker(artifact_dir=artifact_dir, enable_trained_model=True).score_details([feature], profile="gaming")

        self.assertEqual(details[0]["serving_threshold"], 2.0)
        self.assertTrue(details[0]["passes_serving_threshold"])

    def test_score_details_uses_bucket_threshold_and_fallback(self) -> None:
        reaction = CandidateFeatures(candidate_id="reaction", hook_strength=1.0, source="reaction")
        general = CandidateFeatures(candidate_id="general", hook_strength=1.0, source="unknown")
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            (artifact_dir / "gaming_ranker.json").write_text(
                json.dumps(
                    {
                        "version": 3,
                        "profile": "gaming",
                        "model_type": "linear",
                        "feature_names": ["hook_strength", "content_bucket__reaction"],
                        "weights": {"hook_strength": 3.0, "content_bucket__reaction": 2.0},
                        "scales": {
                            "hook_strength": {"mean": 0.0, "std": 1.0},
                            "content_bucket__reaction": {"mean": 0.0, "std": 1.0},
                        },
                        "training_stats": {
                            "rows": 100,
                            "positive_rows": 40,
                            "negative_rows": 60,
                            "train_rows": 80,
                            "validation_rows": 20,
                        },
                        "validation": {
                            "accuracy": 0.75,
                            "precision_at_5": 0.8,
                            "ndcg_at_5": 0.8,
                            "ranking_groups": 2,
                            "ranking_metric_min": 0.5,
                            "ranking_metric_p25": 0.6,
                        },
                        "filter_gate": {
                            "passed": True,
                            "bucket_gates": {"reaction": {"passed": True}, "general": {"passed": False}},
                            "serving_threshold_per_bucket": {"reaction": 1.25, "general": 9.0},
                            "selected_model_per_bucket": {"reaction": "trained", "general": "fallback_heuristic"},
                        },
                        "intercept": 0.0,
                        "blend": 0.65,
                        "serving_threshold": 0.0,
                    }
                ),
                encoding="utf-8",
            )
            details = Ranker(artifact_dir=artifact_dir, enable_trained_model=True).score_details(
                [reaction, general],
                profile="gaming",
            )

        by_id = {item["candidate_id"]: item for item in details}
        self.assertEqual(by_id["reaction"]["content_bucket"], "reaction")
        self.assertEqual(by_id["reaction"]["bucket_model"], "trained")
        self.assertEqual(by_id["reaction"]["serving_threshold"], 1.25)
        self.assertTrue(by_id["reaction"]["trained_model_used"])
        self.assertEqual(by_id["general"]["bucket_model"], "fallback_heuristic")
        self.assertFalse(by_id["general"]["trained_model_used"])

    def test_inspect_artifact_reports_rejected_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            (artifact_dir / "gaming_ranker.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "profile": "gaming",
                        "feature_names": ["quoteability_score"],
                        "weights": {"quoteability_score": 5.0},
                        "scales": {"quoteability_score": {"mean": 0.0, "std": 1.0}},
                        "training_stats": {
                            "rows": 100,
                            "positive_rows": 40,
                            "negative_rows": 60,
                            "train_rows": 80,
                            "validation_rows": 20,
                        },
                        "validation": {
                            "accuracy": 0.8,
                            "precision_at_5": 0.8,
                            "ndcg_at_5": 0.8,
                            "ranking_groups": 1,
                        },
                        "intercept": 0.0,
                        "blend": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            inspection = Ranker(artifact_dir=artifact_dir, enable_trained_model=True).inspect_artifact("gaming")

        self.assertEqual(inspection["status"], "rejected")
        self.assertEqual(inspection["reason"], "validation_ranking_groups_below_minimum")

    def test_inspect_artifact_accepts_filter_gate_despite_low_accuracy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            (artifact_dir / "gaming_ranker.json").write_text(
                json.dumps(
                    {
                        "version": 3,
                        "profile": "gaming",
                        "feature_names": ["quoteability_score"],
                        "weights": {"quoteability_score": 5.0},
                        "scales": {"quoteability_score": {"mean": 0.0, "std": 1.0}},
                        "training_stats": {
                            "rows": 100,
                            "positive_rows": 40,
                            "negative_rows": 60,
                            "train_rows": 80,
                            "validation_rows": 20,
                        },
                        "validation": {
                            "accuracy": 0.42,
                            "precision_at_5": 0.7,
                            "ndcg_at_5": 0.8,
                            "ranking_groups": 2,
                            "ndcg_at_5_min": 0.5,
                            "ndcg_at_5_p25": 0.6,
                        },
                        "filter_gate": {
                            "precision": 0.52,
                            "recall": 0.75,
                            "f1": 0.61,
                            "selected_rate": 0.7,
                        },
                        "intercept": 0.0,
                        "blend": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            inspection = Ranker(artifact_dir=artifact_dir, enable_trained_model=True).inspect_artifact("gaming")

        self.assertEqual(inspection["status"], "usable")

    def test_inspect_artifact_rejects_weak_filter_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            (artifact_dir / "gaming_ranker.json").write_text(
                json.dumps(
                    {
                        "version": 3,
                        "profile": "gaming",
                        "feature_names": ["quoteability_score"],
                        "weights": {"quoteability_score": 5.0},
                        "scales": {"quoteability_score": {"mean": 0.0, "std": 1.0}},
                        "training_stats": {
                            "rows": 100,
                            "positive_rows": 40,
                            "negative_rows": 60,
                            "train_rows": 80,
                            "validation_rows": 20,
                        },
                        "validation": {
                            "accuracy": 0.58,
                            "precision_at_5": 0.7,
                            "ndcg_at_5": 0.8,
                            "ranking_groups": 2,
                            "ndcg_at_5_min": 0.5,
                            "ndcg_at_5_p25": 0.6,
                        },
                        "filter_gate": {
                            "precision": 0.49,
                            "recall": 0.75,
                            "f1": 0.61,
                            "selected_rate": 0.7,
                        },
                        "intercept": 0.0,
                        "blend": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            inspection = Ranker(artifact_dir=artifact_dir, enable_trained_model=True).inspect_artifact("gaming")

        self.assertEqual(inspection["status"], "rejected")
        self.assertEqual(inspection["reason"], "validation_filter_precision_below_minimum")

    def test_inspect_artifact_rejects_fallback_heuristic_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            (artifact_dir / "gaming_ranker.json").write_text(
                json.dumps(
                    {
                        "version": 4,
                        "profile": "gaming",
                        "selected_model": "fallback_heuristic",
                        "semantic_score_config": semantic_service.semantic_score_config(),
                        "feature_names": ["quoteability_score"],
                        "weights": {"quoteability_score": 5.0},
                        "scales": {"quoteability_score": {"mean": 0.0, "std": 1.0}},
                        "training_stats": {
                            "rows": 100,
                            "positive_rows": 40,
                            "negative_rows": 60,
                            "train_rows": 80,
                            "validation_rows": 20,
                        },
                        "validation": {
                            "accuracy": 0.58,
                            "precision_at_5": 0.7,
                            "ndcg_at_5": 0.8,
                            "ranking_groups": 2,
                            "ndcg_at_5_min": 0.5,
                            "ndcg_at_5_p25": 0.6,
                        },
                        "filter_gate": {
                            "precision": 0.7,
                            "recall": 0.75,
                            "f1": 0.72,
                            "selected_rate": 0.4,
                        },
                        "intercept": 0.0,
                        "blend": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            inspection = Ranker(artifact_dir=artifact_dir, enable_trained_model=True).inspect_artifact("gaming")

        self.assertEqual(inspection["status"], "rejected")
        self.assertEqual(inspection["reason"], "fallback_heuristic_selected")

    def test_inspect_artifact_rejects_offline_only_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            (artifact_dir / "gaming_ranker.json").write_text(
                json.dumps(
                    {
                        "version": 3,
                        "profile": "gaming",
                        "model_type": "linear",
                        "feature_names": ["quoteability_score", "neighbor_positive_density"],
                        "weights": {"quoteability_score": 2.0, "neighbor_positive_density": 1.0},
                        "scales": {
                            "quoteability_score": {"mean": 0.0, "std": 1.0},
                            "neighbor_positive_density": {"mean": 0.0, "std": 1.0},
                        },
                        "training_stats": {
                            "rows": 100,
                            "positive_rows": 40,
                            "negative_rows": 60,
                            "train_rows": 80,
                            "validation_rows": 20,
                        },
                        "validation": {"accuracy": 0.8, "ndcg_at_5": 0.8, "precision_at_5": 0.8},
                        "intercept": 0.0,
                        "blend": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            inspection = Ranker(artifact_dir=artifact_dir, enable_trained_model=True).inspect_artifact("gaming")

        self.assertEqual(inspection["status"], "rejected")
        self.assertEqual(inspection["reason"], "artifact_references_offline_only_features")

    def test_inspect_artifact_rejects_semantic_config_mismatch_for_v4_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            (artifact_dir / "gaming_ranker.json").write_text(
                json.dumps(
                    {
                        "version": 4,
                        "profile": "gaming",
                        "model_type": "linear",
                        "feature_names": ["quoteability_score"],
                        "semantic_score_config": {"regex_weight": 0.5, "llm_weight": 0.5},
                        "weights": {"quoteability_score": 2.0},
                        "scales": {"quoteability_score": {"mean": 0.0, "std": 1.0}},
                        "training_stats": {
                            "rows": 100,
                            "positive_rows": 40,
                            "negative_rows": 60,
                            "train_rows": 80,
                            "validation_rows": 20,
                        },
                        "validation": {"accuracy": 0.8, "ndcg_at_5": 0.8, "precision_at_5": 0.8},
                        "intercept": 0.0,
                        "blend": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            inspection = Ranker(artifact_dir=artifact_dir, enable_trained_model=True).inspect_artifact("gaming")

        self.assertEqual(inspection["status"], "rejected")
        self.assertEqual(inspection["reason"], "semantic_score_config_mismatch")

    def test_inspect_artifact_falls_back_to_mlflow_registry_when_local_file_is_missing(self) -> None:
        registry_artifact = {
            "version": 4,
            "profile": "gaming",
            "model_type": "linear",
            "feature_names": ["quoteability_score"],
            "semantic_score_config": ranker_module.semantic_service.semantic_score_config(),
            "weights": {"quoteability_score": 2.0},
            "scales": {"quoteability_score": {"mean": 0.0, "std": 1.0}},
            "training_stats": {
                "rows": 100,
                "positive_rows": 40,
                "negative_rows": 60,
                "train_rows": 80,
                "validation_rows": 20,
            },
            "validation": {
                "accuracy": 0.8,
                "ndcg_at_5": 0.8,
                "precision_at_5": 0.8,
                "ranking_groups": 2,
                "ndcg_at_5_min": 0.5,
                "ndcg_at_5_p25": 0.6,
            },
            "intercept": 0.0,
            "blend": 1.0,
            "_artifact_base_dir": "/tmp/mlflow-registry",
        }

        fake_mlflow_tracking = types.ModuleType("ml.mlflow_tracking")
        fake_mlflow_tracking.load_registered_ranker_artifact = lambda profile, stage="Production": dict(registry_artifact)

        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict(sys.modules, {"ml.mlflow_tracking": fake_mlflow_tracking}), \
             patch.object(ml_package, "mlflow_tracking", fake_mlflow_tracking, create=True):
            artifact_dir = Path(tmp)
            inspection = Ranker(artifact_dir=artifact_dir, enable_trained_model=True).inspect_artifact("gaming")

        self.assertEqual(inspection["status"], "usable")
        self.assertEqual(inspection["source"], "mlflow_registry")

    def test_inspect_artifact_falls_back_to_mlflow_registry_when_local_file_is_rejected(self) -> None:
        registry_artifact = {
            "version": 4,
            "profile": "gaming",
            "model_type": "linear",
            "feature_names": ["quoteability_score"],
            "semantic_score_config": ranker_module.semantic_service.semantic_score_config(),
            "weights": {"quoteability_score": 2.0},
            "scales": {"quoteability_score": {"mean": 0.0, "std": 1.0}},
            "training_stats": {
                "rows": 100,
                "positive_rows": 40,
                "negative_rows": 60,
                "train_rows": 80,
                "validation_rows": 20,
            },
            "validation": {
                "accuracy": 0.8,
                "ndcg_at_5": 0.8,
                "precision_at_5": 0.8,
                "ranking_groups": 2,
                "ndcg_at_5_min": 0.5,
                "ndcg_at_5_p25": 0.6,
            },
            "intercept": 0.0,
            "blend": 1.0,
            "_artifact_base_dir": "/tmp/mlflow-registry",
        }

        fake_mlflow_tracking = types.ModuleType("ml.mlflow_tracking")
        fake_mlflow_tracking.load_registered_ranker_artifact = lambda profile, stage="Production": dict(registry_artifact)

        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict(sys.modules, {"ml.mlflow_tracking": fake_mlflow_tracking}), \
             patch.object(ml_package, "mlflow_tracking", fake_mlflow_tracking, create=True):
            artifact_dir = Path(tmp)
            (artifact_dir / "gaming_ranker.json").write_text(
                json.dumps(
                    {
                        "version": 4,
                        "profile": "gaming",
                        "model_type": "linear",
                        "feature_names": ["quoteability_score"],
                        "semantic_score_config": {"regex_weight": 0.5, "llm_weight": 0.5},
                        "weights": {"quoteability_score": 2.0},
                        "scales": {"quoteability_score": {"mean": 0.0, "std": 1.0}},
                        "training_stats": {
                            "rows": 100,
                            "positive_rows": 40,
                            "negative_rows": 60,
                            "train_rows": 80,
                            "validation_rows": 20,
                        },
                        "validation": {"accuracy": 0.8, "ndcg_at_5": 0.8, "precision_at_5": 0.8},
                        "intercept": 0.0,
                        "blend": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            inspection = Ranker(artifact_dir=artifact_dir, enable_trained_model=True).inspect_artifact("gaming")

        self.assertEqual(inspection["status"], "usable")
        self.assertEqual(inspection["source"], "mlflow_registry")

    @unittest.skipUnless(joblib is not None, "joblib is required for artifact tests")
    def test_inspect_artifact_rejects_unloadable_joblib_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            (artifact_dir / "gaming_ranker.json").write_text(
                json.dumps(
                    {
                        "version": 3,
                        "profile": "gaming",
                        "model_type": "gradient_boosting",
                        "feature_names": ["quoteability_score"],
                        "weights": {},
                        "scales": {"quoteability_score": {"mean": 0.0, "std": 1.0}},
                        "model_path": "missing_model.joblib",
                        "training_stats": {
                            "rows": 100,
                            "positive_rows": 40,
                            "negative_rows": 60,
                            "train_rows": 80,
                            "validation_rows": 20,
                        },
                        "validation": {"accuracy": 0.8, "ndcg_at_5": 0.8, "precision_at_5": 0.8},
                        "blend": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(ranker_module.logger, "warning") as warning_mock:
                inspection = Ranker(artifact_dir=artifact_dir, enable_trained_model=True).inspect_artifact("gaming")

        self.assertEqual(inspection["status"], "rejected")
        self.assertEqual(inspection["reason"], "artifact_model_unloadable")
        warning_mock.assert_called()

    @unittest.skipUnless(joblib is not None, "joblib is required for artifact tests")
    def test_ranker_ignores_undertrained_artifact(self) -> None:
        feature = CandidateFeatures(
            candidate_id="ml",
            duration=12.0,
            hook_delay_sec=1.0,
            hook_strength=0.2,
            max_hype=1.0,
            mean_hype=1.0,
            arc_score=1.0,
            chat_peak=0.0,
            motion_peak=0.0,
            face_reaction_peak=0.0,
            speaker_turns=0,
            exclamation_count=0,
            laughter_count=0,
            question_open=False,
            context_cost=0.0,
            payoff_score=1.0,
            context_efficiency=0.5,
            quoteability_score=1.0,
            surprise_score=0.0,
            contradiction_score=0.0,
            standalone_score=1.0,
            semantic_density=1.0,
            profile_fit_score=1.0,
            source="quote",
        )
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            (artifact_dir / "gaming_ranker.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "profile": "gaming",
                        "feature_names": ["quoteability_score"],
                        "weights": {"quoteability_score": 50.0},
                        "scales": {"quoteability_score": {"mean": 0.0, "std": 1.0}},
                        "training_stats": {
                            "rows": 10,
                            "positive_rows": 5,
                            "negative_rows": 5,
                            "train_rows": 8,
                            "validation_rows": 2,
                        },
                        "validation": {"accuracy": 0.5, "precision": 0.5, "recall": 0.5},
                        "intercept": 0.0,
                        "blend": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            score = Ranker(artifact_dir=artifact_dir, enable_trained_model=True).score([feature], profile="gaming")[0][1]

        self.assertGreater(score, 1.0)

    @unittest.skipUnless(joblib is not None, "joblib is required for artifact tests")
    def test_ranker_rejects_high_ndcg_artifact_when_accuracy_is_low(self) -> None:
        feature = CandidateFeatures(
            candidate_id="ml",
            duration=12.0,
            hook_delay_sec=1.0,
            hook_strength=0.2,
            max_hype=1.0,
            mean_hype=1.0,
            arc_score=1.0,
            chat_peak=0.0,
            motion_peak=0.0,
            face_reaction_peak=0.0,
            speaker_turns=0,
            exclamation_count=0,
            laughter_count=0,
            question_open=False,
            context_cost=0.0,
            payoff_score=1.0,
            context_efficiency=0.5,
            quoteability_score=1.0,
            surprise_score=0.0,
            contradiction_score=0.0,
            standalone_score=1.0,
            semantic_density=1.0,
            profile_fit_score=1.0,
            source="quote",
        )
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            (artifact_dir / "gaming_ranker.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "profile": "gaming",
                        "feature_names": ["quoteability_score"],
                        "weights": {"quoteability_score": 5.0},
                        "scales": {"quoteability_score": {"mean": 0.0, "std": 1.0}},
                        "training_stats": {
                            "rows": 100,
                            "positive_rows": 40,
                            "negative_rows": 60,
                            "train_rows": 80,
                            "validation_rows": 20,
                        },
                        "validation": {"accuracy": 0.57, "ndcg_at_5": 1.0},
                        "intercept": 0.0,
                        "blend": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            score = Ranker(artifact_dir=artifact_dir, enable_trained_model=True).score([feature], profile="gaming")[0][1]

        self.assertGreater(score, 1.0)

    @unittest.skipUnless(joblib is not None and GradientBoostingClassifier is not None, "joblib and sklearn are required for artifact tests")
    def test_ranker_scores_with_gradient_boosting_artifact(self) -> None:
        model = GradientBoostingClassifier(random_state=0)
        model.fit([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]], [0, 0, 0, 1, 1, 1])

        feature = CandidateFeatures(
            candidate_id="ml",
            duration=12.0,
            hook_delay_sec=1.0,
            hook_strength=0.2,
            max_hype=1.0,
            mean_hype=1.0,
            arc_score=1.0,
            chat_peak=0.0,
            motion_peak=0.0,
            face_reaction_peak=0.0,
            speaker_turns=0,
            exclamation_count=0,
            laughter_count=0,
            question_open=False,
            context_cost=0.0,
            payoff_score=1.0,
            context_efficiency=0.5,
            quoteability_score=3.0,
            surprise_score=0.0,
            contradiction_score=0.0,
            standalone_score=0.0,
            semantic_density=0.0,
            profile_fit_score=1.0,
            source="quote",
        )
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            joblib.dump(model, artifact_dir / "gaming_ranker_model.joblib")
            (artifact_dir / "gaming_ranker.json").write_text(
                json.dumps(
                    {
                        "version": 2,
                        "profile": "gaming",
                        "model_type": "gradient_boosting",
                        "feature_names": ["quoteability_score"],
                        "weights": {},
                        "scales": {"quoteability_score": {"mean": 0.0, "std": 1.0}},
                        "model_path": "gaming_ranker_model.joblib",
                        "training_stats": {
                            "rows": 100,
                            "positive_rows": 40,
                            "negative_rows": 60,
                            "train_rows": 80,
                            "validation_rows": 20,
                        },
                        "validation": {"accuracy": 0.65, "ndcg_at_5": 0.8, "precision_at_5": 0.8},
                        "blend": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            score = Ranker(artifact_dir=artifact_dir, enable_trained_model=True).score([feature], profile="gaming")[0][1]

        self.assertGreater(score, 1.0)

    @unittest.skipUnless(joblib is not None, "joblib is required for artifact tests")
    def test_ranker_scores_with_lightgbm_ranker_artifact(self) -> None:
        feature = CandidateFeatures(
            candidate_id="ml",
            duration=12.0,
            hook_delay_sec=1.0,
            hook_strength=0.2,
            max_hype=1.0,
            mean_hype=1.0,
            arc_score=1.0,
            chat_peak=0.0,
            motion_peak=0.0,
            face_reaction_peak=0.0,
            speaker_turns=0,
            exclamation_count=0,
            laughter_count=0,
            question_open=False,
            context_cost=0.0,
            payoff_score=1.0,
            context_efficiency=0.5,
            quoteability_score=2.0,
            surprise_score=0.0,
            contradiction_score=0.0,
            standalone_score=0.0,
            semantic_density=0.0,
            profile_fit_score=1.0,
            source="quote",
        )
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            joblib.dump(_FakeLightGBMModel(), artifact_dir / "gaming_ranker_model.joblib")
            (artifact_dir / "gaming_ranker.json").write_text(
                json.dumps(
                    {
                        "version": 2,
                        "profile": "gaming",
                        "model_type": "lightgbm_ranker",
                        "feature_names": ["quoteability_score"],
                        "weights": {},
                        "scales": {"quoteability_score": {"mean": 0.0, "std": 1.0}},
                        "model_path": "gaming_ranker_model.joblib",
                        "training_stats": {
                            "rows": 100,
                            "positive_rows": 40,
                            "negative_rows": 60,
                            "train_rows": 80,
                            "validation_rows": 20,
                        },
                        "validation": {"accuracy": 0.65, "ndcg_at_5": 0.8, "precision_at_5": 0.8},
                        "blend": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            score = Ranker(artifact_dir=artifact_dir, enable_trained_model=True).score([feature], profile="gaming")[0][1]

        self.assertGreater(score, 1.0)

    def test_ranker_feature_flag_can_disable_trained_artifact(self) -> None:
        feature = CandidateFeatures(
            candidate_id="ml",
            hook_strength=0.1,
            payoff_score=1.0,
            max_hype=1.0,
            quoteability_score=1.0,
            standalone_score=1.0,
            semantic_density=1.0,
            profile_fit_score=1.0,
            source="quote",
        )
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            (artifact_dir / "gaming_ranker.json").write_text(
                json.dumps(
                    {
                        "version": 3,
                        "profile": "gaming",
                        "model_type": "linear",
                        "feature_names": ["quoteability_score"],
                        "weights": {"quoteability_score": 10.0},
                        "scales": {"quoteability_score": {"mean": 0.0, "std": 1.0}},
                        "training_stats": {
                            "rows": 100,
                            "positive_rows": 40,
                            "negative_rows": 60,
                            "train_rows": 80,
                            "validation_rows": 20,
                        },
                        "validation": {"accuracy": 0.7, "ndcg_at_5": 0.8, "precision_at_5": 0.8},
                        "blend": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            disabled_score = Ranker(artifact_dir=artifact_dir, enable_trained_model=False).score([feature], profile="gaming")[0][1]
            enabled_score = Ranker(artifact_dir=artifact_dir, enable_trained_model=True).score([feature], profile="gaming")[0][1]

        self.assertNotEqual(disabled_score, enabled_score)

    def test_ranker_rewards_chapter_coherence_and_topic_shift(self) -> None:
        coherent = CandidateFeatures(
            candidate_id="coherent",
            hook_strength=0.5,
            payoff_score=4.0,
            max_hype=4.0,
            context_efficiency=0.6,
            standalone_score=0.6,
            profile_fit_score=0.6,
            chapter_coherence_score=0.85,
            topic_shift_strength=0.7,
            chapter_local_density=0.6,
            clip_overlap_count=0.0,
            source="reaction",
        )
        fragmented = CandidateFeatures(
            candidate_id="fragmented",
            hook_strength=0.5,
            payoff_score=4.0,
            max_hype=4.0,
            context_efficiency=0.6,
            standalone_score=0.6,
            profile_fit_score=0.6,
            chapter_coherence_score=0.2,
            topic_shift_strength=0.1,
            chapter_local_density=0.2,
            clip_overlap_count=2.5,
            source="reaction",
        )

        scored = Ranker().score_with_rationale([coherent, fragmented], profile="general")
        score_map = {candidate_id: (score, rationale) for candidate_id, score, rationale in scored}

        self.assertGreater(score_map["coherent"][0], score_map["fragmented"][0])
        self.assertIn("chapter_coherent", score_map["coherent"][1])
        self.assertIn("topic_shift_hook", score_map["coherent"][1])
        self.assertIn("overlap_risk", score_map["fragmented"][1])

    def test_ranker_rewards_trend_aligned_candidates(self) -> None:
        trend_aligned = CandidateFeatures(
            candidate_id="trend",
            hook_strength=0.4,
            payoff_score=3.5,
            max_hype=3.5,
            context_efficiency=0.55,
            standalone_score=0.55,
            profile_fit_score=0.55,
            trend_hook_score=0.72,
            trend_topic_score=0.6,
            trend_prior_score=0.8,
            source="reaction",
        )
        plain = CandidateFeatures(
            candidate_id="plain",
            hook_strength=0.4,
            payoff_score=3.5,
            max_hype=3.5,
            context_efficiency=0.55,
            standalone_score=0.55,
            profile_fit_score=0.55,
            trend_hook_score=0.0,
            trend_topic_score=0.0,
            trend_prior_score=0.0,
            source="reaction",
        )

        scored = Ranker().score_with_rationale([trend_aligned, plain], profile="general")
        score_map = {candidate_id: (score, rationale) for candidate_id, score, rationale in scored}

        self.assertGreater(score_map["trend"][0], score_map["plain"][0])
        self.assertIn("trend_aligned", score_map["trend"][1])


if __name__ == "__main__":
    unittest.main()
