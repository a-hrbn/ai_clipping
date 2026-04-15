import json
import tempfile
import unittest
from pathlib import Path

try:
    import joblib
except ModuleNotFoundError:  # pragma: no cover - optional ML deps in test env
    joblib = None

try:
    from sklearn.ensemble import GradientBoostingClassifier
except ModuleNotFoundError:  # pragma: no cover - optional ML deps in test env
    GradientBoostingClassifier = None

from services import selector
from services.ranker import Ranker
from services.timeline_types import Candidate, CandidateFeatures


class _FakeLightGBMModel:
    def predict(self, matrix):
        return [float(row[0]) * 2.0 for row in matrix]


class RankerTests(unittest.TestCase):
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
