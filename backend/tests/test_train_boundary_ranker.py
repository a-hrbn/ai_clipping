import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))

from ml import train_boundary_ranker
from services import boundary_scorer
from services.timeline_types import BoundaryFeatures


def _row(
    candidate_id: str,
    creator_id: str,
    *,
    label_class: str,
    start_error: float,
    end_error: float,
    duration: float,
    hook_strength: float,
    payoff_score: float,
    payoff_clarity: float,
    standalone: float,
    context_efficiency: float,
    slow_open_penalty: float,
    context_dependency_score: float,
    weak_resolution_penalty: float,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "creator_id": creator_id,
        "job_id": f"job-{candidate_id}",
        "label": 1 if label_class == "positive" else 0,
        "label_class": label_class,
        "sample_weight": 1.0,
        "boundary_reference_start": 10.0,
        "boundary_reference_end": 10.0 + duration,
        "boundary_abs_start_error_sec": abs(start_error),
        "boundary_abs_end_error_sec": abs(end_error),
        "boundary_abs_total_error_sec": abs(start_error) + abs(end_error),
        "features": {
            "duration": duration,
            "hook_delay_sec": 2.5,
            "hook_strength": hook_strength,
            "payoff_score": payoff_score,
            "payoff_clarity_score": payoff_clarity,
            "standalone_score": standalone,
            "context_efficiency": context_efficiency,
            "slow_open_penalty": slow_open_penalty,
            "context_dependency_score": context_dependency_score,
            "weak_resolution_penalty": weak_resolution_penalty,
            "speaker_turns": 1,
            "motion_peak": 0.3,
            "topic_shift_strength": 0.2,
            "boundary_left_context_sec": 1.4,
            "boundary_right_context_sec": 1.6,
            "quoteability_score": 0.5,
            "post_hook_hype": 3.0,
            "chat_peak": 0.2,
            "face_reaction_peak": 0.2,
            "arc_score": 2.5,
        },
    }


class TrainBoundaryRankerTests(unittest.TestCase):
    def test_train_boundary_ranker_writes_artifact(self) -> None:
        rows = [
            _row("p1", "c1", label_class="positive", start_error=0.0, end_error=0.0, duration=18.0, hook_strength=0.85, payoff_score=6.0, payoff_clarity=0.8, standalone=0.8, context_efficiency=0.8, slow_open_penalty=0.1, context_dependency_score=0.1, weak_resolution_penalty=0.1),
            _row("p2", "c2", label_class="positive", start_error=0.0, end_error=0.0, duration=20.0, hook_strength=0.8, payoff_score=5.8, payoff_clarity=0.78, standalone=0.75, context_efficiency=0.76, slow_open_penalty=0.12, context_dependency_score=0.12, weak_resolution_penalty=0.12),
            _row("p3", "c3", label_class="positive", start_error=0.0, end_error=0.0, duration=22.0, hook_strength=0.82, payoff_score=6.2, payoff_clarity=0.81, standalone=0.79, context_efficiency=0.82, slow_open_penalty=0.09, context_dependency_score=0.08, weak_resolution_penalty=0.1),
            _row("n1", "c1", label_class="boundary_near_miss_negative", start_error=1.2, end_error=1.0, duration=15.0, hook_strength=0.35, payoff_score=2.5, payoff_clarity=0.3, standalone=0.35, context_efficiency=0.32, slow_open_penalty=0.7, context_dependency_score=0.72, weak_resolution_penalty=0.65),
            _row("n2", "c2", label_class="boundary_near_miss_negative", start_error=1.0, end_error=1.3, duration=27.0, hook_strength=0.4, payoff_score=2.9, payoff_clarity=0.35, standalone=0.38, context_efficiency=0.34, slow_open_penalty=0.62, context_dependency_score=0.68, weak_resolution_penalty=0.6),
            _row("n3", "c3", label_class="boundary_near_miss_negative", start_error=1.4, end_error=1.1, duration=29.0, hook_strength=0.3, payoff_score=2.2, payoff_clarity=0.28, standalone=0.3, context_efficiency=0.28, slow_open_penalty=0.75, context_dependency_score=0.7, weak_resolution_penalty=0.72),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            status = train_boundary_ranker.train_from_rows(rows, "general", artifact_dir)

            self.assertEqual(status, 0)
            artifact_path = artifact_dir / "general_boundary_ranker.json"
            self.assertTrue(artifact_path.exists())
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            self.assertIn("start_model", artifact)
            self.assertIn("end_model", artifact)
            self.assertIn("pair_model", artifact)
            self.assertIn("blend", artifact)

    def test_boundary_scorer_blends_with_artifact(self) -> None:
        artifact = {
            "version": 1,
            "profile": "general",
            "model_type": "boundary_logistic",
            "blend": {"start": 0.5, "end": 0.0, "pair": 0.0},
            "start_model": {
                "model_type": "logistic",
                "feature_names": list(boundary_scorer.START_MODEL_FEATURE_NAMES),
                "weights": {"clean_open_score": 4.0, "mid_thought_risk": -8.0},
                "intercept": -1.0,
                "scales": {
                    name: {"mean": 0.0, "std": 1.0}
                    for name in boundary_scorer.START_MODEL_FEATURE_NAMES
                },
                "validation": {"accuracy": 0.8, "f1": 0.7},
                "training_rows": 12,
            },
            "end_model": {
                "model_type": "logistic",
                "feature_names": list(boundary_scorer.END_MODEL_FEATURE_NAMES),
                "weights": {name: 0.1 for name in boundary_scorer.END_MODEL_FEATURE_NAMES},
                "intercept": 0.0,
                "scales": {
                    name: {"mean": 0.0, "std": 1.0}
                    for name in boundary_scorer.END_MODEL_FEATURE_NAMES
                },
                "validation": {"accuracy": 0.8, "f1": 0.7},
                "training_rows": 12,
            },
            "pair_model": {
                "model_type": "logistic",
                "feature_names": list(boundary_scorer.PAIR_MODEL_FEATURE_NAMES),
                "weights": {name: 0.1 for name in boundary_scorer.PAIR_MODEL_FEATURE_NAMES},
                "intercept": 0.0,
                "scales": {
                    name: {"mean": 0.0, "std": 1.0}
                    for name in boundary_scorer.PAIR_MODEL_FEATURE_NAMES
                },
                "validation": {"accuracy": 0.8, "f1": 0.7},
                "training_rows": 12,
            },
        }
        features = [
            BoundaryFeatures(
                boundary_id="b1",
                candidate_id="c1",
                role="start",
                clean_open_score=0.9,
                mid_thought_risk=0.0,
                context_dependency_risk=0.1,
                hook_proximity_score=0.8,
                duration_prior_score=0.8,
                speaker_transition_score=0.0,
                scene_transition_score=0.0,
                pause_strength_score=0.0,
                lexical_break_score=0.4,
            ),
            BoundaryFeatures(
                boundary_id="b2",
                candidate_id="c1",
                role="start",
                clean_open_score=0.52,
                mid_thought_risk=0.7,
                context_dependency_risk=0.2,
                hook_proximity_score=0.45,
                duration_prior_score=0.55,
                speaker_transition_score=0.0,
                scene_transition_score=0.0,
                pause_strength_score=0.0,
                lexical_break_score=0.1,
            ),
        ]
        heuristic = boundary_scorer._heuristic_start_scores(features)
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(boundary_scorer, "_DEFAULT_ARTIFACT_DIR", Path(tmp)), \
             patch.object(boundary_scorer.settings, "enable_boundary_ranker", True):
            (Path(tmp) / "general_boundary_ranker.json").write_text(json.dumps(artifact), encoding="utf-8")
            boundary_scorer._ARTIFACT_CACHE.clear()
            blended = boundary_scorer.score_start_boundaries(features, "general")

        self.assertGreater(blended["b1"], heuristic["b1"])
        self.assertLess(blended["b2"], heuristic["b2"])


if __name__ == "__main__":
    unittest.main()
