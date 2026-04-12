import sys
import types
import unittest
from unittest.mock import patch

sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))

from services import feature_builder
from services.timeline_types import Candidate
from services.transcriber import RichSegment


class FeatureBuilderTests(unittest.TestCase):
    def test_build_features_extracts_expected_signals(self) -> None:
        candidate = Candidate(id="x", start=0.0, end=10.0, source="reaction", anchor_time=2.5)
        segments = [
            RichSegment(start=0.0, end=2.0, text="What now?", emotions=["question"], speaker="STREAMER", hype_score=2.0),
            RichSegment(
                start=2.0,
                end=5.0,
                text="No way!",
                emotions=["exclamation", "audio_laughter", "face_reaction"],
                speaker="GUEST_1",
                hype_score=6.0,
                arc_score=4.0,
                chat_spike_score=0.6,
                max_motion_score=0.7,
                max_face_reaction=0.8,
            ),
            RichSegment(start=5.0, end=8.0, text="hahaha", emotions=["laughter"], speaker="STREAMER", hype_score=4.0),
        ]

        features = feature_builder.build_features([candidate], segments)[0]

        self.assertEqual(features.candidate_id, "x")
        self.assertTrue(features.question_open)
        self.assertGreater(features.max_hype, 5.0)
        self.assertGreater(features.payoff_score, 1.0)
        self.assertEqual(features.speaker_turns, 2)
        self.assertGreater(features.hook_strength, 0.2)
        self.assertGreaterEqual(features.standalone_score, 0.4)
        self.assertGreaterEqual(features.semantic_density, 0.0)
        self.assertEqual(features.feature_build_version, 2)

    def test_build_features_includes_similarity_features(self) -> None:
        candidate = Candidate(id="x", start=0.0, end=10.0, source="reaction", anchor_time=2.5)
        segments = [
            RichSegment(start=0.0, end=2.0, text="hook text", speaker="STREAMER", hype_score=2.0),
            RichSegment(start=2.0, end=5.0, text="payoff text", speaker="STREAMER", hype_score=6.0),
        ]

        with patch(
            "services.feature_builder.clip_similarity.score_candidate_window",
            return_value={
                "winner_similarity_max": 0.91,
                "winner_similarity_mean": 0.85,
                "loser_similarity_max": 0.22,
                "similarity_margin": 0.69,
                "neighbor_quality_score": 1.4,
            },
        ):
            features = feature_builder.build_features([candidate], segments)[0]

        self.assertEqual(features.winner_similarity_max, 0.91)
        self.assertEqual(features.similarity_margin, 0.69)
        self.assertEqual(features.neighbor_quality_score, 1.4)
        self.assertTrue(features.has_vector_similarity)

    def test_build_features_can_skip_llm_semantics_outside_shortlist(self) -> None:
        candidate = Candidate(id="x", start=0.0, end=10.0, source="reaction", anchor_time=2.5)
        segments = [
            RichSegment(start=0.0, end=2.0, text="hook text", speaker="STREAMER", hype_score=2.0),
            RichSegment(start=2.0, end=5.0, text="payoff text", speaker="STREAMER", hype_score=6.0),
        ]

        with patch("services.feature_builder.semantic_service.score_window") as score_window_mock:
            score_window_mock.return_value = {
                "quoteability": 0.1,
                "surprise": 0.2,
                "contradiction": 0.3,
                "standalone": 0.4,
                "density": 0.5,
                "hook_strength": 0.6,
            }
            feature_builder.build_features(
                [candidate],
                segments,
                "gaming",
                True,
                {"other"},
            )

        self.assertEqual(score_window_mock.call_args.kwargs["use_llm"], False)

    def test_build_features_tracks_llm_semantic_usage(self) -> None:
        candidate = Candidate(id="x", start=0.0, end=10.0, source="reaction", anchor_time=2.5)
        segments = [
            RichSegment(start=0.0, end=2.0, text="hook text", speaker="STREAMER", hype_score=2.0),
            RichSegment(start=2.0, end=5.0, text="payoff text", speaker="STREAMER", hype_score=6.0),
        ]

        with patch("services.feature_builder.semantic_service.score_window") as score_window_mock:
            score_window_mock.return_value = {
                "quoteability": 0.1,
                "surprise": 0.2,
                "contradiction": 0.3,
                "standalone": 0.4,
                "density": 0.5,
                "hook_strength": 0.6,
                "llm_used": True,
            }
            features = feature_builder.build_features([candidate], segments, "gaming")[0]

        self.assertTrue(features.has_llm_semantics)

    def test_build_features_passes_llm_provider(self) -> None:
        candidate = Candidate(id="x", start=0.0, end=10.0, source="reaction", anchor_time=2.5)
        segments = [
            RichSegment(start=0.0, end=2.0, text="hook text", speaker="STREAMER", hype_score=2.0),
            RichSegment(start=2.0, end=5.0, text="payoff text", speaker="STREAMER", hype_score=6.0),
        ]

        with patch("services.feature_builder.semantic_service.score_window") as score_window_mock:
            score_window_mock.return_value = {
                "quoteability": 0.1,
                "surprise": 0.2,
                "contradiction": 0.3,
                "standalone": 0.4,
                "density": 0.5,
                "hook_strength": 0.6,
            }
            feature_builder.build_features([candidate], segments, "gaming", llm_provider="ollama")

        self.assertEqual(score_window_mock.call_args.kwargs["llm_provider"], "ollama")


if __name__ == "__main__":
    unittest.main()
