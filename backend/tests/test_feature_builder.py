import sys
import types
import unittest
from unittest.mock import patch

sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))
sys.modules.setdefault("ollama", types.SimpleNamespace(Client=lambda *_args, **_kwargs: None))

from services import feature_builder
from services.timeline_types import Candidate, Chapter
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
        self.assertGreater(features.trend_hook_score, 0.0)
        self.assertGreater(features.trend_prior_score, 0.0)
        self.assertEqual(features.feature_build_version, 5)
        self.assertGreaterEqual(features.payoff_clarity_score, 0.0)
        self.assertGreaterEqual(features.slow_open_penalty, 0.0)
        self.assertGreaterEqual(features.context_dependency_score, 0.0)
        self.assertGreaterEqual(features.weak_resolution_penalty, 0.0)

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

    def test_build_features_populates_chapter_features(self) -> None:
        candidates = [
            Candidate(id="a", start=2.0, end=8.0, source="reaction", anchor_time=4.0),
            Candidate(id="b", start=6.0, end=12.0, source="reaction", anchor_time=8.0),
        ]
        segments = [
            RichSegment(start=0.0, end=3.0, text="setup before a big reveal", speaker="STREAMER", hype_score=2.0),
            RichSegment(
                start=3.0,
                end=6.0,
                text="big reveal payoff moment",
                speaker="STREAMER",
                hype_score=5.0,
                chat_spike_score=0.6,
                has_scene_change=True,
            ),
            RichSegment(start=6.0, end=10.0, text="new topic starts here", speaker="GUEST_1", hype_score=3.0),
            RichSegment(start=10.0, end=14.0, text="follow up on the new topic", speaker="GUEST_1", hype_score=2.5),
        ]
        chapters = [
            Chapter(id="chapter_00", start=0.0, end=10.0, index=0, label="reaction", confidence=0.8),
            Chapter(id="chapter_01", start=10.0, end=14.0, index=1, label="discussion", confidence=0.7),
        ]

        features = feature_builder.build_features(candidates, segments, chapters=chapters)

        self.assertEqual(len(features), 2)
        self.assertGreater(features[0].chapter_position, 0.0)
        self.assertLess(features[0].chapter_position, 1.0)
        self.assertGreater(features[0].topic_shift_strength, 0.0)
        self.assertGreater(features[0].chapter_local_density, 0.0)
        self.assertGreater(features[0].chapter_boundary_distance, 0.0)
        self.assertGreater(features[0].chapter_coherence_score, 0.0)
        self.assertGreater(features[0].clip_overlap_count, 0.0)
        self.assertGreaterEqual(features[0].trend_prior_score, 0.0)
        self.assertGreater(features[1].vod_progress, features[0].vod_progress)

    def test_build_features_flags_context_dependency_and_slow_open(self) -> None:
        candidate = Candidate(id="ctx", start=0.0, end=12.0, source="quote", anchor_time=5.0)
        segments = [
            RichSegment(start=0.0, end=4.0, text="and then he said this thing", speaker="STREAMER", hype_score=0.8),
            RichSegment(start=4.0, end=8.0, text="so yeah that's what happened", speaker="STREAMER", hype_score=1.1),
            RichSegment(start=8.0, end=12.0, text="okay", speaker="STREAMER", hype_score=0.7),
        ]

        with patch("services.feature_builder.semantic_service.score_window") as score_window_mock:
            score_window_mock.return_value = {
                "quoteability": 0.2,
                "surprise": 0.1,
                "contradiction": 0.0,
                "standalone": 0.2,
                "density": 0.1,
                "hook_strength": 0.15,
                "context_dependency": 0.8,
                "payoff_clarity": 0.1,
                "llm_used": False,
            }
            features = feature_builder.build_features([candidate], segments)[0]

        self.assertGreater(features.slow_open_penalty, 0.4)
        self.assertGreater(features.context_dependency_score, 0.5)
        self.assertGreater(features.weak_resolution_penalty, 0.4)


if __name__ == "__main__":
    unittest.main()
