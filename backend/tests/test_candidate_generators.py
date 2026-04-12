import sys
import types
import unittest
from unittest.mock import patch

sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))

from services import candidate_generators
from services.timeline_types import Chapter
from services.transcriber import RichSegment


class CandidateGeneratorTests(unittest.TestCase):
    def test_generate_candidates_defaults_to_semantic_candidates(self) -> None:
        segments = [
            RichSegment(start=0.0, end=2.0, text="setup", speaker="STREAMER", hype_score=1.0),
            RichSegment(
                start=2.0,
                end=5.0,
                text="what is happening?!",
                speaker="STREAMER",
                emotions=["exclamation"],
                hype_score=5.5,
                arc_score=3.0,
                chat_spike_score=0.7,
                max_face_reaction=0.5,
            ),
            RichSegment(start=5.0, end=7.0, text="haha no way", speaker="GUEST_1", emotions=["laughter"], hype_score=4.5),
            RichSegment(start=7.0, end=10.0, text="listen to this take.", speaker="STREAMER", arc_score=3.2),
        ]

        candidates = candidate_generators.generate_candidates(segments)
        sources = {candidate.source for candidate in candidates}

        self.assertIn("quote", sources)
        self.assertTrue(sources.issubset({"quote", "tension_build", "narrative_reveal", "controversy"}))

    def test_generate_semantic_candidates_only_uses_semantic_sources(self) -> None:
        segments = [
            RichSegment(start=0.0, end=2.0, text="setup", speaker="STREAMER", hype_score=1.0),
            RichSegment(
                start=2.0,
                end=5.0,
                text="listen to this insane take!",
                speaker="STREAMER",
                emotions=["exclamation"],
                hype_score=5.0,
                arc_score=3.2,
            ),
            RichSegment(start=5.0, end=8.0, text="nobody expected that reveal", speaker="GUEST_1", hype_score=2.0),
        ]

        candidates = candidate_generators.generate_semantic_candidates(segments)
        sources = {candidate.source for candidate in candidates}

        self.assertTrue(candidates)
        self.assertTrue(sources.issubset({"quote", "tension_build", "narrative_reveal", "controversy"}))

    def test_merge_prioritized_candidates_keeps_peak_when_semantic_candidate_overlaps(self) -> None:
        peak = candidate_generators.Candidate(
            id="peak:0:1000",
            start=8.0,
            end=24.0,
            source="peak",
            anchor_time=12.0,
            score_hint=8.0,
        )
        overlapping_semantic = candidate_generators.Candidate(
            id="quote:0:1100",
            start=9.0,
            end=22.0,
            source="quote",
            anchor_time=12.5,
            score_hint=6.0,
        )
        distant_semantic = candidate_generators.Candidate(
            id="controversy:0:5000",
            start=50.0,
            end=68.0,
            source="controversy",
            anchor_time=56.0,
            score_hint=5.0,
        )

        merged = candidate_generators.merge_prioritized_candidates([peak], [overlapping_semantic, distant_semantic])

        self.assertEqual([candidate.id for candidate in merged], ["peak:0:1000", "controversy:0:5000"])

    def test_dense_window_proposals_add_recall_for_gradual_build(self) -> None:
        segments = [
            RichSegment(start=0.0, end=4.0, text="setup", hype_score=1.2, arc_score=0.6),
            RichSegment(start=4.0, end=8.0, text="getting tense", hype_score=2.3, arc_score=1.5),
            RichSegment(start=8.0, end=12.0, text="chat is waking up", hype_score=3.2, arc_score=2.2, chat_spike_score=0.35),
            RichSegment(start=12.0, end=16.0, text="no way this works", hype_score=4.8, arc_score=3.4, max_face_reaction=0.6),
            RichSegment(start=16.0, end=20.0, text="that's insane", hype_score=5.5, arc_score=4.0, chat_spike_score=0.55, emotions=["exclamation"]),
        ]

        candidates = candidate_generators._generate_dense_window_candidates(segments)

        self.assertTrue(candidates)
        self.assertTrue(any(candidate.source == "dense_window" for candidate in candidates))

    def test_generate_fallback_candidates_preserves_generic_recall_sources(self) -> None:
        segments = [
            RichSegment(start=0.0, end=2.0, text="setup", speaker="STREAMER", hype_score=1.0),
            RichSegment(
                start=2.0,
                end=5.0,
                text="what is happening?!",
                speaker="STREAMER",
                emotions=["exclamation"],
                hype_score=5.5,
                arc_score=3.0,
                chat_spike_score=0.7,
                max_face_reaction=0.5,
            ),
            RichSegment(start=5.0, end=7.0, text="haha no way", speaker="GUEST_1", emotions=["laughter"], hype_score=4.5),
            RichSegment(start=7.0, end=10.0, text="listen to this take.", speaker="STREAMER", arc_score=3.2),
        ]

        candidates = candidate_generators.generate_fallback_candidates(segments)
        sources = {candidate.source for candidate in candidates}

        self.assertIn("hype_peak", sources)
        self.assertIn("chat_sync", sources)
        self.assertIn("reaction", sources)

    def test_generate_semantic_candidates_runs_per_chapter_when_provided(self) -> None:
        segments = [
            RichSegment(start=0.0, end=2.0, text="first chapter"),
            RichSegment(start=2.0, end=4.0, text="still first chapter"),
            RichSegment(start=10.0, end=12.0, text="second chapter"),
            RichSegment(start=12.0, end=14.0, text="still second chapter"),
        ]
        chapters = [
            Chapter(id="chapter_00", start=0.0, end=4.0, index=0),
            Chapter(id="chapter_01", start=10.0, end=14.0, index=1),
        ]

        with patch("services.candidate_generators._generate_semantic_candidates_for_segments") as helper_mock:
            helper_mock.side_effect = [
                [candidate_generators.Candidate(id="quote:0:100", start=0.0, end=4.0, source="quote", anchor_time=1.0)],
                [candidate_generators.Candidate(id="quote:1:1100", start=10.0, end=14.0, source="quote", anchor_time=11.0)],
            ]
            candidates = candidate_generators.generate_semantic_candidates(segments, chapters=chapters)

        self.assertEqual(len(candidates), 2)
        self.assertEqual(helper_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
