import unittest
import sys
import types

sys.modules.setdefault("ollama", types.SimpleNamespace(Client=object))
sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))

from services import analyzer
from services.timeline_types import Candidate
from services.transcriber import RichSegment


class AnalyzerValidationTests(unittest.TestCase):
    def test_validate_snaps_times_to_known_boundaries(self) -> None:
        segments = [
            RichSegment(start=0.0, end=5.0, text="a"),
            RichSegment(start=5.0, end=10.0, text="b"),
        ]

        clips = analyzer._validate(
            {
                "clips": [
                    {
                        "rank": 1,
                        "start_time": 0.4,
                        "end_time": 9.6,
                        "viral_score": 11.5,
                        "reason": "peak",
                    }
                ]
            },
            segments,
        )

        self.assertIsNotNone(clips)
        self.assertEqual(clips[0]["start_time"], 0.0)
        self.assertEqual(clips[0]["end_time"], 10.0)
        self.assertEqual(clips[0]["viral_score"], 10.0)

    def test_validate_rejects_invalid_ranges(self) -> None:
        segments = [RichSegment(start=10.0, end=20.0, text="a")]

        clips = analyzer._validate(
            {
                "clips": [
                    {
                        "rank": 1,
                        "start_time": 25.0,
                        "end_time": 26.0,
                        "viral_score": 6.0,
                        "reason": "bad",
                    }
                ]
            },
            segments,
        )

        self.assertIsNone(clips)

    def test_enforce_story_shape_expands_peak_only_clip(self) -> None:
        segments = [
            RichSegment(start=0.0, end=2.0, text="Setup."),
            RichSegment(start=2.0, end=4.0, text="Something is happening."),
            RichSegment(start=4.0, end=5.0, text="No way!", emotions=["exclamation"]),
            RichSegment(start=5.0, end=8.0, text="That actually worked."),
            RichSegment(start=8.0, end=12.0, text="Everyone laughs.", emotions=["laughter"]),
        ]
        candidate = Candidate(
            id="c1",
            start=0.0,
            end=12.0,
            source="reaction",
            anchor_time=4.5,
            score_hint=5.0,
            tags=[],
        )

        moment = analyzer._enforce_story_shape(
            {"start_time": 4.0, "end_time": 5.0, "viral_score": 8.0, "reason": "peak"},
            candidate,
            segments,
        )

        self.assertLessEqual(moment["start_time"], 2.0)
        self.assertGreaterEqual(moment["end_time"], 8.0)
        self.assertGreaterEqual(moment["end_time"] - moment["start_time"], 8.0)


if __name__ == "__main__":
    unittest.main()
