import sys
import types
import unittest

sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))

from services import scorer
from services.transcriber import RichSegment


class ScoreTimelineTests(unittest.TestCase):
    def test_build_score_timeline_adds_smoothed_virality_scores(self) -> None:
        segments = scorer.score_segments(
            [
                RichSegment(start=0.0, end=2.0, text="setup", words_per_second=1.0),
                RichSegment(
                    start=2.0,
                    end=4.0,
                    text="wait what happened?!",
                    emotions=["exclamation", "question", "audio_laughter"],
                    words_per_second=3.5,
                    chat_spike_score=0.6,
                    max_face_reaction=0.5,
                ),
                RichSegment(start=4.0, end=6.0, text="calm reset", words_per_second=1.2),
            ]
        )

        timeline = scorer.build_score_timeline(segments)

        self.assertEqual(len(timeline), 3)
        self.assertGreater(timeline[1].virality_score, timeline[0].virality_score)
        self.assertGreater(timeline[1].smoothed_score, 0.0)
        self.assertAlmostEqual(
            timeline[1].smoothed_score,
            round((timeline[0].virality_score + timeline[1].virality_score + timeline[2].virality_score) / 3.0, 3),
        )


if __name__ == "__main__":
    unittest.main()
