import sys
import types
import unittest

sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))

from services import candidate_normalizer
from services.timeline_types import Candidate
from services.transcriber import RichSegment


class CandidateNormalizerTests(unittest.TestCase):
    def test_normalize_candidates_snaps_and_dedupes(self) -> None:
        segments = [
            RichSegment(start=0.0, end=2.0, text="Intro."),
            RichSegment(start=2.0, end=4.0, text="Wait what?"),
            RichSegment(start=4.0, end=8.0, text="Big reaction!"),
            RichSegment(start=8.0, end=12.0, text="Payoff."),
        ]
        candidates = [
            Candidate(id="a", start=0.3, end=7.5, source="hype_peak", anchor_time=3.5, score_hint=5.0),
            Candidate(id="b", start=0.8, end=7.8, source="reaction", anchor_time=3.6, score_hint=4.0),
        ]

        normalized = candidate_normalizer.normalize_candidates(candidates, segments, max_duration=30.0)

        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0].start, 2.0)
        self.assertEqual(normalized[0].end, 12.0)


if __name__ == "__main__":
    unittest.main()
