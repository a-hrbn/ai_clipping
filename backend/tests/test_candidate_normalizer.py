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

    def test_dense_candidates_keep_setup_before_anchor(self) -> None:
        segments = [
            RichSegment(start=0.0, end=4.0, text="Here is the setup."),
            RichSegment(start=4.0, end=8.0, text="and this needs context"),
            RichSegment(start=8.0, end=12.0, text="Wait for it?"),
            RichSegment(start=12.0, end=16.0, text="Huge payoff!"),
        ]
        candidate = Candidate(
            id="dense_inventory:20:0:0:0",
            start=0.5,
            end=15.0,
            source="dense_inventory",
            anchor_time=12.5,
            score_hint=4.0,
        )

        normalized = candidate_normalizer.normalize_candidates([candidate], segments, max_duration=30.0)

        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0].start, 0.0)
        self.assertEqual(normalized[0].end, 16.0)

    def test_mid_thought_start_repairs_to_clean_context(self) -> None:
        segments = [
            RichSegment(start=0.0, end=3.0, text="Clean setup."),
            RichSegment(start=3.0, end=6.0, text="and then the actual point"),
            RichSegment(start=6.0, end=10.0, text="Big payoff!"),
        ]
        candidate = Candidate(
            id="dense_window:20:0:0",
            start=3.2,
            end=9.5,
            source="dense_window",
            anchor_time=7.0,
            score_hint=4.0,
        )

        normalized = candidate_normalizer.normalize_candidates([candidate], segments, max_duration=30.0)

        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0].start, 0.0)


if __name__ == "__main__":
    unittest.main()
