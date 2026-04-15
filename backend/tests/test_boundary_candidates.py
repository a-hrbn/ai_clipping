import sys
import types
import unittest

sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))

from services import boundary_candidates
from services.timeline_types import Candidate
from services.transcriber import RichSegment


class BoundaryCandidateTests(unittest.TestCase):
    def test_build_boundary_points_merges_scene_speaker_pause_signals(self) -> None:
        segments = [
            RichSegment(start=0.0, end=2.0, text="Setup.", speaker="STREAMER"),
            RichSegment(
                start=3.4,
                end=5.0,
                text="No way!",
                speaker="GUEST_1",
                has_scene_change=True,
            ),
            RichSegment(start=5.0, end=7.0, text="That actually worked.", speaker="GUEST_1"),
        ]
        candidate = Candidate(
            id="cand",
            start=0.0,
            end=7.0,
            source="reaction",
            anchor_time=4.2,
        )

        boundaries = boundary_candidates.build_boundary_points(segments, candidate)
        by_time = {boundary.time: boundary for boundary in boundaries}

        self.assertIn(3.4, by_time)
        self.assertIn(5.0, by_time)
        self.assertTrue(by_time[3.4].is_sentence_start)
        self.assertTrue(by_time[3.4].is_speaker_turn)
        self.assertTrue(by_time[3.4].is_scene_cut)
        self.assertTrue(by_time[3.4].is_pause_break)
        self.assertTrue(by_time[5.0].is_sentence_end)


if __name__ == "__main__":
    unittest.main()
