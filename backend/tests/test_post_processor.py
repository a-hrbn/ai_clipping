import sys
import types
import unittest

sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))

from services import post_processor
from services.transcriber import RichSegment


class PostProcessorTests(unittest.TestCase):
    def test_snap_boundaries_extends_too_short_clip(self) -> None:
        segments = [
            RichSegment(start=0.0, end=2.0, text="Setup."),
            RichSegment(start=2.0, end=4.0, text="Question?"),
            RichSegment(start=4.0, end=5.0, text="Peak!"),
            RichSegment(start=5.0, end=8.0, text="Payoff."),
        ]

        snapped = post_processor.snap_boundaries(
            [{"start_time": 4.0, "end_time": 5.0, "viral_score": 8.0, "reason": "short"}],
            segments,
        )

        self.assertEqual(len(snapped), 1)
        self.assertGreaterEqual(snapped[0]["end_time"] - snapped[0]["start_time"], 6.0)


if __name__ == "__main__":
    unittest.main()
