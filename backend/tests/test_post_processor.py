import importlib
import sys
import types
import unittest

sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))

from services.moment_finder import _post_processor
from services.transcriber import RichSegment


class PostProcessorTests(unittest.TestCase):
    def test_snap_boundaries_extends_too_short_clip(self) -> None:
        segments = [
            RichSegment(start=0.0, end=2.0, text="Setup."),
            RichSegment(start=2.0, end=4.0, text="Question?"),
            RichSegment(start=4.0, end=5.0, text="Peak!"),
            RichSegment(start=5.0, end=8.0, text="Payoff."),
        ]

        snapped = _post_processor.snap_boundaries(
            [{"start_time": 4.0, "end_time": 5.0, "viral_score": 8.0, "reason": "short"}],
            segments,
        )

        self.assertEqual(len(snapped), 1)
        self.assertGreaterEqual(snapped[0]["end_time"] - snapped[0]["start_time"], 6.0)

    def test_post_processor_is_an_internal_seam_not_a_top_level_service(self) -> None:
        # The seam now lives behind the moment_finder package; it must no longer
        # be importable as a top-level service. Renaming a writer there can never
        # silently strand a reader, because moment_finder is the only importer.
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("services.post_processor")

        self.assertTrue(hasattr(_post_processor, "snap_boundaries"))
        self.assertTrue(hasattr(_post_processor, "enforce_diversity"))


if __name__ == "__main__":
    unittest.main()
