import sys
import types
import unittest

sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))

from services import semantic_service
from services.transcriber import RichSegment


class SemanticServiceTests(unittest.TestCase):
    def test_semantic_scores_detect_surprise_and_quoteability(self) -> None:
        window = [
            RichSegment(start=0.0, end=2.0, text="Listen, the truth is this is crazy!", emotions=["exclamation"]),
            RichSegment(start=2.0, end=5.0, text="I thought we lose, but no way this worked."),
        ]

        self.assertGreater(semantic_service.quoteability_score(window), 0.4)
        self.assertGreater(semantic_service.surprise_score(window), 0.3)
        self.assertGreater(semantic_service.contradiction_score(window), 0.2)
        self.assertGreater(semantic_service.standalone_score(window, 1.0), 0.5)


if __name__ == "__main__":
    unittest.main()
