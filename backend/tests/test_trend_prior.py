import unittest
from unittest.mock import patch

from services import trend_prior
from services.transcriber import RichSegment


class TrendPriorTests(unittest.TestCase):
    def test_scores_hook_and_topic_matches(self) -> None:
        window = [
            RichSegment(start=0.0, end=2.0, text="No way, what happened here?"),
            RichSegment(start=2.0, end=5.0, text="This reveal is actually insane and chat went crazy."),
        ]

        scores = trend_prior.score_candidate_window(window, profile="general")

        self.assertGreater(scores["trend_hook_score"], 0.0)
        self.assertGreater(scores["trend_topic_score"], 0.0)
        self.assertGreater(scores["trend_prior_score"], 0.0)
        self.assertIn("disbelief_reaction", scores["matched_labels"])

    def test_can_disable_trend_prior(self) -> None:
        window = [RichSegment(start=0.0, end=2.0, text="No way, what happened here?")]

        with patch.object(trend_prior.settings, "enable_trend_prior", False):
            scores = trend_prior.score_candidate_window(window, profile="general")

        self.assertEqual(scores["trend_hook_score"], 0.0)
        self.assertEqual(scores["trend_topic_score"], 0.0)
        self.assertEqual(scores["trend_prior_score"], 0.0)
        self.assertEqual(scores["matched_labels"], [])


if __name__ == "__main__":
    unittest.main()
