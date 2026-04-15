import unittest

from services import clip_strategy


class ClipStrategyTests(unittest.TestCase):
    def test_normalize_strategy_builds_goal_family_and_keywords(self) -> None:
        strategy = clip_strategy.normalize_strategy(
            {
                "platform": "TikTok",
                "goal": "funny reaction moments",
                "topic": "ranked gameplay",
                "must_include": ["clutch", "rage"],
                "exclude": ["sponsor"],
                "clip_style": "reaction_first",
                "variant_count": 2,
                "time_range": {"start_sec": 30.0, "end_sec": 90.0},
            }
        )

        self.assertEqual(strategy["platform"], "tiktok")
        self.assertEqual(strategy["goal_family"], "reaction")
        self.assertEqual(strategy["clip_style"], "reaction_first")
        self.assertEqual(strategy["variant_count"], 2)
        self.assertIn("clutch", strategy["keywords"])
        self.assertIn("sponsor", strategy["exclude_keywords"])

    def test_build_variant_specs_prefers_goal_and_platform_specific_variants(self) -> None:
        specs = clip_strategy.build_variant_specs(
            {
                "platform": "youtube_shorts",
                "goal": "best quote from the debate",
                "clip_style": "quote_first",
                "variant_count": 3,
            }
        )

        keys = [spec["key"] for spec in specs]
        self.assertEqual(keys[0], "quote_first")
        self.assertIn("balanced", keys)
        self.assertIn("context_balanced", keys)

    def test_candidate_time_range_and_relevance_respect_constraints(self) -> None:
        strategy = {
            "goal": "crazy reveal",
            "must_include": ["reveal"],
            "exclude": ["boring"],
            "time_range": {"start_sec": 10.0, "end_sec": 20.0},
        }

        self.assertTrue(clip_strategy.candidate_in_time_range(9.0, 12.0, strategy))
        self.assertFalse(clip_strategy.candidate_in_time_range(21.0, 24.0, strategy))
        self.assertGreater(clip_strategy.text_relevance_score("crazy reveal payoff", strategy), 0.0)
        self.assertEqual(clip_strategy.text_relevance_score("boring sponsor segment", strategy), 0.0)


if __name__ == "__main__":
    unittest.main()
