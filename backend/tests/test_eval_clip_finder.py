import unittest

from ml import eval_clip_finder


class PredCoverageRatioTests(unittest.TestCase):
    def test_fraction_of_prediction_inside_gt(self) -> None:
        gt = {"start": 100.0, "end": 300.0}
        pred = {"start_time": 150.0, "end_time": 160.0}  # 10s, fully inside

        self.assertAlmostEqual(eval_clip_finder.pred_coverage_ratio(gt, pred), 1.0)

    def test_partial_overlap_at_edge(self) -> None:
        gt = {"start": 100.0, "end": 300.0}
        pred = {"start_time": 96.0, "end_time": 104.0}  # 8s, 4s overlap

        self.assertAlmostEqual(eval_clip_finder.pred_coverage_ratio(gt, pred), 0.5)

    def test_zero_length_prediction_is_zero(self) -> None:
        gt = {"start": 100.0, "end": 300.0}
        pred = {"start_time": 150.0, "end_time": 150.0}

        self.assertEqual(eval_clip_finder.pred_coverage_ratio(gt, pred), 0.0)


class IsHitTests(unittest.TestCase):
    def test_short_clip_fully_inside_long_gt_window_is_a_hit(self) -> None:
        # A precise 10s clip inside a 200s loose Iboclipss-repost window:
        # GT-coverage is tiny, pred-coverage is full -> the second direction hits.
        gt = {"start": 100.0, "end": 300.0}
        pred = {"start_time": 150.0, "end_time": 160.0}

        self.assertLess(eval_clip_finder.coverage_ratio(gt, pred), 0.5)
        self.assertTrue(eval_clip_finder.is_hit(gt, pred))

    def test_symmetric_overlap_is_a_hit(self) -> None:
        gt = {"start": 100.0, "end": 120.0}
        pred = {"start_time": 105.0, "end_time": 125.0}  # both directions 0.75

        self.assertTrue(eval_clip_finder.is_hit(gt, pred))

    def test_trivial_overlap_is_not_a_hit(self) -> None:
        gt = {"start": 100.0, "end": 200.0}
        pred = {"start_time": 190.0, "end_time": 260.0}  # 10s overlap, both < 0.5

        self.assertLess(eval_clip_finder.coverage_ratio(gt, pred), 0.5)
        self.assertLess(eval_clip_finder.pred_coverage_ratio(gt, pred), 0.5)
        self.assertFalse(eval_clip_finder.is_hit(gt, pred))

    def test_eight_second_clip_needs_real_overlap_to_hit(self) -> None:
        # The finder's 8s length discipline keeps the lower edge safe: a degenerate
        # mini-prediction can't game pred-coverage. An 8s clip must overlap >= 4s.
        gt = {"start": 100.0, "end": 300.0}
        below = {"start_time": 95.0, "end_time": 103.0}   # 3s overlap -> 0.375
        at = {"start_time": 96.0, "end_time": 104.0}      # 4s overlap -> 0.5

        self.assertFalse(eval_clip_finder.is_hit(gt, below))
        self.assertTrue(eval_clip_finder.is_hit(gt, at))


class EvaluateVodBidirectionalTests(unittest.TestCase):
    def test_short_clip_inside_long_window_hits_with_gt_centric_coverage(self) -> None:
        gt_moments = [{"start": 100.0, "end": 300.0, "views": 500_000}]
        pred_moments = [{"start_time": 150.0, "end_time": 160.0, "rank": 1}]

        result = eval_clip_finder.evaluate_vod(gt_moments, pred_moments, ks=(5, 10, 20))

        self.assertEqual(result["hit_count"], 1)
        hit = result["gt_hits"][0]
        self.assertTrue(hit["hit"])
        self.assertEqual(hit["hit_rank"], 1)
        # reported coverage stays GT-centric: 10s / 200s = 0.05
        self.assertEqual(hit["coverage"], 0.05)

    def test_trivial_overlap_is_a_miss(self) -> None:
        gt_moments = [{"start": 100.0, "end": 200.0}]
        pred_moments = [{"start_time": 190.0, "end_time": 260.0, "rank": 1}]

        result = eval_clip_finder.evaluate_vod(gt_moments, pred_moments, ks=(5, 10, 20))

        self.assertEqual(result["hit_count"], 0)
        hit = result["gt_hits"][0]
        self.assertFalse(hit["hit"])
        self.assertIsNone(hit["hit_rank"])
        self.assertEqual(hit["coverage"], 0.1)  # GT-centric: 10s / 100s

    def test_symmetric_case_still_hits(self) -> None:
        gt_moments = [{"start": 10.0, "end": 30.0}]
        pred_moments = [{"start_time": 12.0, "end_time": 28.0, "rank": 2}]

        result = eval_clip_finder.evaluate_vod(gt_moments, pred_moments, ks=(5, 10, 20))

        self.assertEqual(result["hit_count"], 1)
        self.assertEqual(result["gt_hits"][0]["hit_rank"], 2)


class FilterManifestTiersTests(unittest.TestCase):
    def test_filter_keeps_only_requested_tiers(self) -> None:
        manifest = {
            "123": [
                {"start": 1.0, "end": 2.0, "gt_tier": "gold"},
                {"start": 3.0, "end": 4.0, "gt_tier": "review"},
                {"start": 5.0, "end": 6.0},  # unverified — excluded under a filter
            ],
            "456": [
                {"start": 7.0, "end": 8.0, "gt_tier": "silver"},
            ],
        }

        filtered = eval_clip_finder.filter_manifest_tiers(manifest, {"gold", "silver"})

        self.assertEqual(len(filtered["123"]), 1)
        self.assertEqual(filtered["123"][0]["gt_tier"], "gold")
        self.assertEqual(len(filtered["456"]), 1)


if __name__ == "__main__":
    unittest.main()
