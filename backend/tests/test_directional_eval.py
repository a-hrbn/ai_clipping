import unittest

from ml import directional_eval


def _vod(vod_id, gt_hits):
    return {"vod_id": vod_id, "has_gt": bool(gt_hits), "gt_hits": gt_hits}


def _hit(views, hit, rank):
    return {"views": float(views), "hit": hit, "hit_rank": rank}


class AggregateTierTests(unittest.TestCase):
    def setUp(self) -> None:
        # One gold moment that only a deep prediction (rank 45) recovers — the
        # signature of a *ranking* gap (surfaced but not top-K), not a
        # generation gap (never surfaced).
        self.per_vod = [
            _vod("A", [
                _hit(200_000, True, 3),     # hit shallow
                _hit(50_000, True, 8),      # hit, below headline tier
                _hit(500_000, True, 45),    # hit only deep (recall@50, not @10)
                _hit(300_000, False, None), # generation miss
            ]),
        ]
        self.ks = (5, 10, 20, 50)

    def test_floor_counts_all_gold_and_exposes_ranking_gap(self) -> None:
        agg = directional_eval.aggregate_tier(self.per_vod, self.ks, min_views=0.0)

        self.assertEqual(agg["gt_count"], 4)
        self.assertEqual(agg["recall"]["5"], 0.25)   # only rank 3
        self.assertEqual(agg["recall"]["10"], 0.5)   # ranks 3, 8
        self.assertEqual(agg["recall"]["20"], 0.5)
        self.assertEqual(agg["recall"]["50"], 0.75)  # ranks 3, 8, 45
        # raw recall@50 strictly above recall@10 == ranking gap, not generation
        self.assertGreater(agg["recall"]["50"], agg["recall"]["10"])

    def test_headline_tier_filters_by_views(self) -> None:
        agg = directional_eval.aggregate_tier(self.per_vod, self.ks, min_views=100_000)

        self.assertEqual(agg["gt_count"], 3)  # the 50k moment is dropped
        self.assertAlmostEqual(agg["recall"]["10"], 1 / 3, places=6)
        self.assertAlmostEqual(agg["recall"]["50"], 2 / 3, places=6)

    def test_empty_tier_yields_none_recall(self) -> None:
        agg = directional_eval.aggregate_tier(self.per_vod, self.ks, min_views=10_000_000)

        self.assertEqual(agg["gt_count"], 0)
        self.assertIsNone(agg["recall"]["10"])
        self.assertIsNone(agg["avg_rank"])


class TwoTierRecallTests(unittest.TestCase):
    def test_reports_floor_and_headline_with_threshold(self) -> None:
        per_vod = [_vod("A", [_hit(50_000, True, 2), _hit(300_000, True, 40)])]

        result = directional_eval.two_tier_recall(per_vod, (10, 50), headline_min_views=100_000)

        self.assertEqual(result["headline_min_views"], 100_000)
        self.assertEqual(result["floor"]["gt_count"], 2)
        self.assertEqual(result["headline"]["gt_count"], 1)
        # floor: rank 2 (<=10) and rank 40 (<=50) -> r@10 0.5, r@50 1.0
        self.assertEqual(result["floor"]["recall"]["10"], 0.5)
        self.assertEqual(result["floor"]["recall"]["50"], 1.0)
        # headline keeps only the 300k moment, hit at rank 40 (deep only)
        self.assertEqual(result["headline"]["recall"]["10"], 0.0)
        self.assertEqual(result["headline"]["recall"]["50"], 1.0)


class MissingPredictionVodsTests(unittest.TestCase):
    def test_surfaces_gold_vods_without_predictions(self) -> None:
        gold = {"A": [{"start": 1, "end": 2}], "B": [{"start": 3, "end": 4}]}
        predictions = {"A": [{"start_time": 1, "end_time": 2}], "B": []}

        missing = directional_eval.missing_prediction_vods(gold, predictions)

        self.assertEqual(missing, ["B"])

    def test_vods_without_gold_are_ignored(self) -> None:
        gold = {"A": [], "B": [{"start": 3, "end": 4}]}

        missing = directional_eval.missing_prediction_vods(gold, {})

        self.assertEqual(missing, ["B"])  # A has no gold, not a coverage gap


class GoldManifestTests(unittest.TestCase):
    def test_keeps_only_gold_and_drops_empty_vods(self) -> None:
        manifest = {
            "A": [
                {"start": 1, "end": 2, "gt_tier": "gold"},
                {"start": 3, "end": 4, "gt_tier": "silver"},        # not_confirmed
                {"start": 5, "end": 6, "gt_tier": "unverifiable"},  # vod_unavailable
            ],
            "B": [{"start": 7, "end": 8, "gt_tier": "review"}],     # no gold -> dropped
        }

        gold = directional_eval.gold_manifest(manifest)

        self.assertEqual(list(gold), ["A"])
        self.assertEqual(len(gold["A"]), 1)
        self.assertEqual(gold["A"][0]["gt_tier"], "gold")


class RunTests(unittest.TestCase):
    def test_end_to_end_two_tier_with_missing_predictions(self) -> None:
        manifest = {
            "A": [
                {"start": 10.0, "end": 30.0, "views": 50_000, "gt_tier": "gold"},
                {"start": 100.0, "end": 120.0, "views": 999_999, "gt_tier": "silver"},
            ],
            "B": [{"start": 10.0, "end": 30.0, "views": 300_000, "gt_tier": "gold"}],
        }
        predictions = {"A": [{"start_time": 12.0, "end_time": 28.0, "rank": 2}]}

        result = directional_eval.run(
            manifest, predictions, ks=(5, 10, 20, 50), headline_min_views=100_000
        )

        self.assertEqual(result["gold_vods"], 2)
        self.assertEqual(result["missing_prediction_vods"], ["B"])
        self.assertEqual(result["headline_min_views"], 100_000)
        # floor: A's gold hit at rank 2, B's gold a miss (no preds); silver excluded
        self.assertEqual(result["floor"]["gt_count"], 2)
        self.assertEqual(result["floor"]["recall"]["5"], 0.5)
        # headline (>=100k): only B's 300k gold moment, which is a miss
        self.assertEqual(result["headline"]["gt_count"], 1)
        self.assertEqual(result["headline"]["recall"]["5"], 0.0)

    def test_report_renders_both_tiers(self) -> None:
        manifest = {"A": [{"start": 10.0, "end": 30.0, "views": 200_000, "gt_tier": "gold"}]}
        predictions = {"A": [{"start_time": 11.0, "end_time": 29.0, "rank": 1}]}

        result = directional_eval.run(manifest, predictions, ks=(10, 50), headline_min_views=100_000)
        report = directional_eval.format_report(result)

        self.assertIn("FLOOR", report.upper())
        self.assertIn("HEADLINE", report.upper())
        self.assertIn("r@50", report)


if __name__ == "__main__":
    unittest.main()
