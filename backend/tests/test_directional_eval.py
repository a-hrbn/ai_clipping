import json
import pathlib
import tempfile
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


class GenerationCeilingTests(unittest.TestCase):
    def test_raw_any_covers_regardless_of_rank(self) -> None:
        gold = {"A": [{"start": 100.0, "end": 120.0, "views": 300_000}]}
        raw = {"A": [
            {"start_time": 500.0, "end_time": 520.0},  # no cover
            {"start_time": 101.0, "end_time": 119.0},  # covers, deep in the list
        ]}

        ceiling = directional_eval.generation_ceiling(gold, raw, min_views=0.0)

        self.assertEqual(ceiling["gt_count"], 1)
        self.assertEqual(ceiling["covered"], 1)
        self.assertEqual(ceiling["recall_any"], 1.0)

    def test_headline_filter_by_views(self) -> None:
        gold = {"A": [
            {"start": 100.0, "end": 120.0, "views": 50_000},
            {"start": 200.0, "end": 220.0, "views": 300_000},
        ]}
        raw = {"A": [{"start_time": 201.0, "end_time": 219.0}]}  # covers only the 300k one

        ceiling = directional_eval.generation_ceiling(gold, raw, min_views=100_000)

        self.assertEqual(ceiling["gt_count"], 1)  # 50k moment filtered out
        self.assertEqual(ceiling["recall_any"], 1.0)


class TwoSourceRunTests(unittest.TestCase):
    def test_raw_covers_beyond_topk_is_ceiling_hit_but_product_miss(self) -> None:
        # The signature ranking gap: a covering candidate exists in the raw set
        # but is truncated out of final_moments — product misses, ceiling hits.
        manifest = {"A": [{"start": 100.0, "end": 120.0, "views": 300_000, "gt_tier": "gold"}]}
        final = {"A": [{"start_time": 500.0, "end_time": 520.0, "rank": 1}]}  # no cover
        raw = {"A": [
            {"start_time": 500.0, "end_time": 520.0},
            {"start_time": 101.0, "end_time": 119.0},  # covers gold
        ]}

        result = directional_eval.run(manifest, final, raw, ks=(5, 10, 20, 50))

        self.assertEqual(result["floor"]["recall"]["20"], 0.0)        # product miss
        self.assertEqual(result["generation"]["floor"]["recall_any"], 1.0)  # ceiling hit
        self.assertEqual(result["generation"]["headline"]["recall_any"], 1.0)

    def test_raw_defaults_to_final_backward_compatible(self) -> None:
        manifest = {"A": [{"start": 100.0, "end": 120.0, "views": 50_000, "gt_tier": "gold"}]}
        final = {"A": [{"start_time": 101.0, "end_time": 119.0, "rank": 1}]}

        result = directional_eval.run(manifest, final, ks=(5, 10, 20))

        self.assertEqual(result["floor"]["recall"]["20"], 1.0)
        # raw defaulted to final → ceiling matches the product here
        self.assertEqual(result["generation"]["floor"]["recall_any"], 1.0)

    def test_report_shows_product_ceiling_and_gap(self) -> None:
        manifest = {"A": [{"start": 100.0, "end": 120.0, "views": 300_000, "gt_tier": "gold"}]}
        final = {"A": [{"start_time": 500.0, "end_time": 520.0, "rank": 1}]}
        raw = {"A": [{"start_time": 101.0, "end_time": 119.0}]}

        result = directional_eval.run(manifest, final, raw, ks=(5, 10, 20, 50))
        report = directional_eval.format_report(result)

        self.assertIn("PRODUCT", report.upper())
        self.assertIn("raw@any", report)
        self.assertIn("RANKING GAP", report.upper())


class CoverageGapDenominatorTests(unittest.TestCase):
    def test_no_pred_gold_vod_is_coverage_gap_not_a_miss(self) -> None:
        # B has gold but no finder predictions: it is a coverage gap, surfaced
        # separately, and must NOT drag the with-pred recall denominator.
        manifest = {
            "A": [{"start": 10.0, "end": 30.0, "views": 300_000, "gt_tier": "gold"}],
            "B": [{"start": 10.0, "end": 30.0, "views": 300_000, "gt_tier": "gold"}],
        }
        final = {"A": [{"start_time": 11.0, "end_time": 29.0, "rank": 1}]}

        result = directional_eval.run(manifest, final, ks=(5, 10, 20))

        self.assertEqual(result["gold_vods"], 2)
        self.assertEqual(result["scored_vods"], 1)          # only A is scored
        self.assertEqual(result["missing_prediction_vods"], ["B"])
        # floor counts only A's gold moment (a hit) — B does not appear as a miss
        self.assertEqual(result["floor"]["gt_count"], 1)
        self.assertEqual(result["floor"]["recall"]["20"], 1.0)

    def test_report_shows_coverage_gap_not_a_miss(self) -> None:
        manifest = {
            "A": [{"start": 10.0, "end": 30.0, "views": 300_000, "gt_tier": "gold"}],
            "B": [{"start": 10.0, "end": 30.0, "views": 300_000, "gt_tier": "gold"}],
        }
        final = {"A": [{"start_time": 11.0, "end_time": 29.0, "rank": 1}]}

        result = directional_eval.run(manifest, final, ks=(10, 20))
        report = directional_eval.format_report(result)

        self.assertIn("coverage gap", report.lower())
        self.assertIn("excluded from recall", report.lower())
        self.assertIn("B", report)  # the gap VOD id is listed


class AnchorRestrictionTests(unittest.TestCase):
    def test_non_anchor_gold_vod_excluded_from_canonical(self) -> None:
        manifest = {
            "A": [{"start": 10.0, "end": 30.0, "views": 300_000, "gt_tier": "gold"}],
            "2749590872": [{"start": 10.0, "end": 30.0, "views": 300_000, "gt_tier": "gold"}],
        }
        preds = {
            "A": [{"start_time": 11.0, "end_time": 29.0, "rank": 1}],
            "2749590872": [{"start_time": 11.0, "end_time": 29.0, "rank": 1}],
        }

        result = directional_eval.run(manifest, preds, anchor_vod_ids=["A"], ks=(5, 10, 20))

        self.assertTrue(result["anchor_restricted"])
        self.assertEqual(result["gold_vods"], 1)            # the non-anchor VOD is gone
        self.assertEqual(result["scored_vods"], 1)
        self.assertEqual(result["floor"]["gt_count"], 1)    # only A's moment counted

    def test_no_anchor_arg_keeps_all_gold(self) -> None:
        manifest = {
            "A": [{"start": 10.0, "end": 30.0, "views": 300_000, "gt_tier": "gold"}],
            "C": [{"start": 10.0, "end": 30.0, "views": 300_000, "gt_tier": "gold"}],
        }
        preds = {
            "A": [{"start_time": 11.0, "end_time": 29.0, "rank": 1}],
            "C": [{"start_time": 11.0, "end_time": 29.0, "rank": 1}],
        }

        result = directional_eval.run(manifest, preds, ks=(5, 10, 20))

        self.assertFalse(result["anchor_restricted"])
        self.assertEqual(result["gold_vods"], 2)


class RunTests(unittest.TestCase):
    def test_end_to_end_two_tier_with_coverage_gap(self) -> None:
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
        self.assertEqual(result["scored_vods"], 1)
        self.assertEqual(result["missing_prediction_vods"], ["B"])  # coverage gap, not a miss
        self.assertEqual(result["headline_min_views"], 100_000)
        # floor: only A is scored (B is a coverage gap); A's 50k gold hits at rank 2
        self.assertEqual(result["floor"]["gt_count"], 1)
        self.assertEqual(result["floor"]["recall"]["5"], 1.0)
        # headline (>=100k): A's gold is 50k (below tier), B excluded as a gap -> empty tier
        self.assertEqual(result["headline"]["gt_count"], 0)
        self.assertIsNone(result["headline"]["recall"]["5"])

    def test_report_renders_both_tiers(self) -> None:
        manifest = {"A": [{"start": 10.0, "end": 30.0, "views": 200_000, "gt_tier": "gold"}]}
        predictions = {"A": [{"start_time": 11.0, "end_time": 29.0, "rank": 1}]}

        result = directional_eval.run(manifest, predictions, ks=(10, 50), headline_min_views=100_000)
        report = directional_eval.format_report(result)

        self.assertIn("FLOOR", report.upper())
        self.assertIn("HEADLINE", report.upper())
        self.assertIn("r@50", report)


class OutputArtifactTests(unittest.TestCase):
    def test_output_writes_valid_baseline_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            manifest_path = tmp_path / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps({
                    "vod_id": "A",
                    "gt_moments": [{"start": 10.0, "end": 30.0, "views": 300_000, "gt_tier": "gold"}],
                }) + "\n",
                encoding="utf-8",
            )
            anchor_csv = tmp_path / "anchors.csv"
            anchor_csv.write_text("vod_id\nA\n", encoding="utf-8")
            preds_dir = tmp_path / "storage"
            preds_dir.mkdir()
            out_path = tmp_path / "baseline.json"

            rc = directional_eval.main([
                "--manifest", str(manifest_path),
                "--anchor-csv", str(anchor_csv),
                "--predictions-dir", str(preds_dir),
                "--output", str(out_path),
                "--json",
            ])

            self.assertEqual(rc, 0)
            self.assertTrue(out_path.exists())
            artifact = json.loads(out_path.read_text(encoding="utf-8"))
            for key in (
                "ks", "gold_tiers", "anchor_restricted", "gold_vods", "scored_vods",
                "missing_prediction_vods", "headline_min_views", "floor", "headline",
                "generation", "coverage_threshold", "computed_at",
            ):
                self.assertIn(key, artifact)
            # product recall@20 (headline column) present on both tiers
            self.assertIn(str(directional_eval.PRODUCT_HEADLINE_K), artifact["floor"]["recall"])
            self.assertIn(str(directional_eval.PRODUCT_HEADLINE_K), artifact["headline"]["recall"])
            # generation ceiling (raw@any) present
            self.assertIn("recall_any", artifact["generation"]["floor"])
            self.assertTrue(artifact["anchor_restricted"])


if __name__ == "__main__":
    unittest.main()
