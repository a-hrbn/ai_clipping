import pathlib
import tempfile
import unittest

from ml import gt_match_verify
from ml import match_cross_platform_audio_fingerprints as audio_matcher
from ml import verify_gt_matches as verifier


def _match_row(vod_id, start, end, *, url, views=0, sim=0.7, kind="single_window", title="clip"):
    return {
        "vod_id": vod_id,
        "window_start": start,
        "window_end": end,
        "source_url": url,
        "views": views,
        "transcript_similarity": sim,
        "transcript_match_kind": kind,
        "repost_title": title,
    }


# --------------------------------------------------------------------------
# Pure decision seam: captured VODs → matcher vod_rows pointing at artifacts
# --------------------------------------------------------------------------
class BuildCapturedVodRowsTests(unittest.TestCase):
    def test_points_each_row_at_its_stored_fingerprint_artifact(self) -> None:
        root = pathlib.Path("/cache")
        rows = gt_match_verify.build_captured_vod_rows(["A", "B"], root)

        self.assertEqual([r["vod_id"] for r in rows], ["A", "B"])
        self.assertEqual(rows[0]["fingerprint_path"], str(root / "A" / "fingerprint.json"))
        self.assertEqual(rows[0]["vod_url"], "https://www.twitch.tv/videos/A")

    def test_dedups_and_skips_blank_ids(self) -> None:
        rows = gt_match_verify.build_captured_vod_rows(["A", "A", "", None, "B"], pathlib.Path("/c"))
        self.assertEqual([r["vod_id"] for r in rows], ["A", "B"])


# --------------------------------------------------------------------------
# Pure decision seam: match rows → per-VOD deduped manifest
# --------------------------------------------------------------------------
class BuildManifestFromMatchesTests(unittest.TestCase):
    def test_groups_by_vod_and_dedups_overlapping_windows(self) -> None:
        rows = [
            _match_row("V1", 100.0, 120.0, url="u1", views=10),
            _match_row("V1", 105.0, 124.0, url="u2", views=5),   # overlaps u1 → deduped
            _match_row("V1", 400.0, 420.0, url="u3", views=7),
            _match_row("V2", 10.0, 30.0, url="u4", views=3),
        ]
        manifest = gt_match_verify.build_manifest_from_matches(rows)

        by_vod = {row["vod_id"]: row["gt_moments"] for row in manifest}
        self.assertEqual(len(by_vod["V1"]), 2)        # overlap collapsed, distant kept
        self.assertEqual(len(by_vod["V2"]), 1)
        # moments carry the fields the verifier reads
        moment = by_vod["V2"][0]
        self.assertEqual(moment["source_url"], "u4")
        self.assertIn("match_similarity", moment)

    def test_skips_rows_without_a_vod_id(self) -> None:
        manifest = gt_match_verify.build_manifest_from_matches([_match_row("", 1.0, 2.0, url="u")])
        self.assertEqual(manifest, [])


# --------------------------------------------------------------------------
# Pure decision seam: gold-filtered manifest (the AC-3 core)
# --------------------------------------------------------------------------
class FilterGoldManifestTests(unittest.TestCase):
    def _manifest(self):
        return [
            {
                "vod_id": "V1",
                "gt_moments": [
                    {"start": 1.0, "end": 2.0, "gt_tier": "gold"},
                    {"start": 3.0, "end": 4.0, "gt_tier": "silver"},
                    {"start": 5.0, "end": 6.0, "gt_tier": "review"},        # not_confirmed (weak)
                    {"start": 7.0, "end": 8.0, "gt_tier": "unverifiable"},  # vod_unavailable
                ],
            },
            {
                "vod_id": "V2",
                "gt_moments": [
                    {"start": 1.0, "end": 2.0, "gt_tier": "review"},
                    {"start": 3.0, "end": 4.0, "gt_tier": "unverifiable"},
                ],
            },
        ]

    def test_keeps_only_gold_and_silver_and_drops_emptied_vods(self) -> None:
        filtered = gt_match_verify.filter_gold_manifest(self._manifest())

        self.assertEqual([r["vod_id"] for r in filtered], ["V1"])  # V2 emptied → dropped
        tiers = [m["gt_tier"] for m in filtered[0]["gt_moments"]]
        self.assertEqual(tiers, ["gold", "silver"])  # review + unverifiable excluded

    def test_keep_silver_false_keeps_only_gold(self) -> None:
        filtered = gt_match_verify.filter_gold_manifest(self._manifest(), keep_silver=False)
        tiers = [m["gt_tier"] for m in filtered[0]["gt_moments"]]
        self.assertEqual(tiers, ["gold"])


class VerdictCountsTests(unittest.TestCase):
    def test_counts_by_tier(self) -> None:
        rows = [
            {"gt_tier": "gold"}, {"gt_tier": "gold"},
            {"gt_tier": "silver"}, {"gt_tier": "review"}, {"gt_tier": "unverifiable"},
        ]
        counts = gt_match_verify.verdict_counts(rows)
        self.assertEqual(counts["gold"], 2)
        self.assertEqual(counts["silver"], 1)
        self.assertEqual(counts["review"], 1)
        self.assertEqual(counts["unverifiable"], 1)


# --------------------------------------------------------------------------
# Match+verify core orchestration (thin shell exercised through injected fakes)
# --------------------------------------------------------------------------
class MatchAndVerifyTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.cache_root = pathlib.Path(self._tmp.name) / "captures"

    def _tier_for(self, url):
        return {
            "gold-url": "gold",
            "silver-url": "silver",
            "review-url": "review",
            "dead-url": "unverifiable",
        }[url]

    def _fake_verify(self, vod_id, moment, *, full_vod_fp, null_source_fp):
        tier = self._tier_for(moment["source_url"])
        verdict = {
            "gold": "confirmed",
            "silver": "not_confirmed",
            "review": "not_confirmed",
            "unverifiable": "vod_unavailable",
        }[tier]
        return {
            "key": verifier.moment_key(vod_id, moment),
            "vod_id": vod_id,
            "gt_tier": tier,
            "verdict": verdict,
            "full_vod_fp_present": full_vod_fp is not None,
        }

    def test_each_match_receives_a_verdict_and_gold_manifest_is_filtered(self) -> None:
        match_rows = [
            _match_row("V1", 100.0, 120.0, url="gold-url", views=900_000),
            _match_row("V1", 400.0, 420.0, url="silver-url", views=120_000),
            _match_row("V1", 700.0, 720.0, url="review-url", views=80_000),
            _match_row("V2", 10.0, 30.0, url="dead-url", views=50_000),
        ]
        result = gt_match_verify.match_and_verify(
            reposts=[{"source_url": "gold-url"}],
            captured_vod_ids=["V1", "V2"],
            cache_root=self.cache_root,
            match_reposts=lambda vod_rows, reposts: match_rows,
            load_vod_fingerprint=lambda vod_id: None,
            verify_moment=self._fake_verify,
        )

        # AC-1: every matched moment received a verdict.
        self.assertEqual(len(result.verifications), 4)
        all_moments = [m for row in result.manifest for m in row["gt_moments"]]
        self.assertTrue(all("gt_tier" in m for m in all_moments))
        self.assertEqual(result.counts["gold"], 1)

        # AC-2 + AC-3: review (not_confirmed) and unverifiable (vod_unavailable)
        # are excluded from the gold manifest; gold + silver survive.
        gold_tiers = sorted(m["gt_tier"] for row in result.gold_manifest for m in row["gt_moments"])
        self.assertEqual(gold_tiers, ["gold", "silver"])
        gold_vods = {row["vod_id"] for row in result.gold_manifest}
        self.assertNotIn("V2", gold_vods)  # only a dead match → dropped entirely

    def test_confirms_from_stored_fingerprint_when_vod_media_is_gone(self) -> None:
        # AC-4: a VOD captured earlier (fingerprint artifact on disk, NO media)
        # still feeds verification — the default loader reads the stored artifact.
        fp = audio_matcher.Fingerprint(duration=100.0, values=list(range(800)))
        fp_path = self.cache_root / "V1" / "fingerprint.json"
        fp_path.parent.mkdir(parents=True, exist_ok=True)
        import json
        fp_path.write_text(json.dumps({"duration": fp.duration, "fingerprint": fp.values}), encoding="utf-8")

        seen = {}

        def recording_verify(vod_id, moment, *, full_vod_fp, null_source_fp):
            seen["fp_present"] = full_vod_fp is not None
            seen["fp_len"] = len(full_vod_fp.values) if full_vod_fp else 0
            return self._fake_verify(vod_id, moment, full_vod_fp=full_vod_fp, null_source_fp=null_source_fp)

        result = gt_match_verify.match_and_verify(
            reposts=[{"source_url": "gold-url"}],
            captured_vod_ids=["V1"],
            cache_root=self.cache_root,
            match_reposts=lambda vod_rows, reposts: [
                _match_row("V1", 100.0, 120.0, url="gold-url", views=500_000)
            ],
            verify_moment=recording_verify,
        )

        self.assertTrue(seen["fp_present"])
        self.assertEqual(seen["fp_len"], 800)
        self.assertEqual(result.counts["gold"], 1)
        self.assertEqual(len(result.gold_manifest), 1)


if __name__ == "__main__":
    unittest.main()
