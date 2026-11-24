import json
import pathlib
import tempfile
import unittest

from ml import gt_match_verify
from ml import gt_match_verify_weekly as weekly


def _repost(clip_id, *, url=None, views=0, title="clip"):
    return {
        "clip_id": clip_id,
        "source_url": url or f"https://youtube.com/shorts/{clip_id}",
        "views": views,
        "title": title,
    }


def _manifest_row(vod_id, moments):
    return {"vod_id": vod_id, "gt_moments": moments}


def _moment(start, end, *, url, tier, views=0):
    return {
        "start": start,
        "end": end,
        "source_url": url,
        "views": views,
        "gt_tier": tier,
    }


# --------------------------------------------------------------------------
# Pure decision seam: new-repost selection diff (AC-2 — the headline)
# --------------------------------------------------------------------------
class SelectNewRepostsTests(unittest.TestCase):
    def test_excludes_already_matched_and_returns_only_new(self) -> None:
        reposts = [_repost("C"), _repost("B"), _repost("A")]
        new = weekly.select_new_reposts(reposts, {"A", "B"})

        self.assertEqual([r["clip_id"] for r in new], ["C"])

    def test_empty_inputs_yield_nothing(self) -> None:
        self.assertEqual(weekly.select_new_reposts([], set()), [])
        self.assertEqual(weekly.select_new_reposts([], {"A"}), [])

    def test_deduplicates_keeping_first_occurrence(self) -> None:
        reposts = [_repost("C", views=1), _repost("C", views=2), _repost("D")]
        new = weekly.select_new_reposts(reposts, set())

        self.assertEqual([(r["clip_id"], r["views"]) for r in new], [("C", 1), ("D", 0)])

    def test_skips_records_without_a_usable_id(self) -> None:
        reposts = [{"clip_id": ""}, {"clip_id": None}, {"foo": "bar"}, _repost("E")]
        new = weekly.select_new_reposts(reposts, set())

        self.assertEqual([r["clip_id"] for r in new], ["E"])

    def test_falls_back_to_source_url_when_no_clip_id(self) -> None:
        # A repost already matched by its URL (no clip_id) must not be re-matched.
        reposts = [{"source_url": "u-seen"}, {"source_url": "u-new"}]
        new = weekly.select_new_reposts(reposts, {"u-seen"})

        self.assertEqual([r["source_url"] for r in new], ["u-new"])


# --------------------------------------------------------------------------
# Pure decision seam: accumulating manifest merge across weekly runs
# --------------------------------------------------------------------------
class MergeManifestsTests(unittest.TestCase):
    def test_appends_new_moments_per_vod_and_preserves_existing(self) -> None:
        existing = [_manifest_row("V1", [_moment(10, 20, url="a", tier="gold")])]
        new = [
            _manifest_row("V1", [_moment(50, 60, url="b", tier="silver")]),
            _manifest_row("V2", [_moment(5, 15, url="c", tier="gold")]),
        ]
        merged = weekly.merge_manifests(existing, new)

        by_vod = {r["vod_id"]: r["gt_moments"] for r in merged}
        self.assertEqual({m["source_url"] for m in by_vod["V1"]}, {"a", "b"})
        self.assertEqual([m["source_url"] for m in by_vod["V2"]], ["c"])
        # moments are ordered by start within a VOD
        self.assertEqual([m["start"] for m in by_vod["V1"]], [10, 50])

    def test_new_verdict_for_same_repost_overwrites_old(self) -> None:
        existing = [_manifest_row("V1", [_moment(10, 20, url="a", tier="review")])]
        new = [_manifest_row("V1", [_moment(10, 20, url="a", tier="gold")])]
        merged = weekly.merge_manifests(existing, new)

        moments = merged[0]["gt_moments"]
        self.assertEqual(len(moments), 1)
        self.assertEqual(moments[0]["gt_tier"], "gold")  # new wins


    def test_overlapping_moment_from_another_repost_keeps_the_better_verdict(self) -> None:
        # #130: cross-week dedup must not let a higher-view "review" duplicate
        # push a confirmed gold moment out of the gold manifest.
        existing = [_manifest_row("V1", [_moment(10, 20, url="a", tier="gold", views=10)])]
        new = [_manifest_row("V1", [_moment(11, 21, url="b", tier="review", views=99)])]
        merged = weekly.merge_manifests(existing, new)

        self.assertEqual([(m["source_url"], m["gt_tier"]) for m in merged[0]["gt_moments"]], [("a", "gold")])


# --------------------------------------------------------------------------
# Pure decision seam: gold-accrual vs target (AC-4)
# --------------------------------------------------------------------------
class GoldAccrualTests(unittest.TestCase):
    def test_counts_gold_moments_and_vods_against_target(self) -> None:
        gold_manifest = [
            _manifest_row("V1", [_moment(1, 2, url="a", tier="gold"),
                                 _moment(3, 4, url="b", tier="silver")]),
            _manifest_row("V2", [_moment(1, 2, url="c", tier="gold")]),
        ]
        accrual = weekly.gold_accrual(gold_manifest, target_gold=10, target_vods=4)

        self.assertEqual(accrual["gold"], 2)      # silver not counted
        self.assertEqual(accrual["vods"], 2)
        self.assertEqual(accrual["target_gold"], 10)
        self.assertEqual(accrual["gold_pct"], 20.0)
        self.assertEqual(accrual["vods_pct"], 50.0)
        self.assertFalse(accrual["target_reached"])

    def test_target_reached_when_both_thresholds_met(self) -> None:
        gold_manifest = [
            _manifest_row(f"V{i}", [_moment(1, 2, url=f"u{i}", tier="gold")])
            for i in range(5)
        ]
        accrual = weekly.gold_accrual(gold_manifest, target_gold=5, target_vods=5)
        self.assertTrue(accrual["target_reached"])


# --------------------------------------------------------------------------
# State persistence (resumable / idempotent)
# --------------------------------------------------------------------------
class MatchStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.state_path = pathlib.Path(self._tmp.name) / "state.json"

    def test_missing_state_loads_empty(self) -> None:
        state = weekly.load_match_state(self.state_path)
        self.assertEqual(weekly.matched_ids(state), set())

    def test_save_then_load_roundtrips(self) -> None:
        state = weekly.load_match_state(self.state_path)
        state.setdefault("matched", {})["A"] = {"clip_id": "A"}
        weekly.save_match_state(self.state_path, state)

        reloaded = weekly.load_match_state(self.state_path)
        self.assertEqual(weekly.matched_ids(reloaded), {"A"})
        self.assertFalse((self.state_path.parent / "state.json.tmp").exists())

    def test_unreadable_state_falls_back_to_empty(self) -> None:
        self.state_path.write_text("{ not json", encoding="utf-8")
        state = weekly.load_match_state(self.state_path)
        self.assertEqual(weekly.matched_ids(state), set())


# --------------------------------------------------------------------------
# Weekly orchestration (thin shell exercised through injected fakes)
# --------------------------------------------------------------------------
class RunWeeklyMatchVerifyTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = pathlib.Path(self._tmp.name)
        self.state_path = root / "state.json"
        self.cache_root = root / "captures"
        self.manifest_path = root / "matched.jsonl"
        self.gold_path = root / "gold.jsonl"
        self.match_calls: list[list[str]] = []

    def _fake_match_and_verify_factory(self, tier_by_url):
        def _fake(reposts, captured_vod_ids, *, cache_root, **kwargs):
            self.match_calls.append([r.get("clip_id") or r.get("source_url") for r in reposts])
            manifest = []
            match_rows = []
            verifications = []
            counts: dict = {}
            for i, r in enumerate(reposts):
                url = r["source_url"]
                tier = tier_by_url.get(url)
                if tier is None:
                    continue
                vod_id = r.get("vod_id", "V1")
                # Distinct VOD position per repost in a batch, unless the test pins one ("at").
                start = r.get("at", 10 + 100 * i)
                match_rows.append({"vod_id": vod_id, "source_url": url})
                manifest.append(_manifest_row(
                    vod_id, [_moment(start, start + 10, url=url, tier=tier, views=r.get("views", 0))]))
                verifications.append({"vod_id": vod_id, "gt_tier": tier})
                counts[tier] = counts.get(tier, 0) + 1
            gold_manifest = gt_match_verify.filter_gold_manifest(manifest)
            return gt_match_verify.MatchVerifyResult(
                manifest=manifest,
                gold_manifest=gold_manifest,
                verifications=verifications,
                counts=counts,
                match_rows=match_rows,
            )
        return _fake

    def _run(self, reposts, captured_vod_ids, *, match=None, **overrides):
        kwargs = dict(
            scan_reposts=lambda: reposts,
            captured_vod_ids=captured_vod_ids,
            state_path=self.state_path,
            cache_root=self.cache_root,
            manifest_path=self.manifest_path,
            gold_manifest_path=self.gold_path,
            match_and_verify=match,
            target_gold=10,
            target_vods=2,
            matched_at="2026-06-11T00:00:00+00:00",
            download_repost=lambda url: self.cache_root / "short_form" / "clip.mp4",
        )
        kwargs.update(overrides)
        return weekly.run_weekly_match_verify(**kwargs)

    def test_matches_new_reposts_writes_gold_manifest_and_summarizes(self) -> None:
        reposts = [
            dict(_repost("g1", url="gold-url", views=900_000), vod_id="V1"),
            dict(_repost("s1", url="silver-url", views=120_000), vod_id="V1"),
            dict(_repost("r1", url="review-url", views=80_000), vod_id="V2"),
        ]
        match = self._fake_match_and_verify_factory(
            {"gold-url": "gold", "silver-url": "silver", "review-url": "review"}
        )
        summary = self._run(reposts, ["V1", "V2"], match=match)

        # AC-3: match+verify ran and the gold-filtered manifest was written to disk.
        self.assertEqual(summary["run_counts"]["gold"], 1)
        self.assertEqual(summary["run_counts"]["silver"], 1)
        self.assertEqual(summary["run_counts"]["review"], 1)
        gold_rows = [json.loads(line) for line in self.gold_path.read_text().splitlines() if line]
        gold_tiers = sorted(m["gt_tier"] for row in gold_rows for m in row["gt_moments"])
        self.assertEqual(gold_tiers, ["gold", "silver"])  # review excluded from gold manifest

        # AC-4: summary reports new reposts + gold-accrual vs target.
        self.assertEqual(summary["new_reposts"], 3)
        self.assertEqual(summary["accrual"]["gold"], 1)
        self.assertEqual(summary["accrual"]["target_gold"], 10)
        self.assertFalse(summary["scan_zeroed"])

    def test_only_new_reposts_are_matched_and_manifest_accumulates(self) -> None:
        match = self._fake_match_and_verify_factory({"u-g1": "gold", "u-g2": "gold"})
        first = [dict(_repost("g1", url="u-g1"), vod_id="V1")]
        self._run(first, ["V1"], match=match)
        self.assertEqual(self.match_calls[-1], ["g1"])

        # Next week: g1 already matched, only g2 is new → only g2 is re-matched.
        second = [dict(_repost("g1", url="u-g1"), vod_id="V1"),
                  dict(_repost("g2", url="u-g2"), vod_id="V2")]
        summary = self._run(second, ["V1", "V2"], match=match)
        self.assertEqual(self.match_calls[-1], ["g2"])

        # Manifest accumulated both golds across the two runs.
        self.assertEqual(summary["accrual"]["gold"], 2)
        self.assertEqual(summary["accrual"]["vods"], 2)

    def test_reports_new_vods_captured_since_last_run(self) -> None:
        match = self._fake_match_and_verify_factory({})
        self._run([], ["V1"], match=match)  # first run sees V1
        summary = self._run([], ["V1", "V2"], match=match)  # V2 captured since

        self.assertEqual(summary["new_vods"], ["V2"])
        self.assertEqual(summary["total_captured_vods"], 2)

    def test_zeroed_discovery_does_not_advance_state_or_clobber_manifest(self) -> None:
        # AC-1: a bot-protection-zeroed scan must not be read as "0 new reposts".
        match = self._fake_match_and_verify_factory({"u-g1": "gold"})
        self._run([dict(_repost("g1", url="u-g1"), vod_id="V1")], ["V1"], match=match)
        gold_before = self.gold_path.read_text()

        def _zeroed():
            raise weekly.RepostDiscoveryZeroed("try again later")

        summary = self._run([], ["V1"], match=match, scan_reposts=_zeroed)

        self.assertTrue(summary["scan_zeroed"])
        self.assertEqual(summary["new_reposts"], 0)
        # State + gold manifest untouched: next week retries the same scan.
        self.assertEqual(self.gold_path.read_text(), gold_before)
        self.assertEqual(weekly.matched_ids(weekly.load_match_state(self.state_path)), {"g1"})

    def test_reposts_are_downloaded_before_matching_and_failed_downloads_retry_next_run(self) -> None:
        # #126: the matcher (like the real one) yields nothing for a repost
        # without video_path, so each new repost must be downloaded first; a
        # repost whose download fails is not marked done and is retried.
        seen_by_matcher: list[dict] = []

        def _media_only_match(reposts, captured_vod_ids, *, cache_root, **kwargs):
            seen_by_matcher.extend(reposts)
            with_media = [r for r in reposts if r.get("video_path")]
            return self._fake_match_and_verify_factory({r["source_url"]: "gold" for r in with_media})(
                with_media, captured_vod_ids, cache_root=cache_root
            )

        down_urls = {"u-bad"}
        download_calls: list[str] = []

        def _download(url):
            download_calls.append(url)
            if url in down_urls:
                raise RuntimeError("HTTP Error 429: try again later")
            return pathlib.Path(f"/cache/short_form/{url}.mp4")

        reposts = [dict(_repost("ok", url="u-ok"), vod_id="V1"),
                   dict(_repost("bad", url="u-bad"), vod_id="V2")]
        first = self._run(reposts, ["V1", "V2"], match=_media_only_match, download_repost=_download)

        self.assertEqual(
            [(r["clip_id"], r.get("video_path")) for r in seen_by_matcher],
            [("ok", "/cache/short_form/u-ok.mp4")],
        )
        self.assertEqual(first["accrual"]["gold"], 1)
        self.assertEqual(first["download_failed"], 1)

        # Next week the download works: only the previously failed repost is retried and matched.
        down_urls.clear()
        download_calls.clear()
        seen_by_matcher.clear()
        second = self._run(reposts, ["V1", "V2"], match=_media_only_match, download_repost=_download)

        self.assertEqual(download_calls, ["u-bad"])
        self.assertEqual([r["clip_id"] for r in seen_by_matcher], ["bad"])
        self.assertEqual(second["accrual"]["gold"], 2)
        self.assertEqual(second["download_failed"], 0)

    def test_same_moment_from_two_weeks_appears_once_in_gold_manifest(self) -> None:
        # #130: two different reposts of the same VOD moment, matched a week apart.
        match = self._fake_match_and_verify_factory({"u-week1": "gold", "u-week2": "gold"})
        self._run([dict(_repost("w1", url="u-week1", views=50_000), vod_id="V1", at=100)], ["V1"],
                  match=match, matched_at="2026-06-11T00:00:00+00:00")
        summary = self._run([dict(_repost("w2", url="u-week2", views=90_000), vod_id="V1", at=102)], ["V1"],
                            match=match, matched_at="2026-06-18T00:00:00+00:00")

        gold_rows = [json.loads(line) for line in self.gold_path.read_text().splitlines() if line]
        self.assertEqual([len(r["gt_moments"]) for r in gold_rows], [1])
        self.assertEqual(summary["accrual"]["gold"], 1)
        # Both reposts produced a match, so neither is retried.
        self.assertEqual(weekly.matched_ids(weekly.load_match_state(self.state_path)), {"w1", "w2"})

    def _captured_only_match(self):
        """Fake matcher that, like the real one, only hits a repost once its VOD is captured."""
        def _match(reposts, captured_vod_ids, *, cache_root, **kwargs):
            hits = {r["source_url"]: "gold" for r in reposts if r.get("vod_id") in set(captured_vod_ids)}
            return self._fake_match_and_verify_factory(hits)(reposts, captured_vod_ids, cache_root=cache_root)
        return _match

    def test_repost_without_a_hit_stays_open_and_is_retried_against_newly_captured_vods(self) -> None:
        # #127: the repost's VOD is only captured the day after the weekly run.
        late = [dict(_repost("late", url="u-late"), vod_id="V2")]
        first = self._run(late, ["V1"], match=self._captured_only_match(),
                          matched_at="2026-06-11T00:00:00+00:00")

        self.assertEqual(first["accrual"]["gold"], 0)
        self.assertNotIn("late", weekly.matched_ids(weekly.load_match_state(self.state_path)))

        second = self._run(late, ["V1", "V2"], match=self._captured_only_match(),
                           matched_at="2026-06-18T00:00:00+00:00")

        self.assertEqual(self.match_calls[-1], ["late"])
        self.assertEqual(second["accrual"]["gold"], 1)
        self.assertIn("late", weekly.matched_ids(weekly.load_match_state(self.state_path)))

    def test_repost_without_a_hit_is_finally_done_after_14_days(self) -> None:
        never = [dict(_repost("never", url="u-never"), vod_id="V9")]
        self._run(never, ["V1"], match=self._captured_only_match(), matched_at="2026-06-01T00:00:00+00:00")
        self._run(never, ["V1"], match=self._captured_only_match(), matched_at="2026-06-14T00:00:00+00:00")
        self.assertNotIn("never", weekly.matched_ids(weekly.load_match_state(self.state_path)))

        self._run(never, ["V1"], match=self._captured_only_match(), matched_at="2026-06-16T00:00:00+00:00")
        self.assertIn("never", weekly.matched_ids(weekly.load_match_state(self.state_path)))

        # Done for good: a later run no longer hands it to the matcher.
        self.match_calls.clear()
        self._run(never, ["V1"], match=self._captured_only_match(), matched_at="2026-06-23T00:00:00+00:00")
        self.assertEqual(self.match_calls, [[]])

    def test_repost_whose_match_lost_the_batch_dedup_is_still_done(self) -> None:
        # #127: "done" is decided on match rows, not the post-dedup manifest.
        def _dedup_match(reposts, captured_vod_ids, *, cache_root, **kwargs):
            manifest = [_manifest_row("V1", [_moment(100, 110, url="u-winner", tier="gold")])]
            return gt_match_verify.MatchVerifyResult(
                manifest=manifest,
                gold_manifest=gt_match_verify.filter_gold_manifest(manifest),
                verifications=[{"vod_id": "V1", "gt_tier": "gold"}],
                counts={"gold": 1},
                match_rows=[{"vod_id": "V1", "source_url": "u-winner"},
                            {"vod_id": "V1", "source_url": "u-loser"}],
            )

        reposts = [dict(_repost("winner", url="u-winner"), vod_id="V1"),
                   dict(_repost("loser", url="u-loser"), vod_id="V1")]
        self._run(reposts, ["V1"], match=_dedup_match)

        self.assertEqual(weekly.matched_ids(weekly.load_match_state(self.state_path)), {"winner", "loser"})

    def test_repost_uploaded_more_than_14_days_ago_without_a_hit_is_done_on_first_try(self) -> None:
        # #127: "älter als 14 Tage" is the repost's age, not time since first attempt.
        old = [dict(_repost("old", url="u-old"), vod_id="V9", upload_date="20260520")]
        fresh = [dict(_repost("fresh", url="u-fresh"), vod_id="V9", upload_date="20260605")]
        self._run(old + fresh, ["V1"], match=self._captured_only_match(), matched_at="2026-06-11T00:00:00+00:00")

        self.assertEqual(weekly.matched_ids(weekly.load_match_state(self.state_path)), {"old"})


if __name__ == "__main__":
    unittest.main()
