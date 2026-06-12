"""Tests for the gap-decomposition + baseline-delta report (ml/gap_decomposition.py).

The report partitions every gold moment into hard-miss / accidental-only /
clean-but-buried and reports the canonical metrics (raw@any, floor/headline
recall@K, avg_rank) as a delta against the frozen composite baseline. These
tests pin the classification heuristic and the aggregation through the public
interface: small hand-built fixtures in, exact bucket counts out.
"""
import json
import os
import pathlib
import sys
import tempfile
import types
import unittest

# Defensive: building the deduped pool reaches the serving post-processor, which
# can pull the heavy transcription stack. Prior art (test_replay_harness) stubs
# whisper the same way so the import stays cheap in a bare test env.
sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_a, **_k: None))

from ml import eval_clip_finder, gap_decomposition as gd

_STORAGE_DIR = eval_clip_finder.DEFAULT_STORAGE_DIR
_BASELINE_PATH = gd.DEFAULT_BASELINE


def _real_artifacts_present() -> bool:
    """The reproduction guard needs the (gitignored) pipeline_debug artifacts + baseline."""
    return _BASELINE_PATH.exists() and any(_STORAGE_DIR.glob("pipeline_debug_*"))


def _gold(start, end, views=0.0):
    return {"start": float(start), "end": float(end), "views": float(views), "gt_tier": "gold"}


def _cand(start, end):
    return {"start_time": float(start), "end_time": float(end)}


class IsRealCandidateTests(unittest.TestCase):
    def test_clip_sized_candidate_matching_gold_duration_is_real(self) -> None:
        # 30s gold, 20s candidate: inside [8,90]s and 0.5-3x the gold window.
        self.assertTrue(gd.is_real_candidate(_gold(100, 130), _cand(102, 122)))

    def test_oversized_window_is_not_real(self) -> None:
        # A 300s accidental window covers the gold but is far past the 90s cap.
        self.assertFalse(gd.is_real_candidate(_gold(100, 130), _cand(0, 300)))

    def test_duration_outside_gold_ratio_is_not_real(self) -> None:
        # 10s gold, 40s candidate: inside [8,90]s but 4x the gold window (> 3x).
        self.assertFalse(gd.is_real_candidate(_gold(100, 110), _cand(100, 140)))

    def test_too_short_candidate_is_not_real(self) -> None:
        # 5s candidate is below the 8s floor regardless of the gold window.
        self.assertFalse(gd.is_real_candidate(_gold(100, 110), _cand(100, 105)))


class ClassifyGoldTests(unittest.TestCase):
    def test_no_raw_candidate_covers_is_hard_miss(self) -> None:
        # Hard-miss is decided on the RAW pool — nothing covers the gold window,
        # so generation never surfaced it (reconciles with raw@any).
        gold = _gold(100, 130)
        raw = [_cand(0, 20), _cand(500, 520)]  # neither overlaps [100,130]

        result = gd.classify_gold(gold, raw, ranked_pool=raw)

        self.assertEqual(result["bucket"], gd.HARD_MISS)
        self.assertIsNone(result["buried_rank"])

    def test_only_oversized_window_covers_is_accidental_only(self) -> None:
        # A 300s window covers the gold, but no real (clip-sized) candidate does.
        gold = _gold(100, 130)
        oversized = _cand(0, 300)
        raw = [oversized]

        result = gd.classify_gold(gold, raw, ranked_pool=[oversized])

        self.assertEqual(result["bucket"], gd.ACCIDENTAL_ONLY)
        self.assertIsNone(result["buried_rank"])

    def test_real_covering_candidate_is_clean_but_buried_with_rank(self) -> None:
        # A real 20s candidate covers the gold; its rank in the deduped pool is
        # the buried_rank (1-based position). Here it sits at rank 3.
        gold = _gold(100, 130)
        real = _cand(102, 122)
        raw = [real]
        ranked = [_cand(500, 520), _cand(600, 620), real]  # real is 3rd

        result = gd.classify_gold(gold, raw, ranked_pool=ranked)

        self.assertEqual(result["bucket"], gd.CLEAN_BUT_BURIED)
        self.assertEqual(result["buried_rank"], 3)

    def test_buried_rank_is_the_best_rank_among_real_covering(self) -> None:
        # Two real covering candidates; buried_rank is the shallowest (min) one.
        gold = _gold(100, 130)
        real_shallow = _cand(101, 123)
        real_deep = _cand(103, 121)
        ranked = [real_shallow, _cand(500, 520), real_deep]

        result = gd.classify_gold(gold, [real_shallow, real_deep], ranked_pool=ranked)

        self.assertEqual(result["bucket"], gd.CLEAN_BUT_BURIED)
        self.assertEqual(result["buried_rank"], 1)


# ── Hand-computed decomposition fixture ──────────────────────────────────────
# Two VODs, four gold moments, one of each bucket plus a second buried one.
# shortlist_depth is set to 3 so the within/beyond cut is checkable on a small
# pool. Every expected count below is computed by hand from these windows.
_GOLD1 = _gold(100, 130, views=300_000)   # A: clean-but-buried, real cand @ rank 2
_GOLD2 = _gold(200, 230, views=50_000)    # A: hard-miss, nothing covers it
_GOLD3 = _gold(400, 430, views=500_000)   # B: accidental-only, 350s window covers
_GOLD4 = _gold(6000, 6030, views=80_000)  # B: clean-but-buried, real cand @ rank 5

_RANKED_A = [_cand(500, 520), _cand(102, 122)]              # real1 at rank 2
_RANKED_B = [
    _cand(300, 650),     # rank 1: 350s oversized window over GOLD3 (accidental)
    _cand(1000, 1020), _cand(1100, 1120), _cand(1200, 1220),
    _cand(6002, 6022),   # rank 5: real window over GOLD4 (beyond shortlist 3)
]


def _fixture():
    gold_by_vod = {"A": [_GOLD1, _GOLD2], "B": [_GOLD3, _GOLD4]}
    # raw == ranked here: hard-miss coverage is read off the raw pool, and only
    # GOLD2 has no covering candidate anywhere.
    return gold_by_vod, {"A": _RANKED_A, "B": _RANKED_B}, {"A": _RANKED_A, "B": _RANKED_B}


class DecomposeTierTests(unittest.TestCase):
    def test_floor_bucket_counts_match_hand_computation(self) -> None:
        gold, raw, pool = _fixture()

        tier = gd.decompose_tier(gold, raw, pool, min_views=0.0, shortlist_depth=3)

        self.assertEqual(tier["gold_count"], 4)
        self.assertEqual(tier["buckets"], {
            gd.HARD_MISS: 1, gd.ACCIDENTAL_ONLY: 1, gd.CLEAN_BUT_BURIED: 2,
        })
        self.assertEqual(tier["shares"][gd.HARD_MISS], 0.25)
        self.assertEqual(tier["shares"][gd.CLEAN_BUT_BURIED], 0.5)

    def test_floor_buried_rank_stats_and_shortlist_cut(self) -> None:
        gold, raw, pool = _fixture()

        tier = gd.decompose_tier(gold, raw, pool, min_views=0.0, shortlist_depth=3)

        # buried ranks are [2, 5] -> median 3.5, min 2, max 5, n 2
        self.assertEqual(tier["buried_rank"]["median"], 3.5)
        self.assertEqual(tier["buried_rank"]["min"], 2)
        self.assertEqual(tier["buried_rank"]["max"], 5)
        self.assertEqual(tier["buried_rank"]["n"], 2)
        # shortlist_depth=3: rank 2 is reachable, rank 5 is truncated away
        self.assertEqual(tier["shortlist_depth"], 3)
        self.assertEqual(tier["clean_within_shortlist"], 1)
        self.assertEqual(tier["clean_beyond_shortlist"], 1)

    def test_headline_filters_by_views(self) -> None:
        gold, raw, pool = _fixture()

        tier = gd.decompose_tier(gold, raw, pool, min_views=100_000, shortlist_depth=3)

        # Only GOLD1 (300k, buried) and GOLD3 (500k, accidental) survive.
        self.assertEqual(tier["gold_count"], 2)
        self.assertEqual(tier["buckets"], {
            gd.HARD_MISS: 0, gd.ACCIDENTAL_ONLY: 1, gd.CLEAN_BUT_BURIED: 1,
        })
        self.assertEqual(tier["buried_rank"]["n"], 1)
        self.assertEqual(tier["buried_rank"]["median"], 2)


class BaselineDeltaTests(unittest.TestCase):
    def _baseline(self):
        return {
            "floor": {"recall": {"5": 0.1, "20": 0.2}, "avg_rank": 8.0},
            "headline": {"recall": {"5": 0.1, "20": 0.1}, "avg_rank": 5.0},
            "generation": {"floor": {"recall_any": 0.6}, "headline": {"recall_any": 0.6}},
        }

    def test_reports_raw_any_recall_and_avg_rank_deltas(self) -> None:
        current = {
            "floor": {"recall": {"5": 0.15, "20": 0.3}, "avg_rank": 6.0},
            "headline": {"recall": {"5": 0.1, "20": 0.2}, "avg_rank": None},
            "generation": {"floor": {"recall_any": 0.65}, "headline": {"recall_any": 0.6}},
        }

        delta = gd.baseline_delta(current, self._baseline())

        # raw@any (generation ceiling) per tier
        self.assertAlmostEqual(delta["raw_any"]["floor"]["delta"], 0.05, places=6)
        self.assertEqual(delta["raw_any"]["floor"]["current"], 0.65)
        self.assertEqual(delta["raw_any"]["floor"]["baseline"], 0.6)
        self.assertAlmostEqual(delta["raw_any"]["headline"]["delta"], 0.0, places=6)
        # recall@K per tier (the column the proposal moves)
        self.assertAlmostEqual(delta["recall"]["floor"]["20"]["delta"], 0.1, places=6)
        self.assertAlmostEqual(delta["recall"]["headline"]["5"]["delta"], 0.0, places=6)
        # avg_rank, with None propagated when a tier has no hits
        self.assertAlmostEqual(delta["avg_rank"]["floor"]["delta"], -2.0, places=6)
        self.assertIsNone(delta["avg_rank"]["headline"]["delta"])


def _baseline_result():
    return {
        "floor": {"recall": {"5": 0.1, "20": 0.1}, "avg_rank": 8.0},
        "headline": {"recall": {"5": 0.1, "20": 0.1}, "avg_rank": 5.0},
        "generation": {"floor": {"recall_any": 0.5}, "headline": {"recall_any": 0.5}},
    }


class RunTests(unittest.TestCase):
    def test_surfaces_coverage_gap_and_decomposes_only_scored_gold(self) -> None:
        # A is scored (a real covering candidate sits at pool rank 2 → buried);
        # B is gold but has no finder predictions → a coverage gap, surfaced and
        # excluded from the denominator, never counted as a miss.
        manifest = {"A": [_gold(100, 130, 300_000)], "B": [_gold(100, 130, 300_000)]}
        raw = {"A": [_cand(102, 122)]}
        pool = {"A": [_cand(500, 520), _cand(102, 122)]}
        final = {"A": [{"start_time": 102.0, "end_time": 122.0, "rank": 1}]}

        result = gd.run(
            manifest, raw, pool, final, _baseline_result(),
            anchor_vod_ids=["A", "B"], ks=(5, 10, 20),
        )

        self.assertEqual(result["gold_vods"], 2)
        self.assertEqual(result["scored_vods"], 1)
        self.assertEqual(result["missing_prediction_vods"], ["B"])  # gap, not a miss
        floor = result["decomposition"]["floor"]
        self.assertEqual(floor["gold_count"], 1)  # B is not folded in
        self.assertEqual(floor["buckets"][gd.CLEAN_BUT_BURIED], 1)
        # the canonical metrics + their delta against the frozen baseline ride along
        self.assertIn("delta", result["baseline_delta"]["raw_any"]["floor"])
        self.assertEqual(result["thresholds"]["min_duration_s"], 8.0)
        self.assertEqual(result["shortlist_depth"], gd.DEFAULT_SHORTLIST_DEPTH)


def _row(start, end, composite, *, reason):
    virality = composite / 10.0
    return {
        "start_time": start, "end_time": end, "viral_score": composite,
        "composite_score": composite, "llm_virality": virality,
        "hook": virality, "flow": virality, "value": virality, "trend": virality,
        "reason": reason,
    }


def _write_job_dir(storage, vod_id, ts, *, raw, segments, mtime):
    job_dir = storage / f"pipeline_debug_{ts}_{vod_id}"
    debug = job_dir / "debug"
    debug.mkdir(parents=True)
    (debug / "llm_moment_finder_moments_raw.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in raw), encoding="utf-8")
    (debug / "segments.jsonl").write_text(
        "".join(json.dumps(s) + "\n" for s in segments), encoding="utf-8")
    os.utime(job_dir, (mtime, mtime))


class CliTests(unittest.TestCase):
    def test_main_writes_valid_report_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            manifest_path = tmp_path / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps({
                    "vod_id": "V",
                    "gt_moments": [{"start": 0.0, "end": 20.0, "views": 300_000, "gt_tier": "gold"}],
                }) + "\n", encoding="utf-8")
            anchor_csv = tmp_path / "anchors.csv"
            anchor_csv.write_text("vod_id\nV\n", encoding="utf-8")
            baseline_path = tmp_path / "baseline.json"
            baseline_path.write_text(json.dumps(_baseline_result()), encoding="utf-8")
            storage = tmp_path / "storage"
            storage.mkdir()
            _write_job_dir(
                storage, "V", "20260101T000000000000Z",
                raw=[_row(1.0, 19.0, 8.0, reason="hit")],
                segments=[{"start": 0.0, "end": 20.0, "text": "A clean opening sentence here."}],
                mtime=1000.0)
            out_path = tmp_path / "report.json"

            rc = gd.main([
                "--manifest", str(manifest_path),
                "--anchor-csv", str(anchor_csv),
                "--predictions-dir", str(storage),
                "--baseline", str(baseline_path),
                "--output", str(out_path),
                "--json",
            ])

            self.assertEqual(rc, 0)
            self.assertTrue(out_path.exists())
            artifact = json.loads(out_path.read_text(encoding="utf-8"))
            for key in (
                "decomposition", "baseline_delta", "missing_prediction_vods",
                "scored_vods", "shortlist_depth", "thresholds", "computed_at",
            ):
                self.assertIn(key, artifact)
            # V's single gold is recovered as clean-but-buried (the only candidate)
            self.assertEqual(artifact["decomposition"]["floor"]["gold_count"], 1)
            self.assertEqual(
                artifact["decomposition"]["floor"]["buckets"][gd.CLEAN_BUT_BURIED], 1)


class DocumentedDecompositionGuardTests(unittest.TestCase):
    @unittest.skipUnless(
        _real_artifacts_present(),
        "real pipeline_debug artifacts / baseline not present (tmp/storage is gitignored)",
    )
    def test_reproduces_the_prd_floor_decomposition(self) -> None:
        # Criterion #2: replaying the real artifacts under the identity rank must
        # reproduce the PRD #46 diagnosis — floor 19 hard-miss / 10 accidental /
        # 22 clean-but-buried of 51 gold, buried-real median rank ~141 (min 2).
        manifest = eval_clip_finder.load_manifest()
        anchors = eval_clip_finder.read_anchor_vod_ids()
        baseline = json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))

        raw, pool, final, _gaps = gd.build_inputs(anchors, _STORAGE_DIR)
        result = gd.run(manifest, raw, pool, final, baseline, anchor_vod_ids=anchors)

        floor = result["decomposition"]["floor"]
        self.assertEqual(floor["gold_count"], 51)
        self.assertEqual(floor["buckets"], {
            gd.HARD_MISS: 19, gd.ACCIDENTAL_ONLY: 10, gd.CLEAN_BUT_BURIED: 22,
        })
        # hard-miss reconciles exactly with the frozen raw@any miss (51 − 32 covered)
        gen = baseline["generation"]["floor"]
        self.assertEqual(floor["buckets"][gd.HARD_MISS], gen["gt_count"] - gen["covered"])
        # buried-real median rank ~141 (documented), shallowest at rank 2
        self.assertEqual(floor["buried_rank"]["min"], 2)
        self.assertAlmostEqual(floor["buried_rank"]["median"], 141, delta=5)
        # the 5 missing_prediction_VODs stay a visible coverage gap, not misses
        self.assertEqual(result["missing_prediction_vods"], baseline["missing_prediction_vods"])
        # identity rank == baseline: every metric delta is zero (the skew guard)
        self.assertEqual(result["baseline_delta"]["recall"]["floor"]["20"]["delta"], 0.0)
        self.assertEqual(result["baseline_delta"]["raw_any"]["floor"]["delta"], 0.0)


if __name__ == "__main__":
    unittest.main()
