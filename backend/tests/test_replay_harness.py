"""Tests for the offline-replay harness (ml/replay_harness.py).

The harness *calls* the real serving post-processing functions (snap_boundaries,
dedupe_moments_by_overlap, enforce_diversity, top_k) under a pluggable
rank_score_fn, then feeds directional_eval — so the offline number equals the
serving number. These tests pin that contract through the public interface:
small fixtures in, observable post-processing/eval behaviour out.
"""
import json
import os
import pathlib
import sys
import tempfile
import types
import unittest

# Defensive: the moment_finder package reaches RichSegment, which can pull the
# heavy transcription stack. Prior art (test_post_processor) stubs whisper the
# same way so the import stays cheap in a bare test env.
sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_a, **_k: None))

from ml import eval_clip_finder, replay_harness
from services.moment_finder import _post_processor, _rank_post_processed_rows
from services.transcriber import RichSegment

_STORAGE_DIR = eval_clip_finder.DEFAULT_STORAGE_DIR
_BASELINE_PATH = eval_clip_finder.BACKEND_ROOT / "ml" / "artifacts" / "finder_recall_baseline.json"


def _real_artifacts_present() -> bool:
    """The skew guard needs the (gitignored) pipeline_debug artifacts + baseline."""
    return _BASELINE_PATH.exists() and any(_STORAGE_DIR.glob("pipeline_debug_*"))


def _segments() -> list[RichSegment]:
    # Two well-separated sentence pairs so snap lands on clean boundaries and the
    # two moments stay far enough apart to both survive diversity.
    return [
        RichSegment(start=0.0, end=4.0, text="Hello there everyone."),
        RichSegment(start=4.0, end=8.0, text="This is a test."),
        RichSegment(start=100.0, end=104.0, text="Another topic here."),
        RichSegment(start=104.0, end=108.0, text="It continues now."),
    ]


def _row(rank, start, end, composite, *, reason) -> dict:
    virality = composite / 10.0
    return {
        "rank": rank,
        "start_time": start,
        "end_time": end,
        "viral_score": composite,
        "composite_score": composite,
        "llm_virality": virality,
        "hook": virality,
        "flow": virality,
        "value": virality,
        "trend": virality,
        "reason": reason,
    }


class ReplayPostprocessIdentityTests(unittest.TestCase):
    def test_identity_is_bit_equal_to_the_serving_path(self) -> None:
        # No overlapping rows -> dedupe is a no-op, so the reference is the exact
        # public serving sequence snap -> diversity -> rank -> top_k. The harness
        # under the identity rank_score_fn must reproduce it byte-for-byte (the
        # eval/serving skew guard).
        segments = _segments()
        rows = [
            _row(1, 0.0, 8.0, 8.0, reason="A"),
            _row(2, 100.0, 108.0, 7.0, reason="B"),
        ]

        ref = _post_processor.snap_boundaries(rows, segments)
        ref = _post_processor.enforce_diversity(ref)
        ref = _rank_post_processed_rows(ref)

        got = replay_harness.replay_postprocess(rows, segments)

        self.assertEqual(got, ref)


class RankScoreFnTests(unittest.TestCase):
    def test_alternative_rank_score_fn_changes_final_order(self) -> None:
        # Identity ranks by composite (A=8.0 above B=7.0), so A leads. A
        # rank_score_fn that rescores by start_time (later wins) must flip the
        # final order to B-then-A — the single variation point actually steers
        # the served ranking.
        segments = _segments()
        rows = [
            _row(1, 0.0, 8.0, 8.0, reason="A"),
            _row(2, 100.0, 108.0, 7.0, reason="B"),
        ]

        def by_start_time(deduped: list[dict]) -> list[dict]:
            return [dict(r, composite_score=r["start_time"]) for r in deduped]

        baseline = replay_harness.replay_postprocess(rows, segments)
        rescored = replay_harness.replay_postprocess(rows, segments, rank_score_fn=by_start_time)

        self.assertEqual(baseline[0]["reason"], "A")
        self.assertEqual(rescored[0]["reason"], "B")


class TopKTests(unittest.TestCase):
    def test_top_k_truncates_to_the_k_highest_ranked(self) -> None:
        segments = _segments()
        rows = [
            _row(1, 0.0, 8.0, 8.0, reason="A"),
            _row(2, 100.0, 108.0, 7.0, reason="B"),
        ]

        kept = replay_harness.replay_postprocess(rows, segments, top_k=1)

        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["reason"], "A")  # the higher-composite moment


class DedupeTests(unittest.TestCase):
    def test_overlapping_row_is_deduped_in_the_replay(self) -> None:
        # Two rows that snap to the same segment window become a >=0.8-overlap
        # duplicate; the real dedupe must drop the lower-virality one. A third,
        # far-away row is untouched. Proves the harness calls the genuine
        # dedupe_moments_by_overlap, not a copy.
        segments = [
            RichSegment(start=0.0, end=30.0, text="A long opening statement that ends here."),
            RichSegment(start=200.0, end=230.0, text="A far away separate topic ending here."),
        ]
        rows = [
            _row(1, 2.0, 25.0, 8.0, reason="keep"),
            _row(2, 5.0, 28.0, 7.0, reason="drop"),  # snaps onto "keep" -> duplicate
            _row(3, 205.0, 225.0, 6.0, reason="far"),
        ]

        kept = replay_harness.replay_postprocess(rows, segments)

        reasons = {r["reason"] for r in kept}
        self.assertEqual(reasons, {"keep", "far"})
        self.assertNotIn("drop", reasons)


class ReplayRankedPoolTests(unittest.TestCase):
    def test_returns_full_deduped_pool_in_rank_order_without_top_k(self) -> None:
        # The deduped-pool seam (the judge-shortlist currency): snap -> dedupe ->
        # rank, but NO enforce_diversity / top_k. Both rows survive dedupe and
        # neither is truncated, so the pool holds both in composite-rank order —
        # while the top_k path would cut to one.
        segments = _segments()
        rows = [
            _row(1, 0.0, 8.0, 8.0, reason="A"),
            _row(2, 100.0, 108.0, 7.0, reason="B"),
        ]

        pool = replay_harness.replay_ranked_pool(rows, segments)

        self.assertEqual([r["reason"] for r in pool], ["A", "B"])  # ranked by composite
        self.assertEqual(len(replay_harness.replay_postprocess(rows, segments, top_k=1)), 1)
        self.assertEqual(len(pool), 2)  # the pool ignores top_k entirely

    def test_rank_score_fn_reorders_the_pool(self) -> None:
        segments = _segments()
        rows = [
            _row(1, 0.0, 8.0, 8.0, reason="A"),
            _row(2, 100.0, 108.0, 7.0, reason="B"),
        ]

        def by_start_time(deduped: list[dict]) -> list[dict]:
            return [dict(r, composite_score=r["start_time"]) for r in deduped]

        pool = replay_harness.replay_ranked_pool(rows, segments, rank_score_fn=by_start_time)

        self.assertEqual(pool[0]["reason"], "B")  # later start wins under the override


class LoadSegmentsChatParityTests(unittest.TestCase):
    """Skew guard (#65): the segment loader must preserve the chat emote-class
    scores so a re-gen sees the SAME enriched prompt marker as production."""

    def test_load_segments_round_trips_chat_marker_identically(self) -> None:
        from ml.llm_moment_finder import _build_chat_marker

        # A segment whose chat spike carries a dominant emote class + a voice burst:
        # _build_chat_marker emits ' | chat=0.82 hype | vol=0.60'. That class label
        # only survives if chat_hype_score round-trips through segments.jsonl — the
        # loader currently drops it, degrading the re-gen prompt vs production.
        original = RichSegment(
            start=10.0, end=14.0, text="that is actually insane",
            chat_spike_score=0.82, chat_hype_score=0.9, chat_funny_score=0.1,
            chat_shock_score=0.0, voice_energy_burst_score=0.6)
        expected_marker = _build_chat_marker(original)
        self.assertIn("hype", expected_marker)   # the fixture exercises the class label

        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "segments.jsonl"
            path.write_text(json.dumps({
                "start": 10.0, "end": 14.0, "text": "that is actually insane",
                "chat_spike_score": 0.82, "chat_hype_score": 0.9,
                "chat_funny_score": 0.1, "chat_shock_score": 0.0,
                "voice_energy_burst_score": 0.6,
            }) + "\n", encoding="utf-8")
            loaded = replay_harness._load_segments(path)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(_build_chat_marker(loaded[0]), expected_marker)


def _write_job_dir(storage: pathlib.Path, vod_id: str, ts: str, *, raw, segments, mtime: float) -> pathlib.Path:
    """Write a pipeline_debug_<ts>_<vod> job dir; omit an artifact by passing None."""
    job_dir = storage / f"pipeline_debug_{ts}_{vod_id}"
    debug = job_dir / "debug"
    debug.mkdir(parents=True)
    if raw is not None:
        (debug / "llm_moment_finder_moments_raw.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in raw), encoding="utf-8"
        )
    if segments is not None:
        (debug / "segments.jsonl").write_text(
            "".join(json.dumps(s) + "\n" for s in segments), encoding="utf-8"
        )
    os.utime(job_dir, (mtime, mtime))
    return job_dir


class LoadReplayInputsTests(unittest.TestCase):
    def test_picks_newest_job_dir_that_has_both_artifacts(self) -> None:
        # The newest run for V is missing segments; an older run has both. The
        # loader must fall back to the newest run *with both artifacts*, not the
        # newest run outright.
        with tempfile.TemporaryDirectory() as tmp:
            storage = pathlib.Path(tmp)
            _write_job_dir(
                storage, "V", "20260101T000000000000Z",
                raw=[{"start_time": 1.0, "end_time": 9.0, "composite_score": 5.0}],
                segments=[{"start": 0.0, "end": 10.0, "text": "Older complete run."}],
                mtime=1000.0,
            )
            _write_job_dir(
                storage, "V", "20260202T000000000000Z",
                raw=[{"start_time": 2.0, "end_time": 8.0, "composite_score": 9.0}],
                segments=None,  # newer but incomplete
                mtime=2000.0,
            )

            loaded = replay_harness.load_replay_inputs("V", storage)

            self.assertIsNotNone(loaded)
            rows, segments = loaded
            self.assertEqual([r["composite_score"] for r in rows], [5.0])  # the complete (older) run
            self.assertEqual(len(segments), 1)
            self.assertEqual(segments[0].text, "Older complete run.")

    def test_missing_artifacts_surface_as_a_coverage_gap_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = pathlib.Path(tmp)
            # W has raw but no segments; Z has no job dir at all.
            _write_job_dir(
                storage, "W", "20260101T000000000000Z",
                raw=[{"start_time": 1.0, "end_time": 9.0}], segments=None, mtime=1000.0,
            )

            self.assertIsNone(replay_harness.load_replay_inputs("W", storage))
            self.assertIsNone(replay_harness.load_replay_inputs("Z", storage))


class RunReplayTests(unittest.TestCase):
    def test_scores_replayable_vods_and_surfaces_artifactless_vods_as_gaps(self) -> None:
        # V is replayable and its replayed clip covers V's gold moment; Z is a
        # gold VOD with no artifacts. Z must surface as a coverage gap (excluded
        # from the denominator), never dragged in as a miss.
        with tempfile.TemporaryDirectory() as tmp:
            storage = pathlib.Path(tmp)
            _write_job_dir(
                storage, "V", "20260101T000000000000Z",
                raw=[_row(1, 1.0, 9.0, 8.0, reason="hit")],
                segments=[{"start": 0.0, "end": 10.0, "text": "A clean sentence."}],
                mtime=1000.0,
            )
            manifest = {
                "V": [{"start": 0.0, "end": 10.0, "views": 300_000, "gt_tier": "gold"}],
                "Z": [{"start": 0.0, "end": 10.0, "views": 300_000, "gt_tier": "gold"}],
            }

            result = replay_harness.run_replay(["V", "Z"], storage, manifest)

            self.assertEqual(result["gold_vods"], 2)
            self.assertEqual(result["scored_vods"], 1)
            self.assertEqual(result["missing_prediction_vods"], ["Z"])  # gap, not a miss
            self.assertEqual(result["floor"]["recall"]["20"], 1.0)      # V's gold hit


class BaselineSkewGuardTests(unittest.TestCase):
    @unittest.skipUnless(
        _real_artifacts_present(),
        "real pipeline_debug artifacts / baseline not present (tmp/storage is gitignored)",
    )
    def test_identity_reproduces_frozen_finder_recall_baseline(self) -> None:
        # The eval/serving skew + regression guard: replaying the real artifacts
        # under the identity rank_score_fn must reproduce the committed baseline's
        # product recall (floor & headline @5/@10/@20) exactly. Since the harness
        # *calls* the serving functions, any drift here is a real serving change.
        manifest = eval_clip_finder.load_manifest()
        anchors = eval_clip_finder.read_anchor_vod_ids()

        result = replay_harness.run_replay(anchors, _STORAGE_DIR, manifest)
        baseline = json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))

        for k in ("5", "10", "20"):
            self.assertEqual(result["floor"]["recall"][k], baseline["floor"]["recall"][k])
            self.assertEqual(result["headline"]["recall"][k], baseline["headline"]["recall"][k])
        self.assertEqual(result["scored_vods"], baseline["scored_vods"])
        self.assertEqual(result["missing_prediction_vods"], baseline["missing_prediction_vods"])


class CliTests(unittest.TestCase):
    def test_main_writes_valid_replay_result_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            manifest_path = tmp_path / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps({
                    "vod_id": "V",
                    "gt_moments": [{"start": 0.0, "end": 10.0, "views": 300_000, "gt_tier": "gold"}],
                }) + "\n",
                encoding="utf-8",
            )
            anchor_csv = tmp_path / "anchors.csv"
            anchor_csv.write_text("vod_id\nV\n", encoding="utf-8")
            storage = tmp_path / "storage"
            storage.mkdir()
            _write_job_dir(
                storage, "V", "20260101T000000000000Z",
                raw=[_row(1, 1.0, 9.0, 8.0, reason="hit")],
                segments=[{"start": 0.0, "end": 10.0, "text": "A clean sentence."}],
                mtime=1000.0,
            )
            out_path = tmp_path / "replay.json"

            rc = replay_harness.main([
                "--manifest", str(manifest_path),
                "--anchor-csv", str(anchor_csv),
                "--predictions-dir", str(storage),
                "--output", str(out_path),
                "--json",
            ])

            self.assertEqual(rc, 0)
            self.assertTrue(out_path.exists())
            artifact = json.loads(out_path.read_text(encoding="utf-8"))
            for key in (
                "ks", "floor", "headline", "generation",
                "missing_prediction_vods", "scored_vods", "computed_at",
            ):
                self.assertIn(key, artifact)
            self.assertEqual(artifact["floor"]["recall"]["20"], 1.0)  # V's gold hit


if __name__ == "__main__":
    unittest.main()
