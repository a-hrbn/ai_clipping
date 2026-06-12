"""Tests for the grouped-by-VOD OOF eval-runner (ml/oof_eval_runner.py).

The runner splits the eval VODs into K folds with the *VOD* as the group, trains a
pluggable model per fold on the other folds, ranks each eval VOD with the fold that
did NOT see it (out-of-fold), and feeds the result through the existing replay
harness (#47) -> gap_decomposition (#49) -> directional_eval to emit a two-metric
delta report. It *calls* those functions, never reimplements them. These tests pin
the contract through the public interface: small fixtures in, observable fold /
leakage / metric behaviour out. No paid call.
"""
import json
import os
import pathlib
import sys
import tempfile
import types
import unittest

from ml import directional_eval, eval_clip_finder, gap_decomposition

_STORAGE_DIR = eval_clip_finder.DEFAULT_STORAGE_DIR
_BASELINE_PATH = gap_decomposition.DEFAULT_BASELINE


def _real_artifacts_present() -> bool:
    """The skew guard needs the (gitignored) pipeline_debug artifacts + baseline."""
    return _BASELINE_PATH.exists() and any(_STORAGE_DIR.glob("pipeline_debug_*"))

# Defensive: build_oof_inputs reaches the serving post-processor, which can pull
# the heavy transcription stack. Prior art (test_replay_harness, test_gap_
# decomposition) stubs whisper so the import stays cheap in a bare test env.
sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_a, **_k: None))

from ml import oof_eval_runner as oof


def _gold(start, end, views=0.0):
    return {"start": float(start), "end": float(end), "views": float(views), "gt_tier": "gold"}


def _cand(start, end, **extra):
    return {"start_time": float(start), "end_time": float(end), **extra}


# ── Fixture helpers (mirror test_replay_harness / test_gap_decomposition) ─────

def _segments() -> list[dict]:
    # Two well-separated sentence pairs so snap lands on clean boundaries and the
    # two moments stay far enough apart to both survive dedupe + diversity.
    return [
        {"start": 0.0, "end": 4.0, "text": "Hello there everyone."},
        {"start": 4.0, "end": 8.0, "text": "This is a test."},
        {"start": 100.0, "end": 104.0, "text": "Another topic here."},
        {"start": 104.0, "end": 108.0, "text": "It continues now."},
    ]


def _row(start, end, composite, *, reason) -> dict:
    virality = composite / 10.0
    return {
        "start_time": start, "end_time": end, "viral_score": composite,
        "composite_score": composite, "llm_virality": virality,
        "hook": virality, "flow": virality, "value": virality, "trend": virality,
        "reason": reason,
    }


def _two_moment_rows() -> list[dict]:
    return [_row(0.0, 8.0, 8.0, reason="A"), _row(100.0, 108.0, 7.0, reason="B")]


def _write_job_dir(storage, vod_id, ts, *, raw, segments, mtime) -> None:
    job_dir = storage / f"pipeline_debug_{ts}_{vod_id}"
    debug = job_dir / "debug"
    debug.mkdir(parents=True)
    (debug / "llm_moment_finder_moments_raw.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in raw), encoding="utf-8")
    (debug / "segments.jsonl").write_text(
        "".join(json.dumps(s) + "\n" for s in segments), encoding="utf-8")
    os.utime(job_dir, (mtime, mtime))


def _write_vods(storage: pathlib.Path, vod_ids) -> None:
    """Write a replayable job dir per VOD (identical small two-moment pool)."""
    for i, vod_id in enumerate(vod_ids):
        _write_job_dir(
            storage, vod_id, "20260101T000000000000Z",
            raw=_two_moment_rows(), segments=_segments(), mtime=1000.0 + i)


class _SpyTrainer:
    """A stub TrainFn that records each fold's training ids and stamps the rows it
    ranks with the id of the model that produced them — so a test can read back,
    for every eval VOD, *which* model ranked it and *what* that model trained on.
    """

    def __init__(self) -> None:
        self.trained: dict[int, frozenset] = {}

    def __call__(self, train_vods):
        model_id = len(self.trained)
        self.trained[model_id] = frozenset(train_vods)

        def rank_score_fn(rows: list[dict]) -> list[dict]:
            return [dict(r, _model_id=model_id) for r in rows]

        return rank_score_fn


class AssignFoldsTests(unittest.TestCase):
    def test_each_vod_lands_in_exactly_one_fold_partitioning_the_set(self) -> None:
        # Group = VOD: every VOD is assigned exactly one fold index, and the
        # assignment covers the whole anchor set (no VOD dropped, none doubled).
        folds = oof.assign_folds(["a", "b", "c", "d", "e"], k=2)

        self.assertEqual(set(folds), {"a", "b", "c", "d", "e"})
        self.assertTrue(all(0 <= f < 2 for f in folds.values()))
        self.assertEqual(sorted(set(folds.values())), [0, 1])  # both folds used

    def test_assignment_is_deterministic_regardless_of_input_order(self) -> None:
        # The split must be reproducible run-to-run: a shuffled anchor list gives
        # the identical {vod: fold} map (so the frozen baseline comparison holds).
        a = oof.assign_folds(["c", "a", "b", "d"], k=2)
        b = oof.assign_folds(["d", "b", "a", "c"], k=2)

        self.assertEqual(a, b)


class LeakageGuardTests(unittest.TestCase):
    def test_no_eval_vod_is_ranked_by_a_fold_that_trained_on_it(self) -> None:
        # The core OOF guarantee. Four VODs, k=2: each eval VOD must be ranked by
        # the fold model whose training set EXCLUDES it. A spy trainer stamps every
        # row with the model that ranked it, so we can read, per VOD, the model and
        # cross-check its recorded training ids — catching a runner that ranks a
        # VOD with the wrong (leaking) fold's model.
        vods = ["v0", "v1", "v2", "v3"]
        folds = oof.assign_folds(vods, k=2)
        spy = _SpyTrainer()
        with tempfile.TemporaryDirectory() as tmp:
            storage = pathlib.Path(tmp)
            _write_vods(storage, vods)

            raw, pool, final, gaps, fold_train = oof.build_oof_inputs(
                vods, storage, folds, spy)

        self.assertEqual(gaps, [])
        # Every fold's model trained on exactly the anchors NOT in that fold.
        for fold, train_vods in fold_train.items():
            held_out = {v for v in vods if folds[v] == fold}
            self.assertEqual(set(train_vods), set(vods) - held_out)
        # Per VOD: the model that actually ranked it never saw it in training.
        for vod_id in vods:
            model_ids = {r["_model_id"] for r in pool[vod_id]}
            self.assertEqual(len(model_ids), 1)  # one model ranked the whole VOD
            ranking_model = next(iter(model_ids))
            self.assertNotIn(vod_id, spy.trained[ranking_model])  # no leakage


# A 1-VOD fixture exercising all three #49 buckets. Only the clean-but-buried gold
# (a real, clip-sized candidate covers it and survives dedupe) belongs to the
# engineering denominator — the set a ranker can actually move.
_CBB_GOLD = _gold(100, 130, views=300_000)   # clean-but-buried: real cand (102,122)
_HARD_MISS_GOLD = _gold(200, 230)             # hard-miss: nothing covers [200,230]
_ACCIDENTAL_GOLD = _gold(400, 430)            # accidental: only a 400s window covers
_REAL_CAND = _cand(102, 122)                  # clip-sized, gold-proportioned
_OVERSIZED_CAND = _cand(300, 700)             # covers the accidental gold, not real


class CleanButBuriedRecallTests(unittest.TestCase):
    def _fixture(self):
        gold = {"V": [_CBB_GOLD, _HARD_MISS_GOLD, _ACCIDENTAL_GOLD]}
        raw = {"V": [_REAL_CAND, _OVERSIZED_CAND]}
        pool = {"V": [_REAL_CAND, _OVERSIZED_CAND]}
        return gold, raw, pool

    def test_denominator_is_only_the_clean_but_buried_gold(self) -> None:
        # Hard-miss (no candidate) and accidental-only (oversized window) are
        # excluded — the metric is un-confounded by the generation ceiling.
        gold, raw, pool = self._fixture()
        final = {"V": [_REAL_CAND]}  # the real cand is served at rank 1

        result = oof.clean_but_buried_recall(gold, raw, pool, final, ks=(5, 10, 20))

        self.assertEqual(result["denominator"], 1)  # only the CBB gold counts
        self.assertEqual(result["hits"]["20"], 1)
        self.assertEqual(result["recall"]["20"], 1.0)

    def test_buried_clean_gold_not_served_scores_zero_recall(self) -> None:
        # The real candidate exists in the pool (so the gold is in the denominator)
        # but the served final set misses it → recall 0 over a denominator of 1.
        gold, raw, pool = self._fixture()
        final = {"V": []}  # nothing served

        result = oof.clean_but_buried_recall(gold, raw, pool, final, ks=(5, 10, 20))

        self.assertEqual(result["denominator"], 1)
        self.assertEqual(result["recall"]["20"], 0.0)


def _reverse_reorder_train_fn(_train_vods):
    """A canned fake model: rescore each candidate by its start time so the later
    moment wins, flipping the composite order (mirrors the harness's by_start_time)."""

    def rank_score_fn(rows: list[dict]) -> list[dict]:
        return [dict(r, composite_score=r["start_time"]) for r in rows]

    return rank_score_fn


class FakeReorderSensitivityTests(unittest.TestCase):
    def test_fake_reorder_flips_the_served_order_and_moves_recall(self) -> None:
        # The two-moment pool (A composite 8 @ [0,8], B composite 7 @ [100,108])
        # with top_k=1: identity serves A and buries B, a reorder serves B. The gold
        # is covered only by B, so the model seam visibly steers the served ranking
        # AND the engineering metric moves 0 → 1.0 (metric sensitivity).
        vods = ["v0", "v1"]  # >= 2 VODs, the group=VOD split
        folds = oof.assign_folds(vods, k=2)
        gold = {v: [_gold(100, 108, views=300_000)] for v in vods}  # covered only by B
        with tempfile.TemporaryDirectory() as tmp:
            storage = pathlib.Path(tmp)
            _write_vods(storage, vods)
            id_raw, id_pool, id_final, _g, _f = oof.build_oof_inputs(
                vods, storage, folds, oof.identity_train_fn, top_k=1)
            fk_raw, fk_pool, fk_final, _g2, _f2 = oof.build_oof_inputs(
                vods, storage, folds, _reverse_reorder_train_fn, top_k=1)

        for v in vods:  # the served order flips A -> B under the fake model
            self.assertEqual(id_final[v][0]["reason"], "A")
            self.assertEqual(fk_final[v][0]["reason"], "B")
        id_cbb = oof.clean_but_buried_recall(gold, id_raw, id_pool, id_final, ks=(20,))
        fk_cbb = oof.clean_but_buried_recall(gold, fk_raw, fk_pool, fk_final, ks=(20,))
        self.assertEqual(id_cbb["recall"]["20"], 0.0)  # identity buries the clean gold
        self.assertEqual(fk_cbb["recall"]["20"], 1.0)  # the reorder serves it


class RunOofTwoMetricDeltaTests(unittest.TestCase):
    def test_report_carries_all_four_metrics_as_baseline_deltas(self) -> None:
        # One VOD, three gold (one of each #49 bucket). The frozen composite
        # baseline serves nothing; the OOF run lifts the clean gold into the served
        # set. The report must show BOTH gates plus the guardrail and avg_rank, each
        # as a current/baseline/delta cell against the baseline.
        manifest = {"V": [_CBB_GOLD, _HARD_MISS_GOLD, _ACCIDENTAL_GOLD]}
        raw = {"V": [_REAL_CAND, _OVERSIZED_CAND]}
        pool = {"V": [_REAL_CAND, _OVERSIZED_CAND]}  # SET invariant to the model
        baseline_final = {"V": []}                    # composite baseline serves nothing
        oof_final = {"V": [_REAL_CAND]}               # OOF serves the clean gold @ rank 1
        # The frozen baseline IS the identity/composite directional_eval result.
        frozen = directional_eval.run(
            manifest, baseline_final, raw, anchor_vod_ids=["V"],
            ks=(5, 10, 20), headline_min_views=100_000)

        report = oof.run_oof(
            manifest, raw, pool, oof_final, pool, baseline_final, frozen,
            anchor_vod_ids=["V"], folds={"V": 0}, fold_train_vods={0: []},
            coverage_gaps=[], ks=(5, 10, 20), headline_min_views=100_000)

        tm = report["two_metric_delta"]
        self.assertEqual(tm["k"], 20)
        # Ship-Gate: full-denominator headline recall@20, 0 → 1.0
        ship = tm["ship_headline_recall"]
        self.assertEqual(ship["baseline"], 0.0)
        self.assertEqual(ship["current"], 1.0)
        self.assertAlmostEqual(ship["delta"], 1.0, places=6)
        # Engineering-Gate: clean_but_buried recall@20 (floor tier), 0 → 1.0
        eng = tm["engineering_clean_but_buried_recall"]["floor"]
        self.assertEqual(eng["baseline"], 0.0)
        self.assertEqual(eng["current"], 1.0)
        self.assertAlmostEqual(eng["delta"], 1.0, places=6)
        # Guardrail: floor recall@20 (1 of 3 gold served) rises, not falls
        guard = tm["guardrail_floor_recall"]
        self.assertAlmostEqual(guard["current"], 1 / 3, places=6)
        self.assertGreaterEqual(guard["delta"], 0.0)
        # avg_rank cell present for both tiers
        self.assertIn("floor", tm["avg_rank"])
        self.assertIn("headline", tm["avg_rank"])


class CoverageGapAndFoldLoggingTests(unittest.TestCase):
    def test_artifactless_gold_vod_is_a_logged_gap_not_a_miss(self) -> None:
        # V is replayable; Z is a gold VOD with no artifacts. Z must surface as a
        # coverage gap (in gaps AND missing_prediction_vods), never folded in as a
        # miss, and every fold bound must be logged through on_fold.
        anchors = ["V", "Z"]
        folds = oof.assign_folds(anchors, k=2)
        records: list[dict] = []
        with tempfile.TemporaryDirectory() as tmp:
            storage = pathlib.Path(tmp)
            _write_vods(storage, ["V"])  # only V gets a job dir
            raw, pool, final, gaps, fold_train = oof.build_oof_inputs(
                anchors, storage, folds, oof.identity_train_fn, on_fold=records.append)

            manifest = {"V": [_gold(0, 8, 300_000)], "Z": [_gold(0, 8, 300_000)]}
            frozen = directional_eval.run(
                manifest, final, raw, anchor_vod_ids=anchors, ks=(5, 10, 20))
            report = oof.run_oof(
                manifest, raw, pool, final, pool, final, frozen,
                anchor_vod_ids=anchors, folds=folds, fold_train_vods=fold_train,
                coverage_gaps=gaps, ks=(5, 10, 20))

        self.assertEqual(gaps, ["Z"])
        self.assertEqual(report["coverage_gap_vods"], ["Z"])
        self.assertIn("Z", report["missing_prediction_vods"])  # gap, not a miss
        self.assertEqual(report["scored_vods"], ["V"])
        # One logged record per fold, each carrying the fold bound.
        self.assertEqual(len(records), 2)
        for record in records:
            self.assertEqual({"fold", "train_vods", "eval_vods"}, set(record))


class IdentityReproducesBaselineTests(unittest.TestCase):
    def test_oof_under_identity_matches_the_plain_composite_replay(self) -> None:
        # Under the identity stub the OOF dispatch is a no-op: each VOD is ranked by
        # a fold model that ignores training, so the pools + served sets are bit-
        # equal to the plain composite replay (the #49 build_inputs). That is what
        # makes the identity OOF reproduce the frozen baseline recall.
        vods = ["v0", "v1", "v2", "v3"]
        folds = oof.assign_folds(vods, k=2)
        with tempfile.TemporaryDirectory() as tmp:
            storage = pathlib.Path(tmp)
            _write_vods(storage, vods)
            _braw, base_pool, base_final, base_gaps = gap_decomposition.build_inputs(vods, storage)
            _oraw, oof_pool, oof_final, oof_gaps, _ft = oof.build_oof_inputs(
                vods, storage, folds, oof.identity_train_fn)

        self.assertEqual(oof_gaps, base_gaps)
        for v in vods:
            self.assertEqual(oof_final[v], base_final[v])
            self.assertEqual(oof_pool[v], base_pool[v])

    @unittest.skipUnless(
        _real_artifacts_present(),
        "real pipeline_debug artifacts / baseline not present (tmp/storage is gitignored)",
    )
    def test_identity_oof_reproduces_frozen_finder_recall_baseline(self) -> None:
        # AC#2 / the #47 skew + regression guard: the full identity OOF report over
        # the real artifacts must reproduce the committed baseline's product recall
        # (floor & headline @5/@10/@20) and leave every gate delta at exactly zero.
        manifest = eval_clip_finder.load_manifest()
        anchors = eval_clip_finder.read_anchor_vod_ids()
        frozen = json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))

        report = oof.build_report(anchors, _STORAGE_DIR, manifest, frozen)

        metrics = report["decomposition"]["metrics"]
        for k in ("5", "10", "20"):
            self.assertEqual(metrics["floor"]["recall"][k], frozen["floor"]["recall"][k])
            self.assertEqual(metrics["headline"]["recall"][k], frozen["headline"]["recall"][k])
        self.assertEqual(metrics["scored_vods"], frozen["scored_vods"])  # same 18 gold-scored
        self.assertEqual(report["missing_prediction_vods"], frozen["missing_prediction_vods"])
        # identity OOF == composite baseline → every gate delta is exactly zero.
        tm = report["two_metric_delta"]
        self.assertEqual(tm["ship_headline_recall"]["delta"], 0.0)
        self.assertEqual(tm["guardrail_floor_recall"]["delta"], 0.0)
        self.assertEqual(tm["engineering_clean_but_buried_recall"]["floor"]["delta"], 0.0)


class CliTests(unittest.TestCase):
    def test_main_writes_valid_two_metric_report_json_with_no_paid_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            manifest_path = tmp_path / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps({
                    "vod_id": "V",
                    "gt_moments": [{"start": 100.0, "end": 108.0, "views": 300_000, "gt_tier": "gold"}],
                }) + "\n", encoding="utf-8")
            anchor_csv = tmp_path / "anchors.csv"
            anchor_csv.write_text("vod_id\nV\n", encoding="utf-8")
            baseline_path = tmp_path / "baseline.json"
            # A directional_eval-shaped frozen baseline (serves nothing for V).
            base = directional_eval.run(
                {"V": [{"start": 100.0, "end": 108.0, "views": 300_000, "gt_tier": "gold"}]},
                {"V": []}, {"V": _two_moment_rows()}, anchor_vod_ids=["V"], ks=(5, 10, 20))
            baseline_path.write_text(json.dumps(base), encoding="utf-8")
            storage = tmp_path / "storage"
            storage.mkdir()
            _write_vods(storage, ["V"])
            out_path = tmp_path / "report.json"

            rc = oof.main([
                "--manifest", str(manifest_path),
                "--anchor-csv", str(anchor_csv),
                "--predictions-dir", str(storage),
                "--baseline", str(baseline_path),
                "--k", "2",
                "--output", str(out_path),
                "--json",
            ])

            self.assertEqual(rc, 0)
            self.assertTrue(out_path.exists())
            artifact = json.loads(out_path.read_text(encoding="utf-8"))
            for key in (
                "two_metric_delta", "decomposition", "clean_but_buried",
                "folds", "fold_train_vods", "coverage_gap_vods",
                "missing_prediction_vods", "scored_vods", "computed_at",
            ):
                self.assertIn(key, artifact)
            tm = artifact["two_metric_delta"]
            for metric in ("ship_headline_recall", "guardrail_floor_recall",
                           "engineering_clean_but_buried_recall", "avg_rank"):
                self.assertIn(metric, tm)


if __name__ == "__main__":
    unittest.main()
