"""Tests for the within-VOD pairwise numpy ranker + PU-dataset (ml/pairwise_ranker.py).

Slice A of PRD #51: a from-scratch PU dataset (training unit = a candidate in its
VOD's own deduped pool; positive = a #49 real-only gold-covering candidate; the rest
of the pool = unlabeled) and a within-VOD pairwise logistic ranker with explicit
interaction terms. The model plugs into the #52 OOF runner as a rank_score_fn and is
read against the pre-fixed A-Gate. These tests pin the contract through the public
interface with small hand-computed fixtures — no paid call, no network, no pickle.
"""
import json
import os
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

from ml import pairwise_ranker as pr

# build_oof_inputs reaches the serving post-processor, which can pull the heavy
# transcription stack. Stub whisper so the import stays cheap (mirrors the OOF /
# gap_decomposition tests) before importing the runner.
sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_a, **_k: None))

from ml import directional_eval  # noqa: E402
from ml import oof_eval_runner as oof  # noqa: E402


def _gold(start, end, views=0.0):
    return {"start": float(start), "end": float(end), "views": float(views), "gt_tier": "gold"}


def _cand(start, end, **extra):
    return {"start_time": float(start), "end_time": float(end), **extra}


def _scored(start, end, *, hook=0.0, flow=0.0, value=0.0, trend=0.0, **extra):
    return _cand(start, end, hook=hook, flow=flow, value=value, trend=trend, **extra)


# ── Replayable-job-dir fixtures (mirror test_oof_eval_runner) ──────────────────

def _segments() -> list[dict]:
    return [
        {"start": 0.0, "end": 4.0, "text": "Hello there everyone."},
        {"start": 4.0, "end": 8.0, "text": "This is a test."},
        {"start": 100.0, "end": 104.0, "text": "Another topic here."},
        {"start": 104.0, "end": 108.0, "text": "It continues now."},
    ]


def _finder_row(start, end, composite, *, reason) -> dict:
    virality = composite / 10.0
    return {
        "start_time": start, "end_time": end, "viral_score": composite,
        "composite_score": composite, "llm_virality": virality,
        "hook": virality, "flow": virality, "value": virality, "trend": virality,
        "reason": reason,
    }


def _two_moment_rows() -> list[dict]:
    # A has the higher composite (served first by identity); B is the gold-covering
    # one. Their feature signatures differ (A all-axes 0.8, B all-axes 0.7), so a
    # within-VOD ranker can separate them.
    return [_finder_row(0.0, 8.0, 8.0, reason="A"), _finder_row(100.0, 108.0, 7.0, reason="B")]


def _write_vods(storage: pathlib.Path, vod_ids) -> None:
    for i, vod_id in enumerate(vod_ids):
        job_dir = storage / f"pipeline_debug_20260101T000000000000Z_{vod_id}"
        debug = job_dir / "debug"
        debug.mkdir(parents=True)
        (debug / "llm_moment_finder_moments_raw.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in _two_moment_rows()), encoding="utf-8")
        (debug / "segments.jsonl").write_text(
            "".join(json.dumps(s) + "\n" for s in _segments()), encoding="utf-8")
        os.utime(job_dir, (1000.0 + i, 1000.0 + i))


# A 1-VOD pool exercising the #49 realness split. Only a clip-sized, gold-proportioned
# candidate that covers a gold is a positive; an oversized window that happens to cover
# (accidental) and a candidate that covers no gold are both unlabeled (PU).
_CBB_GOLD = _gold(100, 130, views=300_000)   # covered by a real cand (102,122)
_ACCIDENTAL_GOLD = _gold(400, 430)            # only an oversized window covers it
_REAL_CAND = _cand(102, 122)                  # 20s, 0.67x gold -> real, covers CBB gold
_OVERSIZED_CAND = _cand(300, 700)             # 400s window -> not real, covers accidental
_NONCOVER_CAND = _cand(0, 20)                 # covers no gold -> unlabeled


class LabelPoolTests(unittest.TestCase):
    def test_only_a_real_covering_candidate_is_a_positive(self) -> None:
        # The real, gold-proportioned candidate that covers a gold is the lone
        # positive; the oversized accidental cover and the non-covering sibling are
        # both unlabeled (PU). The positive carries the covered gold's views so the
        # headline up-weighting can read it later.
        gold = [_CBB_GOLD, _ACCIDENTAL_GOLD]
        pool = [_REAL_CAND, _OVERSIZED_CAND, _NONCOVER_CAND]

        labeled = pr.label_pool(gold, pool)

        labels = [c["label"] for c in labeled]
        self.assertEqual(labels, [pr.POSITIVE, pr.UNLABELED, pr.UNLABELED])
        self.assertEqual(labeled[0]["gold_views"], 300_000)
        self.assertIsNone(labeled[1]["gold_views"])


class BuildPuDatasetTests(unittest.TestCase):
    def test_headline_positive_is_up_weighted_over_a_floor_positive(self) -> None:
        # The rest of the pool is unlabeled (PU). A positive that covers a headline
        # gold (views >= the headline floor) is up-weighted; a positive covering only
        # a sub-headline gold keeps the base weight. The up-weight is strict.
        pool_by_vod = {"H": [_REAL_CAND, _NONCOVER_CAND], "F": [_REAL_CAND]}
        gold_by_vod = {"H": [_gold(100, 130, views=300_000)], "F": [_gold(100, 130, views=0)]}

        ds = pr.build_pu_dataset(pool_by_vod, gold_by_vod, headline_min_views=100_000)

        h_pos = ds["vods"]["H"][0]
        f_pos = ds["vods"]["F"][0]
        h_unlabeled = ds["vods"]["H"][1]
        self.assertEqual(h_pos["label"], pr.POSITIVE)
        self.assertEqual(h_unlabeled["label"], pr.UNLABELED)
        self.assertGreater(h_pos["weight"], f_pos["weight"])  # headline up-weighted
        self.assertEqual(f_pos["weight"], pr.BASE_WEIGHT)
        self.assertEqual(h_unlabeled["weight"], pr.BASE_WEIGHT)

    def test_vod_with_no_real_positive_is_train_dropped_but_eval_kept(self) -> None:
        # A VOD whose pool has no real covering candidate yields no positive: it is
        # dropped from the training VODs (no pairs can be formed) but its pool is
        # kept in the dataset so the eval denominator still sees it (train-drop,
        # eval-keep).
        pool_by_vod = {"P": [_REAL_CAND], "D": [_NONCOVER_CAND, _OVERSIZED_CAND]}
        gold_by_vod = {"P": [_CBB_GOLD], "D": [_gold(900, 930)]}  # D's gold uncovered

        ds = pr.build_pu_dataset(pool_by_vod, gold_by_vod)

        self.assertIn("P", ds["train_vods"])
        self.assertNotIn("D", ds["train_vods"])      # no positive -> not trainable
        self.assertIn("D", ds["train_dropped_vods"])
        self.assertIn("D", ds["vods"])               # eval-keep: pool retained
        self.assertTrue(all(c["label"] == pr.UNLABELED for c in ds["vods"]["D"]))


class WithinVodPairsTests(unittest.TestCase):
    def test_pairs_are_positive_over_sibling_within_the_same_vod_only(self) -> None:
        # Each positive is paired with each non-positive sibling in its OWN VOD; a
        # train-dropped VOD (no positive) contributes none, and no pair ever crosses
        # a VOD boundary — the comparison the issue fixes (positive > pool sibling).
        pool_by_vod = {
            "A": [_cand(102, 122, tag="A_pos"), _cand(0, 20, tag="A_u1"), _cand(500, 520, tag="A_u2")],
            "B": [_cand(102, 122, tag="B_pos"), _cand(0, 20, tag="B_u1")],
            "D": [_cand(0, 20, tag="D_u1")],  # no positive -> train-dropped
        }
        gold_by_vod = {"A": [_CBB_GOLD], "B": [_CBB_GOLD], "D": [_gold(900, 930)]}
        ds = pr.build_pu_dataset(pool_by_vod, gold_by_vod)

        pairs = pr.within_vod_pairs(ds)

        # A: 1 positive x 2 siblings = 2 pairs; B: 1 x 1 = 1; D: none. Total 3.
        self.assertEqual(len(pairs), 3)
        for pair in pairs:
            self.assertEqual(pair["pos"]["label"], pr.POSITIVE)
            self.assertEqual(pair["neg"]["label"], pr.UNLABELED)
            # both members carry the same VOD prefix tag -> never cross-VOD
            self.assertEqual(pair["pos"]["row"]["tag"][0], pair["neg"]["row"]["tag"][0])
            self.assertEqual(pair["vod_id"][0], pair["pos"]["row"]["tag"][0])
        self.assertNotIn("D", {p["vod_id"] for p in pairs})


class FeatureRowTests(unittest.TestCase):
    def test_feature_row_carries_base_features_and_explicit_interactions(self) -> None:
        # The feature vector is row-only (no segments at the scoring seam): the four
        # LLM axes + normalized duration, PLUS the explicit nonlinear interaction
        # terms the linear composite sweep could not represent — value x duration
        # bucket and value x trend. Hand-computed for a 30s clip (bucket 1).
        row = _scored(100, 130, hook=0.9, flow=0.6, value=0.8, trend=0.7)

        feats = pr.feature_row(row)

        self.assertEqual(set(feats), set(pr.FEATURES))
        self.assertEqual(feats["value"], 0.8)
        self.assertEqual(feats["dur_n"], 0.5)                 # 30s / 60
        self.assertEqual(pr.duration_bucket(30.0), 1)         # 20 <= 30 < 45
        self.assertEqual(feats["value_x_durbucket"], 0.8)     # value(0.8) * bucket(1)
        self.assertAlmostEqual(feats["value_x_trend"], 0.56)  # 0.8 * 0.7
        self.assertIn("value_x_durbucket", pr.FEATURES)
        self.assertIsInstance(pr.FEATURE_BUILD_VERSION, int)


def _pair(pos_row, neg_row, weight=1.0):
    return {"vod_id": "X", "pos": {"row": pos_row}, "neg": {"row": neg_row}, "weight": weight}


class TrainPairwiseLogisticTests(unittest.TestCase):
    def _separable_pairs(self):
        # Two VODs, each a positive with high `value` over a sibling with low `value`,
        # everything else equal — a separable within-VOD comparison the model must
        # learn to order.
        return [
            _pair(_scored(100, 130, value=0.9, hook=0.5, flow=0.5, trend=0.5),
                  _scored(100, 130, value=0.1, hook=0.5, flow=0.5, trend=0.5)),
            _pair(_scored(100, 130, value=0.8, hook=0.4, flow=0.4, trend=0.4),
                  _scored(100, 130, value=0.2, hook=0.4, flow=0.4, trend=0.4)),
        ]

    def test_training_orders_the_separable_positive_above_its_sibling(self) -> None:
        # After fitting, every training positive must score strictly above its paired
        # sibling, and the weight vector is finite and aligned to FEATURES.
        pairs = self._separable_pairs()

        weights = pr.train_pairwise_logistic(pairs, l1=0.0, epochs=400)

        self.assertEqual(len(weights), len(pr.FEATURES))
        self.assertTrue(all(abs(w) < 1e6 for w in weights))
        for pair in pairs:
            pos_s = pr.score_features(weights, pair["pos"]["row"])
            neg_s = pr.score_features(weights, pair["neg"]["row"])
            self.assertGreater(pos_s, neg_s)

    def test_training_is_deterministic(self) -> None:
        # Zeros-init full-batch descent — no seed needed; two fits agree bit-for-bit.
        pairs = self._separable_pairs()
        a = pr.train_pairwise_logistic(pairs, l1=0.01, epochs=200)
        b = pr.train_pairwise_logistic(pairs, l1=0.01, epochs=200)
        self.assertEqual(a, b)

    def test_l1_shrinks_the_weight_vector(self) -> None:
        # L1 regularisation pulls the weight L1-norm down vs the unregularised fit.
        pairs = self._separable_pairs()
        unreg = pr.train_pairwise_logistic(pairs, l1=0.0, epochs=400)
        reg = pr.train_pairwise_logistic(pairs, l1=0.2, epochs=400)
        self.assertLess(sum(abs(w) for w in reg), sum(abs(w) for w in unreg))


class PairwiseModelArtifactTests(unittest.TestCase):
    def test_artifact_is_diffable_json_and_round_trips_scores(self) -> None:
        # The artifact is a plain JSON weight vector + feature list +
        # feature_build_version (no pickle/joblib): json.dumps must succeed, the keys
        # are pinned, and reloading from the parsed JSON reproduces the scores exactly.
        weights = [0.1, 0.0, 0.7, 0.0, 0.0, 0.5, 0.0]
        model = pr.PairwiseModel(weights=weights)
        row = _scored(100, 130, hook=0.9, flow=0.6, value=0.8, trend=0.7)

        blob = json.dumps(model.to_dict())  # must be JSON-native (would raise on pickle types)
        reloaded = pr.PairwiseModel.from_dict(json.loads(blob))

        as_dict = model.to_dict()
        self.assertEqual(set(as_dict), {"feature_build_version", "features", "weights"})
        self.assertEqual(as_dict["features"], list(pr.FEATURES))
        self.assertEqual(as_dict["feature_build_version"], pr.FEATURE_BUILD_VERSION)
        self.assertEqual(reloaded.score(row), model.score(row))
        self.assertEqual(model.score(row), pr.score_features(weights, row))


class InteractionChangesRankingTests(unittest.TestCase):
    def test_nonzero_interaction_weight_flips_the_order_deterministically(self) -> None:
        # The headline acceptance: an interaction term with weight != 0 changes the
        # ranking. Row A has the higher `value`; B has a lower value but a longer
        # duration (bucket 2). A pure-`value` (linear) model ranks A over B; switching
        # ON the value_x_durbucket weight flips it to B over A — a move the linear
        # composite sweep could never make. Same input -> same output (deterministic).
        idx = {name: i for i, name in enumerate(pr.FEATURES)}
        w_linear = [0.0] * len(pr.FEATURES)
        w_linear[idx["value"]] = 1.0
        w_interaction = list(w_linear)
        w_interaction[idx["value_x_durbucket"]] = 1.0

        row_a = _scored(100, 110, value=0.50)   # 10s -> bucket 0
        row_b = _scored(100, 150, value=0.45)   # 50s -> bucket 2
        rows = [row_a, row_b]

        linear = pr.rank_score_fn(pr.PairwiseModel(weights=w_linear))(rows)
        inter = pr.rank_score_fn(pr.PairwiseModel(weights=w_interaction))(rows)
        inter_again = pr.rank_score_fn(pr.PairwiseModel(weights=w_interaction))(rows)

        # linear: A scores above B; interaction: the order flips to B above A.
        self.assertGreater(linear[0]["composite_score"], linear[1]["composite_score"])
        self.assertLess(inter[0]["composite_score"], inter[1]["composite_score"])
        # deterministic, and the input rows are not mutated in place
        self.assertEqual([r["composite_score"] for r in inter],
                         [r["composite_score"] for r in inter_again])
        self.assertNotIn("composite_score", row_a)


class FakeAdapterTests(unittest.TestCase):
    def test_fake_train_fn_is_canned_deterministic_and_ignores_training(self) -> None:
        # The Fake adapter (Two-Adapter discipline): a #52 TrainFn whose rank_score_fn
        # is canned — no network, no disk, no fit. It ignores the training VOD ids, so
        # different (or empty) training sets produce identical scores, run-to-run.
        rows = [_scored(100, 130, value=0.8, trend=0.5), _scored(0, 10, value=0.2, trend=0.5)]

        out_a = pr.fake_train_fn(["v0", "v1"])([dict(r) for r in rows])
        out_b = pr.fake_train_fn([])([dict(r) for r in rows])
        out_a2 = pr.fake_train_fn(["x"])([dict(r) for r in rows])

        scores_a = [r["composite_score"] for r in out_a]
        self.assertEqual(scores_a, [r["composite_score"] for r in out_b])    # canned
        self.assertEqual(scores_a, [r["composite_score"] for r in out_a2])   # deterministic
        self.assertGreater(out_a[0]["composite_score"], out_a[1]["composite_score"])  # really scores


class RealAdapterOofIntegrationTests(unittest.TestCase):
    def test_real_train_fn_plugs_into_oof_and_moves_the_served_order(self) -> None:
        # The real adapter, used as the #52 TrainFn: it replays each fold's training
        # VODs, builds the from-scratch PU dataset, fits the within-VOD pairwise model
        # and returns a rank_score_fn. Two VODs, k=2 (group=VOD) → each VOD is ranked
        # by the model trained on the OTHER (leakage-free). Identity serves the higher-
        # composite moment A; the trained ranker — taught that the gold-covering B
        # should outrank its sibling — flips the served top-1 to B.
        vods = ["v0", "v1"]
        folds = oof.assign_folds(vods, k=2)
        manifest = {v: [_gold(100, 108, views=300_000)] for v in vods}  # gold covers B
        with tempfile.TemporaryDirectory() as tmp:
            storage = pathlib.Path(tmp)
            _write_vods(storage, vods)
            train_fn = pr.make_oof_train_fn(storage, manifest)

            id_raw, id_pool, id_final, _g, _f = oof.build_oof_inputs(
                vods, storage, folds, oof.identity_train_fn, top_k=1)
            re_raw, re_pool, re_final, _g2, _f2 = oof.build_oof_inputs(
                vods, storage, folds, train_fn, top_k=1)

        for v in vods:
            self.assertEqual(id_final[v][0]["reason"], "A")   # identity serves A
            self.assertEqual(re_final[v][0]["reason"], "B")   # trained ranker flips to B
        # And the move is visible on the engineering metric: the clean gold (covered
        # only by B) goes from buried (0) to served (1.0) at K=1.
        gold = {v: [_gold(100, 108, views=300_000)] for v in vods}
        id_cbb = oof.clean_but_buried_recall(gold, id_raw, id_pool, id_final, ks=(20,))
        re_cbb = oof.clean_but_buried_recall(gold, re_raw, re_pool, re_final, ks=(20,))
        self.assertEqual(id_cbb["recall"]["20"], 0.0)
        self.assertEqual(re_cbb["recall"]["20"], 1.0)


def _tm(cbb_floor, guardrail_floor, avg_rank_floor):
    """A minimal #52 two_metric_delta with just the cells the A-Gate reads."""
    return {
        "k": 20,
        "engineering_clean_but_buried_recall": {"floor": {"delta": cbb_floor}, "headline": {"delta": None}},
        "guardrail_floor_recall": {"delta": guardrail_floor},
        "avg_rank": {"floor": {"delta": avg_rank_floor}, "headline": {"delta": None}},
    }


class AGateTests(unittest.TestCase):
    def test_ship_when_cbb_rises_by_a_majority_floor_holds_avg_rank_ok(self) -> None:
        # The pre-fixed ship rule, all three conditions met: clean_but_buried recall@20
        # rises by a majority of the re-rankable set (> 0.5), floor recall@20 does not
        # fall, and avg_rank does not explode.
        gate = pr.evaluate_a_gate(_tm(cbb_floor=0.6, guardrail_floor=0.1, avg_rank_floor=-2.0))

        self.assertEqual(gate["verdict"], pr.SHIP)
        self.assertTrue(gate["ship"])
        self.assertIn("clean_but_buried", gate["reason"])

    def test_probe_when_cbb_lift_is_flat(self) -> None:
        # A sub-majority cbb lift (the "flat" case) → Probe, regardless of guardrails.
        gate = pr.evaluate_a_gate(_tm(cbb_floor=0.10, guardrail_floor=0.2, avg_rank_floor=-5.0))
        self.assertEqual(gate["verdict"], pr.PROBE)
        self.assertFalse(gate["conditions"]["cbb_majority"])

    def test_probe_when_floor_recall_falls(self) -> None:
        # A strong cbb lift that costs floor recall@20 is vetoed by the guardrail.
        gate = pr.evaluate_a_gate(_tm(cbb_floor=0.6, guardrail_floor=-0.05, avg_rank_floor=0.0))
        self.assertEqual(gate["verdict"], pr.PROBE)
        self.assertFalse(gate["conditions"]["floor_not_falling"])

    def test_probe_when_avg_rank_explodes(self) -> None:
        # A strong cbb lift that explodes avg_rank is vetoed by the guardrail.
        gate = pr.evaluate_a_gate(
            _tm(cbb_floor=0.6, guardrail_floor=0.0, avg_rank_floor=50.0),
            max_avg_rank_increase=5.0)
        self.assertEqual(gate["verdict"], pr.PROBE)
        self.assertFalse(gate["conditions"]["avg_rank_not_exploding"])


class FromScratchTests(unittest.TestCase):
    def test_dataset_and_training_are_pure_in_memory_no_disk_reads(self) -> None:
        # The from-scratch guarantee (AC#3): building the PU dataset and fitting the
        # model touch NO file — so they cannot be reading the circular
        # jynxzi_training_rows.jsonl. Patch open/read_text to blow up; the in-memory
        # build off the supplied pools must still complete.
        pool_by_vod = {"V": [
            _scored(102, 122, value=0.9, hook=0.5, flow=0.5, trend=0.5),  # covers CBB gold -> positive
            _scored(0, 20, value=0.1, hook=0.5, flow=0.5, trend=0.5),     # unlabeled sibling
        ]}
        gold_by_vod = {"V": [_CBB_GOLD]}

        def boom(*_a, **_k):
            raise AssertionError("no file I/O allowed during the from-scratch build")

        with mock.patch("builtins.open", boom), mock.patch("pathlib.Path.read_text", boom):
            model = pr.train_model_from_pools(pool_by_vod, gold_by_vod)

        self.assertEqual(len(model.weights), len(pr.FEATURES))

    def test_circular_corpus_is_named_as_a_deprecated_non_source(self) -> None:
        # The circular corpus is explicitly recorded as NOT a training source, and a
        # deprecation marker sits next to it for anyone browsing the artifacts.
        self.assertEqual(pr.DEPRECATED_CIRCULAR_TRAINING_ROWS, "jynxzi_training_rows.jsonl")
        marker = (pathlib.Path(pr.__file__).resolve().parent / "artifacts"
                  / "jynxzi_training_rows.jsonl.DEPRECATED.md")
        self.assertTrue(marker.exists(), "expected a deprecation marker beside the circular corpus")


class RunSliceATests(unittest.TestCase):
    def test_run_slice_a_drives_oof_and_attaches_the_gate_verdict(self) -> None:
        # The deliverable end-to-end: run_slice_a fits the real ranker, drives the #52
        # OOF runner with it, and attaches the mechanically-read A-Gate. The report
        # carries the two-metric delta AND an a_gate verdict — no paid call.
        vods = ["v0", "v1"]
        manifest = {v: [_gold(100, 108, views=300_000)] for v in vods}
        with tempfile.TemporaryDirectory() as tmp:
            storage = pathlib.Path(tmp)
            _write_vods(storage, vods)
            # Frozen composite baseline: identity serves A, so the clean gold (B) is
            # buried — the ranker has room to move it.
            baseline = directional_eval.run(
                manifest, {v: [] for v in vods},
                {v: _two_moment_rows() for v in vods},
                anchor_vod_ids=vods, ks=(5, 10, 20), headline_min_views=100_000)

            # top_k=1 forces burial: identity serves only A, so the clean gold B is
            # buried and the ranker has room to move it.
            report = pr.run_slice_a(vods, storage, manifest, baseline, k=2, top_k=1)

        self.assertIn("two_metric_delta", report)
        self.assertIn("a_gate", report)
        self.assertIn(report["a_gate"]["verdict"], (pr.SHIP, pr.PROBE))
        self.assertIn("reason", report["a_gate"])
        # The ranker lifts the buried clean gold into the served set -> engineering
        # cbb recall@20 moves up off the identity baseline.
        eng = report["two_metric_delta"]["engineering_clean_but_buried_recall"]["floor"]
        self.assertGreater(eng["current"], eng["baseline"])


class CliTests(unittest.TestCase):
    def test_main_writes_a_report_json_with_the_gate_and_no_paid_call(self) -> None:
        vods = ["v0", "v1"]
        manifest_rows = {v: [_gold(100, 108, views=300_000)] for v in vods}
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            manifest_path = tmp_path / "manifest.jsonl"
            manifest_path.write_text(
                "".join(json.dumps({"vod_id": v, "gt_moments": manifest_rows[v]}) + "\n" for v in vods),
                encoding="utf-8")
            anchor_csv = tmp_path / "anchors.csv"
            anchor_csv.write_text("vod_id\n" + "\n".join(vods) + "\n", encoding="utf-8")
            baseline = directional_eval.run(
                manifest_rows, {v: [] for v in vods},
                {v: _two_moment_rows() for v in vods},
                anchor_vod_ids=vods, ks=(5, 10, 20), headline_min_views=100_000)
            baseline_path = tmp_path / "baseline.json"
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            storage = tmp_path / "storage"
            storage.mkdir()
            _write_vods(storage, vods)
            out_path = tmp_path / "report.json"

            rc = pr.main([
                "--manifest", str(manifest_path), "--anchor-csv", str(anchor_csv),
                "--predictions-dir", str(storage), "--baseline", str(baseline_path),
                "--k", "2", "--top-k", "1", "--output", str(out_path), "--json",
            ])

            self.assertEqual(rc, 0)
            artifact = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertIn("a_gate", artifact)
            self.assertIn(artifact["a_gate"]["verdict"], (pr.SHIP, pr.PROBE))
            for key in ("two_metric_delta", "decomposition", "model", "computed_at"):
                self.assertIn(key, artifact)


if __name__ == "__main__":
    unittest.main()
