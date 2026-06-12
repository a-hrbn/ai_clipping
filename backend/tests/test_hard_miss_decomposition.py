"""Tests for the hard_miss Root-Cause-Decomposition skeleton (ml/hard_miss_decomposition.py).

The thinnest tracer of the Generation-Track (PRD #56, Phase 1): it composes the
existing seams — `directional_eval` (raw@any / the measurement seam), the #49
`gap_decomposition` hard_miss set, and the `replay_harness` loaders (#47) — to
report the hard_miss *denominator* before any detector exists. In this slice it
classifies NOTHING: every hard_miss gold carries `bucket = "unclassified"`.

The decisive test is the reuse guard: the module's hard_miss set is the #49
hard_miss set (it sub-classifies an already-verified set, never re-classifies).
These tests pin that, the floor/headline split and the coverage gap through the
public interface on hand-built fixtures — never a network or paid call.
"""
from __future__ import annotations

import json
import os
import sys
import types

import pytest

# Defensive: building the deduped pool reaches the serving post-processor, which
# can pull the heavy transcription stack. Prior art (test_gap_decomposition)
# stubs whisper the same way so the import stays cheap in a bare test env.
sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_a, **_k: None))

from ml import eval_clip_finder, gap_decomposition as gd, hard_miss_decomposition as hmd


def _gold(start, end, views=0.0, desc=""):
    return {"start": float(start), "end": float(end), "views": float(views),
            "gt_tier": "gold", "desc": desc}


def _cand(start, end):
    return {"start_time": float(start), "end_time": float(end)}


# ── Hand-computed fixture spanning all three #49 buckets ─────────────────────
# A: one hard-miss (nothing covers) + one clean-but-buried (real cand covers).
# B: one accidental-only (only a 350s oversized window covers) + one hard-miss.
_HARD1 = _gold(200, 230, views=50_000, desc="A hard-miss")
_CLEAN = _gold(100, 130, views=300_000, desc="A clean-but-buried")
_ACC = _gold(400, 430, views=500_000, desc="B accidental-only")
_HARD2 = _gold(6000, 6030, views=80_000, desc="B hard-miss")

_RAW_A = [_cand(102, 122)]                      # covers _CLEAN only
_RAW_B = [_cand(300, 650)]                       # 350s window covers _ACC only


def _fixture():
    gold_by_vod = {"A": [_HARD1, _CLEAN], "B": [_ACC, _HARD2]}
    raw_by_vod = {"A": _RAW_A, "B": _RAW_B}
    pool_by_vod = {"A": _RAW_A, "B": _RAW_B}
    return gold_by_vod, raw_by_vod, pool_by_vod


# ─────────────────────────────────────────────────────────────────────────────
# Slice 1 — reuse guard: the hard_miss set is exactly #49's, each unclassified
# ─────────────────────────────────────────────────────────────────────────────

def test_hard_miss_golds_returns_only_the_49_hard_miss_set_unclassified() -> None:
    gold_by_vod, raw_by_vod, pool_by_vod = _fixture()

    golds = hmd.hard_miss_golds(gold_by_vod, raw_by_vod, pool_by_vod)

    # Only the two windows #49 classifies as hard_miss come back — the buried and
    # accidental golds (which #49 puts in other buckets) are excluded.
    windows = {(g["vod_id"], g["start"], g["end"]) for g in golds}
    assert windows == {("A", 200.0, 230.0), ("B", 6000.0, 6030.0)}
    # This slice classifies nothing: every hard_miss carries bucket="unclassified".
    assert all(g["bucket"] == hmd.UNCLASSIFIED for g in golds)
    assert {g["desc"] for g in golds} == {"A hard-miss", "B hard-miss"}


def test_hard_miss_set_is_bit_identical_to_gap_decomposition_classify_gold() -> None:
    # The reuse guard, stated directly against #49's classifier: the module's
    # hard_miss windows are exactly the golds gap_decomposition.classify_gold puts
    # in HARD_MISS — it sub-classifies that set, it never re-derives it.
    gold_by_vod, raw_by_vod, pool_by_vod = _fixture()

    expected = {
        (vod_id, g["start"], g["end"])
        for vod_id, moments in gold_by_vod.items()
        for g in moments
        if gd.classify_gold(g, raw_by_vod[vod_id], ranked_pool=pool_by_vod[vod_id])["bucket"]
        == gd.HARD_MISS
    }

    golds = hmd.hard_miss_golds(gold_by_vod, raw_by_vod, pool_by_vod)
    assert {(g["vod_id"], g["start"], g["end"]) for g in golds} == expected


# ─────────────────────────────────────────────────────────────────────────────
# Slice 2 — run() splits the hard_miss denominator into floor vs headline
# ─────────────────────────────────────────────────────────────────────────────

def test_run_splits_hard_miss_into_floor_and_headline_tiers() -> None:
    # Two hard-miss golds on one VOD: one below the headline floor (50k), one above
    # it (300k). Raw covers neither, so both are hard_miss at the floor; only the
    # 300k gold survives the headline view filter.
    manifest = {"A": [_gold(200, 230, views=50_000, desc="long-tail"),
                      _gold(800, 830, views=300_000, desc="headline")]}
    raw = {"A": [_cand(0, 20)]}                       # covers neither gold
    pool = {"A": [_cand(0, 20)]}

    result = hmd.run(manifest, raw, pool, anchor_vod_ids=["A"], headline_min_views=100_000)

    floor = result["hard_miss"]["floor"]
    headline = result["hard_miss"]["headline"]
    assert floor["count"] == 2
    assert {g["desc"] for g in floor["golds"]} == {"long-tail", "headline"}
    # The headline tier keeps only the high-view hard_miss — the routing weight sits
    # on the North-Star (headline), reported separately, with its count named.
    assert headline["count"] == 1
    assert [g["desc"] for g in headline["golds"]] == ["headline"]
    assert result["headline_min_views"] == 100_000


# ─────────────────────────────────────────────────────────────────────────────
# Slice 3 — coverage gap: no-prediction gold VODs are surfaced, never counted
# ─────────────────────────────────────────────────────────────────────────────

def test_run_surfaces_missing_prediction_vods_excluded_from_denominator() -> None:
    # A is scored (a hard-miss gold); GAP is gold but has no finder predictions —
    # a coverage gap, surfaced separately and dropped from the hard_miss
    # denominator, never folded in as a silent miss.
    manifest = {"A": [_gold(200, 230, views=300_000, desc="A hard-miss")],
                "GAP": [_gold(100, 130, views=900_000, desc="uncovered")]}
    raw = {"A": [_cand(0, 20)]}                       # GAP has no raw predictions
    pool = {"A": [_cand(0, 20)]}

    result = hmd.run(manifest, raw, pool, anchor_vod_ids=["A", "GAP"], headline_min_views=100_000)

    assert result["gold_vods"] == 2
    assert result["scored_vods"] == 1                 # GAP is not in the denominator
    assert result["missing_prediction_vods"] == ["GAP"]
    # Only A's hard-miss is counted; GAP's gold is never folded in as a miss.
    assert result["hard_miss"]["floor"]["count"] == 1
    assert [g["vod_id"] for g in result["hard_miss"]["floor"]["golds"]] == ["A"]


# ── Job-dir writer for the CLI test (the pipeline_debug_* replay layout) ──────
def _row(start, end, composite=8.0):
    virality = composite / 10.0
    return {"start_time": float(start), "end_time": float(end),
            "viral_score": composite, "composite_score": composite,
            "llm_virality": virality, "hook": virality, "flow": virality,
            "value": virality, "trend": virality, "reason": "row"}


def _write_job_dir(storage, vod_id, ts, *, raw, segments, mtime):
    debug = storage / f"pipeline_debug_{ts}_{vod_id}" / "debug"
    debug.mkdir(parents=True)
    (debug / "llm_moment_finder_moments_raw.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in raw), encoding="utf-8")
    (debug / "segments.jsonl").write_text(
        "".join(json.dumps(s) + "\n" for s in segments), encoding="utf-8")
    os.utime(debug.parent, (mtime, mtime))


# ─────────────────────────────────────────────────────────────────────────────
# Slice 4 — the CLI replays the real artifact layout and writes a valid report
# ─────────────────────────────────────────────────────────────────────────────

def test_main_reads_replay_artifacts_and_writes_valid_report(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps({
            "vod_id": "V",
            "gt_moments": [{"start": 500.0, "end": 530.0, "views": 300_000,
                            "gt_tier": "gold", "desc": "V hard-miss"}],
        }) + "\n", encoding="utf-8")
    anchor_csv = tmp_path / "anchors.csv"
    anchor_csv.write_text("vod_id\nV\n", encoding="utf-8")
    storage = tmp_path / "storage"
    storage.mkdir()
    # V's only raw moment sits at [0,20] — it never covers the gold at [500,530],
    # so the gold is a hard-miss (decided on the raw pool, reconciling with #49).
    _write_job_dir(
        storage, "V", "20260101T000000000000Z",
        raw=[_row(1.0, 19.0)],
        segments=[{"start": 0.0, "end": 20.0, "text": "A clean opening sentence here."}],
        mtime=1000.0)
    out_path = tmp_path / "report.json"

    rc = hmd.main([
        "--manifest", str(manifest_path),
        "--anchor-csv", str(anchor_csv),
        "--predictions-dir", str(storage),
        "--output", str(out_path),
        "--json",
    ])

    assert rc == 0
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    for key in ("gold_vods", "scored_vods", "missing_prediction_vods",
                "headline_min_views", "hard_miss", "computed_at"):
        assert key in artifact
    # V's single gold is a hard-miss, carried unclassified in this slice.
    floor = artifact["hard_miss"]["floor"]
    assert floor["count"] == 1
    assert floor["golds"][0]["desc"] == "V hard-miss"
    assert floor["golds"][0]["bucket"] == hmd.UNCLASSIFIED


# ─────────────────────────────────────────────────────────────────────────────
# Guard — reconcile the hard_miss set against #49 on the real artifacts
# ─────────────────────────────────────────────────────────────────────────────

_BASELINE_PATH = gd.DEFAULT_BASELINE
_STORAGE_DIR = eval_clip_finder.DEFAULT_STORAGE_DIR
_MISSING_PREDICTION_VODS = ["2756055866", "2764976309", "2770356205", "2771207666", "2773650615"]


def _real_artifacts_present() -> bool:
    """The (gitignored) pipeline_debug artifacts + the frozen baseline."""
    return _BASELINE_PATH.exists() and any(_STORAGE_DIR.glob("pipeline_debug_*"))


@pytest.mark.skipif(
    not _real_artifacts_present(),
    reason="real pipeline_debug artifacts / baseline not present (tmp/storage is gitignored)",
)
def test_reuse_guard_hard_miss_set_equals_49_on_real_artifacts() -> None:
    # The decisive reuse guard: replaying the real artifacts, the skeleton's
    # hard_miss set must be bit-identical to #49's — same count AND same windows,
    # both tiers — proving the decomposition sub-classifies an already-verified set
    # and never re-derives membership. Reuses one build so both seams see one input.
    manifest = eval_clip_finder.load_manifest()
    anchors = eval_clip_finder.read_anchor_vod_ids()
    baseline = json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))
    raw, pool, final, _gaps = gd.build_inputs(anchors, _STORAGE_DIR)

    result = hmd.run(manifest, raw, pool, anchor_vod_ids=anchors)
    gd_result = gd.run(manifest, raw, pool, final, baseline, anchor_vod_ids=anchors)

    # Count reconciliation — the documented PRD #56 denominator (floor 19, headline 9).
    for tier in ("floor", "headline"):
        assert (result["hard_miss"][tier]["count"]
                == gd_result["decomposition"][tier]["buckets"][gd.HARD_MISS])
    assert result["hard_miss"]["floor"]["count"] == 19
    assert result["hard_miss"]["headline"]["count"] == 9
    # hard_miss reconciles exactly with the frozen raw@any miss (gt − covered).
    gen = baseline["generation"]["floor"]
    assert result["hard_miss"]["floor"]["count"] == gen["gt_count"] - gen["covered"]

    # Window-level bit-identity: the skeleton's hard_miss windows are exactly the
    # golds gd.classify_gold puts in HARD_MISS over the same scored gold.
    gold = directional_gold(manifest, anchors)
    expected = {
        (vod_id, float(g["start"]), float(g["end"]))
        for vod_id, moments in gold.items() if raw.get(vod_id)
        for g in moments
        if gd.classify_gold(g, raw[vod_id], ranked_pool=pool.get(vod_id, []))["bucket"]
        == gd.HARD_MISS
    }
    assert {(g["vod_id"], g["start"], g["end"])
            for g in result["hard_miss"]["floor"]["golds"]} == expected

    # The 5 missing-prediction VODs stay a visible coverage gap, not misses.
    assert result["missing_prediction_vods"] == _MISSING_PREDICTION_VODS
    assert result["missing_prediction_vods"] == baseline["missing_prediction_vods"]
    # This slice classifies nothing: every hard_miss, both tiers, is unclassified.
    for tier in ("floor", "headline"):
        assert all(g["bucket"] == hmd.UNCLASSIFIED for g in result["hard_miss"][tier]["golds"])


def directional_gold(manifest, anchors):
    """The anchor-restricted gold manifest #49/#48 score over (test-local helper)."""
    from ml import directional_eval
    gold = directional_eval.gold_manifest(manifest)
    anchor_set = set(anchors)
    return {vod_id: moments for vod_id, moments in gold.items() if vod_id in anchor_set}
