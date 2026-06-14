"""Tests for the WALL-Re-Test verdict/compare harness (ml/wall_retest_compare.py).

Slice 1 of PRD #90: a deterministic harness that compares two prediction dirs
(base vs neu) and reads the *pre-fixed* verdict off the recall@20 movement —
``>= 0.25 WALL_FALLS`` / ``<= 0.15 ESCALATE_PASS2`` / between ``INCONCLUSIVE``.
The bands were calibrated against the headline tier (shipped recall@20 = 0.125,
ceiling raw@any = 0.625), so the verdict keys on the headline movement; floor is
reported alongside. raw@any (generation lift) + clean_but_buried (sorting lift)
ride along for both dirs.

These tests pin the verdict bands + the diff through the public interface on
hand-built numbers, and the measurement on canned pipeline_debug artifacts —
never a network or paid call.
"""
from __future__ import annotations

import sys
import types

# Defensive: measuring reaches the serving post-processor (replay), which can
# pull the heavy transcription stack. Prior art stubs whisper the same way so the
# import stays cheap in a bare test env.
sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_a, **_k: None))

import json

import pathlib

import pytest

from ml import eval_clip_finder, wall_retest_compare as wrc


# ─────────────────────────────────────────────────────────────────────────────
# Slice 1 — verdict_for_movement: the pre-fixed bands, boundaries included
# ─────────────────────────────────────────────────────────────────────────────

def test_movement_at_or_above_025_is_wall_falls() -> None:
    # A recall@20 jump of >= 0.25 means the cheap lever recovered the signal —
    # the WALL falls. Pre-fixed in code, read off the movement, never rationalized.
    assert wrc.verdict_for_movement(0.30) == wrc.WALL_FALLS


def test_movement_at_or_below_015_is_escalate_pass2() -> None:
    # A movement of <= 0.15 means even the best-shot lever barely moved recall —
    # escalate to pass 2 (the WALL holds for now). base-vs-base lands here (0.0).
    assert wrc.verdict_for_movement(0.10) == wrc.ESCALATE_PASS2
    assert wrc.verdict_for_movement(0.0) == wrc.ESCALATE_PASS2


def test_movement_strictly_between_the_bands_is_inconclusive() -> None:
    # 0.15 < movement < 0.25: the lever moved recall, but not past the WALL_FALLS
    # bar — neither a clean recovery nor a clean failure.
    assert wrc.verdict_for_movement(0.20) == wrc.INCONCLUSIVE


def test_lower_boundary_015_is_escalate_not_inconclusive() -> None:
    # The band is closed at 0.15: exactly 0.15 escalates (<= 0.15), it does not
    # fall into the inconclusive middle. The pre-fixed boundary, pinned.
    assert wrc.verdict_for_movement(0.15) == wrc.ESCALATE_PASS2


def test_upper_boundary_025_is_wall_falls_not_inconclusive() -> None:
    # The band is closed at 0.25: exactly 0.25 fells the WALL (>= 0.25), it does
    # not fall into the inconclusive middle. The pre-fixed boundary, pinned.
    assert wrc.verdict_for_movement(0.25) == wrc.WALL_FALLS


# ─────────────────────────────────────────────────────────────────────────────
# Slice 2 — compare(base, neu): the headline recall@20 movement → verdict, with
# raw@any (generation lift) + clean_but_buried (sorting lift) carried for both.
# ─────────────────────────────────────────────────────────────────────────────

def _measure(*, r_floor, r_head, raw_floor, raw_head, cbb_floor, cbb_head):
    """A hand-built per-dir measure (the shape measure_dir returns)."""
    return {
        "recall_at_20": {"floor": r_floor, "headline": r_head},
        "raw_any": {"floor": raw_floor, "headline": raw_head},
        "clean_but_buried": {"floor": cbb_floor, "headline": cbb_head},
    }


def test_compare_keys_verdict_on_headline_recall20_movement() -> None:
    # base headline recall@20 = 0.125 (the shipped baseline), neu = 0.40 → a
    # +0.275 headline movement → WALL_FALLS. The movement that drove it is surfaced.
    base = _measure(r_floor=0.12, r_head=0.125, raw_floor=0.6, raw_head=0.625,
                    cbb_floor=0.2, cbb_head=0.2)
    neu = _measure(r_floor=0.30, r_head=0.40, raw_floor=0.7, raw_head=0.75,
                   cbb_floor=0.3, cbb_head=0.35)

    result = wrc.compare(base, neu)

    assert result["verdict"] == wrc.WALL_FALLS
    assert result["verdict_tier"] == "headline"
    assert result["movement"] == 0.275
    assert result["recall_at_20"]["headline"] == {"base": 0.125, "neu": 0.40, "lift": 0.275}


def test_compare_carries_generation_and_sorting_lifts_for_both_tiers() -> None:
    # raw@any is the generation lift (did neu surface more gold at any rank);
    # clean_but_buried is the sorting lift (more real-but-buried gold). Both are
    # always reported, both tiers, both dirs — independent of the verdict.
    base = _measure(r_floor=0.12, r_head=0.125, raw_floor=0.6, raw_head=0.625,
                    cbb_floor=0.20, cbb_head=0.10)
    neu = _measure(r_floor=0.30, r_head=0.40, raw_floor=0.72, raw_head=0.75,
                   cbb_floor=0.35, cbb_head=0.30)

    result = wrc.compare(base, neu)

    # generation lift (raw@any)
    assert result["raw_any"]["floor"] == {"base": 0.6, "neu": 0.72, "lift": 0.12}
    assert result["raw_any"]["headline"] == {"base": 0.625, "neu": 0.75, "lift": 0.125}
    # sorting lift (clean_but_buried share)
    assert result["clean_but_buried"]["floor"] == {"base": 0.20, "neu": 0.35, "lift": 0.15}
    assert result["clean_but_buried"]["headline"] == {"base": 0.10, "neu": 0.30, "lift": 0.20}
    # the floor recall@20 rides along even though the headline movement drives
    assert result["recall_at_20"]["floor"] == {"base": 0.12, "neu": 0.30, "lift": 0.18}


def test_compare_identical_measures_is_zero_movement_escalate_pass2() -> None:
    # base-vs-base: identical measures → 0.0 headline movement → ESCALATE_PASS2
    # ("WALL hält"). The deterministic null comparison the CLI demo exercises.
    m = _measure(r_floor=0.12, r_head=0.125, raw_floor=0.6, raw_head=0.625,
                 cbb_floor=0.2, cbb_head=0.2)

    result = wrc.compare(m, m)

    assert result["movement"] == 0.0
    assert result["verdict"] == wrc.ESCALATE_PASS2
    assert result["raw_any"]["headline"]["lift"] == 0.0
    assert result["clean_but_buried"]["headline"]["lift"] == 0.0


def test_compare_none_headline_recall_is_inconclusive() -> None:
    # A subset with no headline gold on either side → headline recall@20 is None →
    # the movement can't be computed → INCONCLUSIVE (never silently ESCALATE).
    base = _measure(r_floor=0.12, r_head=None, raw_floor=0.6, raw_head=None,
                    cbb_floor=0.2, cbb_head=None)
    neu = _measure(r_floor=0.40, r_head=None, raw_floor=0.7, raw_head=None,
                   cbb_floor=0.3, cbb_head=None)

    result = wrc.compare(base, neu)

    assert result["movement"] is None
    assert result["verdict"] == wrc.INCONCLUSIVE
    # the floor lift is still carried for transparency
    assert result["recall_at_20"]["floor"]["lift"] == 0.28


# ─────────────────────────────────────────────────────────────────────────────
# Slice 3 — measure_dir: replay one prediction dir through the real eval seams
# (gap_decomposition.build_inputs → directional_eval.run → decompose_tier) and
# extract recall@20 + raw@any + clean_but_buried per tier. Canned artifacts only.
# ─────────────────────────────────────────────────────────────────────────────

def _row(start, end, composite, *, reason="r"):
    """One raw finder moment row (the eval_recall post-processor row form)."""
    virality = composite / 10.0
    return {
        "start_time": float(start), "end_time": float(end),
        "viral_score": composite, "composite_score": composite,
        "llm_virality": virality, "hook": virality, "flow": virality,
        "value": virality, "trend": virality, "reason": reason,
    }


def _mgold(start, end, views, desc=""):
    return {"start": float(start), "end": float(end), "views": float(views),
            "gt_tier": "gold", "desc": desc}


def _write_job_dir(storage, vod_id, ts, *, raw, segments, mtime):
    debug = storage / f"pipeline_debug_{ts}_{vod_id}" / "debug"
    debug.mkdir(parents=True)
    (debug / "llm_moment_finder_moments_raw.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in raw), encoding="utf-8")
    (debug / "segments.jsonl").write_text(
        "".join(json.dumps(s) + "\n" for s in segments), encoding="utf-8")
    import os
    os.utime(debug.parent, (mtime, mtime))


def test_measure_dir_extracts_recall20_raw_any_and_clean_but_buried(tmp_path) -> None:
    # One VOD, two golds: A [100,130] (300k, headline) is covered by a real
    # clip-sized candidate that survives the serving postprocess into the top-20;
    # B [5000,5030] (50k, floor-only) has no covering candidate at all → hard-miss.
    # Floor (both golds): recall@20 0.5, raw@any 0.5, clean_but_buried 0.5.
    # Headline (A only):  recall@20 1.0, raw@any 1.0, clean_but_buried 1.0.
    storage = tmp_path / "storage"
    storage.mkdir()
    _write_job_dir(
        storage, "V", "20260101T000000000000Z",
        raw=[_row(102, 122, 9.0, reason="covers A")],   # nothing covers B
        segments=[
            {"start": 100.0, "end": 130.0, "text": "the gold A window speech here"},
            {"start": 5000.0, "end": 5030.0, "text": "some unrelated later speech"},
        ],
        mtime=1000.0)
    manifest = {"V": [_mgold(100, 130, 300_000, "A"), _mgold(5000, 5030, 50_000, "B")]}

    measure = wrc.measure_dir(["V"], storage, manifest)

    assert measure["recall_at_20"] == {"floor": 0.5, "headline": 1.0}
    assert measure["raw_any"] == {"floor": 0.5, "headline": 1.0}
    assert measure["clean_but_buried"] == {"floor": 0.5, "headline": 1.0}


# ─────────────────────────────────────────────────────────────────────────────
# Slice 4 — the CLI: base-dir vs neu-dir → --json verdict + per-dir decomposition.
# base stays untouched (its own dir); neu in --predictions-dir.
# ─────────────────────────────────────────────────────────────────────────────

def _write_manifest(path, vod_id, golds):
    path.write_text(
        json.dumps({"vod_id": vod_id, "gt_moments": golds}) + "\n", encoding="utf-8")


def test_main_base_vs_base_is_zero_movement_escalate_pass2(tmp_path) -> None:
    # The demo: base-vs-base on the SAME dir → identical measures → 0.0 headline
    # movement → ESCALATE_PASS2 ("WALL hält"). Deterministic, net-free.
    storage = tmp_path / "storage"
    storage.mkdir()
    _write_job_dir(
        storage, "V", "20260101T000000000000Z",
        raw=[_row(102, 122, 9.0, reason="covers A")],
        segments=[{"start": 100.0, "end": 130.0, "text": "the gold A window speech here"}],
        mtime=1000.0)
    manifest_path = tmp_path / "manifest.jsonl"
    _write_manifest(manifest_path, "V", [_mgold(100, 130, 300_000, "A")])
    anchor_csv = tmp_path / "anchors.csv"
    anchor_csv.write_text("vod_id\nV\n", encoding="utf-8")
    out_path = tmp_path / "report.json"

    rc = wrc.main([
        "--base-dir", str(storage), "--predictions-dir", str(storage),
        "--manifest", str(manifest_path), "--anchor-csv", str(anchor_csv),
        "--output", str(out_path), "--json",
    ])

    assert rc == 0
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["movement"] == 0.0
    assert report["verdict"] == wrc.ESCALATE_PASS2
    # both dirs' decompositions are carried, raw@any + clean_but_buried equal
    assert report["base"]["recall_at_20"]["headline"] == 1.0
    assert report["neu"]["recall_at_20"]["headline"] == 1.0
    assert report["raw_any"]["headline"]["lift"] == 0.0
    assert "computed_at" in report


def test_main_neu_recovers_uncovered_gold_is_wall_falls(tmp_path) -> None:
    # base's finder never covers gold A (hard-miss → headline recall@20 0.0); the
    # neu dir (its own --predictions-dir, base untouched) covers it → 1.0. The
    # +1.0 headline movement >= 0.25 → WALL_FALLS, read straight off the two dirs.
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    _write_job_dir(
        base_dir, "V", "20260101T000000000000Z",
        raw=[_row(7000, 7020, 8.0, reason="misses A")],   # nowhere near gold A
        segments=[{"start": 100.0, "end": 130.0, "text": "the gold A window speech here"},
                  {"start": 7000.0, "end": 7020.0, "text": "some unrelated chatter"}],
        mtime=1000.0)
    neu_dir = tmp_path / "neu"
    neu_dir.mkdir()
    _write_job_dir(
        neu_dir, "V", "20260101T000000000000Z",
        raw=[_row(102, 122, 9.0, reason="covers A")],
        segments=[{"start": 100.0, "end": 130.0, "text": "the gold A window speech here"}],
        mtime=1000.0)
    manifest_path = tmp_path / "manifest.jsonl"
    _write_manifest(manifest_path, "V", [_mgold(100, 130, 300_000, "A")])
    anchor_csv = tmp_path / "anchors.csv"
    anchor_csv.write_text("vod_id\nV\n", encoding="utf-8")
    out_path = tmp_path / "report.json"

    rc = wrc.main([
        "--base-dir", str(base_dir), "--predictions-dir", str(neu_dir),
        "--manifest", str(manifest_path), "--anchor-csv", str(anchor_csv),
        "--output", str(out_path), "--json",
    ])

    assert rc == 0
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["base"]["recall_at_20"]["headline"] == 0.0
    assert report["neu"]["recall_at_20"]["headline"] == 1.0
    assert report["movement"] == 1.0
    assert report["verdict"] == wrc.WALL_FALLS
    # generation lift: base never generated a covering candidate, neu did
    assert report["raw_any"]["headline"] == {"base": 0.0, "neu": 1.0, "lift": 1.0}


# ─────────────────────────────────────────────────────────────────────────────
# Demoable — base-vs-base on the real pipeline_debug_* artifacts → diff 0 → the
# deterministic null verdict (WALL hält). tmp/storage is gitignored, so the test
# self-skips where the artifacts are absent (mirrors the prior-art guards).
# ─────────────────────────────────────────────────────────────────────────────

_STORAGE_DIR = eval_clip_finder.DEFAULT_STORAGE_DIR
_MANIFEST_PATH = eval_clip_finder.DEFAULT_MANIFEST


def _real_artifacts_present() -> bool:
    return _MANIFEST_PATH.exists() and any(_STORAGE_DIR.glob("pipeline_debug_*"))


@pytest.mark.skipif(
    not _real_artifacts_present(),
    reason="no real pipeline_debug_* artifacts (tmp/storage is gitignored)",
)
def test_base_vs_base_on_real_artifacts_is_zero_movement_escalate_pass2() -> None:
    # Replaying the SAME real dir as both base and neu is the deterministic null
    # comparison: identical measures → 0.0 movement on every tier → ESCALATE_PASS2
    # ("WALL hält"). Proves the harness composes the real seams end-to-end and that
    # the verdict floors at the status quo when nothing changed.
    manifest = eval_clip_finder.load_manifest()
    anchors = eval_clip_finder.read_anchor_vod_ids()

    base = wrc.measure_dir(anchors, _STORAGE_DIR, manifest)
    neu = wrc.measure_dir(anchors, _STORAGE_DIR, manifest)
    result = wrc.compare(base, neu)

    assert base["recall_at_20"] == neu["recall_at_20"]
    assert base["raw_any"] == neu["raw_any"]
    assert base["clean_but_buried"] == neu["clean_but_buried"]
    assert result["movement"] == 0.0
    assert result["verdict"] == wrc.ESCALATE_PASS2
    # every reported lift is a clean zero — base is untouched, neu == base here
    for metric in ("recall_at_20", "raw_any", "clean_but_buried"):
        for tier in ("floor", "headline"):
            assert result[metric][tier]["lift"] in (0.0, None)
