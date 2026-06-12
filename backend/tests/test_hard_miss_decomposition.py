"""Tests for the hard_miss Root-Cause-Decomposition (ml/hard_miss_decomposition.py).

A read-only tracer of the Generation-Track (PRD #56, Phase 1): it composes the
existing seams — `directional_eval` (raw@any / the measurement seam), the #49
`gap_decomposition` hard_miss set, and the `replay_harness` loaders (#47) — and
routes each hard_miss gold into one priority-resolved root-cause bucket
(no-transcript → truncation → llm-skipped).

Two guards anchor it: the reuse guard pins that the module's hard_miss set is the
#49 set (it sub-classifies an already-verified set, never re-classifies), and the
detectors are pure — fixed fixtures (segments + chunked response + gold) in,
exact buckets out, hand-computed, through the public interface, never a network
or paid call. The skipif-guarded reconciliation test pins the live distribution
on the real artifacts (floor 19 / headline 9, ChainsFR by name in llm_skipped).
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


def _seg(start, end, text):
    return {"start": float(start), "end": float(end), "text": str(text)}


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

def test_hard_miss_golds_returns_only_the_49_hard_miss_set() -> None:
    gold_by_vod, raw_by_vod, pool_by_vod = _fixture()

    golds = hmd.hard_miss_golds(gold_by_vod, raw_by_vod, pool_by_vod)

    # Only the two windows #49 classifies as hard_miss come back — the buried and
    # accidental golds (which #49 puts in other buckets) are excluded.
    windows = {(g["vod_id"], g["start"], g["end"]) for g in golds}
    assert windows == {("A", 200.0, 230.0), ("B", 6000.0, 6030.0)}
    # Each hard_miss now carries a resolved root-cause bucket. With no segments /
    # response supplied the content detectors can't fire, so the residuum applies.
    assert all(g["bucket"] in hmd.BUCKETS for g in golds)
    assert all(g["bucket"] == hmd.LLM_SKIPPED for g in golds)
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


# ═════════════════════════════════════════════════════════════════════════════
# Issue #58 — the three root-cause detectors (no-transcript → truncation →
# llm-skipped). Each detector is a pure function: fixed fixtures in, exact
# verdicts out, hand-computed. No network, no paid call.
# ═════════════════════════════════════════════════════════════════════════════

# ── Detector 1: no-transcript / visual (checked first, model-agnostic) ────────

def test_no_transcript_detector_flags_window_with_near_empty_segment_text() -> None:
    # A gold window whose overlapping segments carry essentially no transcript
    # text is visual — no transcript hook can ever recover it. Measured on the
    # absolute non-whitespace char count inside the window.
    gold = _gold(200, 230)
    silent_segments = [_seg(195, 235, "   \n  ")]      # whitespace only → 0 chars

    assert hmd.is_no_transcript(gold, silent_segments, min_chars=40) is True


def test_no_transcript_detector_passes_window_with_real_transcript() -> None:
    # A gold window densely covered by speech is NOT visual — the transcript is
    # present, so it routes onward to truncation / llm-skipped, never here.
    gold = _gold(200, 230)
    spoken_segments = [
        _seg(198, 214, "did he actually do the face reveal i cannot believe this"),
        _seg(214, 232, "five million subscribers and he finally showed his face"),
    ]

    assert hmd.is_no_transcript(gold, spoken_segments, min_chars=40) is False


# ── Detector 2: truncation (strict — the covering chunk block aborts mid-JSON) ─
# The single-chunk case: no `===== CHUNK i/N =====` delimiter, so the whole
# response.txt is the one block covering every gold.
_CLEAN_RESPONSE = json.dumps({"moments": [
    {"start": 1, "end": 2, "hook": 0.7, "rationale": "a complete moment"}]})
# A real gemini truncation: the JSON aborts mid-string with no closing brackets.
_TRUNCATED_RESPONSE = (
    '{\n  "moments": [\n    {"start": 1, "end": 2, "rationale": "first ok"},\n'
    '    {"start": 50, "end": 80, "rationale": "second cut off mid-stre')


def test_truncation_detector_flags_single_chunk_response_aborted_mid_json() -> None:
    # response.txt is one block (no chunk delimiter) and it aborts mid-moment —
    # the covering chunk is truncated, so the gold is a truncation miss.
    gold = _gold(60, 75)
    segments = [_seg(60, 75, "talking through the second moment that got cut off")]

    assert hmd.is_truncated(gold, segments, _TRUNCATED_RESPONSE) is True


def test_truncation_detector_passes_cleanly_closed_single_chunk_response() -> None:
    # A response that parses to complete JSON is never truncation — even a high
    # completion-token count does not override a cleanly-closed block.
    gold = _gold(1, 2)
    segments = [_seg(1, 2, "a complete moment with real speech in the window")]

    assert hmd.is_truncated(gold, segments, _CLEAN_RESPONSE) is False


# A 2-chunk response (dur=2700): chunk 1 [0,2700) closes cleanly; chunk 2
# [2700,5400) aborts mid-moment. This is the ChainsFR shape — a clean covering
# chunk while a DIFFERENT chunk in the same VOD truncated.
_TWO_CHUNK_RESPONSE = (
    "===== CHUNK 1/2 =====\n"
    + json.dumps({"moments": [{"start": 100, "end": 130, "rationale": "chunk 1 ok"}]})
    + "\n\n===== CHUNK 2/2 =====\n"
    '{\n  "moments": [\n    {"start": 3000, "end": 3030, "rationale": "cut off mid-stre'
)
# Segments pin the timeline t0=0 and put real speech under both golds.
_TWO_CHUNK_SEGMENTS = [
    _seg(0, 130, "speech under the first gold inside the clean opening chunk"),
    _seg(2990, 3035, "speech under the second gold inside the truncated chunk"),
]


def test_truncation_detector_maps_gold_to_its_covering_chunk_not_any_chunk() -> None:
    # The gold at 100s sits in chunk 1, which closed cleanly — it is NOT
    # truncation even though chunk 2 truncated. Proves the mapping picks the
    # COVERING chunk, not "any chunk in the VOD truncated".
    gold_in_clean_chunk = _gold(100, 130)

    assert hmd.is_truncated(
        gold_in_clean_chunk, _TWO_CHUNK_SEGMENTS, _TWO_CHUNK_RESPONSE,
        chunk_duration_sec=2700) is False


def test_truncation_detector_flags_gold_in_the_truncated_chunk() -> None:
    # The gold at 3000s sits in chunk 2's span, which aborted mid-JSON — its
    # candidate was lost to a mechanical cut, so it is a truncation miss.
    gold_in_truncated_chunk = _gold(3000, 3030)

    assert hmd.is_truncated(
        gold_in_truncated_chunk, _TWO_CHUNK_SEGMENTS, _TWO_CHUNK_RESPONSE,
        chunk_duration_sec=2700) is True


def test_covering_chunk_index_maps_chainsfr_to_chunk_9() -> None:
    # The real ChainsFR Face Reveal geometry (VOD 2772745645): t0=361.3, the
    # 14-chunk split at dur=2700 puts the gold at 22950.7 in chunk 9 — which on
    # the real artifact closed cleanly, so ChainsFR is llm-skipped, not
    # truncation. floor((22950.7-361.3)/2700)+1 = 8+1 = 9, hand-computed.
    chainsfr = _gold(22950.7, 22994.6, views=1_434_649, desc="ChainsFR Face Reveal")

    idx = hmd.covering_chunk_index(
        chainsfr, t0=361.3, chunk_duration_sec=2700.0, n_chunks=14)

    assert idx == 9


def test_covering_chunk_index_clamps_a_gold_past_the_last_window() -> None:
    # A gold beyond the last window still maps to a real block (clamped to N).
    late = _gold(99_999, 100_010)

    assert hmd.covering_chunk_index(
        late, t0=0.0, chunk_duration_sec=2700.0, n_chunks=14) == 14


# ── Detector 3 + priority resolution: visual → truncation → llm-skipped ───────
# One 2-chunk response: chunk 1 [0,2700) TRUNCATED, chunk 2 [2700,5400) clean.
# Three golds sharing this fixture exercise every precedence rung.
_PRIORITY_RESPONSE = (
    "===== CHUNK 1/2 =====\n"
    '{\n  "moments": [\n    {"start": 200, "end": 230, "rationale": "cut off mid-stre'
    "\n\n===== CHUNK 2/2 =====\n"
    + json.dumps({"moments": [{"start": 3000, "end": 3030, "rationale": "clean"}]})
)
_PRIORITY_SEGMENTS = [
    _seg(95, 135, "   \n  "),                                   # visual window: blank
    _seg(195, 235, "real speech under the gold in the truncated chunk one"),
    _seg(2990, 3035, "real speech under the gold in the clean chunk two here"),
]


def test_classify_no_transcript_beats_truncation() -> None:
    # A blank-transcript gold that ALSO sits in the truncated chunk 1 is visual —
    # no-transcript is checked first and wins, because no text hook recovers it.
    visual_gold = _gold(100, 130, desc="blank window in a truncated chunk")

    bucket = hmd.classify_root_cause(
        visual_gold, segments=_PRIORITY_SEGMENTS, response_text=_PRIORITY_RESPONSE,
        chunk_duration_sec=2700, min_transcript_chars=40)

    assert bucket == hmd.VISUAL


def test_classify_truncation_beats_llm_skipped() -> None:
    # Transcript present AND the covering chunk truncated → truncation (it beats
    # the llm-skipped residuum).
    trunc_gold = _gold(200, 230, desc="spoken window in the truncated chunk")

    bucket = hmd.classify_root_cause(
        trunc_gold, segments=_PRIORITY_SEGMENTS, response_text=_PRIORITY_RESPONSE,
        chunk_duration_sec=2700, min_transcript_chars=40)

    assert bucket == hmd.TRUNCATION


def test_classify_llm_skipped_is_the_residuum() -> None:
    # Transcript present AND the covering chunk closed cleanly → the model simply
    # skipped it: llm-skipped, the residuum a stronger model could address.
    skipped_gold = _gold(3000, 3030, desc="spoken window in a clean chunk")

    bucket = hmd.classify_root_cause(
        skipped_gold, segments=_PRIORITY_SEGMENTS, response_text=_PRIORITY_RESPONSE,
        chunk_duration_sec=2700, min_transcript_chars=40)

    assert bucket == hmd.LLM_SKIPPED


# ─────────────────────────────────────────────────────────────────────────────
# run() now classifies — per-bucket counts per tier, summing to the hard_miss
# count (the buckets are mutually exclusive and exhaustive over the set).
# ─────────────────────────────────────────────────────────────────────────────

def test_run_reports_per_bucket_counts_summing_to_hard_miss_count() -> None:
    # Three hard-miss golds on one VOD, one per bucket: a blank-window visual, a
    # spoken gold in the truncated chunk 1, a spoken gold in the clean chunk 2.
    # The 50k skip gold drops out of the headline tier.
    manifest = {"V": [
        _gold(100, 130, views=300_000, desc="visual"),
        _gold(200, 230, views=300_000, desc="trunc"),
        _gold(3000, 3030, views=50_000, desc="skip"),
    ]}
    raw = {"V": [_cand(0, 5)]}                          # covers no gold → all hard_miss
    pool = {"V": [_cand(0, 5)]}

    result = hmd.run(
        manifest, raw, pool, anchor_vod_ids=["V"], headline_min_views=100_000,
        segments_by_vod={"V": _PRIORITY_SEGMENTS},
        response_by_vod={"V": _PRIORITY_RESPONSE})

    floor = result["hard_miss"]["floor"]
    headline = result["hard_miss"]["headline"]
    assert floor["count"] == 3
    assert floor["buckets"] == {hmd.VISUAL: 1, hmd.TRUNCATION: 1, hmd.LLM_SKIPPED: 1}
    assert sum(floor["buckets"].values()) == floor["count"]
    # The 50k spoken-clean gold is the only one below the headline floor.
    assert headline["count"] == 2
    assert headline["buckets"] == {hmd.VISUAL: 1, hmd.TRUNCATION: 1, hmd.LLM_SKIPPED: 0}
    assert sum(headline["buckets"].values()) == headline["count"]


# ─────────────────────────────────────────────────────────────────────────────
# The report surfaces per-bucket counts (floor + headline), every gold by name
# under its bucket, the generator-model fact, and the truncation caveat.
# ─────────────────────────────────────────────────────────────────────────────

def test_format_report_surfaces_buckets_chainsfr_model_fact_and_caveat() -> None:
    manifest = {"V": [
        _gold(100, 130, views=300_000, desc="blank visual window"),
        _gold(200, 230, views=300_000, desc="spoken truncated window"),
        _gold(3000, 3030, views=1_434_649,
              desc="Jynxzi Reacts to ChainsFR's Face Reveal! 😱👀"),
    ]}
    raw = {"V": [_cand(0, 5)]}
    pool = {"V": [_cand(0, 5)]}

    result = hmd.run(
        manifest, raw, pool, anchor_vod_ids=["V"], headline_min_views=100_000,
        segments_by_vod={"V": _PRIORITY_SEGMENTS},
        response_by_vod={"V": _PRIORITY_RESPONSE})
    report = hmd.format_report(result)

    # Per-bucket counts, named, for both tiers.
    for bucket in (hmd.VISUAL, hmd.TRUNCATION, hmd.LLM_SKIPPED):
        assert bucket in report
    # The 1.43M ChainsFR Face Reveal appears by name (it lands in llm_skipped).
    assert "ChainsFR" in report
    # The generator-model fact: flash-lite generates, Sonnet ranked.
    assert hmd.GENERATOR_MODEL in report
    assert hmd.RANKING_MODEL in report
    # The caveat: only the truncation subset moves the shipped product metric.
    assert "Caveat" in report
    assert "product recall@20" in report


# ── Job-dir writer for the CLI test (the pipeline_debug_* replay layout) ──────
def _row(start, end, composite=8.0):
    virality = composite / 10.0
    return {"start_time": float(start), "end_time": float(end),
            "viral_score": composite, "composite_score": composite,
            "llm_virality": virality, "hook": virality, "flow": virality,
            "value": virality, "trend": virality, "reason": "row"}


def _write_job_dir(storage, vod_id, ts, *, raw, segments, mtime, response=None, stage=None):
    debug = storage / f"pipeline_debug_{ts}_{vod_id}" / "debug"
    debug.mkdir(parents=True)
    (debug / "llm_moment_finder_moments_raw.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in raw), encoding="utf-8")
    (debug / "segments.jsonl").write_text(
        "".join(json.dumps(s) + "\n" for s in segments), encoding="utf-8")
    if response is not None:
        (debug / "llm_moment_finder_response.txt").write_text(response, encoding="utf-8")
    if stage is not None:
        (debug / "llm_moment_finder_stage.json").write_text(
            json.dumps(stage), encoding="utf-8")
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
    # V's single gold is a hard-miss, now carried with a resolved root-cause bucket.
    floor = artifact["hard_miss"]["floor"]
    assert floor["count"] == 1
    assert floor["golds"][0]["desc"] == "V hard-miss"
    assert floor["golds"][0]["bucket"] in hmd.BUCKETS
    assert sum(floor["buckets"].values()) == floor["count"]


def test_main_classifies_a_truncated_gold_as_truncation(tmp_path) -> None:
    # End-to-end: the CLI must load response.txt through the replay loaders and
    # route a spoken gold whose covering chunk aborted mid-JSON into truncation.
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps({
            "vod_id": "T",
            "gt_moments": [{"start": 500.0, "end": 530.0, "views": 300_000,
                            "gt_tier": "gold", "desc": "T truncated gold"}],
        }) + "\n", encoding="utf-8")
    anchor_csv = tmp_path / "anchors.csv"
    anchor_csv.write_text("vod_id\nT\n", encoding="utf-8")
    storage = tmp_path / "storage"
    storage.mkdir()
    # Raw covers nothing → hard-miss. Real speech under the gold → not visual.
    # The single-chunk response.txt aborts mid-JSON → the covering chunk truncated.
    truncated_response = (
        '{\n  "moments": [\n    {"start": 500, "end": 530, "rationale": "cut off mid-stre')
    _write_job_dir(
        storage, "T", "20260101T000000000000Z",
        raw=[_row(1.0, 19.0)],
        segments=[{"start": 495.0, "end": 535.0,
                   "text": "lots of real spoken words filling the gold window here"}],
        response=truncated_response,
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
    floor = json.loads(out_path.read_text(encoding="utf-8"))["hard_miss"]["floor"]
    assert floor["buckets"][hmd.TRUNCATION] == 1
    assert floor["golds"][0]["bucket"] == hmd.TRUNCATION


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
    # Reuse guard only pins membership: with no segments/response threaded here,
    # the content detectors don't fire (residuum). The real per-bucket
    # distribution is asserted in the decomposition reconciliation test below.
    for tier in ("floor", "headline"):
        assert sum(result["hard_miss"][tier]["buckets"].values()) == result["hard_miss"][tier]["count"]
        assert all(g["bucket"] in hmd.BUCKETS for g in result["hard_miss"][tier]["golds"])


@pytest.mark.skipif(
    not _real_artifacts_present(),
    reason="real pipeline_debug artifacts / baseline not present (tmp/storage is gitignored)",
)
def test_decomposition_buckets_reconcile_and_surface_chainsfr_on_real_artifacts() -> None:
    # Feeds the REAL segments + chunked responses through the classifier and pins
    # the live root-cause distribution. The buckets partition the #49 hard_miss
    # set exactly (mutually exclusive + exhaustive over floor 19 / headline 9).
    manifest = eval_clip_finder.load_manifest()
    anchors = eval_clip_finder.read_anchor_vod_ids()
    (raw, pool, segments_by_vod, response_by_vod,
     chunk_duration_by_vod, _gaps) = hmd.build_inputs(anchors, _STORAGE_DIR)

    result = hmd.run(
        manifest, raw, pool,
        segments_by_vod=segments_by_vod, response_by_vod=response_by_vod,
        chunk_duration_by_vod=chunk_duration_by_vod, anchor_vod_ids=anchors)

    floor = result["hard_miss"]["floor"]
    headline = result["hard_miss"]["headline"]
    for tier in (floor, headline):
        assert all(g["bucket"] in hmd.BUCKETS for g in tier["golds"])
        assert sum(tier["buckets"].values()) == tier["count"]
    assert floor["count"] == 19 and headline["count"] == 9

    # The live diagnosis (frozen artifacts, default 40-char visual threshold):
    #   FLOOR    visual=0  truncation=2  llm_skipped=17
    #   HEADLINE visual=0  truncation=0  llm_skipped=9
    # visual=0 is an empirical finding, NOT a threshold artifact: the sparsest
    # hard_miss gold holds 63 transcript chars, well above the 40-char floor — so
    # the PRD's "visual / ChainsFR" hypothesis is refuted, every hard_miss gold
    # has speech. llm_skipped dominates → the routing weight is on the gated paid
    # strong-model re-gen probe; truncation is the small floor-only mechanical win.
    assert floor["buckets"] == {hmd.VISUAL: 0, hmd.TRUNCATION: 2, hmd.LLM_SKIPPED: 17}
    assert headline["buckets"] == {hmd.VISUAL: 0, hmd.TRUNCATION: 0, hmd.LLM_SKIPPED: 9}

    # The 1.43M ChainsFR Face Reveal appears by name in EXACTLY ONE bucket — and
    # it is llm_skipped, not visual: its window holds ~545 transcript chars and
    # its covering chunk (9/14) closed cleanly, so flash-lite saw it and skipped
    # it. That is the bucket a stronger model could address.
    chainsfr = [g for g in floor["golds"]
                if "ChainsFR" in g["desc"] or round(g["views"]) == 1_434_649]
    assert len(chainsfr) == 1
    assert chainsfr[0]["bucket"] == hmd.LLM_SKIPPED


def directional_gold(manifest, anchors):
    """The anchor-restricted gold manifest #49/#48 score over (test-local helper)."""
    from ml import directional_eval
    gold = directional_eval.gold_manifest(manifest)
    anchor_set = set(anchors)
    return {vod_id: moments for vod_id, moments in gold.items() if vod_id in anchor_set}
