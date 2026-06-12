"""Tests for the strong-model Window-Probe data assembly (ml/strong_model_window_probe.py).

The free, offline Phase-1 tracer of PRD #60: it de-risks the data assembly *before*
any paid Sonnet call (#62) or separation analysis (#63). It composes existing seams —
the #56 `hard_miss_decomposition` `llm_skipped` bucket (the gold windows), the delivered
flash-lite top-20 (`final_moments.json`) filtered to non-gold via #49 `is_hit` (the
control), and the #47 `replay_harness` loaders — and builds, for every window (gold AND
control, treated identically), a modest symmetric transcript excerpt from `segments.jsonl`.

These tests pin the excerpt builder, the control filter and the llm_skipped reuse guard
through the public interface on hand-built fixtures — never a network or paid call.
"""
from __future__ import annotations

import json
import os
import sys
import types

import pytest

# Defensive: the assembly reaches hard_miss_decomposition -> the serving
# post-processor, which can pull the heavy transcription stack. Prior art
# (test_gap_decomposition) stubs whisper the same way so the import stays cheap.
sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_a, **_k: None))

from ml import eval_clip_finder, hard_miss_decomposition as hmd, strong_model_window_probe as swp


def _seg(start, end, text):
    return {"start": float(start), "end": float(end), "text": str(text)}


def _gold(start, end, views=0.0, desc=""):
    return {"start": float(start), "end": float(end), "views": float(views),
            "gt_tier": "gold", "desc": desc}


def _moment(start, end, composite=8.0):
    """One delivered top-20 moment (final_moments.json shape: start_time/end_time)."""
    return {"start_time": float(start), "end_time": float(end),
            "composite_score": float(composite), "reason": "delivered"}


# ─────────────────────────────────────────────────────────────────────────────
# Slice 1 — build_excerpt grabs the segments symmetrically around a window
# ─────────────────────────────────────────────────────────────────────────────

def test_build_excerpt_grabs_segments_within_the_symmetric_pad() -> None:
    # Window [200,230], pad 90s -> excerpt window [110, 320]. Only the segments
    # overlapping that span contribute; the ones just outside are excluded.
    segments = [
        _seg(100, 110, "before_excluded"),     # ends at 110 -> no overlap with [110,320]
        _seg(110, 120, "lead_in"),             # inside the left pad
        _seg(200, 210, "the_gold_itself"),     # the window
        _seg(310, 320, "tail_out"),            # inside the right pad
        _seg(320, 330, "after_excluded"),      # starts at 320 -> no overlap
        _seg(500, 510, "far_away"),            # well outside
    ]

    excerpt = swp.build_excerpt(200, 230, segments, pad_sec=90)

    assert excerpt["excerpt_start"] == 110.0
    assert excerpt["excerpt_end"] == 320.0
    assert "lead_in" in excerpt["text"]
    assert "the_gold_itself" in excerpt["text"]
    assert "tail_out" in excerpt["text"]
    for excluded in ("before_excluded", "after_excluded", "far_away"):
        assert excluded not in excerpt["text"]


# ─────────────────────────────────────────────────────────────────────────────
# Slice 2 — the excerpt is clamped to the VOD's first/last segment bounds
# ─────────────────────────────────────────────────────────────────────────────

def test_build_excerpt_clamps_to_vod_start_and_end_bounds() -> None:
    # Timeline runs [0, 200]. A window near the start would pad to -40 (before the
    # VOD) and one near the end would pad past 200 — both clamp to the real bounds,
    # so the excerpt never references time outside the transcript.
    segments = [
        _seg(0, 10, "vod_start"),
        _seg(60, 70, "near_start_window"),
        _seg(160, 170, "near_end_window"),
        _seg(190, 200, "vod_end"),
    ]

    head = swp.build_excerpt(50, 80, segments, pad_sec=90)     # raw [-40,170] -> [0,170]
    assert head["excerpt_start"] == 0.0                        # clamped up from -40
    assert head["excerpt_end"] == 170.0                        # 170 < 200, untouched
    assert "vod_start" in head["text"] and "vod_end" not in head["text"]

    tail = swp.build_excerpt(150, 180, segments, pad_sec=90)   # raw [60,270] -> [60,200]
    assert tail["excerpt_start"] == 60.0                       # 60 > 0, untouched
    assert tail["excerpt_end"] == 200.0                        # clamped down from 270
    assert "vod_end" in tail["text"] and "vod_start" not in tail["text"]


# ─────────────────────────────────────────────────────────────────────────────
# Slice 3 — the control set is the delivered top-20, filtered to non-gold
# ─────────────────────────────────────────────────────────────────────────────

def test_control_windows_drops_gold_overlapping_delivered_moments() -> None:
    # The delivered top-20 (product rank order). m1 covers the gold (>=50% of the
    # 30s GT window) so it is a hit and dropped; the two distant moments are the
    # genuine non-gold controls. The dropped count is surfaced for logging.
    gold = _gold(100, 130)
    delivered = [
        _moment(105, 125, composite=9.0),     # covers the gold -> dropped
        _moment(500, 520, composite=8.0),     # control
        _moment(600, 620, composite=7.0),     # control
    ]

    result = swp.control_windows(delivered, [gold], threshold=0.5)

    controls = result["controls"]
    assert result["dropped_gold_overlap"] == 1
    assert [(c["start"], c["end"]) for c in controls] == [(500.0, 520.0), (600.0, 620.0)]
    # Product rank is preserved (1-based position in the delivered order).
    assert [c["rank"] for c in controls] == [2, 3]
    assert [c["composite_score"] for c in controls] == [8.0, 7.0]


# ─────────────────────────────────────────────────────────────────────────────
# Slice 4 — gold windows ARE the #56 llm_skipped bucket (reuse guard)
# ─────────────────────────────────────────────────────────────────────────────

def _cand(start, end):
    return {"start_time": float(start), "end_time": float(end)}


# A 2-chunk response (dur=2700): chunk 1 [t0,t0+2700) aborts mid-JSON, chunk 2
# closes cleanly. A spoken gold in chunk 1 -> truncation; one in chunk 2 ->
# llm_skipped. The probe must pick ONLY the llm_skipped gold.
_RESPONSE = (
    "===== CHUNK 1/2 =====\n"
    '{\n  "moments": [\n    {"start": 200, "end": 230, "rationale": "cut off mid-stre'
    "\n\n===== CHUNK 2/2 =====\n"
    '{"moments": [{"start": 3000, "end": 3030, "rationale": "clean"}]}'
)
_SEGMENTS = [
    _seg(195, 235, "real spoken words under the gold in the truncated first chunk"),
    _seg(2990, 3035, "real spoken words under the gold in the clean second chunk"),
]


def test_gold_windows_are_exactly_the_56_llm_skipped_golds_not_reclassified() -> None:
    # Two hard-miss golds: one truncation, one llm_skipped (per #56). The probe
    # consumes the #56 hard_miss result and must surface ONLY the llm_skipped gold
    # as a gold window — it sub-selects an already-classified set, never re-derives.
    manifest = {"V": [_gold(200, 230, views=300_000, desc="truncation gold"),
                      _gold(3000, 3030, views=300_000, desc="llm_skipped gold")]}
    raw = {"V": [_cand(0, 5)]}                          # covers neither -> both hard_miss
    pool = {"V": [_cand(0, 5)]}

    hm = hmd.run(manifest, raw, pool, anchor_vod_ids=["V"], headline_min_views=100_000,
                 segments_by_vod={"V": _SEGMENTS}, response_by_vod={"V": _RESPONSE})

    golds = swp.llm_skipped_golds(hm["hard_miss"], "floor")

    assert [(g["vod_id"], g["start"], g["end"]) for g in golds] == [("V", 3000.0, 3030.0)]
    assert golds[0]["desc"] == "llm_skipped gold"
    # Pinned directly against #56: exactly the golds hmd put in the llm_skipped bucket.
    expected = [g for g in hm["hard_miss"]["floor"]["golds"] if g["bucket"] == hmd.LLM_SKIPPED]
    assert golds == expected


# ─────────────────────────────────────────────────────────────────────────────
# Slice 5 — run() assembles per-VOD gold + control windows, each with an excerpt
# ─────────────────────────────────────────────────────────────────────────────

# VOD "V": gold A [200,230] truncation (chunk 1), gold B [3000,3030] llm_skipped
# (chunk 2). Segments carry speech under both golds AND under two control windows.
_V_SEGMENTS = [
    _seg(195, 235, "gold A speech inside the truncated first chunk with plenty of real spoken words"),
    _seg(495, 525, "control one speech delivered non gold"),
    _seg(595, 625, "control two speech delivered non gold"),
    _seg(2990, 3035, "gold B speech inside the clean second chunk with plenty of real spoken words"),
]


def test_run_assembles_gold_and_control_windows_with_excerpts() -> None:
    manifest = {"V": [_gold(200, 230, views=300_000, desc="truncation gold"),
                      _gold(3000, 3030, views=300_000, desc="llm_skipped gold")]}
    raw = {"V": [_cand(0, 5)]}                          # covers neither -> both hard_miss
    pool = {"V": [_cand(0, 5)]}
    # Delivered top-20: one overlaps gold A (dropped), two are genuine controls.
    final = {"V": [_moment(205, 225, composite=9.0),
                   _moment(500, 520, composite=8.0),
                   _moment(600, 620, composite=7.0)]}

    result = swp.run(
        manifest, raw, pool, final, anchor_vod_ids=["V"], headline_min_views=100_000,
        segments_by_vod={"V": _V_SEGMENTS}, response_by_vod={"V": _RESPONSE}, pad_sec=90)

    vod = result["vods"]["V"]
    # Gold window = only the llm_skipped gold, with its symmetric excerpt.
    assert [(g["start"], g["end"]) for g in vod["gold_windows"]] == [(3000.0, 3030.0)]
    assert "gold B speech" in vod["gold_windows"][0]["excerpt"]["text"]
    assert vod["gold_windows"][0]["headline"] is True            # 300k >= 100k

    # Control windows = the delivered non-gold top-20, each with its own excerpt
    # built by the SAME builder (identical treatment to gold).
    assert [(c["start"], c["end"]) for c in vod["control_windows"]] == [(500.0, 520.0), (600.0, 620.0)]
    assert "control one speech" in vod["control_windows"][0]["excerpt"]["text"]
    assert "control two speech" in vod["control_windows"][1]["excerpt"]["text"]
    # The gold-overlapping delivered moment is dropped and the bound is logged.
    assert vod["control_dropped_gold_overlap"] == 1
    assert result["counts"]["gold_windows"] == {"floor": 1, "headline": 1}
    assert result["counts"]["control_windows"] == 2


def test_run_surfaces_vods_without_final_moments_as_a_coverage_gap() -> None:
    # V has an llm_skipped gold but no loadable final_moments.json — its controls
    # can't be assembled. That is a surfaced coverage gap, never a silent miss: the
    # gold window still stands, the control set is empty, and V is named in gaps.
    manifest = {"V": [_gold(200, 230, views=300_000, desc="truncation gold"),
                      _gold(3000, 3030, views=300_000, desc="llm_skipped gold")]}
    raw = {"V": [_cand(0, 5)]}
    pool = {"V": [_cand(0, 5)]}

    result = swp.run(
        manifest, raw, pool, final_by_vod={}, anchor_vod_ids=["V"], headline_min_views=100_000,
        segments_by_vod={"V": _V_SEGMENTS}, response_by_vod={"V": _RESPONSE}, pad_sec=90)

    assert result["gaps"]["no_final_moments"] == ["V"]
    vod = result["vods"]["V"]
    assert len(vod["gold_windows"]) == 1                  # the gold window still stands
    assert vod["control_windows"] == []                  # no controls assembled
    assert vod["control_dropped_gold_overlap"] == 0
    assert result["counts"]["control_windows"] == 0


# ═════════════════════════════════════════════════════════════════════════════
# Issue #62 — the paid Sonnet window-rating + persist/replay. rate_windows is the
# core: gold AND control windows (one set) scored identically through the reused
# repost_virality_judge compact prompt + parser, raw persisted BEFORE parsing.
# All tests run net-free over the Fake adapter / persisted raw-response fixtures.
# ═════════════════════════════════════════════════════════════════════════════

def _win(start, end, kind, text, **extra):
    return {"vod_id": "V", "start": float(start), "end": float(end),
            "kind": kind, "excerpt": {"text": text}, **extra}


def test_rate_windows_scores_gold_and_control_by_one_based_id() -> None:
    # gold + control in ONE scored set: the 1-based id maps each window to its
    # score, identical prompt/score form for both. The raw 0-1 repost_virality is
    # attached (not the 1-10 pipeline rescale — the #63 separation reads the raw).
    windows = [
        _win(3000, 3030, "gold", "the gold transcript excerpt under the skipped window"),
        _win(500, 520, "control", "the delivered control transcript excerpt here"),
    ]
    raw = '{"scores": [{"id": 1, "repost_virality": 0.8}, {"id": 2, "repost_virality": 0.3}]}'

    saved: list[str] = []
    result = swp.rate_windows(windows, call_provider=lambda _p: raw, persist_raw=saved.append)

    assert [w["score"] for w in result["scored"]] == [0.8, 0.3]
    assert [w["kind"] for w in result["scored"]] == ["gold", "control"]
    assert result["dropped"] == []
    # Persisted the raw response BEFORE parsing (the re-evaluable artifact).
    assert saved == [raw]


def test_rate_windows_salvages_truncated_json_and_drops_unscored_fail_closed() -> None:
    # A real Sonnet truncation: the compact score array aborts mid-item after two
    # complete entries. The reused salvage recovers ids 1 and 2; window 3 has no
    # score and is DROPPED (fail-closed) — never read as "scored everything". The
    # full truncated raw is still persisted (persist happens before the parse).
    windows = [
        _win(3000, 3030, "gold", "gold window one"),
        _win(500, 520, "control", "control window two"),
        _win(600, 620, "control", "control window three never scored"),
    ]
    truncated = ('{"scores": [{"id": 1, "repost_virality": 0.7}, '
                 '{"id": 2, "repost_virality": 0.4}, {"id": 3, "repost_vira')

    saved: list[str] = []
    result = swp.rate_windows(windows, call_provider=lambda _p: truncated, persist_raw=saved.append)

    assert [(w["kind"], w["score"]) for w in result["scored"]] == [("gold", 0.7), ("control", 0.4)]
    # Window 3 is surfaced as dropped, not silently scored.
    assert [(w["start"], w["end"]) for w in result["dropped"]] == [(600.0, 620.0)]
    # The whole truncated raw is persisted regardless — diagnosable, replayable.
    assert saved == [truncated]


def test_paid_rater_persists_raw_and_scores_then_replays_identically(tmp_path) -> None:
    # The full persist/replay round-trip, net-free: the paid factory (provider
    # injected with a fake — the real wiring binds Sonnet) persists the raw BEFORE
    # parsing and the per-window scores AFTER; the replay factory re-scores
    # bit-identical from the persisted raw, never touching the network.
    windows = [
        _win(3000, 3030, "gold", "the gold transcript excerpt", views=300_000, desc="ChainsFR"),
        _win(500, 520, "control", "the control transcript excerpt", rank=3),
    ]
    raw = '{"scores": [{"id": 1, "repost_virality": 0.9}, {"id": 2, "repost_virality": 0.2}]}'

    recorded: list[str] = []
    paid = swp.paid_make_rater(tmp_path, provider=lambda prompt: recorded.append(prompt) or raw)
    paid_out = paid("V", windows)

    # raw.txt persisted verbatim; scores.json carries the per-window scores + kind.
    assert (tmp_path / "V.raw.txt").read_text(encoding="utf-8") == raw
    scores_json = json.loads((tmp_path / "V.scores.json").read_text(encoding="utf-8"))
    assert [(s["kind"], s["score"]) for s in scores_json] == [("gold", 0.9), ("control", 0.2)]
    # gold + control went in ONE prompt (one paid call per VOD), identical form.
    assert len(recorded) == 1

    # Replay binds the persisted raw — no network — and reproduces the scores exactly.
    replay_out = swp.replay_make_rater(tmp_path)("V", windows)
    assert replay_out["scored"] == paid_out["scored"]
    assert [w["score"] for w in replay_out["scored"]] == [0.9, 0.2]


def test_run_rating_scores_each_vod_set_and_surfaces_caps() -> None:
    # The orchestrator rates each VOD's gold+control set through the injected rater
    # and surfaces every cap: a VOD with no windows is named, and per-VOD unscored
    # drops are counted (never silently treated as "scored everything").
    assembled = {"vods": {
        "V1": {
            "gold_windows": [{"vod_id": "V1", "start": 3000.0, "end": 3030.0,
                              "headline": True, "desc": "g", "excerpt": {"text": "gold"}}],
            "control_windows": [
                {"vod_id": "V1", "start": 500.0, "end": 520.0, "rank": 2, "excerpt": {"text": "c1"}},
                {"vod_id": "V1", "start": 600.0, "end": 620.0, "rank": 3, "excerpt": {"text": "c2"}}],
        },
        "V2": {"gold_windows": [], "control_windows": []},   # no windows → a cap
    }}
    # The model scores ids 1 and 2; the third window (2nd control) is dropped.
    raw = '{"scores": [{"id": 1, "repost_virality": 0.8}, {"id": 2, "repost_virality": 0.5}]}'
    make_rater = lambda _vod, windows: swp.rate_windows(
        windows, call_provider=lambda _p: raw, persist_raw=lambda _t: None)

    result = swp.run_rating(assembled, make_rater=make_rater)

    assert result["caps"]["vods_without_windows"] == ["V2"]
    assert result["caps"]["dropped_unscored_by_vod"] == {"V1": 1}
    assert result["counts"] == {"scored": 2, "dropped": 1}
    assert [(w["kind"], w["score"]) for w in result["vods"]["V1"]["scored"]] == [
        ("gold", 0.8), ("control", 0.5)]


# ═════════════════════════════════════════════════════════════════════════════
# Issue #63 — within-set separation + the pre-fixed 0.66 gate (read-only over the
# #62 persisted scores; no paid call). Hand-computed percentiles in, exact gate out.
# ═════════════════════════════════════════════════════════════════════════════

# ── Slice 1: window_separation — fraction of controls the gold strictly beats ──

def test_window_separation_counts_controls_strictly_beaten() -> None:
    # The gold scores 0.5; of three controls it strictly outscores only the 0.3 one
    # (the 0.5 tie does NOT count, the 0.7 beats it). probability-of-superiority =
    # 1/3, the #54 formula on rated windows.
    sep = swp.window_separation(0.5, [0.3, 0.5, 0.7])

    assert sep is not None
    assert sep["gold_score"] == 0.5
    assert sep["sibling_n"] == 3
    assert sep["beaten"] == 1
    assert sep["percentile"] == 1 / 3


def test_window_separation_is_none_without_controls() -> None:
    # No controls in the VOD -> the gold is unmeasurable, surfaced as None (never a
    # 0.0 that would understate separation), fail-closed like #54.
    assert swp.window_separation(0.9, []) is None


# ── Slice 2: the pre-fixed 0.66 gate, read asymmetrically ─────────────────────

def test_gate_separated_routes_to_escalate_b() -> None:
    # Median percentile at/above the 0.66 bar (#54's cut) -> getrennt -> escalate
    # (B), the full paid_regen_probe. 0.66 itself separates (>=).
    assert swp.gate(0.70)["separated"] is True
    assert swp.gate(0.70)["route"] == swp.ESCALATE_B
    assert swp.gate(0.66)["separated"] is True


def test_gate_not_separated_routes_to_pivot_c() -> None:
    # Below 0.66 -> nicht getrennt -> decided exogenous-to-transcript -> Pivot (C).
    verdict = swp.gate(0.50)
    assert verdict["separated"] is False
    assert verdict["route"] == swp.PIVOT_C


def test_gate_fails_closed_on_unmeasurable_median() -> None:
    # No measurable gold (None median) is never read as separation -> Pivot (C).
    verdict = swp.gate(None)
    assert verdict["separated"] is False
    assert verdict["route"] == swp.PIVOT_C


# ── Slice 3: analyze_separation — per-tier median + gate, per-window rows ──────

def _scored(vod_id, start, end, kind, score, **extra):
    return {"vod_id": vod_id, "start": float(start), "end": float(end),
            "kind": kind, "score": score, **extra}


def test_analyze_separation_splits_floor_and_headline_and_routes_on_headline() -> None:
    # V1's headline gold beats all 3 of its controls (percentile 1.0); V2's floor
    # gold beats none of its 2 controls (0.0). Floor median([0.0,1.0])=0.5 < 0.66 ->
    # Pivot (C); headline median([1.0])=1.0 >= 0.66 -> escalate (B). Routing weight
    # sits on headline (North-Star).
    rating = {"vods": {
        "V1": {"scored": [
            _scored("V1", 3000, 3030, "gold", 0.9, headline=True, desc="V1 headline gold"),
            _scored("V1", 500, 520, "control", 0.2, rank=2),
            _scored("V1", 600, 620, "control", 0.4, rank=3),
            _scored("V1", 700, 720, "control", 0.6, rank=4),
        ], "dropped": []},
        "V2": {"scored": [
            _scored("V2", 8000, 8030, "gold", 0.3, headline=False, desc="V2 floor gold"),
            _scored("V2", 100, 120, "control", 0.5, rank=1),
            _scored("V2", 200, 220, "control", 0.7, rank=2),
        ], "dropped": []},
    }}

    result = swp.analyze_separation(rating)

    assert result["tiers"]["floor"]["median_percentile"] == 0.5
    assert result["tiers"]["floor"]["gate"]["route"] == swp.PIVOT_C
    assert result["tiers"]["headline"]["median_percentile"] == 1.0
    assert result["tiers"]["headline"]["gate"]["route"] == swp.ESCALATE_B
    # The North-Star headline tier drives the routing verdict.
    assert result["routing_tier"] == "headline"
    assert result["verdict"] == result["tiers"]["headline"]["gate"]["verdict"]
    # Per-window rows (not just the aggregate), identifiable by desc.
    by_desc = {w["desc"]: w for w in result["windows"]}
    assert by_desc["V1 headline gold"]["percentile"] == 1.0
    assert by_desc["V2 floor gold"]["percentile"] == 0.0
    assert by_desc["V1 headline gold"]["separated"] is True
    assert by_desc["V2 floor gold"]["separated"] is False


# ── Slice 4: format_separation — per-window report + the two caveats ──────────

def test_format_separation_surfaces_per_window_tiers_and_both_caveats() -> None:
    rating = {"vods": {
        "C": {"scored": [
            _scored("C", 22950, 22994, "gold", 0.4, headline=True,
                    desc="Jynxzi Reacts to ChainsFR's Face Reveal!"),
            _scored("C", 100, 120, "control", 0.6, rank=1),
            _scored("C", 200, 220, "control", 0.8, rank=2),
        ], "dropped": []},
    }}
    separation = swp.analyze_separation(rating)

    report = swp.format_separation(separation)

    # The 1.43M ChainsFR window is identifiable by name (per-window, not just aggregate).
    assert "ChainsFR" in report
    # Both tiers' gate verdicts are shown, and the headline routing decision.
    assert "FLOOR" in report and "HEADLINE" in report
    assert swp.PIVOT_C in report          # ChainsFR beats neither control -> not separated
    # The thin-n caveat names the tier counts.
    assert "thin" in report.lower()
    # The ship caveat: separated is suggestive, not a ship proof (in-context is B).
    assert "ship" in report.lower()


# ── Slice 6: format_verdict_comment — the markdown verdict for #60 ────────────

def test_format_verdict_comment_carries_route_threshold_and_caveats() -> None:
    rating = {"vods": {
        "C": {"scored": [
            _scored("C", 22950, 22994, "gold", 0.4, headline=True, desc="ChainsFR Face Reveal"),
            _scored("C", 100, 120, "control", 0.6, rank=1),
            _scored("C", 200, 220, "control", 0.8, rank=2),
        ], "dropped": []},
    }}
    separation = swp.analyze_separation(rating)

    comment = swp.format_verdict_comment(separation)

    # The headline routing decision (Pivot C here: ChainsFR beats no control).
    assert swp.PIVOT_C in comment
    # The pre-fixed bar is named, and the routing tier is the North-Star headline.
    assert "0.66" in comment
    assert "headline" in comment.lower()
    # Both caveats travel with the verdict.
    assert "thin" in comment.lower()
    assert "ship" in comment.lower()


# ── Job-dir writer for the CLI test (the pipeline_debug_* replay layout) ──────
def _row(start, end, composite=8.0):
    virality = composite / 10.0
    return {"start_time": float(start), "end_time": float(end),
            "viral_score": composite, "composite_score": composite,
            "llm_virality": virality, "hook": virality, "flow": virality,
            "value": virality, "trend": virality, "reason": "row"}


def _write_job_dir(storage, vod_id, ts, *, raw, segments, response, final, mtime):
    debug = storage / f"pipeline_debug_{ts}_{vod_id}" / "debug"
    debug.mkdir(parents=True)
    (debug / "llm_moment_finder_moments_raw.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in raw), encoding="utf-8")
    (debug / "segments.jsonl").write_text(
        "".join(json.dumps(s) + "\n" for s in segments), encoding="utf-8")
    (debug / "llm_moment_finder_response.txt").write_text(response, encoding="utf-8")
    (debug / "final_moments.json").write_text(
        json.dumps({"moments": final}), encoding="utf-8")
    os.utime(debug.parent, (mtime, mtime))


# ─────────────────────────────────────────────────────────────────────────────
# Slice 6 — the CLI replays the real artifact layout into a window-probe report
# ─────────────────────────────────────────────────────────────────────────────

def test_main_assembles_windows_from_replay_artifacts(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps({
            "vod_id": "V",
            "gt_moments": [{"start": 3000.0, "end": 3030.0, "views": 300_000,
                            "gt_tier": "gold", "desc": "V llm_skipped"}],
        }) + "\n", encoding="utf-8")
    anchor_csv = tmp_path / "anchors.csv"
    anchor_csv.write_text("vod_id\nV\n", encoding="utf-8")
    storage = tmp_path / "storage"
    storage.mkdir()
    # Single-chunk clean response -> the gold is llm_skipped (transcript present,
    # nothing truncated). Raw covers nothing -> hard_miss. Delivered top-20 holds
    # one gold-overlapping moment (dropped) and one genuine non-gold control.
    clean_response = json.dumps({"moments": [{"start": 10, "end": 30, "rationale": "ok"}]})
    _write_job_dir(
        storage, "V", "20260101T000000000000Z",
        raw=[_row(1.0, 19.0)],
        segments=[
            _seg(2990, 3035, "the llm skipped gold window with plenty of real spoken words here"),
            _seg(495, 525, "the delivered control window speech here"),
        ],
        response=clean_response,
        final=[_moment(3005, 3025, composite=9.0),   # overlaps the gold -> dropped
               _moment(500, 520, composite=8.0)],     # genuine non-gold control
        mtime=1000.0)
    out_path = tmp_path / "report.json"

    rc = swp.main([
        "--manifest", str(manifest_path),
        "--anchor-csv", str(anchor_csv),
        "--predictions-dir", str(storage),
        "--output", str(out_path),
        "--json",
    ])

    assert rc == 0
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    vod = artifact["vods"]["V"]
    assert [(g["start"], g["end"]) for g in vod["gold_windows"]] == [(3000.0, 3030.0)]
    assert "llm skipped gold window" in vod["gold_windows"][0]["excerpt"]["text"]
    assert [(c["start"], c["end"]) for c in vod["control_windows"]] == [(500.0, 520.0)]
    assert "delivered control window" in vod["control_windows"][0]["excerpt"]["text"]
    assert vod["control_dropped_gold_overlap"] == 1
    assert "computed_at" in artifact


def test_main_replay_rescoreswindows_from_persisted_raw_no_network(tmp_path) -> None:
    # End-to-end --replay: the CLI assembles the windows (1 gold + 1 control for V),
    # then re-scores them from a persisted V.raw.txt — NO network, NO paid call. The
    # report carries the per-window scores by kind, exactly what #63 will read.
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps({
            "vod_id": "V",
            "gt_moments": [{"start": 3000.0, "end": 3030.0, "views": 300_000,
                            "gt_tier": "gold", "desc": "V llm_skipped"}],
        }) + "\n", encoding="utf-8")
    anchor_csv = tmp_path / "anchors.csv"
    anchor_csv.write_text("vod_id\nV\n", encoding="utf-8")
    storage = tmp_path / "storage"
    storage.mkdir()
    clean_response = json.dumps({"moments": [{"start": 10, "end": 30, "rationale": "ok"}]})
    _write_job_dir(
        storage, "V", "20260101T000000000000Z",
        raw=[_row(1.0, 19.0)],
        segments=[_seg(2990, 3035, "the llm skipped gold window with real spoken words"),
                  _seg(495, 525, "the delivered control window speech here")],
        response=clean_response,
        final=[_moment(3005, 3025, composite=9.0),   # overlaps the gold -> dropped
               _moment(500, 520, composite=8.0)],     # genuine non-gold control
        mtime=1000.0)
    # The persisted paid response: window 1 = gold, window 2 = control (assembly order).
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    (artifacts_dir / "V.raw.txt").write_text(
        '{"scores": [{"id": 1, "repost_virality": 0.85}, {"id": 2, "repost_virality": 0.30}]}',
        encoding="utf-8")
    out_path = tmp_path / "report.json"

    rc = swp.main([
        "--manifest", str(manifest_path),
        "--anchor-csv", str(anchor_csv),
        "--predictions-dir", str(storage),
        "--artifacts-dir", str(artifacts_dir),
        "--replay",
        "--output", str(out_path),
        "--json",
    ])

    assert rc == 0
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    assert artifact["rating_mode"] == "replay"
    scored = artifact["rating"]["vods"]["V"]["scored"]
    assert [(w["kind"], w["score"]) for w in scored] == [("gold", 0.85), ("control", 0.30)]
    assert artifact["rating"]["counts"] == {"scored": 2, "dropped": 0}


def test_main_replay_computes_separation_and_gate(tmp_path) -> None:
    # --replay also applies the #63 separation + 0.66 gate over the re-scored
    # windows, net-free. The gold (0.85) beats its single control (0.30) -> pct 1.0
    # -> headline separated -> escalate (B). The gate is reproducible from disk.
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps({
            "vod_id": "V",
            "gt_moments": [{"start": 3000.0, "end": 3030.0, "views": 300_000,
                            "gt_tier": "gold", "desc": "V llm_skipped"}],
        }) + "\n", encoding="utf-8")
    anchor_csv = tmp_path / "anchors.csv"
    anchor_csv.write_text("vod_id\nV\n", encoding="utf-8")
    storage = tmp_path / "storage"
    storage.mkdir()
    clean_response = json.dumps({"moments": [{"start": 10, "end": 30, "rationale": "ok"}]})
    _write_job_dir(
        storage, "V", "20260101T000000000000Z",
        raw=[_row(1.0, 19.0)],
        segments=[_seg(2990, 3035, "the llm skipped gold window with real spoken words"),
                  _seg(495, 525, "the delivered control window speech here")],
        response=clean_response,
        final=[_moment(3005, 3025, composite=9.0), _moment(500, 520, composite=8.0)],
        mtime=1000.0)
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    (artifacts_dir / "V.raw.txt").write_text(
        '{"scores": [{"id": 1, "repost_virality": 0.85}, {"id": 2, "repost_virality": 0.30}]}',
        encoding="utf-8")
    out_path = tmp_path / "report.json"

    rc = swp.main([
        "--manifest", str(manifest_path), "--anchor-csv", str(anchor_csv),
        "--predictions-dir", str(storage), "--artifacts-dir", str(artifacts_dir),
        "--replay", "--output", str(out_path), "--json",
    ])

    assert rc == 0
    separation = json.loads(out_path.read_text(encoding="utf-8"))["separation"]
    assert separation["tiers"]["headline"]["median_percentile"] == 1.0
    assert separation["tiers"]["headline"]["gate"]["route"] == swp.ESCALATE_B
    assert separation["verdict"] == separation["tiers"]["headline"]["gate"]["verdict"]
    assert separation["windows"][0]["percentile"] == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Guard — reconcile the assembly against #56 + the real artifacts (no network)
# ─────────────────────────────────────────────────────────────────────────────

_STORAGE_DIR = eval_clip_finder.DEFAULT_STORAGE_DIR
_CHAINSFR_VOD = "2772745645"
_MISSING_PREDICTION_VODS = ["2756055866", "2764976309", "2770356205", "2771207666", "2773650615"]


def _real_artifacts_present() -> bool:
    return any(_STORAGE_DIR.glob("pipeline_debug_*"))


@pytest.mark.skipif(
    not _real_artifacts_present(),
    reason="real pipeline_debug artifacts not present (tmp/storage is gitignored)",
)
def test_window_probe_reconciles_with_56_on_real_artifacts() -> None:
    # Assemble over the real artifacts and pin: (a) the gold-window set is exactly
    # #56's llm_skipped golds — same counts AND same windows, both tiers (the reuse
    # guard, no re-classification); (b) controls load for every gold VOD; (c)
    # ChainsFR is a named gold window with a real excerpt; (d) the coverage gap is
    # surfaced, not counted.
    manifest = eval_clip_finder.load_manifest()
    anchors = eval_clip_finder.read_anchor_vod_ids()
    (raw, pool, final_by_vod, segments_by_vod, response_by_vod,
     chunk_duration_by_vod, _gaps) = swp.build_inputs(anchors, _STORAGE_DIR)

    result = swp.run(
        manifest, raw, pool, final_by_vod,
        segments_by_vod=segments_by_vod, response_by_vod=response_by_vod,
        chunk_duration_by_vod=chunk_duration_by_vod, anchor_vod_ids=anchors)
    decomposition = hmd.run(
        manifest, raw, pool, segments_by_vod=segments_by_vod, response_by_vod=response_by_vod,
        chunk_duration_by_vod=chunk_duration_by_vod, anchor_vod_ids=anchors)

    # (a) Reuse guard: gold windows == #56 llm_skipped, both tiers (frozen 17 / 9).
    assert result["counts"]["gold_windows"] == {"floor": 17, "headline": 9}
    for tier in ("floor", "headline"):
        expected = {(g["vod_id"], g["start"], g["end"])
                    for g in decomposition["hard_miss"][tier]["golds"]
                    if g["bucket"] == hmd.LLM_SKIPPED}
        got = {(w["vod_id"], w["start"], w["end"])
               for entry in result["vods"].values() for w in entry["gold_windows"]
               if (tier == "floor" or w["headline"])}
        assert got == expected

    # (b) Controls assembled for every gold VOD (none missing final_moments here).
    assert result["gaps"]["no_final_moments"] == []
    assert result["counts"]["control_windows"] > 0
    for entry in result["vods"].values():
        assert len(entry["control_windows"]) > 0

    # (c) ChainsFR is a named gold window with a non-trivial transcript excerpt.
    chainsfr = result["vods"][_CHAINSFR_VOD]["gold_windows"]
    assert any("ChainsFR" in g["desc"] for g in chainsfr)
    assert all(g["excerpt"]["text"] for g in chainsfr)

    # (d) The 5 missing-prediction VODs stay a surfaced coverage gap, not counted.
    assert result["gaps"]["missing_prediction_vods"] == _MISSING_PREDICTION_VODS


_ARTIFACTS_DIR = swp.DEFAULT_ARTIFACTS_DIR


def _probe_artifacts_present() -> bool:
    """The (gitignored) persisted window-probe raw responses from a real paid run."""
    return (_real_artifacts_present() and _ARTIFACTS_DIR.exists()
            and any(_ARTIFACTS_DIR.glob("*.raw.txt")))


@pytest.mark.skipif(
    not _probe_artifacts_present(),
    reason="no persisted window-probe artifacts (no paid Sonnet run yet — #62 is net-free)",
)
def test_replay_rescores_real_window_assembly_net_free() -> None:
    # When a real paid run HAS persisted its raw responses, --replay re-scores the
    # real window assembly net-free and reconciles: every rated VOD's scored +
    # dropped windows account for exactly the assembled gold + control set (the
    # fail-closed invariant), and every score is a real 0-1. No network, no re-pay.
    manifest = eval_clip_finder.load_manifest()
    anchors = eval_clip_finder.read_anchor_vod_ids()
    (raw, pool, final_by_vod, segments_by_vod, response_by_vod,
     chunk_duration_by_vod, _gaps) = swp.build_inputs(anchors, _STORAGE_DIR)
    result = swp.run(
        manifest, raw, pool, final_by_vod,
        segments_by_vod=segments_by_vod, response_by_vod=response_by_vod,
        chunk_duration_by_vod=chunk_duration_by_vod, anchor_vod_ids=anchors)

    # Rate only the VODs that actually have a persisted raw (a partial paid run is
    # honestly partial, never re-priced); reconcile each against its assembly.
    rated = {v: e for v, e in result["vods"].items()
             if swp._raw_path(_ARTIFACTS_DIR, v).exists()}
    assert rated, "guard says a raw exists, but none mapped to an assembled VOD"
    for vod_id, entry in rated.items():
        windows = swp.vod_windows(entry)
        out = swp.replay_make_rater(_ARTIFACTS_DIR)(vod_id, windows)
        assert len(out["scored"]) + len(out["dropped"]) == len(windows)
        assert all(0.0 <= w["score"] <= 1.0 for w in out["scored"])


@pytest.mark.skipif(
    not _probe_artifacts_present(),
    reason="no persisted window-probe artifacts (no paid Sonnet run yet — #62 is net-free)",
)
def test_separation_gate_reconciles_on_real_persisted_scores() -> None:
    # The #63 gate over the REAL persisted scores, net-free: re-score the real
    # assembly from disk, run the within-set separation + 0.66 gate, and pin (a)
    # the gold set against #56 (frozen 17/9, no re-classification), (b) every tier
    # produces a legitimate gate route, (c) the headline verdict drives routing,
    # (d) every measured percentile is a real probability.
    manifest = eval_clip_finder.load_manifest()
    anchors = eval_clip_finder.read_anchor_vod_ids()
    (raw, pool, final_by_vod, segments_by_vod, response_by_vod,
     chunk_duration_by_vod, _gaps) = swp.build_inputs(anchors, _STORAGE_DIR)
    result = swp.run(
        manifest, raw, pool, final_by_vod,
        segments_by_vod=segments_by_vod, response_by_vod=response_by_vod,
        chunk_duration_by_vod=chunk_duration_by_vod, anchor_vod_ids=anchors)

    # (a) Reuse guard: the gold windows are still exactly #56's llm_skipped (17/9).
    assert result["counts"]["gold_windows"] == {"floor": 17, "headline": 9}

    # Re-score only the VODs with a persisted raw (a partial paid run stays honest),
    # then run the separation + gate over that rating.
    rated_vods: dict[str, dict] = {}
    for vod_id, entry in result["vods"].items():
        if not swp._raw_path(_ARTIFACTS_DIR, vod_id).exists():
            continue
        out = swp.replay_make_rater(_ARTIFACTS_DIR)(vod_id, swp.vod_windows(entry))
        rated_vods[vod_id] = {"scored": out["scored"], "dropped": out["dropped"]}
    assert rated_vods, "guard says a raw exists, but none mapped to an assembled VOD"

    separation = swp.analyze_separation({"vods": rated_vods})

    # (b)/(c) Each tier resolves to a real route; the headline drives the verdict.
    for tier in ("floor", "headline"):
        assert separation["tiers"][tier]["gate"]["route"] in (swp.ESCALATE_B, swp.PIVOT_C)
    assert separation["routing_tier"] == "headline"
    assert separation["verdict"] == separation["tiers"]["headline"]["gate"]["verdict"]
    # (d) Every gold window is a real probability or a surfaced unmeasurable.
    for w in separation["windows"]:
        assert w["percentile"] is None or 0.0 <= w["percentile"] <= 1.0
