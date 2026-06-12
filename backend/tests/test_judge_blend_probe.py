"""Tests for the Judge-Blend Survival-Probe (ml/judge_blend_probe.py).

The free, read-only #55 follow-up to the #54 separability probe. #54 proved the
paid judge separates the buried gold *within the pool* (headline median 0.737);
this probe asks the gating question PRD #51 never ran: does that within-pool
signal **survive the snap→dedupe→diversity→top_k cut into the global top-20** when
the judge is *softly* blended (bounded-α) instead of hard top-concentrated (#50)?

A ``rank_score_fn`` rank-normalizes the finder composite and the judge virality
within each VOD pool and convex-blends them:  score = (1-α)·finder' + α·judge'.
Run through the real harness (replay → #49 → directional_eval) over an α-sweep,
the curve drives a pre-fixed 3-way verdict: B(distill) / B(gating-feature) /
Generation. These tests pin the blend math and the verdict on hand-built fixtures
— never a network or paid call.
"""
from __future__ import annotations

import json
import os
import sys
import types

import pytest

# Defensive: the replay chain reaches RichSegment, which can pull the heavy
# transcription stack. Prior art stubs whisper so the import stays cheap.
sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_a, **_k: None))

from ml import eval_clip_finder, gap_decomposition, judge_blend_probe as jbp, separability_probe as sp


def _row(start: float, end: float, composite: float) -> dict:
    """A deduped candidate row the blend rescore receives (window + finder composite)."""
    return {"start_time": float(start), "end_time": float(end),
            "composite_score": float(composite)}


def _judged(start: float, end: float, score: float) -> dict:
    """One persisted judged candidate (judge_spike/*.scores.json shape): the window
    plus the judge's repost-virality on the 1-10 composite_score scale."""
    return {"start_time": float(start), "end_time": float(end),
            "composite_score": float(score), "reason": ""}


# ─────────────────────────────────────────────────────────────────────────────
# Slice 1 — rank01: within-pool rank-normalize to [0,1], ties → average rank
# ─────────────────────────────────────────────────────────────────────────────

def test_rank01_normalizes_to_unit_interval_with_avg_rank_for_ties() -> None:
    # Distinct ascending values map evenly onto [0,1]: lowest 0.0, highest 1.0.
    assert jbp.rank01([10.0, 20.0, 30.0]) == [0.0, 0.5, 1.0]

    # Order-independent: a value keeps its normalized rank wherever it sits.
    assert jbp.rank01([30.0, 10.0, 20.0]) == [1.0, 0.0, 0.5]

    # Ties share the average of their ranks. [5,5,3]: 3 is the lowest (rank 1 →
    # 0.0); the two 5s share ranks 2 and 3 → avg 2.5 → (2.5-1)/2 = 0.75.
    assert jbp.rank01([5.0, 5.0, 3.0]) == [0.75, 0.75, 0.0]


# ─────────────────────────────────────────────────────────────────────────────
# Slice 2 — the blend convex-combines rank-normalized finder ⊕ judge per window
# ─────────────────────────────────────────────────────────────────────────────

def test_blend_convex_combines_rank_normalized_finder_and_judge() -> None:
    # Three deduped candidates; the judge scored the same windows. finder and judge
    # disagree on the order, so the blend at α=0.5 is a genuine mix.
    rows = [_row(100, 120, 9.0), _row(200, 220, 5.0), _row(300, 320, 1.0)]
    judged = [_judged(100, 120, 2.0), _judged(200, 220, 8.0), _judged(300, 320, 6.0)]

    out = jbp.make_blend_rank_score_fn(judged, alpha=0.5)(rows)

    # finder01 = rank01([9,5,1]) = [1.0, 0.5, 0.0]
    # judge01  = rank01([2,8,6]) = [0.0, 1.0, 0.5]   (joined to each row by window)
    # α=0.5 →   [.5·1+.5·0, .5·.5+.5·1, .5·0+.5·.5] = [0.5, 0.75, 0.25]
    assert [r["composite_score"] for r in out] == [0.5, 0.75, 0.25]

    # The rescore overrides composite_score only — window + row order are preserved
    # (the downstream rank step does the reordering, not this fn).
    assert [(r["start_time"], r["end_time"]) for r in out] == [(100, 120), (200, 220), (300, 320)]

    # The input rows are left untouched (new dicts returned).
    assert [r["composite_score"] for r in rows] == [9.0, 5.0, 1.0]


# ─────────────────────────────────────────────────────────────────────────────
# Slice 3 — α=0 ranks by finder alone (judge ignored): the baseline reproduction
# ─────────────────────────────────────────────────────────────────────────────

def test_alpha_zero_ranks_by_finder_alone_ignoring_judge() -> None:
    rows = [_row(100, 120, 3.0), _row(200, 220, 9.0), _row(300, 320, 1.0), _row(400, 420, 7.0)]
    # Judge scores that would reshuffle the order entirely IF they leaked at α=0.
    judged = [_judged(100, 120, 9.0), _judged(200, 220, 1.0),
              _judged(300, 320, 8.0), _judged(400, 420, 2.0)]

    out = jbp.make_blend_rank_score_fn(judged, alpha=0.0)(rows)

    # At α=0 the blend is finder' alone — a monotonic image of the composite — so
    # ranking by the blended score reproduces the pure finder order, judge ignored.
    ranked_windows = [r["start_time"] for r in sorted(out, key=lambda r: r["composite_score"], reverse=True)]
    assert ranked_windows == [200.0, 400.0, 100.0, 300.0]  # composites 9 > 7 > 3 > 1


# ─────────────────────────────────────────────────────────────────────────────
# Slice 4 — a judge-missing candidate keeps its pure finder' (invariant to α)
# ─────────────────────────────────────────────────────────────────────────────

def test_judge_missing_candidate_falls_back_to_pure_finder() -> None:
    rows = [_row(100, 120, 10.0), _row(200, 220, 7.0), _row(300, 320, 4.0), _row(400, 420, 1.0)]
    # The judge scored every window EXCEPT 300-320 (it sat beyond the shortlist).
    judged = [_judged(100, 120, 3.0), _judged(200, 220, 9.0), _judged(400, 420, 6.0)]

    out_half = jbp.make_blend_rank_score_fn(judged, alpha=0.5)(rows)
    out_low = jbp.make_blend_rank_score_fn(judged, alpha=0.3)(rows)

    # finder01 = rank01([10,7,4,1]) = [1.0, 2/3, 1/3, 0.0]. The unmatched 300-320
    # candidate keeps finder' = 1/3 at BOTH α — never pushed down or boosted.
    assert out_half[2]["composite_score"] == pytest.approx(1 / 3)
    assert out_low[2]["composite_score"] == pytest.approx(1 / 3)

    # A matched candidate DOES move with α (the blend is live), so r3 being fixed is
    # the fallback at work, not a dead blend. r1: judge'=0.0 → (1-α)·1.0.
    assert out_half[0]["composite_score"] == pytest.approx(0.5)
    assert out_low[0]["composite_score"] == pytest.approx(0.7)


# ─────────────────────────────────────────────────────────────────────────────
# Slice 5 — oracle scoping: blend ONLY known gold-covering candidates (leakage)
# ─────────────────────────────────────────────────────────────────────────────

def test_oracle_blends_only_gold_covering_candidates() -> None:
    rows = [_row(100, 120, 9.0), _row(200, 220, 5.0), _row(300, 320, 1.0)]
    judged = [_judged(100, 120, 2.0), _judged(200, 220, 8.0), _judged(300, 320, 6.0)]
    # r1 covers G1, r3 covers G2; r2 (200-220) covers no gold — the eval-view labels.
    gold = [{"start": 100.0, "end": 130.0}, {"start": 300.0, "end": 330.0}]

    uniform = jbp.make_blend_rank_score_fn(judged, alpha=0.5)(rows)
    oracle = jbp.make_blend_rank_score_fn(judged, alpha=0.5, gold=gold)(rows)

    # Uniform blends every matched candidate, so the non-gold r2 still gets a judge'
    # (0.75, from slice 2's fixture).
    assert uniform[1]["composite_score"] == pytest.approx(0.75)
    # Oracle restricts the judge boost to the labeled gold-covering candidates: the
    # non-gold r2 falls back to pure finder' = 0.5. This is the deliberate leakage
    # the upper-bound pass is built on.
    assert oracle[1]["composite_score"] == pytest.approx(0.5)
    # The gold-covering r1 is still blended — judge'=0.0 pulls it off its finder' 1.0.
    assert oracle[0]["composite_score"] == pytest.approx(0.5)


# ── α-curve fixtures for the verdict ─────────────────────────────────────────
def _curve_cell(alpha: float, head: float | None, floor: float | None,
                *, ar_floor: float | None = 0.0, ar_head: float | None = 0.0) -> dict:
    """One α column of the sweep: head/floor recall@20 + avg_rank Δ per tier."""
    return {"alpha": alpha, "head_recall": head, "floor_recall": floor,
            "avg_rank_delta": {"floor": ar_floor, "headline": ar_head}}


def _flat_oracle(head: float = 0.0) -> list[dict]:
    return [_curve_cell(a, head, 0.0) for a in jbp.ALPHA_SWEEP]


# ─────────────────────────────────────────────────────────────────────────────
# Slice 6a — verdict: a uniform α that clears all three gates → B (distill)
# ─────────────────────────────────────────────────────────────────────────────

def test_verdict_uniform_pass_is_b_distill() -> None:
    uniform = [
        _curve_cell(0.0, 0.125, 0.118),                              # baseline column
        _curve_cell(0.1, 0.125, 0.118),
        _curve_cell(0.2, 0.167, 0.118),
        _curve_cell(0.3, 0.208, 0.118, ar_floor=2.0, ar_head=3.0),   # PASS: head≥.208, floor≥.118, Δ≤5
        _curve_cell(0.4, 0.208, 0.078),                              # floor would fail here
        _curve_cell(0.5, 0.250, 0.118),
    ]

    verdict = jbp.classify_verdict(uniform, _flat_oracle())

    assert verdict["verdict"] == jbp.VERDICT_B_DISTILL
    assert verdict["uniform"]["passed"] is True
    assert verdict["uniform"]["alpha"] == 0.3   # the first α that clears every gate
    assert verdict["oracle"]["passed"] is False
    # Every threshold is read explicitly into the report (mechanical, not narrated).
    assert verdict["thresholds"] == {
        "head_recall_gate": 0.208, "floor_recall_gate": 0.118,
        "avg_rank_delta_cap": 5.0, "alpha_cap": 0.5,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Slice 6b — uniform fails but the leakage oracle clears head → B (gating-feature)
# ─────────────────────────────────────────────────────────────────────────────

def test_verdict_oracle_pass_when_uniform_fails_is_b_gating() -> None:
    uniform = [_curve_cell(a, 0.167, 0.118) for a in jbp.ALPHA_SWEEP]  # head 0.167 < 0.208 always
    oracle = [
        _curve_cell(0.0, 0.125, 0.0), _curve_cell(0.1, 0.167, 0.0), _curve_cell(0.2, 0.167, 0.0),
        _curve_cell(0.3, 0.167, 0.0), _curve_cell(0.4, 0.292, 0.0), _curve_cell(0.5, 0.292, 0.0),
    ]

    verdict = jbp.classify_verdict(uniform, oracle)

    assert verdict["verdict"] == jbp.VERDICT_B_GATING
    assert verdict["uniform"]["passed"] is False
    assert verdict["oracle"]["passed"] is True
    assert verdict["oracle"]["alpha"] == 0.4   # the leakage upper bound, never a gate number


# ─────────────────────────────────────────────────────────────────────────────
# Slice 6c — even the oracle upper bound fails → Generation-Track
# ─────────────────────────────────────────────────────────────────────────────

def test_verdict_generation_when_even_oracle_fails() -> None:
    uniform = [_curve_cell(a, 0.167, 0.118) for a in jbp.ALPHA_SWEEP]
    oracle = [_curve_cell(a, 0.167, 0.0) for a in jbp.ALPHA_SWEEP]   # never reaches 0.208

    verdict = jbp.classify_verdict(uniform, oracle)

    assert verdict["verdict"] == jbp.VERDICT_GENERATION
    assert verdict["uniform"]["passed"] is False
    assert verdict["oracle"]["passed"] is False


# ─────────────────────────────────────────────────────────────────────────────
# Slice 6d — the α=0 baseline column is excluded from the PASS search
# ─────────────────────────────────────────────────────────────────────────────

def test_verdict_excludes_alpha_zero_baseline_from_pass_search() -> None:
    # ONLY the α=0 column (no judge contribution) clears the numbers; every α>0 falls
    # short. A win at α=0 is not a judge win — it must not count as a PASS.
    uniform = [_curve_cell(0.0, 0.30, 0.20)] + [_curve_cell(a, 0.125, 0.118) for a in jbp.ALPHA_SWEEP[1:]]
    oracle = [_curve_cell(0.0, 0.30, 0.0)] + [_curve_cell(a, 0.125, 0.0) for a in jbp.ALPHA_SWEEP[1:]]

    verdict = jbp.classify_verdict(uniform, oracle)

    assert verdict["uniform"]["passed"] is False
    assert verdict["oracle"]["passed"] is False
    assert verdict["verdict"] == jbp.VERDICT_GENERATION


# ─────────────────────────────────────────────────────────────────────────────
# Slice 6e — uniform guardrails (floor recall, avg_rank Δ) block an otherwise-pass
# ─────────────────────────────────────────────────────────────────────────────

def test_verdict_uniform_floor_and_avg_rank_guardrails_block_pass() -> None:
    # Head clears at α=0.3, but the floor falls below 0.118 → blocked (un-shippable).
    floor_fail = [_curve_cell(0.3, 0.25, 0.10)]
    assert jbp.classify_verdict(floor_fail, _flat_oracle())["uniform"]["passed"] is False

    # Head + floor clear, but the headline avg_rank Δ is +6.0 (> the +5 cut) → blocked.
    rank_fail = [_curve_cell(0.3, 0.25, 0.118, ar_head=6.0)]
    assert jbp.classify_verdict(rank_fail, _flat_oracle())["uniform"]["passed"] is False

    # The same column with Δ back within +5 passes — proving the guard, not a dead gate.
    ok = [_curve_cell(0.3, 0.25, 0.118, ar_head=4.0)]
    assert jbp.classify_verdict(ok, _flat_oracle())["uniform"]["passed"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Slice 7 — score_curve_cell extracts one α-column through the real #49 harness
# ─────────────────────────────────────────────────────────────────────────────

def _gold(start: float, end: float, *, views: float) -> dict:
    return {"start": float(start), "end": float(end), "views": float(views), "gt_tier": "gold"}


def test_score_curve_cell_reads_recall_and_avg_rank_delta_through_decomposition() -> None:
    # One VOD, two gold: a 300k headline gold (covered at rank 2) and a 50k floor-only
    # gold (uncovered). The metrics come straight out of gap_decomposition.run → no
    # re-implementation here, only extraction.
    manifest = {"V": [_gold(100, 130, views=300_000), _gold(200, 230, views=50_000)]}
    raw = {"V": [_row(102, 122, 8.0), _row(202, 222, 3.0)]}
    pool = raw
    final = {"V": [_row(50, 70, 9.0), _row(102, 122, 8.0)]}  # rank 2 covers the headline gold
    baseline = {
        "floor": {"recall": {"20": 0.0}, "avg_rank": 10.0},
        "headline": {"recall": {"20": 0.0}, "avg_rank": 5.0},
        "generation": {"floor": {"recall_any": 0.0}, "headline": {"recall_any": 0.0}},
    }

    cell = jbp.score_curve_cell(manifest, raw, pool, final, baseline,
                                alpha=0.3, anchor_vod_ids=["V"])

    assert cell["alpha"] == 0.3
    assert cell["head_recall"] == 1.0    # 1/1 headline gold covered
    assert cell["floor_recall"] == 0.5   # 1/2 floor gold covered
    # avg_rank of the single hit is 2.0; Δ vs the baseline avg_ranks (5.0 hd, 10.0 fl).
    assert cell["avg_rank_delta"] == {"headline": -3.0, "floor": -8.0}


# ─────────────────────────────────────────────────────────────────────────────
# Slice 8 — the CLI runs the whole probe over fake storage (no network, no paid)
# ─────────────────────────────────────────────────────────────────────────────

def _moment_row(rank: int, start: float, end: float, composite: float) -> dict:
    """A raw llm_moment_finder row (the replay artifact shape)."""
    virality = composite / 10.0
    return {"rank": rank, "start_time": start, "end_time": end,
            "viral_score": composite, "composite_score": composite, "llm_virality": virality,
            "hook": virality, "flow": virality, "value": virality, "trend": virality, "reason": ""}


def _write_job_dir(storage, vod_id: str, *, raw: list[dict], segments: list[dict]) -> None:
    debug = storage / f"pipeline_debug_20260101T000000000000Z_{vod_id}" / "debug"
    debug.mkdir(parents=True)
    (debug / "llm_moment_finder_moments_raw.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in raw), encoding="utf-8")
    (debug / "segments.jsonl").write_text(
        "".join(json.dumps(s) + "\n" for s in segments), encoding="utf-8")
    os.utime(debug.parent, (1000.0, 1000.0))


def test_main_runs_full_probe_over_fake_storage_and_surfaces_gap(tmp_path) -> None:
    # One judged VOD (V1) with a headline gold + a floor gold, both covered; one gold
    # VOD (GAP) with no artifacts at all → a surfaced coverage gap, never a miss.
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps({"vod_id": "V1", "gt_moments": [
            {"start": 1.0, "end": 7.0, "views": 300_000, "gt_tier": "gold", "desc": "headline"},
            {"start": 101.0, "end": 107.0, "views": 50_000, "gt_tier": "gold", "desc": "floor"},
        ]}) + "\n"
        + json.dumps({"vod_id": "GAP", "gt_moments": [
            {"start": 1.0, "end": 7.0, "views": 900_000, "gt_tier": "gold", "desc": "uncovered"},
        ]}) + "\n", encoding="utf-8")

    anchor_csv = tmp_path / "anchors.csv"
    anchor_csv.write_text("vod_id\nV1\nGAP\n", encoding="utf-8")

    storage = tmp_path / "storage"
    storage.mkdir()
    segments = [{"start": 0.0, "end": 4.0, "text": "Opening line one."},
                {"start": 4.0, "end": 8.0, "text": "Opening line two."},
                {"start": 100.0, "end": 104.0, "text": "Later line one."},
                {"start": 104.0, "end": 108.0, "text": "Later line two."}]
    _write_job_dir(storage, "V1",
                   raw=[_moment_row(1, 0.0, 8.0, 8.0), _moment_row(2, 100.0, 108.0, 6.0)],
                   segments=segments)

    # The already-paid judged pool for V1: the two candidate windows, scored.
    artifacts_dir = tmp_path / "judge_spike"
    artifacts_dir.mkdir()
    (artifacts_dir / "V1.scores.json").write_text(json.dumps([
        {"start_time": 0.0, "end_time": 8.0, "composite_score": 5.0, "reason": ""},
        {"start_time": 100.0, "end_time": 108.0, "composite_score": 7.0, "reason": ""},
    ]) + "\n", encoding="utf-8")

    # Both candidates are always served → recall stays 1.0/1.0 at every α; the α=0
    # column must reproduce this frozen baseline (the built-in sanity check).
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps({
        "floor": {"recall": {"5": 1.0, "10": 1.0, "20": 1.0, "50": 1.0}, "avg_rank": 1.5},
        "headline": {"recall": {"5": 1.0, "10": 1.0, "20": 1.0, "50": 1.0}, "avg_rank": 1.0},
        "generation": {"floor": {"recall_any": 1.0}, "headline": {"recall_any": 1.0}},
    }) + "\n", encoding="utf-8")

    out_path = tmp_path / "report.json"
    rc = jbp.main([
        "--manifest", str(manifest_path), "--anchor-csv", str(anchor_csv),
        "--predictions-dir", str(storage), "--baseline", str(baseline_path),
        "--artifacts-dir", str(artifacts_dir), "--output", str(out_path), "--json",
    ])

    assert rc == 0
    report = json.loads(out_path.read_text(encoding="utf-8"))

    # The full α-curve, both passes, length 6, alphas in sweep order.
    assert [c["alpha"] for c in report["uniform_curve"]] == list(jbp.ALPHA_SWEEP)
    assert [c["alpha"] for c in report["oracle_curve"]] == list(jbp.ALPHA_SWEEP)

    # GAP is gold but has no judged/replay artifact → surfaced, never counted a miss.
    assert report["scored_vods"] == ["V1"]
    assert report["coverage_gap_vods"] == ["GAP"]

    # The α=0 column reproduces the frozen baseline (head + floor recall@20).
    assert report["alpha0_sanity"]["reproduces_baseline"] is True

    # Recall holds at 1.0/1.0 for every α → the uniform gate clears → B (distill).
    assert report["verdict"]["verdict"] == jbp.VERDICT_B_DISTILL
    assert report["verdict"]["uniform"]["passed"] is True

    for key in ("uniform_curve", "oracle_curve", "verdict", "coverage_gap_vods",
                "alpha0_sanity", "shortlist_depth", "computed_at"):
        assert key in report


# ─────────────────────────────────────────────────────────────────────────────
# Guard — reconcile against the real paid artifacts (skips without them)
# ─────────────────────────────────────────────────────────────────────────────

_BASELINE_PATH = gap_decomposition.DEFAULT_BASELINE
_STORAGE_DIR = eval_clip_finder.DEFAULT_STORAGE_DIR
_UNJUDGED_GOLD_VODS = {"2756055866", "2764976309", "2770356205", "2771207666", "2773650615"}


def _real_artifacts_present() -> bool:
    """The (gitignored) replay artifacts + the frozen baseline + a judged pool."""
    return (
        _BASELINE_PATH.exists()
        and eval_clip_finder.DEFAULT_MANIFEST.exists()
        and any(_STORAGE_DIR.glob("pipeline_debug_*"))
        and any(sp.DEFAULT_ARTIFACTS_DIR.glob("*.scores.json"))
    )


@pytest.mark.skipif(
    not _real_artifacts_present(),
    reason="real pipeline_debug / judge_spike artifacts / baseline not present (gitignored)",
)
def test_alpha0_reproduces_frozen_baseline_on_real_artifacts() -> None:
    # The hard sanity-assert from the PRD: α=0 (rank-normalized finder alone) must
    # reproduce finder_recall_baseline.json exactly — read off the real persisted
    # judged pools the paid spike (#50) wrote, no paid call.
    manifest = eval_clip_finder.load_manifest()
    anchors = eval_clip_finder.read_anchor_vod_ids()
    baseline = json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))
    judged = sp.load_judged_by_vod(sp.DEFAULT_ARTIFACTS_DIR, anchors)

    result = jbp.run(manifest, anchors, _STORAGE_DIR, judged, baseline)

    assert result["alpha0_sanity"]["reproduces_baseline"] is True
    cell0 = result["uniform_curve"][0]
    assert cell0["alpha"] == 0.0
    assert cell0["head_recall"] == baseline["headline"]["recall"]["20"]
    assert cell0["floor_recall"] == baseline["floor"]["recall"]["20"]

    # The 5 un-judged gold VODs surface as the coverage gap — measured-not-counted.
    assert set(result["coverage_gap_vods"]) == _UNJUDGED_GOLD_VODS
