"""Tests for the Judge-Artefakt Separability-Probe (ml/separability_probe.py).

The probe is the free, read-only #54 follow-up to the paid judge spike (#50): it
re-reads the already-paid per-candidate judge scores (judge_spike/*.scores.json)
and asks the one question the blended-rank gate (#50) could not — did the strong
judge lift the clean_but_buried gold (the #49 bucket) ABOVE their non-gold
siblings *within the pool*, independent of the final concentrated rank?

These tests pin the within-pool separation measure and the pre-fixed
getrennt→B / nicht-getrennt→Generation verdict through the public interface, on
hand-built judged-pool fixtures — never a network or paid call.
"""
from __future__ import annotations

import json

import pytest

from ml import eval_clip_finder, separability_probe as sp


def _gold(start: float, end: float, *, views: float = 0.0, desc: str = "") -> dict:
    """A gold moment, the manifest shape the probe reads (start/end/views/tier)."""
    return {"start": float(start), "end": float(end), "views": float(views),
            "gt_tier": "gold", "desc": desc}


def _judged(start: float, end: float, *, score: float, reason: str = "") -> dict:
    """One persisted judged candidate (judge_spike/*.scores.json shape): the window
    plus the judge's repost-virality on the 1-10 composite_score scale."""
    return {"start_time": float(start), "end_time": float(end),
            "composite_score": float(score), "reason": reason}


# ─────────────────────────────────────────────────────────────────────────────
# Slice 1 — the buried gold's covering candidate beats every non-gold sibling
# ─────────────────────────────────────────────────────────────────────────────

def test_separation_percentile_is_one_when_gold_beats_all_siblings() -> None:
    gold = _gold(100, 130)
    # The gold's real (20s, clip-sized) covering candidate scores highest in the
    # pool; three non-gold siblings (far from the gold window) score below it.
    judged = [
        _judged(102, 122, score=8.1),   # real covering candidate for the gold
        _judged(500, 520, score=7.0),
        _judged(600, 620, score=5.0),
        _judged(700, 720, score=3.0),
    ]

    sep = sp.within_pool_separation(gold, [gold], judged)

    assert sep is not None
    assert sep["gold_score"] == 8.1
    assert sep["sibling_n"] == 3
    assert sep["beaten"] == 3
    # Beats all 3 non-gold siblings -> top of the pool -> fully separated.
    assert sep["percentile"] == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Slice 2 — a mid-pack covering candidate; ties count as not-beaten (strictly <)
# ─────────────────────────────────────────────────────────────────────────────

def test_separation_percentile_counts_only_strictly_beaten_siblings() -> None:
    gold = _gold(100, 130)
    # The gold's covering candidate scores 5.0, mid-pack. Of five non-gold siblings
    # two score strictly below (4.0, 3.0); one ties at 5.0 (NOT beaten); two are above.
    judged = [
        _judged(102, 122, score=5.0),   # real covering candidate for the gold
        _judged(500, 520, score=9.0),
        _judged(600, 620, score=7.0),
        _judged(700, 720, score=5.0),   # tie with the gold score -> not beaten
        _judged(800, 820, score=4.0),
        _judged(900, 920, score=3.0),
    ]

    sep = sp.within_pool_separation(gold, [gold], judged)

    assert sep is not None
    assert sep["gold_score"] == 5.0
    assert sep["sibling_n"] == 5
    assert sep["beaten"] == 2          # only 4.0 and 3.0; the 5.0 tie does not count
    assert sep["percentile"] == 0.4    # below the 0.66 cut -> not separated


# ─────────────────────────────────────────────────────────────────────────────
# Slice 3 — no REAL covering candidate in the judged pool -> not measurable (None)
# ─────────────────────────────────────────────────────────────────────────────

def test_separation_is_none_when_no_real_covering_candidate_in_pool() -> None:
    gold = _gold(100, 130)

    # (a) No judged candidate covers the gold window at all — its covering candidate
    # was truncated beyond the judged shortlist. Not measurable here, never a 0.0.
    truncated = [_judged(500, 520, score=9.0), _judged(600, 620, score=8.0)]
    assert sp.within_pool_separation(gold, [gold], truncated) is None

    # (b) Only an oversized (350s) accidental window covers the gold — no real,
    # clip-sized candidate does, so there is no gold candidate to separate.
    accidental_only = [_judged(0, 350, score=9.0), _judged(600, 620, score=8.0)]
    assert sp.within_pool_separation(gold, [gold], accidental_only) is None


# ─────────────────────────────────────────────────────────────────────────────
# Slice 4 — siblings are NON-gold only: a candidate covering another gold is out
# ─────────────────────────────────────────────────────────────────────────────

def test_candidate_covering_another_gold_is_excluded_from_siblings() -> None:
    gold1 = _gold(100, 130)
    gold2 = _gold(500, 530)
    judged = [
        _judged(102, 122, score=6.0),   # covers gold1 (the measured gold)
        _judged(502, 522, score=9.0),   # covers gold2 -> NOT a non-gold sibling
        _judged(700, 720, score=3.0),   # non-gold sibling
        _judged(800, 820, score=4.0),   # non-gold sibling
    ]

    sep = sp.within_pool_separation(gold1, [gold1, gold2], judged)

    assert sep is not None
    assert sep["gold_score"] == 6.0
    # Only the two non-gold windows are siblings; the 9.0 gold2-cand is excluded,
    # so gold1's candidate still beats every sibling (would be 2/3 if mis-counted).
    assert sep["sibling_n"] == 2
    assert sep["percentile"] == 1.0


# ── Hand-computed tier fixture ───────────────────────────────────────────────
# Two VODs. A: one buried gold (measurable) + one gold no candidate covers
# (unmeasurable). B: two buried gold, both measurable. Every count + percentile
# below is computed by hand from these windows + judge scores.
def _tier_fixture() -> tuple[dict, dict]:
    gold_by_vod = {
        "A": [_gold(100, 130, views=300_000), _gold(200, 230, views=50_000)],
        "B": [_gold(400, 430, views=500_000), _gold(600, 630, views=80_000)],
    }
    judged_by_vod = {
        "A": [
            _judged(102, 122, score=8.0),   # covers A's gold1 -> beats both siblings (1.0)
            _judged(500, 520, score=4.0),   # sibling
            _judged(600, 620, score=2.0),   # sibling
        ],                                  # A's gold2 (200-230): no candidate -> unmeasurable
        "B": [
            _judged(402, 422, score=5.0),   # covers B's gold1
            _judged(602, 622, score=6.0),   # covers B's gold2
            _judged(800, 820, score=3.0),   # sibling
            _judged(900, 920, score=9.0),   # sibling
            _judged(1000, 1020, score=1.0),  # sibling
        ],
    }
    return gold_by_vod, judged_by_vod


# ─────────────────────────────────────────────────────────────────────────────
# Slice 5 — probe_tier aggregates the median percentile + measurable counts
# ─────────────────────────────────────────────────────────────────────────────

def test_probe_tier_floor_aggregates_median_and_counts() -> None:
    gold_by_vod, judged_by_vod = _tier_fixture()

    tier = sp.probe_tier(gold_by_vod, judged_by_vod, min_views=0.0)

    # 4 gold at the floor; A's gold2 has no covering candidate -> unmeasurable.
    assert tier["gold_count"] == 4
    assert tier["measurable"] == 3
    assert tier["unmeasurable"] == 1
    # Percentiles: A1 1.0, B1 2/3 (beats 3.0 & 1.0, not 9.0), B2 2/3 -> median 2/3.
    assert tier["median_percentile"] == pytest.approx(2 / 3)


def test_probe_tier_headline_filters_by_views() -> None:
    gold_by_vod, judged_by_vod = _tier_fixture()

    tier = sp.probe_tier(gold_by_vod, judged_by_vod, min_views=100_000)

    # Only A1 (300k) and B1 (500k) clear the headline floor; both measurable.
    assert tier["gold_count"] == 2
    assert tier["measurable"] == 2
    assert tier["unmeasurable"] == 0
    # Percentiles [1.0, 2/3] -> median (1.0 + 2/3) / 2.
    assert tier["median_percentile"] == pytest.approx((1.0 + 2 / 3) / 2)


# ─────────────────────────────────────────────────────────────────────────────
# Slice 6 — classify_separation: the pre-fixed 0.66 verdict, read explicitly
# ─────────────────────────────────────────────────────────────────────────────

def test_classify_separation_applies_the_prefixed_threshold() -> None:
    # Above the upper-third cut -> the signal is separable -> Slice B.
    sep = sp.classify_separation(0.71)
    assert sep["separated"] is True
    assert sep["verdict"] == sp.VERDICT_SEPARATED
    assert "0.66" in sep["reason"]

    # At the cut counts as separated (>=), not below it.
    assert sp.classify_separation(0.66)["separated"] is True

    # Below the cut -> even a strong judge does not lift the buried gold -> Generation.
    miss = sp.classify_separation(0.48)
    assert miss["separated"] is False
    assert miss["verdict"] == sp.VERDICT_NOT_SEPARATED


def test_classify_separation_fails_closed_with_no_measurable_gold() -> None:
    # No measurable gold -> an unmeasurable signal is never read as separated.
    none = sp.classify_separation(None)
    assert none["separated"] is False
    assert none["verdict"] == sp.VERDICT_NOT_SEPARATED


# ─────────────────────────────────────────────────────────────────────────────
# Slice 7 — highest-view buried misses, each followed individually (separated?)
# ─────────────────────────────────────────────────────────────────────────────

def test_top_view_buried_misses_ordered_by_views_each_tagged_separated() -> None:
    gold_by_vod = {
        "A": [
            _gold(100, 130, views=1_400_000, desc="ChainsFR reveal"),  # separated
            _gold(200, 230, views=800_000, desc="buried beat"),        # not separated
            _gold(900, 930, views=5_000_000, desc="uncovered"),        # unmeasurable
        ],
    }
    judged_by_vod = {
        "A": [
            _judged(102, 122, score=9.0),   # covers the reveal -> beats both siblings
            _judged(202, 222, score=2.0),   # covers the beat -> below both siblings
            _judged(500, 520, score=5.0),   # sibling
            _judged(600, 620, score=7.0),   # sibling
        ],                                  # nothing covers the 5M gold -> unmeasurable
    }

    misses = sp.top_view_buried_misses(gold_by_vod, judged_by_vod)

    # The uncovered 5M gold has no candidate in the pool -> not a measurable buried
    # miss; the two measurable ones come back ordered by views desc.
    assert [m["desc"] for m in misses] == ["ChainsFR reveal", "buried beat"]
    assert misses[0]["views"] == 1_400_000.0
    assert misses[0]["percentile"] == 1.0
    assert misses[0]["separated"] is True
    assert misses[1]["percentile"] == 0.0
    assert misses[1]["separated"] is False


# ─────────────────────────────────────────────────────────────────────────────
# Slice 8 — run(): both tier verdicts, the misses table, surfaced coverage gap
# ─────────────────────────────────────────────────────────────────────────────

def test_run_reports_both_tier_verdicts_and_surfaces_coverage_gap() -> None:
    manifest = {
        "A": [
            _gold(100, 130, views=300_000, desc="reveal"),
            _gold(200, 230, views=50_000, desc="beat2"),
        ],
        "GAP": [_gold(100, 130, views=900_000, desc="uncovered")],  # no judged artifact
    }
    judged_by_vod = {
        "A": [
            _judged(102, 122, score=8.0),   # covers reveal -> beats both siblings (1.0)
            _judged(202, 222, score=1.0),   # covers beat2 -> below both siblings (0.0)
            _judged(500, 520, score=3.0),   # sibling
            _judged(600, 620, score=2.0),   # sibling
        ],
    }

    result = sp.run(manifest, judged_by_vod, headline_min_views=100_000)

    # GAP is gold but has no judged artifact -> a surfaced coverage gap, never a miss.
    assert result["gold_vods"] == 2
    assert result["scored_vods"] == ["A"]
    assert result["coverage_gap_vods"] == ["GAP"]
    assert result["shortlist_depth"] == sp.DEFAULT_SHORTLIST_DEPTH

    floor = result["separation"]["floor"]
    headline = result["separation"]["headline"]
    # Floor median of [1.0, 0.0] = 0.5 < 0.66 -> nicht getrennt -> Generation.
    assert floor["median_percentile"] == 0.5
    assert floor["verdict"]["separated"] is False
    assert floor["verdict"]["verdict"] == sp.VERDICT_NOT_SEPARATED
    # Headline keeps only the 300k reveal (percentile 1.0) -> getrennt -> B.
    assert headline["measurable"] == 1
    assert headline["verdict"]["separated"] is True

    # The misses table follows both measurable buried gold, ordered by views.
    assert [m["desc"] for m in result["top_view_buried_misses"]] == ["reveal", "beat2"]


# ─────────────────────────────────────────────────────────────────────────────
# Slice 9 — the CLI re-reads the persisted scores.json artifacts (no network)
# ─────────────────────────────────────────────────────────────────────────────

def test_main_reads_scores_artifacts_and_writes_valid_report(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps({
            "vod_id": "V",
            "gt_moments": [{"start": 100.0, "end": 130.0, "views": 300_000,
                            "gt_tier": "gold", "desc": "reveal"}],
        }) + "\n", encoding="utf-8")
    anchor_csv = tmp_path / "anchors.csv"
    anchor_csv.write_text("vod_id\nV\n", encoding="utf-8")

    # The already-paid judged pool for V: the gold's covering candidate (102-122)
    # tops the pool over its two non-gold siblings.
    artifacts_dir = tmp_path / "judge_spike"
    artifacts_dir.mkdir()
    (artifacts_dir / "V.scores.json").write_text(json.dumps([
        {"start_time": 102.0, "end_time": 122.0, "composite_score": 8.0, "reason": ""},
        {"start_time": 500.0, "end_time": 520.0, "composite_score": 3.0, "reason": ""},
        {"start_time": 600.0, "end_time": 620.0, "composite_score": 2.0, "reason": ""},
    ]) + "\n", encoding="utf-8")
    out_path = tmp_path / "report.json"

    rc = sp.main([
        "--manifest", str(manifest_path),
        "--anchor-csv", str(anchor_csv),
        "--artifacts-dir", str(artifacts_dir),
        "--output", str(out_path),
        "--json",
    ])

    assert rc == 0
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    for key in ("separation", "top_view_buried_misses", "coverage_gap_vods",
                "shortlist_depth", "threshold", "computed_at"):
        assert key in artifact
    floor = artifact["separation"]["floor"]
    # V's gold candidate beats both siblings -> percentile 1.0 -> getrennt -> B.
    assert floor["median_percentile"] == 1.0
    assert floor["verdict"]["separated"] is True
    assert floor["verdict"]["verdict"] == sp.VERDICT_SEPARATED


# ─────────────────────────────────────────────────────────────────────────────
# Guard — reconcile against the real paid spike artifacts (skips without them)
# ─────────────────────────────────────────────────────────────────────────────

_SPIKE_REPORT = sp.DEFAULT_ARTIFACTS_DIR / "spike_report.json"


def _real_artifacts_present() -> bool:
    """The (gitignored) paid-run artifacts: the persisted #49 spike report, the gold
    manifest, and at least one persisted judged pool."""
    return (
        _SPIKE_REPORT.exists()
        and eval_clip_finder.DEFAULT_MANIFEST.exists()
        and any(sp.DEFAULT_ARTIFACTS_DIR.glob("*.scores.json"))
    )


@pytest.mark.skipif(
    not _real_artifacts_present(),
    reason="real judge_spike artifacts / spike_report not present (gitignored)",
)
def test_measurable_set_reconciles_with_49_clean_within_shortlist() -> None:
    # The probe re-reads the same persisted judged pools the paid spike (#50) wrote
    # and must measure exactly the #49 clean_but_buried gold reachable in the judged
    # top-200 shortlist (clean_within_shortlist) — proving the mapping onto the #49
    # buckets, and that the depth cap is accounted for, not silently absorbed.
    manifest = eval_clip_finder.load_manifest()
    anchors = eval_clip_finder.read_anchor_vod_ids()
    judged = sp.load_judged_by_vod(sp.DEFAULT_ARTIFACTS_DIR, anchors)

    result = sp.run(manifest, judged, anchor_vod_ids=anchors)

    spike = json.loads(_SPIKE_REPORT.read_text(encoding="utf-8"))
    dec = spike["decomposition"]["decomposition"]
    for tier in ("floor", "headline"):
        assert result["separation"][tier]["measurable"] == dec[tier]["clean_within_shortlist"]
    # Documented split verdict: the floor does not separate (median < 0.66), the
    # high-view headline tier does (>= 0.66) — read off the persisted scores.
    assert result["separation"]["floor"]["verdict"]["separated"] is False
    assert result["separation"]["headline"]["verdict"]["separated"] is True
