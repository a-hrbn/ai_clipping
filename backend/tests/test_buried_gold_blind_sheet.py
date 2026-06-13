"""Tests for the Buried-Gold Blind-Sheet selection core (ml/buried_gold_blind_sheet.py).

Slice 1 of PRD #69: the read-only selection core that mechanically picks the
population the blind sheet will test — the clean_but_buried golds (buried_rank>20,
gt_tier in {gold,silver}) ranked top-10 by repost views — by *composing* the
existing seams (gap_decomposition.classify_gold, replay_harness, the gold-eval
manifest loader, outcome_view_weighting.view_count). No excerpts/decoys/sheet yet.

These tests pin the selection through the public interface: small hand-built
fixtures in, exact selection out. Never a network or paid call.
"""
from __future__ import annotations

import sys
import types

# Defensive: building the ranked pool reaches the serving post-processor, which
# can pull the heavy transcription stack. Prior art (test_gap_decomposition)
# stubs whisper the same way so the import stays cheap in a bare test env.
sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_a, **_k: None))

from ml import buried_gold_blind_sheet as bgbs  # noqa: E402


def _gold(start, end, *, views=0.0, tier="gold", desc=""):
    """A gold moment in the manifest shape the selector reads."""
    return {"start": float(start), "end": float(end), "views": float(views),
            "gt_tier": tier, "desc": desc}


def _cand(start, end):
    return {"start_time": float(start), "end_time": float(end)}


def _ranked_with_real_at(real, rank):
    """A ranked pool placing the real covering candidate at 1-based ``rank``.

    The (rank-1) preceding decoys are clip-sized windows far from the gold, so
    none of them cover it — the real candidate is the shallowest covering one.
    """
    decoys = [_cand(1000 + i * 100, 1000 + i * 100 + 20) for i in range(rank - 1)]
    return decoys + [real]


# ─────────────────────────────────────────────────────────────────────────────
# Slice 1 — tracer: one clean_but_buried gold at rank 25 is selected
# ─────────────────────────────────────────────────────────────────────────────

def test_selects_single_clean_but_buried_gold_beyond_rank_20() -> None:
    gold = _gold(100, 130, views=300_000, desc="reveal")
    real = _cand(102, 122)  # 20s clip-sized window covers the 30s gold -> real
    raw = [real]
    ranked = _ranked_with_real_at(real, 25)  # buried at rank 25 (> 20)

    result = bgbs.select_buried_golds({"V": [gold]}, {"V": raw}, {"V": ranked})

    assert result["eligible_count"] == 1
    selected = result["selected"]
    assert len(selected) == 1
    chosen = selected[0]
    assert chosen["vod_id"] == "V"
    assert chosen["start"] == 100.0
    assert chosen["end"] == 130.0
    assert chosen["buried_rank"] == 25
    assert chosen["views"] == 300_000.0
    assert chosen["gt_tier"] == "gold"


# ─────────────────────────────────────────────────────────────────────────────
# Slice 2 — the buried_rank > 20 boundary: rank 20 is out, rank 21 is in
# ─────────────────────────────────────────────────────────────────────────────

def test_clean_gold_at_rank_20_is_excluded() -> None:
    # buried_rank == 20 is inside the delivered top-20 -> not a buried miss.
    gold = _gold(100, 130, views=300_000)
    real = _cand(102, 122)
    ranked = _ranked_with_real_at(real, 20)

    result = bgbs.select_buried_golds({"V": [gold]}, {"V": [real]}, {"V": ranked})

    assert result["eligible_count"] == 0
    assert result["selected"] == []


def test_clean_gold_at_rank_21_is_included() -> None:
    # buried_rank == 21 is the first rank outside the top-20 -> selected.
    gold = _gold(100, 130, views=300_000)
    real = _cand(102, 122)
    ranked = _ranked_with_real_at(real, 21)

    result = bgbs.select_buried_golds({"V": [gold]}, {"V": [real]}, {"V": ranked})

    assert result["eligible_count"] == 1
    assert result["selected"][0]["buried_rank"] == 21


# ─────────────────────────────────────────────────────────────────────────────
# Slice 3 — non-buried buckets (hard_miss, accidental_only) are never selected
# ─────────────────────────────────────────────────────────────────────────────

def test_hard_miss_gold_is_excluded() -> None:
    # No raw candidate covers the gold window -> hard_miss (a generation gap),
    # not a buried-discrimination miss.
    gold = _gold(100, 130, views=300_000)
    raw = [_cand(0, 20), _cand(500, 520)]  # neither overlaps [100,130]

    result = bgbs.select_buried_golds({"V": [gold]}, {"V": raw}, {"V": raw})

    assert result["eligible_count"] == 0


def test_accidental_only_gold_is_excluded() -> None:
    # Only an oversized 300s window covers the gold; no real clip-sized candidate
    # does -> accidental_only, never clean_but_buried.
    gold = _gold(100, 130, views=300_000)
    oversized = _cand(0, 300)
    # Push it deep so a (wrong) rank-only selector would still pick it up.
    ranked = [_cand(1000 + i * 100, 1000 + i * 100 + 20) for i in range(24)] + [oversized]

    result = bgbs.select_buried_golds({"V": [gold]}, {"V": [oversized]}, {"V": ranked})

    assert result["eligible_count"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Slice 4 — ranked by repost views (desc), capped at top_n
# ─────────────────────────────────────────────────────────────────────────────

def test_ranks_by_views_desc_and_caps_at_top_n() -> None:
    real = _cand(102, 122)
    ranked = _ranked_with_real_at(real, 25)
    golds = {
        "V1": [_gold(100, 130, views=100_000, desc="low")],
        "V2": [_gold(100, 130, views=500_000, desc="high")],
        "V3": [_gold(100, 130, views=250_000, desc="mid")],
        "V4": [_gold(100, 130, views=300_000, desc="midhigh")],
    }
    raw = {v: [real] for v in golds}
    pool = {v: ranked for v in golds}

    result = bgbs.select_buried_golds(golds, raw, pool, top_n=3)

    # All four are clean_but_buried (eligible); the selection keeps the 3 highest
    # by views, ordered descending — the lowest-view gold is dropped by the cap.
    assert result["eligible_count"] == 4
    assert [g["desc"] for g in result["selected"]] == ["high", "midhigh", "mid"]
    assert [g["views"] for g in result["selected"]] == [500_000.0, 300_000.0, 250_000.0]


# ─────────────────────────────────────────────────────────────────────────────
# Slice 5 — a selection short of top_n is surfaced (logged), never silently capped
# ─────────────────────────────────────────────────────────────────────────────

def test_shortfall_is_surfaced_when_fewer_than_top_n_eligible() -> None:
    gold = _gold(100, 130, views=300_000)
    real = _cand(102, 122)
    ranked = _ranked_with_real_at(real, 25)

    result = bgbs.select_buried_golds({"V": [gold]}, {"V": [real]}, {"V": ranked}, top_n=10)

    # Only 1 eligible buried gold but top_n is 10 -> 9 short, surfaced not capped.
    assert result["eligible_count"] == 1
    assert result["shortfall"] == 9


def test_no_shortfall_when_eligible_meets_top_n() -> None:
    real = _cand(102, 122)
    ranked = _ranked_with_real_at(real, 25)
    golds = {f"V{i}": [_gold(100, 130, views=100_000 + i)] for i in range(3)}
    raw = {v: [real] for v in golds}
    pool = {v: ranked for v in golds}

    result = bgbs.select_buried_golds(golds, raw, pool, top_n=3)

    assert result["eligible_count"] == 3
    assert result["shortfall"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Slice 6 — run() admits only gt_tier in {gold, silver}; review/unverifiable out
# ─────────────────────────────────────────────────────────────────────────────

def _four_tier_fixture():
    """One VOD, four buried golds (gold/silver/review/unverifiable), each with a
    real covering candidate buried at ranks 25..28."""
    reals = [_cand(102, 122), _cand(202, 222), _cand(302, 322), _cand(402, 422)]
    manifest = {"V": [
        _gold(100, 130, views=300_000, tier="gold", desc="g"),
        _gold(200, 230, views=290_000, tier="silver", desc="s"),
        _gold(300, 330, views=280_000, tier="review", desc="r"),
        _gold(400, 430, views=270_000, tier="unverifiable", desc="u"),
    ]}
    decoys = [_cand(1000 + i * 100, 1000 + i * 100 + 20) for i in range(24)]
    raw = {"V": list(reals)}
    pool = {"V": decoys + reals}
    return manifest, raw, pool


def test_run_admits_only_gold_and_silver_tiers() -> None:
    manifest, raw, pool = _four_tier_fixture()

    result = bgbs.run(manifest, raw, pool, anchor_vod_ids=["V"])

    assert {g["desc"] for g in result["selected"]} == {"g", "s"}
    assert all(g["gt_tier"] in {"gold", "silver"} for g in result["selected"])
    assert result["tiers"] == ["gold", "silver"]


# ─────────────────────────────────────────────────────────────────────────────
# Slice 7 — run() surfaces missing-prediction VODs as a coverage gap, not a miss
# ─────────────────────────────────────────────────────────────────────────────

def test_run_surfaces_missing_prediction_vods_as_coverage_gap() -> None:
    real = _cand(102, 122)
    ranked = _ranked_with_real_at(real, 25)
    manifest = {
        "SCORED": [_gold(100, 130, views=300_000, tier="gold", desc="s")],
        # GAP is gold but has no finder predictions (no replayable job dir).
        "GAP": [_gold(100, 130, views=900_000, tier="gold", desc="gap")],
    }
    raw = {"SCORED": [real]}
    pool = {"SCORED": ranked}

    result = bgbs.run(manifest, raw, pool, anchor_vod_ids=["SCORED", "GAP"])

    # GAP is surfaced and excluded from the denominator — never counted as a miss.
    assert result["missing_prediction_vods"] == ["GAP"]
    assert result["gold_vods"] == 2
    assert result["scored_vods"] == 1
    # Despite its higher views, GAP's gold is not selected (no pool to rank it in).
    assert [g["desc"] for g in result["selected"]] == ["s"]


# ─────────────────────────────────────────────────────────────────────────────
# Slice 8 — reuse-guard: the selected set is a subset of the gap_decomposition
# clean_but_buried set (no re-classification, the frozen seam decides)
# ─────────────────────────────────────────────────────────────────────────────

from ml import gap_decomposition as gd  # noqa: E402


def _pool(size, placements, *, base):
    """A ranked pool of ``size`` far (non-covering) decoys, with the covering
    candidates dropped in at their 1-based ranks (via ``placements`` index->cand)."""
    pool = [_cand(base + i * 100, base + i * 100 + 20) for i in range(size)]
    for idx, cand in placements.items():
        pool[idx] = cand
    return pool


def _mixed_fixture():
    real_a1, real_a2 = _cand(102, 122), _cand(152, 172)
    real_b = _cand(402, 422)
    oversized = _cand(550, 950)  # 400s window: covers b2 but not clip-sized (accidental)
    gold = {
        "A": [
            _gold(100, 130, views=300_000, desc="a_buried"),    # real cover @ rank 25
            _gold(150, 180, views=400_000, desc="a_shallow"),   # real cover @ rank 5 (clean, not buried)
            _gold(900, 930, views=500_000, desc="a_hardmiss"),  # nothing covers it
        ],
        "B": [
            _gold(400, 430, views=350_000, desc="b_buried"),     # real cover @ rank 30
            _gold(600, 630, views=600_000, desc="b_accidental"), # only the oversized window covers
        ],
    }
    raw = {"A": [real_a1, real_a2], "B": [real_b, oversized]}
    pool = {
        "A": _pool(25, {4: real_a2, 24: real_a1}, base=2000),
        "B": _pool(30, {0: oversized, 29: real_b}, base=3000),
    }
    return gold, raw, pool


def test_selected_set_is_subset_of_classify_gold_clean_but_buried() -> None:
    gold, raw, pool = _mixed_fixture()

    result = bgbs.select_buried_golds(gold, raw, pool)

    # Independently re-derive the clean_but_buried set straight from the frozen
    # gap_decomposition seam — the selection must never reach outside it.
    clean_keys = set()
    for vod_id, moments in gold.items():
        for moment in moments:
            verdict = gd.classify_gold(moment, raw[vod_id], ranked_pool=pool[vod_id])
            if verdict["bucket"] == gd.CLEAN_BUT_BURIED:
                clean_keys.add((vod_id, moment["start"], moment["end"]))

    selected_keys = {(g["vod_id"], g["start"], g["end"]) for g in result["selected"]}
    assert selected_keys, "the guard is only meaningful with a non-empty selection"
    assert selected_keys <= clean_keys
    # The two buried golds are selected, ordered by views; the shallow clean gold
    # (rank 5), the hard-miss and the accidental gold are all left out.
    assert [g["desc"] for g in result["selected"]] == ["b_buried", "a_buried"]


# ─────────────────────────────────────────────────────────────────────────────
# Slice 9 — the CLI runs offline against pipeline_debug_* artifacts and prints
# the selection (no container, no paid call, no network)
# ─────────────────────────────────────────────────────────────────────────────
import json  # noqa: E402
import os  # noqa: E402


def _row(start, end, composite, *, reason=""):
    """A raw finder row in the llm_moment_finder_moments_raw.jsonl shape."""
    virality = composite / 10.0
    return {
        "start_time": float(start), "end_time": float(end),
        "viral_score": composite, "composite_score": composite, "llm_virality": virality,
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


def test_main_runs_offline_and_writes_selection(tmp_path) -> None:
    # The gold is the [0,30] window; its real covering candidate is the
    # lowest-scored of 26 candidates, so the real replay buries it at rank 26 (>20).
    segments = [
        {"start": float(i * 100), "end": float(i * 100 + 30),
         "text": f"A clean standalone sentence number {i}."}
        for i in range(26)
    ]
    raw = [_row(5.0, 25.0, 1.0, reason="the buried covering candidate")] + [
        _row(i * 100 + 5, i * 100 + 25, 9.0, reason=f"decoy {i}") for i in range(1, 26)
    ]

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(json.dumps({
        "vod_id": "V",
        "gt_moments": [{"start": 0.0, "end": 30.0, "views": 300_000,
                        "gt_tier": "gold", "desc": "reveal"}],
    }) + "\n", encoding="utf-8")
    anchor_csv = tmp_path / "anchors.csv"
    anchor_csv.write_text("vod_id\nV\n", encoding="utf-8")
    storage = tmp_path / "storage"
    storage.mkdir()
    _write_job_dir(storage, "V", "20260101T000000000000Z",
                   raw=raw, segments=segments, mtime=1000.0)
    out_path = tmp_path / "selection.json"

    rc = bgbs.main([
        "--manifest", str(manifest_path),
        "--anchor-csv", str(anchor_csv),
        "--predictions-dir", str(storage),
        "--output", str(out_path),
        "--json",
    ])

    assert rc == 0
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    for key in ("selected", "eligible_count", "shortfall", "scored_vods",
                "missing_prediction_vods", "tiers", "buried_rank_floor", "computed_at"):
        assert key in artifact
    assert artifact["scored_vods"] == 1
    assert artifact["missing_prediction_vods"] == []
    assert artifact["eligible_count"] == 1
    chosen = artifact["selected"][0]
    assert chosen["vod_id"] == "V"
    assert chosen["buried_rank"] == 26  # buried beyond the top-20 by the real replay
    assert chosen["views"] == 300_000.0
    assert chosen["gt_tier"] == "gold"


# ─────────────────────────────────────────────────────────────────────────────
# Slice 10 — reconcile against the real pipeline_debug_* artifacts (skips without
# them): the selection is a subset of the real clean_but_buried baseline, no net
# ─────────────────────────────────────────────────────────────────────────────
import pytest  # noqa: E402

from ml import directional_eval, eval_clip_finder  # noqa: E402

_STORAGE_DIR = eval_clip_finder.DEFAULT_STORAGE_DIR


def _real_artifacts_present() -> bool:
    """The (gitignored) gold manifest + at least one replayable pipeline_debug dir."""
    return (
        eval_clip_finder.DEFAULT_MANIFEST.exists()
        and any(_STORAGE_DIR.glob("pipeline_debug_*"))
    )


@pytest.mark.skipif(
    not _real_artifacts_present(),
    reason="real pipeline_debug artifacts / gold manifest not present (tmp/storage is gitignored)",
)
def test_real_artifact_selection_reconciles_with_clean_but_buried_baseline() -> None:
    manifest = eval_clip_finder.load_manifest()
    anchors = eval_clip_finder.read_anchor_vod_ids()

    raw_by_vod, pool_by_vod, _gaps = bgbs.build_inputs(anchors, _STORAGE_DIR)
    result = bgbs.run(manifest, raw_by_vod, pool_by_vod, anchor_vod_ids=anchors)

    # Independently re-derive the real clean_but_buried set (gold/silver tiers,
    # anchor-restricted like run) and confirm the selection never reaches outside
    # it — no re-classification.
    anchor_set = set(anchors)
    gold = {vod_id: ms
            for vod_id, ms in directional_eval.gold_manifest(manifest, bgbs.GOLD_SILVER_TIERS).items()
            if vod_id in anchor_set}
    clean_keys = set()
    for vod_id, moments in gold.items():
        if not raw_by_vod.get(vod_id):
            continue
        for moment in moments:
            verdict = gd.classify_gold(
                moment, raw_by_vod[vod_id], ranked_pool=pool_by_vod.get(vod_id, []))
            if verdict["bucket"] == gd.CLEAN_BUT_BURIED:
                clean_keys.add((vod_id, float(moment["start"]), float(moment["end"])))

    selected_keys = {(g["vod_id"], g["start"], g["end"]) for g in result["selected"]}
    assert selected_keys <= clean_keys
    # Every acceptance invariant holds on the real artifacts.
    assert all(g["buried_rank"] > bgbs.BURIED_RANK_FLOOR for g in result["selected"])
    assert all(g["gt_tier"] in set(bgbs.GOLD_SILVER_TIERS) for g in result["selected"])
    assert len(result["selected"]) <= bgbs.DEFAULT_TOP_N
    # The missing-prediction VODs stay a surfaced coverage gap, never counted.
    assert result["missing_prediction_vods"] == directional_eval.missing_prediction_vods(
        gold, raw_by_vod)
