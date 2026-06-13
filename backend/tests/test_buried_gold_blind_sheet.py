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


# ═════════════════════════════════════════════════════════════════════════════
# Slice 2 (PRD #69, issue #71) — the blind 8-choice sheet layer
#
# On top of the selection, per selected gold: the covering candidate (the answer)
# + 7 over-ranking non-covering decoys, all rendered through the *same*
# build_excerpt (identical pad_sec, no special treatment), deterministically
# shuffled, the gold hidden. Output: a readable sheet (A-H + answer/recognized)
# + a sealed key in a separate structure.
# ═════════════════════════════════════════════════════════════════════════════

def _seg(start, end, text):
    """A transcript segment in the shape build_excerpt reads (start/end/text)."""
    return {"start": float(start), "end": float(end), "text": text}


def _selected(vod_id, start, end, *, buried_rank, views=0.0, gt_tier="gold", desc=""):
    """A selected buried gold in the shape Slice 1's select_buried_golds emits."""
    return {"vod_id": vod_id, "start": float(start), "end": float(end),
            "buried_rank": buried_rank, "views": float(views),
            "gt_tier": gt_tier, "desc": desc}


def _pool_with_cover_at(cover, rank):
    """Ranked pool: ``rank-1`` far non-covering over-rankers, then ``cover`` at
    1-based ``rank`` (the shallowest covering candidate = the buried_rank)."""
    decoys = [_cand(1000 + i * 100, 1000 + i * 100 + 20) for i in range(rank - 1)]
    return decoys + [cover]


def _segments_for(pool):
    """One distinct standalone sentence segment inside each candidate window, so
    every excerpt carries its own text (windows are spaced so none bleed)."""
    return [_seg(c["start_time"], c["end_time"], f"Sentence number {i} stands alone here.")
            for i, c in enumerate(pool)]


def test_choice_entry_has_eight_choices_with_gold_letter_on_the_covering_candidate() -> None:
    gold = _selected("V", 100, 130, buried_rank=25, views=300_000)
    cover = _cand(102, 122)               # a 20s clip-sized window covering [100,130]
    pool = _pool_with_cover_at(cover, 25)  # 24 far over-rankers + cover at rank 25
    segments = _segments_for(pool)

    entry = bgbs.build_choice_entry(gold, pool, segments, sheet_id="sheet_01")

    # Exactly 8 anonymous lettered choices A-H.
    assert [c["letter"] for c in entry["choices"]] == list("ABCDEFGH")
    # The gold letter names one of them, and it is the covering candidate.
    assert entry["gold_letter"] in bgbs.CHOICE_LETTERS
    gold_choice = next(c for c in entry["choices"] if c["letter"] == entry["gold_letter"])
    assert gold_choice["candidate"]["start_time"] == 102.0
    assert gold_choice["candidate"]["end_time"] == 122.0


def test_gold_letter_is_hit_and_every_decoy_is_not() -> None:
    # The whole point of the sheet: the answer is a *real* covering candidate and
    # not one of the 7 decoys covers the gold — so the human can't win by spotting
    # "the one that overlaps". This is checked against the frozen is_hit seam.
    from ml import eval_clip_finder

    gold = _selected("V", 100, 130, buried_rank=25, views=300_000)
    cover = _cand(102, 122)
    pool = _pool_with_cover_at(cover, 25)
    segments = _segments_for(pool)

    entry = bgbs.build_choice_entry(gold, pool, segments, sheet_id="sheet_01")

    gt = {"start": gold["start"], "end": gold["end"]}
    gold_choice = next(c for c in entry["choices"] if c["letter"] == entry["gold_letter"])
    decoys = [c for c in entry["choices"] if c["letter"] != entry["gold_letter"]]

    assert eval_clip_finder.is_hit(gt, gold_choice["candidate"])
    assert len(decoys) == 7
    assert all(not eval_clip_finder.is_hit(gt, c["candidate"]) for c in decoys)


# ── decoy selection: over-rankers first, fill from deeper, surface the shortfall ─

def test_select_decoys_draws_from_over_rankers_when_enough_available() -> None:
    gold = _selected("V", 100, 130, buried_rank=25)
    cover = _cand(102, 122)
    pool = _pool_with_cover_at(cover, 25)  # 24 over-ranking non-coverers + cover

    pick = bgbs.select_decoys(gold, pool, 25)

    over_rankers = pool[:24]
    assert pick["shortfall"] == 0
    assert pick["overrank_used"] == 7
    assert len(pick["decoys"]) == 7
    # Every decoy is one of the candidates that over-ranked the gold.
    assert all(d in over_rankers for d in pick["decoys"])


def test_select_decoys_fills_from_deeper_pool_when_over_rankers_short() -> None:
    # The covering candidate is shallow (rank 3) so only 2 candidates over-rank it
    # — the remaining 5 decoys are topped up from the deeper non-covering pool.
    gold = _selected("V", 100, 130, buried_rank=3)
    cover = _cand(102, 122)
    over = [_cand(1000, 1020), _cand(1100, 1120)]                        # ranks 1,2
    deeper = [_cand(2000 + i * 100, 2000 + i * 100 + 20) for i in range(7)]  # ranks 4..10
    pool = over + [cover] + deeper

    pick = bgbs.select_decoys(gold, pool, 3)

    assert pick["overrank_used"] == 2   # only two genuine over-rankers existed
    assert len(pick["decoys"]) == 7     # topped up from the deeper pool
    assert pick["shortfall"] == 0
    assert cover not in pick["decoys"]  # the covering candidate is never a decoy


def test_select_decoys_surfaces_shortfall_when_too_few_non_coverers() -> None:
    gold = _selected("V", 100, 130, buried_rank=3)
    cover = _cand(102, 122)
    over = [_cand(1000, 1020), _cand(1100, 1120)]  # 2 over-rankers
    deeper = [_cand(2000, 2020)]                    # only 1 deeper non-coverer
    pool = over + [cover] + deeper

    pick = bgbs.select_decoys(gold, pool, 3)

    assert len(pick["decoys"]) == 3  # 2 over + 1 deeper, all available
    assert pick["shortfall"] == 4    # 7 - 3 surfaced, not silently capped


def test_choice_entry_surfaces_decoy_shortfall_with_fewer_choices() -> None:
    # When the gold can't muster 7 decoys, the block shrinks (gold + available
    # decoys) and the shortfall is carried out — never padded with a covering
    # candidate, never silently capped to 8.
    gold = _selected("V", 100, 130, buried_rank=3)
    cover = _cand(102, 122)
    over = [_cand(1000, 1020), _cand(1100, 1120)]
    deeper = [_cand(2000, 2020)]
    pool = over + [cover] + deeper
    segments = _segments_for(pool)

    entry = bgbs.build_choice_entry(gold, pool, segments, sheet_id="sheet_01")

    assert entry["shortfall"] == 4
    assert len(entry["choices"]) == 4  # 1 covering + 3 available decoys
    assert entry["gold_letter"] in [c["letter"] for c in entry["choices"]]


# ── excerpt parity (no special treatment) + deterministic shuffle ───────────────

def test_all_eight_excerpts_use_build_excerpt_with_identical_pad_sec() -> None:
    from ml import strong_model_window_probe as smwp

    gold = _selected("V", 100, 130, buried_rank=25)
    cover = _cand(102, 122)
    pool = _pool_with_cover_at(cover, 25)
    segments = _segments_for(pool)

    entry = bgbs.build_choice_entry(gold, pool, segments, sheet_id="sheet_01")

    assert len(entry["choices"]) == 8
    for choice in entry["choices"]:
        # Identical pad on every choice — the gold window gets no special treatment.
        assert choice["excerpt"]["pad_sec"] == bgbs.EXCERPT_PAD_SEC == 12.0
        # Each excerpt is exactly what the shared pure build_excerpt produces — same
        # function for gold and decoys, so length/format can't leak the answer.
        cand = choice["candidate"]
        expected = smwp.build_excerpt(
            cand["start_time"], cand["end_time"], segments, pad_sec=12.0)
        assert choice["excerpt"] == expected


def test_shuffle_is_deterministic_for_a_fixed_seed() -> None:
    gold = _selected("V", 100, 130, buried_rank=25)
    cover = _cand(102, 122)
    pool = _pool_with_cover_at(cover, 25)
    segments = _segments_for(pool)

    first = bgbs.build_choice_entry(gold, pool, segments, sheet_id="sheet_01")
    second = bgbs.build_choice_entry(gold, pool, segments, sheet_id="sheet_01")

    # Same gold (same vod_id + window) -> identical arrangement and answer, run to
    # run. Pinned literally so a regression to process-salted randomness is caught:
    # sheet and key are produced in separate invocations and must agree.
    assert first["gold_letter"] == second["gold_letter"] == "D"
    assert ([c["candidate"] for c in first["choices"]]
            == [c["candidate"] for c in second["choices"]])


def test_shuffle_depends_on_the_gold_identity() -> None:
    # A different gold window seeds a different arrangement — the shuffle is keyed
    # on identity, not a global constant.
    cover = _cand(102, 122)
    pool = _pool_with_cover_at(cover, 25)
    segments = _segments_for(pool)
    gold_a = _selected("V", 100, 130, buried_rank=25)
    gold_b = _selected("V", 500, 530, buried_rank=25)

    assert bgbs._shuffle_seed("V", gold_a) != bgbs._shuffle_seed("V", gold_b)


# ── build_sheet: entries + a separate sealed key, shortfalls logged ─────────────

def test_build_sheet_emits_entries_and_a_separate_sealed_key() -> None:
    selected = [_selected("V", 100, 130, buried_rank=25, views=300_000)]
    cover = _cand(102, 122)
    pool = _pool_with_cover_at(cover, 25)
    segments = _segments_for(pool)

    sheet = bgbs.build_sheet(selected, {"V": pool}, {"V": segments})

    assert [e["sheet_id"] for e in sheet["entries"]] == ["sheet_01"]
    entry = sheet["entries"][0]
    # The key is a separate structure: sheet_id -> answer + provenance.
    assert set(sheet["key"]) == {"sheet_01"}
    assert sheet["key"]["sheet_01"] == {
        "gold_letter": entry["gold_letter"], "views": 300_000.0,
        "buried_rank": 25, "vod_id": "V"}
    # The sealed answer points at the covering candidate.
    gold_choice = next(c for c in entry["choices"]
                       if c["letter"] == sheet["key"]["sheet_01"]["gold_letter"])
    assert gold_choice["candidate"] == cover


def test_build_sheet_logs_decoy_shortfall(caplog) -> None:
    import logging

    selected = [_selected("V", 100, 130, buried_rank=3, views=300_000)]
    cover = _cand(102, 122)
    pool = [_cand(1000, 1020), _cand(1100, 1120)] + [cover] + [_cand(2000, 2020)]
    segments = _segments_for(pool)

    with caplog.at_level(logging.INFO, logger="ml.buried_gold_blind_sheet"):
        sheet = bgbs.build_sheet(selected, {"V": pool}, {"V": segments})

    assert sheet["shortfalls"] == [
        {"sheet_id": "sheet_01", "vod_id": "V", "shortfall": 4}]
    assert any("blind_sheet_decoy_shortfall" in r.getMessage() for r in caplog.records)


# ── rendering: caveat header, no spoilers, the key seals the answer ─────────────

def test_sheet_render_carries_caveat_hides_answer_and_key_seals_it() -> None:
    selected = [_selected("VODSECRET", 100, 130, buried_rank=25, views=300_000)]
    cover = _cand(102, 122)
    pool = _pool_with_cover_at(cover, 25)
    segments = _segments_for(pool)
    sheet = bgbs.build_sheet(selected, {"VODSECRET": pool}, {"VODSECRET": segments})

    rendered = bgbs.format_sheet(sheet)
    keytext = bgbs.format_key(sheet)

    # Caveat in the header: upper-bound reading + recognized exclusion + caps.
    assert "UPPER BOUND" in rendered
    assert "recognized" in rendered
    assert "cap" in rendered
    # The sheet shows A–H + an answer/recognized field…
    for letter in "ABCDEFGH":
        assert f"[{letter}]" in rendered
    assert "Your pick (A–H)" in rendered
    # …but none of the sealed-key fields leak onto the sheet.
    assert "gold_letter" not in rendered
    assert "VODSECRET" not in rendered           # vod_id lives only in the key
    assert "300000" not in rendered and "300,000" not in rendered  # views sealed
    # The sealed key (separate output) carries the answer + provenance.
    assert "gold_letter" in keytext
    assert "VODSECRET" in keytext
    assert sheet["key"]["sheet_01"]["gold_letter"] in "ABCDEFGH"


# ── CLI: --sheet-out / --key-out write the sheet + sealed key offline ───────────

def test_main_writes_sheet_and_sealed_key_offline(tmp_path) -> None:
    # The gold is [0,30]; its real covering candidate is the lowest-scored of 26
    # candidates, so the real replay buries it at rank 26 (>20) with 25 distinct
    # over-ranking non-coverers — enough for a full 8-way block.
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
    sheet_path = tmp_path / "sheet.txt"
    key_path = tmp_path / "key.json"

    rc = bgbs.main([
        "--manifest", str(manifest_path),
        "--anchor-csv", str(anchor_csv),
        "--predictions-dir", str(storage),
        "--sheet-out", str(sheet_path),
        "--key-out", str(key_path),
    ])

    assert rc == 0
    assert sheet_path.exists() and key_path.exists()
    key = json.loads(key_path.read_text(encoding="utf-8"))
    assert list(key) == ["sheet_01"]
    assert key["sheet_01"]["gold_letter"] in "ABCDEFGH"
    assert key["sheet_01"]["vod_id"] == "V"
    assert key["sheet_01"]["buried_rank"] == 26
    sheet_text = sheet_path.read_text(encoding="utf-8")
    assert "UPPER BOUND" in sheet_text
    assert "gold_letter" not in sheet_text       # the answer stays sealed
    assert all(f"[{letter}]" in sheet_text for letter in "ABCDEFGH")


# ── reconcile sheet creation against the real pipeline_debug_* artifacts ────────

@pytest.mark.skipif(
    not _real_artifacts_present(),
    reason="real pipeline_debug artifacts / gold manifest not present (tmp/storage is gitignored)",
)
def test_real_artifact_sheet_creation() -> None:
    manifest = eval_clip_finder.load_manifest()
    anchors = eval_clip_finder.read_anchor_vod_ids()

    raw_by_vod, pool_by_vod, segments_by_vod, _gaps = bgbs.build_replay_inputs(
        anchors, _STORAGE_DIR)
    result = bgbs.run(manifest, raw_by_vod, pool_by_vod, anchor_vod_ids=anchors)
    sheet = bgbs.build_sheet(result["selected"], pool_by_vod, segments_by_vod)

    # One block per selected gold; the key seals one answer per block.
    assert len(sheet["entries"]) == len(result["selected"])
    assert set(sheet["key"]) == {e["sheet_id"] for e in sheet["entries"]}
    for entry in sheet["entries"]:
        gt = {"start": next(g["start"] for g in result["selected"]
                            if g["vod_id"] == entry["vod_id"]
                            and g["buried_rank"] == entry["buried_rank"]),
              "end": next(g["end"] for g in result["selected"]
                          if g["vod_id"] == entry["vod_id"]
                          and g["buried_rank"] == entry["buried_rank"])}
        gold_choice = next(c for c in entry["choices"]
                           if c["letter"] == entry["gold_letter"])
        decoys = [c for c in entry["choices"] if c["letter"] != entry["gold_letter"]]
        # The acceptance invariants hold on the real artifacts: the answer covers
        # the gold, every decoy does not, and all excerpts share the same pad.
        assert eval_clip_finder.is_hit(gt, gold_choice["candidate"])
        assert all(not eval_clip_finder.is_hit(gt, c["candidate"]) for c in decoys)
        assert {c["excerpt"]["pad_sec"] for c in entry["choices"]} == {bgbs.EXCERPT_PAD_SEC}
    # The rendered sheet never leaks the answer.
    assert "gold_letter" not in bgbs.format_sheet(sheet)
