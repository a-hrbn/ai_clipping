"""Tests for the B-narrow probe data foundation (ml/b_narrow_probe.py).

The free, offline Phase-1 foundation of PRD #64: it selects the *recoverable
subset* — the llm_skipped golds the #60 strong-model window-probe separated above
the delivered competition (within-set percentile >= 0.66, the same pre-fixed cut)
— mechanically from the #60-persisted scores, never hand-picked. No paid call.

These tests pin the subset selection through the public interface on hand-built
ratings (exact percentiles in, exact subset out) — never a network or paid call.
"""
from __future__ import annotations

import json
import sys
import types

import pytest

# Defensive: the selection reaches strong_model_window_probe -> the serving
# post-processor, which can pull the heavy transcription stack. Prior art stubs
# whisper the same way so the import stays cheap in a bare test env.
sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_a, **_k: None))

from ml import b_narrow_probe as bnp, eval_clip_finder, strong_model_window_probe as swp


def _scored(vod_id, start, end, kind, score, **extra):
    return {"vod_id": vod_id, "start": float(start), "end": float(end),
            "kind": kind, "score": score, **extra}


# ─────────────────────────────────────────────────────────────────────────────
# Slice 3 — select_recoverable: golds with within-set percentile >= 0.66
# ─────────────────────────────────────────────────────────────────────────────

def test_select_recoverable_keeps_only_golds_at_or_above_the_066_cut() -> None:
    # V1: a clearly-separated headline gold (beats all 3 controls -> pct 1.0) and an
    #     exogenous one (beats none -> pct 0.0). V2: a boundary floor gold that beats
    #     2 of 3 controls -> pct 0.667 >= 0.66 (recoverable). Subset = {high, boundary}.
    rating = {"vods": {
        "V1": {"scored": [
            _scored("V1", 3000, 3030, "gold", 0.9, headline=True, desc="recoverable-high"),
            _scored("V1", 6000, 6030, "gold", 0.4, headline=True, desc="exogenous"),
            _scored("V1", 100, 120, "control", 0.5, rank=1),   # gold 0.9 beats all 3 (pct 1.0),
            _scored("V1", 200, 220, "control", 0.6, rank=2),   # gold 0.4 beats none (pct 0.0)
            _scored("V1", 300, 320, "control", 0.7, rank=3),
        ]},
        "V2": {"scored": [
            _scored("V2", 8000, 8030, "gold", 0.5, headline=False, desc="recoverable-boundary"),
            _scored("V2", 400, 420, "control", 0.1, rank=1),
            _scored("V2", 500, 520, "control", 0.2, rank=2),
            _scored("V2", 600, 620, "control", 0.9, rank=3),
        ]},
    }}

    subset = bnp.select_recoverable(rating)

    assert {w["desc"] for w in subset["recoverable"]} == {"recoverable-high", "recoverable-boundary"}
    assert all(w["percentile"] >= 0.66 for w in subset["recoverable"])
    # The exogenous gold (pct 0.0) is excluded — not hand-picked, the cut decides.
    assert "exogenous" not in {w["desc"] for w in subset["recoverable"]}
    # Distinct subset VODs + size surfaced (thin-n explicit: subset of all golds).
    assert subset["vods"] == ["V1", "V2"]
    assert subset["count"] == 2
    assert subset["headline_count"] == 1          # only recoverable-high is headline
    assert subset["total_golds"] == 3             # 3 llm_skipped golds considered


# ═════════════════════════════════════════════════════════════════════════════
# Issue #66 — paid in-context re-gen (gemini-2.5-pro) + persist/replay. Net-free
# unit tests via an injected fake generator / persisted fixtures.
# ═════════════════════════════════════════════════════════════════════════════

def _raw(start, end, composite=8.0):
    """One re-generated raw moment row (the eval_recall raw_moments form)."""
    return {"start_time": float(start), "end_time": float(end),
            "composite_score": float(composite), "viral_score": float(composite),
            "reason": "regen"}


def test_paid_make_regen_persists_raw_rows_then_replays_identically(tmp_path) -> None:
    # The two-adapter / persist-replay contract: the paid factory (generator
    # injected with a fake — the real wiring binds gemini-2.5-pro find_moments)
    # persists the raw pool per VOD, and the replay factory re-reads it
    # bit-identically without re-generating (no network, no re-pay).
    canned = [_raw(100, 120, 9.0), _raw(500, 530, 8.5)]
    calls: list[str] = []
    fake_generate = lambda segments, vod_id: calls.append(vod_id) or canned

    paid_rows = bnp.paid_make_regen(tmp_path, generate=fake_generate)("V", ["seg"])

    assert paid_rows == canned
    assert calls == ["V"]                                  # generated once
    assert (tmp_path / "V.regen_raw.jsonl").exists()       # persisted as jsonl

    # Replay re-reads the persisted pool — no generate call, byte-identical rows.
    replay_calls: list[str] = []
    replay_rows = bnp.replay_make_regen(tmp_path)("V", ["seg"])
    assert replay_rows == canned
    assert replay_calls == []                              # the fake generator is never touched


def test_gemini_pro_generator_swaps_only_the_model_keeping_production_defaults() -> None:
    # The real adapter binds find_moments to the strong model with eval_recall on,
    # and — crucially — passes NO chunking/max_moments/max_tokens, so find_moments'
    # production defaults apply (only the generation model changes vs flash-lite).
    canned_raw = [_raw(100, 120, 9.0)]
    captured: dict = {}

    def fake_find_moments(segments, **kwargs):
        captured.update(kwargs)
        captured["segments"] = segments
        return types.SimpleNamespace(raw_moments=canned_raw)

    generate = bnp.gemini_pro_generator(find_moments_fn=fake_find_moments)
    rows = generate(["seg1", "seg2"], "V")

    assert rows == canned_raw
    assert captured["provider"] == "gemini"
    assert captured["model"] == "gemini-2.5-pro"
    assert captured["eval_recall"] is True
    assert captured["vod_id"] == "V"
    assert captured["profile"] == "jynxzi"
    assert captured["segments"] == ["seg1", "seg2"]
    # Production defaults preserved: the adapter overrides none of these.
    for key in ("max_moments", "max_tokens", "chunk_duration_sec", "chunk_overlap_sec"):
        assert key not in captured


def test_run_regen_generates_each_subset_vod_and_surfaces_caps(tmp_path) -> None:
    # Three subset VODs: V1 generates a pool, V2 has segments but an EMPTY
    # generation, V3 has no loadable segments. Both caps are surfaced, never
    # silently dropped, and the aggregate counts add up.
    segments_by_vod = {"V1": ["s1"], "V2": ["s2"]}        # V3 absent -> no_segments
    make_regen = lambda vod_id, segments: [_raw(100, 120)] if vod_id == "V1" else []

    result = bnp.run_regen(["V1", "V2", "V3"], segments_by_vod, make_regen=make_regen)

    assert result["vods"]["V1"]["count"] == 1
    assert result["vods"]["V2"]["count"] == 0
    assert "V3" not in result["vods"]
    assert result["caps"]["no_segments"] == ["V3"]
    assert result["caps"]["empty_generation"] == ["V2"]
    assert result["counts"] == {"vods": 2, "raw_moments": 1}
    assert result["model"] == "gemini-2.5-pro"


# ─────────────────────────────────────────────────────────────────────────────
# Slice 4 — the CLI reads the persisted #60 scores and reports the subset
# ─────────────────────────────────────────────────────────────────────────────

def test_main_reads_persisted_scores_and_reports_subset(tmp_path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    # One VOD's persisted #60 scores: a recoverable gold (beats both controls) + the
    # controls. The CLI re-derives the subset from disk, net-free.
    scores = [
        {"vod_id": "V", "start": 3000.0, "end": 3030.0, "kind": "gold",
         "headline": True, "rank": None, "desc": "KSI recoverable", "score": 0.9},
        {"vod_id": "V", "start": 100.0, "end": 120.0, "kind": "control",
         "headline": None, "rank": 1, "desc": None, "score": 0.2},
        {"vod_id": "V", "start": 200.0, "end": 220.0, "kind": "control",
         "headline": None, "rank": 2, "desc": None, "score": 0.3},
    ]
    (artifacts / "V.scores.json").write_text(json.dumps(scores), encoding="utf-8")
    out_path = tmp_path / "subset.json"

    rc = bnp.main(["--artifacts-dir", str(artifacts), "--output", str(out_path), "--json"])

    assert rc == 0
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    assert artifact["count"] == 1
    assert artifact["vods"] == ["V"]
    assert artifact["recoverable"][0]["desc"] == "KSI recoverable"
    assert artifact["recoverable"][0]["percentile"] == 1.0
    assert "computed_at" in artifact


def _write_job_dir(storage, vod_id, ts, *, segments, mtime):
    debug = storage / f"pipeline_debug_{ts}_{vod_id}" / "debug"
    debug.mkdir(parents=True)
    # load_replay_inputs requires BOTH artifacts to consider a dir replayable.
    (debug / "llm_moment_finder_moments_raw.jsonl").write_text(
        json.dumps({"start_time": 1.0, "end_time": 9.0, "composite_score": 5.0}) + "\n",
        encoding="utf-8")
    (debug / "segments.jsonl").write_text(
        "".join(json.dumps(s) + "\n" for s in segments), encoding="utf-8")
    import os
    os.utime(debug.parent, (mtime, mtime))


def test_main_replay_regen_rereads_persisted_pool_net_free(tmp_path) -> None:
    # End-to-end --replay: the CLI selects the recoverable subset, loads each
    # subset VOD's segments (parity loader), and RE-READS the persisted re-gen pool
    # from disk — no network, no re-pay. The report carries the per-VOD raw counts.
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "V.scores.json").write_text(json.dumps([
        {"vod_id": "V", "start": 3000.0, "end": 3030.0, "kind": "gold",
         "headline": True, "rank": None, "desc": "KSI recoverable", "score": 0.9},
        {"vod_id": "V", "start": 100.0, "end": 120.0, "kind": "control", "score": 0.2},
        {"vod_id": "V", "start": 200.0, "end": 220.0, "kind": "control", "score": 0.3},
    ]), encoding="utf-8")
    storage = tmp_path / "storage"
    storage.mkdir()
    _write_job_dir(storage, "V", "20260101T000000000000Z",
                   segments=[{"start": 2990.0, "end": 3035.0, "text": "the gold window speech"}],
                   mtime=1000.0)
    regen_dir = tmp_path / "regen"
    regen_dir.mkdir()
    (regen_dir / "V.regen_raw.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in [_raw(3000, 3030, 9.0), _raw(50, 70, 7.0)]),
        encoding="utf-8")
    out_path = tmp_path / "report.json"

    rc = bnp.main([
        "--artifacts-dir", str(artifacts), "--predictions-dir", str(storage),
        "--regen-dir", str(regen_dir), "--replay", "--output", str(out_path), "--json",
    ])

    assert rc == 0
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    assert artifact["regen_mode"] == "replay"
    assert artifact["regen"]["vods"]["V"]["count"] == 2
    assert artifact["regen"]["counts"] == {"vods": 1, "raw_moments": 2}
    assert artifact["regen"]["caps"]["no_segments"] == []


# ─────────────────────────────────────────────────────────────────────────────
# Guard — reconcile the recoverable subset against the #60 0.66 cut on real scores
# ─────────────────────────────────────────────────────────────────────────────

_ARTIFACTS_DIR = swp.DEFAULT_ARTIFACTS_DIR
# The 5 distinct VODs whose golds the #60 strong-model probe separated (pct >= 0.66).
_RECOVERABLE_VODS = ["2748783535", "2752903388", "2757793134", "2758541573", "2759304833"]


def _scores_present() -> bool:
    """The (gitignored) persisted #60 window-probe scores from the paid run."""
    return _ARTIFACTS_DIR.exists() and any(_ARTIFACTS_DIR.glob("*.scores.json"))


@pytest.mark.skipif(
    not _scores_present(),
    reason="no persisted #60 window-probe scores (no paid strong-model run yet)",
)
def test_recoverable_subset_pinned_against_60_cut_on_real_scores() -> None:
    # The recoverable subset, read mechanically from the real persisted #60 scores:
    # exactly the golds the strong model separated (pct >= 0.66) — the bimodal
    # recoverable half the #64 B-narrow re-gen targets.
    rating = swp.load_persisted_rating(_ARTIFACTS_DIR)
    subset = bnp.select_recoverable(rating)

    assert subset["count"] == 5                         # ~5 recoverable golds (PRD #64)
    assert subset["headline_count"] == 4
    assert subset["vods"] == _RECOVERABLE_VODS
    assert all(w["percentile"] >= 0.66 for w in subset["recoverable"])
    # ChainsFR (the named #60 exogenous prediction, pct 0.40) is NOT recoverable.
    assert all("ChainsFR" not in (w.get("desc") or "") for w in subset["recoverable"])
    # A subset of the 17 floor llm_skipped golds — thin-n existence set, not a stat.
    assert subset["total_golds"] == 17


_STORAGE_DIR = eval_clip_finder.DEFAULT_STORAGE_DIR
_REGEN_DIR = bnp.DEFAULT_REGEN_DIR


def _regen_pools_present() -> bool:
    """The (gitignored) persisted gemini-2.5-pro re-gen pools from the paid run."""
    return (_scores_present() and _REGEN_DIR.exists()
            and any(_REGEN_DIR.glob("*.regen_raw.jsonl")))


@pytest.mark.skipif(
    not _regen_pools_present(),
    reason="no persisted re-gen pools (no paid gemini-2.5-pro re-gen run yet)",
)
def test_replay_regen_reconciles_on_real_persisted_pools() -> None:
    # When a real paid re-gen HAS persisted its pools, --replay re-reads them
    # net-free over the real subset: each recoverable VOD with a persisted pool
    # yields raw moments in the eval row shape, no network, no re-pay.
    rating = swp.load_persisted_rating(_ARTIFACTS_DIR)
    subset = bnp.select_recoverable(rating)
    segments_by_vod = bnp.build_regen_inputs(subset["vods"], _STORAGE_DIR)

    regen = bnp.run_regen(
        subset["vods"], segments_by_vod, make_regen=bnp.replay_make_regen(_REGEN_DIR))

    # At least one subset VOD has a replayable persisted pool, and every replayed
    # row carries the eval raw shape (start_time/end_time) the #67 postprocess reads.
    assert regen["counts"]["vods"] >= 1
    assert regen["counts"]["raw_moments"] >= 1
    for entry in regen["vods"].values():
        for row in entry["raw_moments"]:
            assert "start_time" in row and "end_time" in row
