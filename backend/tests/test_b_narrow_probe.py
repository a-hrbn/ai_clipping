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


# ═════════════════════════════════════════════════════════════════════════════
# Issue #67 — postprocess the re-gen pools through the real serving chain and read
# the pre-fixed product-recall@20 gate (>=3/5 recoverable golds reach the delivered
# top-20). Only a top-20 hit is a Pass; raw@any is secondary diagnosis; a regressed
# delivered hit vetoes. Hand-computed numbers in, verdict out — net-free.
# ═════════════════════════════════════════════════════════════════════════════

def test_gate_verdict_passes_iff_majority_reach_top20_and_guardrail_holds() -> None:
    # 3 of 5 recoverable golds reach the delivered top-20, no delivered hit regressed
    # → the pre-fixed >=3/5 gate Passes → escalate (B, full re-gen). raw@any is only
    # diagnosis: the two non-top20 golds split into "buried" (raw covers) vs "not
    # generated" (raw misses) — neither counts as a Pass (the #56 caveat, built in).
    per_gold = [
        {"top20_hit": True, "raw_any_hit": True},
        {"top20_hit": True, "raw_any_hit": True},
        {"top20_hit": True, "raw_any_hit": True},
        {"top20_hit": False, "raw_any_hit": True},    # generated but buried
        {"top20_hit": False, "raw_any_hit": False},   # not generated at all
    ]

    verdict = bnp.gate_verdict(per_gold, regressed=[])

    assert verdict["top20_count"] == 3
    assert verdict["n"] == 5
    assert verdict["guardrail_ok"] is True
    assert verdict["passed"] is True
    assert "escalate" in verdict["verdict"].lower()


def test_gate_verdict_fails_to_c_track_below_majority() -> None:
    # Only 2 of 5 reach top-20 (< 3) → Fail → C-Track. The in-context separation did
    # not survive even for the recoverable subset; the honest kill, no rationalizing.
    per_gold = [
        {"top20_hit": True, "raw_any_hit": True},
        {"top20_hit": True, "raw_any_hit": True},
        {"top20_hit": False, "raw_any_hit": True},
        {"top20_hit": False, "raw_any_hit": True},
        {"top20_hit": False, "raw_any_hit": False},
    ]

    verdict = bnp.gate_verdict(per_gold, regressed=[])

    assert verdict["top20_count"] == 2
    assert verdict["passed"] is False
    assert "C-Track" in verdict["verdict"]


def test_gate_verdict_guardrail_breach_vetoes_a_majority() -> None:
    # 3 of 5 reach top-20 (a majority), BUT the strong model displaced an already-
    # delivered hit → the guardrail vetoes the Pass → C-Track. The guardrail is a
    # hard safety: a real shipped hit must never be traded for a recoverable one.
    per_gold = [{"top20_hit": True, "raw_any_hit": True}] * 3 + \
               [{"top20_hit": False, "raw_any_hit": True}] * 2
    regressed = [{"vod_id": "V", "gt_start": 100.0, "gt_end": 130.0, "desc": "lost hit"}]

    verdict = bnp.gate_verdict(per_gold, regressed=regressed)

    assert verdict["top20_count"] == 3            # a majority on its own
    assert verdict["guardrail_ok"] is False
    assert verdict["passed"] is False             # vetoed by the guardrail
    assert "guardrail" in verdict["verdict"].lower()


def _mgold(start, end, views=300_000.0, desc=""):
    return {"start": float(start), "end": float(end), "views": float(views),
            "gt_tier": "gold", "desc": desc}


def test_measure_subset_marks_top20_hit_vs_buried_per_recoverable_gold() -> None:
    # Two recoverable golds on V: gold A [3000,3030] is covered by the new delivered
    # top-20 (top20_hit); gold B [6000,6030] is covered only in the raw pool, not the
    # top-20 (buried — raw_any_hit but NOT top20_hit, the #56 distinction). The
    # aggregate recall@20 comes from directional_eval (the serving chain → skew-guard).
    recoverable = [
        {"vod_id": "V", "start": 3000.0, "end": 3030.0, "headline": True, "desc": "A reaches"},
        {"vod_id": "V", "start": 6000.0, "end": 6030.0, "headline": True, "desc": "B buried"},
    ]
    manifest = {"V": [_mgold(3000, 3030, desc="A reaches"), _mgold(6000, 6030, desc="B buried")]}
    final_by_vod = {"V": [_raw(3005, 3025, 9.0)]}                  # top-20 covers A only
    raw_by_vod = {"V": [_raw(3005, 3025, 9.0), _raw(6005, 6025, 7.0)]}  # raw covers A AND B

    measured = bnp.measure_subset(recoverable, manifest, final_by_vod, raw_by_vod,
                                  subset_vods=["V"])

    by_desc = {g["desc"]: g for g in measured["per_gold"]}
    assert by_desc["A reaches"]["top20_hit"] is True
    assert by_desc["A reaches"]["raw_any_hit"] is True
    assert by_desc["B buried"]["top20_hit"] is False           # buried, not shipped
    assert by_desc["B buried"]["raw_any_hit"] is True          # generated, just buried
    # The directional_eval skew-guard number is carried (the real serving recall@20).
    assert measured["eval"]["headline"]["recall"]["20"] == 0.5   # 1 of 2 golds in top-20


def test_check_guardrail_flags_a_regressed_delivered_hit_on_subset_vods() -> None:
    # The guardrail reads the frozen baseline's per-gold top-20 hits vs the new eval's.
    # A gold the baseline shipped (hit) that the strong-model top-20 no longer ships
    # is a regression; a still-shipped gold is fine; a non-subset VOD is ignored.
    baseline = {"per_vod": [
        {"vod_id": "V", "gt_hits": [
            {"gt_start": 100.0, "gt_end": 130.0, "hit": True, "desc": "kept hit"},
            {"gt_start": 500.0, "gt_end": 530.0, "hit": True, "desc": "lost hit"},
        ]},
        {"vod_id": "OTHER", "gt_hits": [{"gt_start": 1.0, "gt_end": 2.0, "hit": True, "desc": "not subset"}]},
    ]}
    new_eval = {"per_vod": [
        {"vod_id": "V", "gt_hits": [
            {"gt_start": 100.0, "gt_end": 130.0, "hit": True, "desc": "kept hit"},
            {"gt_start": 500.0, "gt_end": 530.0, "hit": False, "desc": "lost hit"},
        ]},
    ]}

    regressed = bnp.check_guardrail(baseline, new_eval, subset_vods=["V"])

    assert [r["desc"] for r in regressed] == ["lost hit"]   # only the dropped subset hit


def test_main_replay_runs_the_full_gate_end_to_end_net_free(tmp_path) -> None:
    # End-to-end --replay through the REAL serving postprocess chain: the persisted
    # re-gen pool covers the recoverable gold, so it survives snap→dedupe→diversity→
    # top_k into the delivered top-20 → top20_hit. With --min-top20 1 the gate Passes
    # → escalate. No network, no re-pay. Pins the skew-guarded recall@20 too.
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
                   segments=[{"start": 2990.0, "end": 3035.0, "text": "the recoverable gold window speech"}],
                   mtime=1000.0)
    regen_dir = tmp_path / "regen"
    regen_dir.mkdir()
    # The re-gen pool covers the gold [3000,3030] with the top composite → top-20.
    (regen_dir / "V.regen_raw.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in [_raw(3000, 3030, 9.5), _raw(50, 70, 6.0)]),
        encoding="utf-8")
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(json.dumps({
        "vod_id": "V",
        "gt_moments": [{"start": 3000.0, "end": 3030.0, "views": 300_000,
                        "gt_tier": "gold", "desc": "KSI recoverable"}],
    }) + "\n", encoding="utf-8")
    # Baseline where V shipped nothing → no possible regression (clean guardrail).
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps({"per_vod": [
        {"vod_id": "V", "gt_hits": [{"gt_start": 3000.0, "gt_end": 3030.0, "hit": False, "desc": "KSI recoverable"}]}]}),
        encoding="utf-8")
    out_path = tmp_path / "report.json"

    rc = bnp.main([
        "--artifacts-dir", str(artifacts), "--predictions-dir", str(storage),
        "--regen-dir", str(regen_dir), "--manifest", str(manifest_path),
        "--baseline", str(baseline_path), "--min-top20", "1",
        "--replay", "--output", str(out_path), "--json",
    ])

    assert rc == 0
    gate = json.loads(out_path.read_text(encoding="utf-8"))["gate"]
    assert gate["verdict"]["top20_count"] == 1
    assert gate["verdict"]["passed"] is True
    assert "escalate" in gate["verdict"]["verdict"].lower()
    assert gate["per_gold"][0]["top20_hit"] is True
    assert gate["recall_at_20"]["headline"] == 1.0     # skew-guarded serving recall@20


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


@pytest.mark.skipif(
    not _regen_pools_present(),
    reason="no persisted re-gen pools (no paid gemini-2.5-pro re-gen run yet)",
)
def test_gate_verdict_reproduces_on_real_pools_fail_to_c_track() -> None:
    # The #67 verdict, reproduced net-free from the persisted gemini-2.5-pro pools
    # through the real serving chain: only 1/5 recoverable golds (KSI) reaches the
    # delivered top-20 — 3/5 the strong model did NOT even generate in-context, 1/5
    # generated but buried. < 3/5 → FAIL → C-Track. The honest kill the pre-fixed
    # gate locks in: the rate-mode separation does not survive in-context generation.
    rating = swp.load_persisted_rating(_ARTIFACTS_DIR)
    subset = bnp.select_recoverable(rating)
    segments_by_vod = bnp.build_regen_inputs(subset["vods"], _STORAGE_DIR)
    regen = bnp.run_regen(subset["vods"], segments_by_vod,
                          make_regen=bnp.replay_make_regen(_REGEN_DIR))
    manifest = eval_clip_finder.load_manifest()
    baseline = json.loads((bnp.directional_eval.BACKEND_ROOT / "ml" / "artifacts"
                           / "finder_recall_baseline.json").read_text(encoding="utf-8"))

    gate = bnp.run_gate(subset["recoverable"], regen, segments_by_vod, manifest, baseline)

    assert gate["verdict"]["n"] == 5
    assert gate["verdict"]["top20_count"] == 1            # only KSI surfaced
    assert gate["verdict"]["passed"] is False             # < 3/5 → FAIL
    assert "C-Track" in gate["verdict"]["verdict"]
    # Guardrail clean: the strong model displaced no already-delivered hit.
    assert gate["verdict"]["guardrail_ok"] is True
    # raw@any (secondary) is well above product recall@20 — generated-but-buried gap.
    assert gate["raw_at_any"]["floor"] > gate["recall_at_20"]["floor"]
    # The one surfaced gold is KSI (headline); the 3 ungenerated are not in raw@any.
    surfaced = [g for g in gate["per_gold"] if g["top20_hit"]]
    assert len(surfaced) == 1 and "KSI" in surfaced[0]["desc"]
    assert sum(1 for g in gate["per_gold"] if not g["raw_any_hit"]) == 3
