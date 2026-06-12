"""Tests for the paid Judge-Spike orchestrator (ml/judge_spike.py).

The spike wires the repost-virality judge (#48) over the deduped top-~200 of each
anchor VOD into the replay harness (#47) as its rank_score_fn, scores the result
through directional_eval, decomposes the gap (#49) as a delta vs the frozen
composite baseline, and reads a pre-fixed gate. These tests pin the *new* glue —
shortlist scoping, the surfaced cut, free re-evaluation, the gate, and the
highest-view-miss classification — through the public interface, all on the fake
(network-free) judge, never a paid call.
"""
from __future__ import annotations

import sys
import types

# Defensive: the moment_finder package reaches RichSegment, which can pull the
# heavy transcription stack. Prior art (test_replay_harness) stubs whisper the
# same way so the import stays cheap in a bare test env.
sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_a, **_k: None))

from ml import judge_spike, repost_virality_judge
from services.transcriber import RichSegment


def _row(*, start: float, end: float, composite: float, reason: str = "") -> dict:
    """A deduped serving-currency row, the shape the dedupe step emits."""
    virality = composite / 10.0
    return {
        "start_time": start,
        "end_time": end,
        "viral_score": composite,
        "composite_score": composite,
        "llm_virality": virality,
        "hook": virality,
        "flow": virality,
        "value": virality,
        "trend": virality,
        "reason": reason,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Slice 1 — judge_shortlist_rerank scopes the judge to the deduped top-`depth`
# ─────────────────────────────────────────────────────────────────────────────

def test_shortlist_rerank_scores_only_the_top_depth_and_cuts_the_tail() -> None:
    # Three deduped rows in arbitrary (snap) order; composite says C(9) > A(5) > B(3).
    rows = [
        _row(start=100.0, end=120.0, composite=5.0),  # A — composite rank 2
        _row(start=300.0, end=320.0, composite=3.0),  # B — composite rank 3
        _row(start=500.0, end=520.0, composite=9.0),  # C — composite rank 1
    ]
    # A fake judge that would happily score all three if it saw them.
    judge = repost_virality_judge.fake_repost_virality_judge({1: 0.8, 2: 0.7, 3: 0.6})
    rank_score_fn = judge_spike.judge_shortlist_rerank(judge, depth=2)

    out = rank_score_fn([dict(r) for r in rows])

    # Only the top-2 by composite (C rank 1, A rank 2) reach the judge; B (rank 3)
    # is cut — never scored, never back-filled. The judge sees the shortlist in
    # rank order, so id 1 -> C, id 2 -> A.
    assert len(out) == 2
    windows = {(r["start_time"], r["end_time"]) for r in out}
    assert windows == {(500.0, 520.0), (100.0, 120.0)}


# ─────────────────────────────────────────────────────────────────────────────
# Slice 2 — the shortlist cut (and the fail-closed judge drop) is surfaced
# ─────────────────────────────────────────────────────────────────────────────

def test_shortlist_rerank_surfaces_the_cut_not_silently() -> None:
    # A pool of 5 deduped rows; composites 9,7,5,3,1 give a clean rank order.
    rows = [
        _row(start=100.0, end=120.0, composite=9.0),
        _row(start=200.0, end=220.0, composite=7.0),
        _row(start=300.0, end=320.0, composite=5.0),
        _row(start=400.0, end=420.0, composite=3.0),
        _row(start=500.0, end=520.0, composite=1.0),
    ]
    # depth=3 -> shortlist is the top-3; the fake judge only scores ids 1 & 2, so
    # one shortlisted candidate is dropped fail-closed on top of the 2 cut by depth.
    judge = repost_virality_judge.fake_repost_virality_judge({1: 0.9, 2: 0.8})
    cuts: list[dict] = []
    rank_score_fn = judge_spike.judge_shortlist_rerank(judge, depth=3, on_cut=cuts.append)

    out = rank_score_fn([dict(r) for r in rows])

    assert len(out) == 2  # ids 1 & 2 of the shortlist survived
    # The cut is reported once, with both reductions broken out — depth cut AND the
    # fail-closed judge drop — so a short response never reads as "scored everything".
    assert len(cuts) == 1
    assert cuts[0] == {
        "pool_size": 5,
        "depth": 3,
        "shortlist_size": 3,
        "cut": 2,
        "scored": 2,
        "judge_dropped": 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Slice 3 — a persisted raw response re-evaluates for free (no re-pay)
# ─────────────────────────────────────────────────────────────────────────────

def test_persisted_response_replays_paid_scores_without_paying() -> None:
    rows = [
        _row(start=100.0, end=120.0, composite=5.0),
        _row(start=300.0, end=330.0, composite=9.0),
    ]
    raw = '{"scores": [{"id": 1, "repost_virality": 0.9}, {"id": 2, "repost_virality": 0.2}]}'

    # Paid run: the real adapter, provider returns the raw, we persist it.
    saved: list[str] = []
    paid = repost_virality_judge.repost_virality_judge(
        call_provider=lambda _p: raw, persist_raw=saved.append)
    paid_out = paid([dict(r) for r in rows])

    # Re-eval: the persisted raw replays through the SAME real adapter parser; the
    # provider is the replay seam, which never touches the network.
    replay = repost_virality_judge.repost_virality_judge(
        call_provider=judge_spike.persisted_response_provider(saved[0]),
        persist_raw=lambda _t: None)
    replay_out = replay([dict(r) for r in rows])

    assert saved == [raw]
    # Identical scores, decoded by the same parser — re-evaluierbar ohne erneut zu zahlen.
    assert replay_out == paid_out
    assert [r["composite_score"] for r in replay_out] == [9.0, 2.0]


# ─────────────────────────────────────────────────────────────────────────────
# Slice 4 — the pre-fixed gate: headline@20 up AND floor@20 not down
# ─────────────────────────────────────────────────────────────────────────────

def _eval_result(*, floor20: float | None, headline20: float | None) -> dict:
    """The slice of a directional_eval.run result the gate reads (recall@20 per tier)."""
    return {
        "floor": {"recall": {"20": floor20}},
        "headline": {"recall": {"20": headline20}},
    }


def test_gate_passes_iff_headline_up_and_floor_not_down() -> None:
    base = _eval_result(floor20=0.30, headline20=0.40)

    # PASS: headline strictly up, floor flat (not down).
    up = judge_spike.evaluate_gate(_eval_result(floor20=0.30, headline20=0.45), base)
    assert up["passed"] is True
    assert up["headline_recall_at_20"]["delta"] == 0.05
    assert up["floor_recall_at_20"]["delta"] == 0.0

    # FAIL: headline merely flat — the gate needs a measurable rise.
    flat = judge_spike.evaluate_gate(_eval_result(floor20=0.35, headline20=0.40), base)
    assert flat["passed"] is False

    # FAIL: headline up but floor regresses — the floor must not drop.
    floor_drop = judge_spike.evaluate_gate(_eval_result(floor20=0.25, headline20=0.50), base)
    assert floor_drop["passed"] is False


# ─────────────────────────────────────────────────────────────────────────────
# Slice 5 — highest-view misses classified generation vs ranking
# ─────────────────────────────────────────────────────────────────────────────

def _cand(start: float, end: float) -> dict:
    return {"start_time": start, "end_time": end, "composite_score": 5.0}


def test_top_view_misses_classifies_generation_vs_ranking_by_views() -> None:
    gold_by_vod = {
        "v1": [
            # Never generated — no raw candidate covers it -> generation miss.
            {"start": 100.0, "end": 120.0, "views": 1_400_000.0, "desc": "ChainsFR reveal"},
            # A real candidate covers it but ranks deep -> ranking miss.
            {"start": 500.0, "end": 520.0, "views": 200_000.0, "desc": "buried beat"},
            # Highest views, but it IS in the served final set -> must NOT be a miss.
            {"start": 900.0, "end": 920.0, "views": 5_000_000.0, "desc": "covered headliner"},
        ]
    }
    # Raw covers gold2 (real, clip-sized) and gold3, never gold1.
    raw_by_vod = {"v1": [_cand(500.0, 518.0), _cand(900.0, 918.0)]}
    # The deduped ranked pool buries gold2's real candidate at rank 7.
    pool_by_vod = {
        "v1": [
            _cand(0.0, 10.0), _cand(20.0, 30.0), _cand(40.0, 50.0),
            _cand(60.0, 70.0), _cand(80.0, 90.0), _cand(200.0, 210.0),
            _cand(500.0, 518.0),  # rank 7 — the real covering candidate for gold2
        ]
    }
    # The served top-k covers only gold3, so gold1 and gold2 are product misses.
    final_by_vod = {"v1": [_cand(900.0, 918.0)]}

    misses = judge_spike.top_view_misses(gold_by_vod, final_by_vod, raw_by_vod, pool_by_vod)

    # Only the two genuine misses, ordered by views desc; the covered 5M gold is gone.
    assert [m["views"] for m in misses] == [1_400_000.0, 200_000.0]
    assert misses[0]["desc"] == "ChainsFR reveal"
    assert misses[0]["miss_type"] == "generation"
    assert misses[0]["bucket"] == "hard_miss"
    assert misses[0]["buried_rank"] is None
    assert misses[1]["miss_type"] == "ranking"
    assert misses[1]["bucket"] == "clean_but_buried"
    assert misses[1]["buried_rank"] == 7


def test_top_view_misses_excludes_coverage_gap_vods() -> None:
    # A gold VOD with NO finder predictions (absent from raw_by_vod) is a coverage
    # gap, NOT a generation miss — it must not leak in classified as hard_miss.
    gold_by_vod = {"gap": [{"start": 100.0, "end": 120.0, "views": 999_000.0, "desc": "uncovered"}]}
    misses = judge_spike.top_view_misses(gold_by_vod, {}, {}, {})
    assert misses == []


# ─────────────────────────────────────────────────────────────────────────────
# Slice 6 — memoized provider: one paid call per distinct prompt (no double-pay)
# ─────────────────────────────────────────────────────────────────────────────

def test_memoized_provider_calls_underlying_once_per_prompt() -> None:
    # The pool path and the final path both rank the same shortlist and build the
    # identical prompt; memoization means that pays the provider once, not twice.
    calls: list[str] = []

    def base(prompt: str) -> str:
        calls.append(prompt)
        return f"resp:{prompt}"

    provider = judge_spike.memoized_provider(base)

    assert provider("A") == "resp:A"
    assert provider("A") == "resp:A"  # served from cache — no second paid call
    assert provider("B") == "resp:B"
    assert calls == ["A", "B"]  # underlying provider hit once per distinct prompt


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator — prepare the paid report, then re-evaluate it for free
# ─────────────────────────────────────────────────────────────────────────────

def _spike_segments() -> list[RichSegment]:
    # Three well-separated sentence pairs so snap lands cleanly and the three
    # moments stay far enough apart to survive diversity.
    return [
        RichSegment(start=100.0, end=110.0, text="Reveal one."),
        RichSegment(start=110.0, end=120.0, text="It lands."),
        RichSegment(start=300.0, end=315.0, text="Second beat."),
        RichSegment(start=315.0, end=330.0, text="It continues."),
        RichSegment(start=500.0, end=510.0, text="Third moment."),
        RichSegment(start=510.0, end=520.0, text="And done."),
    ]


def _spike_raw_rows() -> list[dict]:
    # Composites give a clean rank order A(8) > B(6) > C(4); the shortlist ids the
    # judge addresses are those rank positions.
    return [
        _row(start=100.0, end=120.0, composite=8.0, reason="A"),
        _row(start=300.0, end=330.0, composite=6.0, reason="B"),
        _row(start=500.0, end=520.0, composite=4.0, reason="C"),
    ]


_SPIKE_MANIFEST = {
    "v1": [{
        "gt_tier": "gold",
        "start": 100.0,
        "end": 120.0,
        "views": 1_400_000.0,
        "desc": "reveal",
        "source_url": "x",
    }]
}

# A directional_eval.run-shaped baseline at zero recall — every move is a rise.
_SPIKE_BASELINE = {
    "floor": {"recall": {"20": 0.0}, "avg_rank": None},
    "headline": {"recall": {"20": 0.0}, "avg_rank": None},
    "generation": {"floor": {"recall_any": 0.0}, "headline": {"recall_any": 0.0}},
}

# The judge addresses the shortlist by rank: A=1, B=2, C=3.
_SPIKE_RESPONSE = (
    '{"scores": [{"id": 1, "repost_virality": 0.9}, '
    '{"id": 2, "repost_virality": 0.5}, {"id": 3, "repost_virality": 0.3}]}'
)


def _load_one_vod(vod_id: str):
    return (_spike_raw_rows(), _spike_segments()) if vod_id == "v1" else None


def test_spike_persists_paid_artifacts_then_replays_them_for_free(tmp_path) -> None:
    # ── Paid run: the real adapter, a recording+memoized provider, raw persisted.
    paid_calls: list[str] = []

    def make_paid_judge(vod_id: str):
        raw_path = tmp_path / f"{vod_id}.raw.txt"
        provider = judge_spike.memoized_provider(
            lambda _p, v=vod_id: (paid_calls.append(v), _SPIKE_RESPONSE)[1])
        return repost_virality_judge.repost_virality_judge(
            call_provider=provider,
            persist_raw=lambda text, p=raw_path: p.write_text(text, encoding="utf-8"))

    report = judge_spike.run_judge_spike(
        ["v1"], _SPIKE_MANIFEST, _SPIKE_BASELINE,
        make_judge=make_paid_judge, load_inputs=_load_one_vod, depth=50)

    # Provider hit exactly once for v1 despite the pool AND final paths both ranking.
    assert paid_calls == ["v1"]
    assert (tmp_path / "v1.raw.txt").read_text(encoding="utf-8") == _SPIKE_RESPONSE
    assert report["scored_vods"] == ["v1"]
    assert report["coverage_gap_vods"] == []
    # The gate verdict is present and computed from the metrics vs the baseline.
    assert "passed" in report["gate"]
    assert report["gate"]["headline_recall_at_20"]["current"] is not None
    # The shortlist cut is recorded per VOD (no silent cap).
    assert report["shortlist_cuts"]["v1"]["pool_size"] == 3

    # ── Replay: the persisted raw re-scores for free; the provider must NOT fire.
    def make_replay_judge(vod_id: str):
        raw = (tmp_path / f"{vod_id}.raw.txt").read_text(encoding="utf-8")
        return repost_virality_judge.repost_virality_judge(
            call_provider=judge_spike.persisted_response_provider(raw),
            persist_raw=lambda _t: None)

    before = list(paid_calls)
    report2 = judge_spike.run_judge_spike(
        ["v1"], _SPIKE_MANIFEST, _SPIKE_BASELINE,
        make_judge=make_replay_judge, load_inputs=_load_one_vod, depth=50)

    assert paid_calls == before  # no new paid calls during replay
    assert report2["gate"] == report["gate"]
    assert report2["top_view_misses"] == report["top_view_misses"]
