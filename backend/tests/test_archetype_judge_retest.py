"""Tests for Arm A of the Archetype-Judge Re-Test (ml/archetype_judge_retest.py).

Arm A is the *neutral control* (PRD #95, issue #96): a gemini-2.5-pro judge reads a
raw transcript excerpt (±12 s, no chat=/spk= markers) per candidate and scores it on
a neutral repost basis (0-1, score-only, NO archetypes, NO anti-loudness clause). It
plugs into the replay harness as the rank_score_fn over the deduped top-200/VOD and is
scored on recall@20 via directional_eval. The raw response is persisted before parsing
so a --replay reproduces the verdict network-free.

Every test runs on the network-free fake provider — never a paid Gemini call.
"""
from __future__ import annotations

import sys
import types

# The moment_finder package reaches RichSegment, which can pull the heavy
# transcription stack. Prior art (test_judge_spike) stubs whisper the same way.
sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_a, **_k: None))

import pytest

from ml import archetype_judge_retest as arm_a
from ml import transcribe_only_rerun as T
from services.transcriber import RichSegment, format_jsonl


def _row(*, start: float, end: float, composite: float = 5.0, text: str = "") -> dict:
    """A deduped serving-currency row; ``text`` (the excerpt) only when provided."""
    virality = composite / 10.0
    row = {
        "start_time": start,
        "end_time": end,
        "viral_score": composite,
        "composite_score": composite,
        "llm_virality": virality,
        "hook": virality,
        "flow": virality,
        "value": virality,
        "trend": virality,
        "reason": "generator rationale that must not crowd out the transcript",
    }
    if text:
        row["text"] = text
    return row


# ─────────────────────────────────────────────────────────────────────────────
# Slice 1 — the neutral rubric prompt: 0-1 score-only, transcript-fed, no
# archetypes, no anti-loudness clause
# ─────────────────────────────────────────────────────────────────────────────

def test_neutral_prompt_is_transcript_fed_score_only_and_carries_no_loudness_or_archetype_bias() -> None:
    rows = [
        _row(start=100.0, end=120.0, text="he calmly explains the en passant trap"),
        _row(start=300.0, end=330.0, text="and then the whole lobby loses it"),
    ]

    prompt = arm_a.build_neutral_judge_prompt(rows)

    # Each candidate is addressable by the same 1-based id the parser maps back, and
    # carries its transcript excerpt — the judge reads the actual content (User Story 4).
    assert "[1]" in prompt and "[2]" in prompt
    assert "he calmly explains the en passant trap" in prompt
    assert "and then the whole lobby loses it" in prompt
    # The response contract the reused parser and prompt must agree on.
    assert '"scores"' in prompt and '"repost_virality"' in prompt
    # Neutral 0-1 scale, scores only — no per-candidate rationale (which truncates JSON).
    assert "0.0" in prompt and "1.0" in prompt
    assert '"rationale"' not in prompt
    # NEUTRAL base: the #50 anti-loudness clause is gone (the deliberately neutralised
    # loudness signal must not return through the rubric), and no archetype is injected
    # (that is Arm B's single variable).
    assert "must not inflate" not in prompt
    assert "loud live reaction" not in prompt.lower()
    assert "archetype" not in prompt.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Slice 2 — the adapter feeds the ±12 s build_excerpt transcript into the prompt
# (raw text, no chat=/spk= markers), persists the raw before parsing, overwrites scores
# ─────────────────────────────────────────────────────────────────────────────

def _seg(start: float, end: float, text: str) -> dict:
    return {"start": start, "end": end, "text": text}


def test_adapter_feeds_pm12s_excerpt_persists_raw_before_parsing_and_overwrites_scores() -> None:
    # Candidate windows with NO transcript on the rows — the adapter must derive it
    # from the segments via build_excerpt(±12 s), not from the row.
    rows = [_row(start=100.0, end=120.0), _row(start=300.0, end=330.0)]
    segments = [
        _seg(95.0, 125.0, "he calmly explains the en passant trap"),
        _seg(305.0, 325.0, "and then the whole lobby loses it"),
        # A segment 30 s before candidate 1, well outside its ±12 s pad — must not leak in.
        _seg(60.0, 70.0, "UNRELATED chatter far away"),
    ]
    raw = '{"scores": [{"id": 1, "repost_virality": 0.9}, {"id": 2, "repost_virality": 0.2}]}'

    events: list[str] = []
    captured: dict[str, str] = {}

    def call_provider(prompt: str) -> str:
        events.append("call")
        captured["prompt"] = prompt
        return raw

    def persist_raw(text: str) -> None:
        events.append("persist")
        captured["persisted"] = text

    judge = arm_a.neutral_repost_judge(
        call_provider=call_provider, persist_raw=persist_raw, segments=segments)
    out = judge([dict(r) for r in rows])

    prompt = captured["prompt"]
    # Each candidate carries the raw transcript excerpt built from the segments…
    assert "he calmly explains the en passant trap" in prompt
    assert "and then the whole lobby loses it" in prompt
    # …only the ±12 s window — the far-away segment is not in candidate 1's excerpt.
    assert "UNRELATED chatter far away" not in prompt
    # Raw excerpt, never the chat=/spk= annotated finder prompt (the neutralised
    # loudness signal must not return through the back door, PRD #95 AC).
    assert "chat=" not in prompt and "spk=" not in prompt
    # The exact unparsed provider output is persisted, AFTER the call and BEFORE any
    # structured decode — a --replay re-reads it bit-for-bit.
    assert captured["persisted"] == raw
    assert events == ["call", "persist"]
    # The judge score (0-1) is written onto composite_score on the pipeline 1-10 scale.
    assert [r["composite_score"] for r in out] == [9.0, 2.0]
    assert [r["start_time"] for r in out] == [100.0, 300.0]
    # The returned pool rows stay clean serving currency — the excerpt is a prompt
    # input, not smuggled into the downstream rows.
    assert "text" not in out[0]


# ─────────────────────────────────────────────────────────────────────────────
# Slice 3 — an unscored candidate is dropped fail-closed (truncation salvage),
# never fabricated; the drop is surfaced
# ─────────────────────────────────────────────────────────────────────────────

def test_truncated_response_drops_the_unscored_candidate_and_surfaces_the_drop(caplog) -> None:
    import json
    import logging

    rows = [
        _row(start=10.0, end=20.0, composite=5.0),
        _row(start=40.0, end=50.0, composite=8.0),  # judge said nothing — must NOT survive
        _row(start=70.0, end=80.0, composite=4.0),
    ]
    # The provider hit max_tokens mid-array: ids 1 and 3 are complete, id 2 absent and
    # the final item is cut off — only balanced items are recovered, the rest dropped.
    raw = (
        '{"scores": ['
        '{"id": 1, "repost_virality": 0.7}, '
        '{"id": 3, "repost_virality": 0.9}, '
        '{"id": 4, "repost_virality": 0.5'
    )
    judge = arm_a.neutral_repost_judge(
        call_provider=lambda _p: raw, persist_raw=lambda _t: None, segments=[])

    with caplog.at_level(logging.INFO, logger="ml.archetype_judge_retest"):
        out = judge(rows)

    # Fail-closed: the unscored candidate (id 2) is dropped, not default-filled on its
    # old 8.0; ids 1 and 3 keep the judge's scores; the truncated id-4 tail is discarded.
    assert [r["start_time"] for r in out] == [10.0, 70.0]
    assert [r["composite_score"] for r in out] == [7.0, 9.0]
    # The drop is accounted for — a truncated run never reads as "scored everything".
    events = [json.loads(r.message) for r in caplog.records if r.name == "ml.archetype_judge_retest"]
    assert {"candidates": 3, "scored": 2, "dropped": 1}.items() <= events[-1].items()


# ─────────────────────────────────────────────────────────────────────────────
# Slice 4 — the Gemini provider seam binds gemini-2.5-pro at temp 0, text only
# ─────────────────────────────────────────────────────────────────────────────

def test_gemini_pro_provider_targets_gemini_2_5_pro_at_temp_zero(monkeypatch) -> None:
    captured: dict = {}

    def fake_call(prompt, *, model, temperature, **kwargs):
        captured["prompt"] = prompt
        captured["model"] = model
        captured["temperature"] = temperature
        return '{"scores": []}', {"input_tokens": 1}

    monkeypatch.setattr(arm_a.d0, "call_gemini_moment_finder", fake_call)

    provider = arm_a.gemini_pro_provider()
    raw = provider("PROMPT")

    # Routes through gemini-2.5-pro deterministically (temp 0), returning just the text
    # — no network is touched (the leaf is monkeypatched), so the test is free.
    assert captured["model"] == "gemini-2.5-pro"
    assert captured["temperature"] == 0.0
    assert captured["prompt"] == "PROMPT"
    assert raw == '{"scores": []}'


# ─────────────────────────────────────────────────────────────────────────────
# Slice 5 — the network-free fake provider replays one canned response
# ─────────────────────────────────────────────────────────────────────────────

def test_fake_gemini_provider_returns_canned_response_for_any_prompt() -> None:
    raw = '{"scores": [{"id": 1, "repost_virality": 0.42}]}'
    provider = arm_a.fake_gemini_provider(raw)

    # Deterministic and prompt-agnostic — no network, the same canned text every call.
    assert provider("anything") == raw
    assert provider("a different prompt entirely") == raw


# ─────────────────────────────────────────────────────────────────────────────
# Slice 6 — run_arm: judge the deduped top-depth, recall@20 over gold∪silver,
# surface the shortlist cut and the coverage gap (never a silent miss)
# ─────────────────────────────────────────────────────────────────────────────

def _arm_segments() -> list[RichSegment]:
    # Three well-separated sentence pairs so snap lands cleanly and the moments stay
    # far enough apart to survive enforce_diversity.
    return [
        RichSegment(start=100.0, end=110.0, text="Reveal one."),
        RichSegment(start=110.0, end=120.0, text="It lands."),
        RichSegment(start=300.0, end=315.0, text="Second beat."),
        RichSegment(start=315.0, end=330.0, text="It continues."),
        RichSegment(start=500.0, end=510.0, text="Third moment."),
        RichSegment(start=510.0, end=520.0, text="And done."),
    ]


def _arm_raw_rows() -> list[dict]:
    # Composites give a clean rank order A(8) > B(6) > C(4); the judge addresses the
    # shortlist by those rank positions.
    return [
        _row(start=100.0, end=120.0, composite=8.0),
        _row(start=300.0, end=330.0, composite=6.0),
        _row(start=500.0, end=520.0, composite=4.0),
    ]


# gold ∪ silver: v1 carries one gold (1.4M) and one silver (50k); the gap VOD carries a
# gold it has NO replayable artifacts for.
_ARM_MANIFEST = {
    "v1": [
        {"gt_tier": "gold", "start": 100.0, "end": 120.0, "views": 1_400_000.0,
         "desc": "reveal", "source_url": "x"},
        {"gt_tier": "silver", "start": 300.0, "end": 330.0, "views": 50_000.0,
         "desc": "banter", "source_url": "x"},
    ],
    "gap": [
        {"gt_tier": "gold", "start": 10.0, "end": 30.0, "views": 900_000.0,
         "desc": "uncovered", "source_url": "x"},
    ],
}

# The judge addresses the shortlist by rank: A=1, B=2, C=3.
_ARM_RESPONSE = (
    '{"scores": [{"id": 1, "repost_virality": 0.9}, '
    '{"id": 2, "repost_virality": 0.5}, {"id": 3, "repost_virality": 0.3}]}'
)


def _arm_load_inputs(vod_id: str):
    return (_arm_raw_rows(), _arm_segments()) if vod_id == "v1" else None


def _arm_make_judge(_vod_id: str, segments: list):
    return arm_a.neutral_repost_judge(
        call_provider=arm_a.fake_gemini_provider(_ARM_RESPONSE),
        persist_raw=lambda _t: None,
        segments=segments,
    )


def test_run_arm_scores_recall_over_gold_and_silver_and_surfaces_the_gap() -> None:
    result = arm_a.run_arm(
        ["v1", "gap"], _ARM_MANIFEST,
        make_judge=_arm_make_judge, load_inputs=_arm_load_inputs)

    # recall@20 is computed via replay_postprocess + directional_eval over gold ∪ silver:
    # the default population is BOTH tiers (gt_count 2 = the gold + the silver), not gold
    # alone, and the judge's top candidates cover both → recall@20 = 1.0.
    assert result["gold_tiers"] == ["gold", "silver"]
    assert result["floor"]["gt_count"] == 2
    assert result["floor"]["recall"]["20"] == 1.0
    # The headline tier (views ≥ 100k) keeps only the 1.4M gold — the silver is sub-threshold.
    assert result["headline"]["gt_count"] == 1

    # The VOD with no replayable artifacts is a SURFACED coverage gap, never folded in
    # as a miss: it is excluded from the scored denominator and named in both surfaces.
    assert result["coverage_gap_vods"] == ["gap"]
    assert "gap" in result["missing_prediction_vods"]
    assert result["scored_vods"] == 1

    # The per-VOD shortlist cut is recorded — no silent cap.
    assert result["shortlist_cuts"]["v1"]["pool_size"] == 3
    assert result["shortlist_cuts"]["v1"]["scored"] == 3


# ─────────────────────────────────────────────────────────────────────────────
# Slice 7 — the paid run persists each raw response; a --replay reproduces the
# arm bit-for-bit network-free
# ─────────────────────────────────────────────────────────────────────────────

def test_paid_arm_persists_raw_then_replays_bit_for_bit_without_paying(tmp_path) -> None:
    # ── Paid run: the real adapter over a recording provider; raw persisted per VOD.
    paid_calls: list[str] = []

    def recording_provider(_prompt: str) -> str:
        paid_calls.append("call")
        return _ARM_RESPONSE

    paid = arm_a.run_arm(
        ["v1", "gap"], _ARM_MANIFEST,
        make_judge=arm_a.paid_make_judge(tmp_path, provider=recording_provider),
        load_inputs=_arm_load_inputs)

    # The exact unparsed response is persisted before parsing — the replayable artifact.
    assert (tmp_path / "v1.raw.txt").read_text(encoding="utf-8") == _ARM_RESPONSE
    assert paid_calls == ["call"]  # one paid call for the one replayable VOD
    assert paid["floor"]["recall"]["20"] == 1.0

    # ── Replay: the persisted raw re-scores for free; the provider must NOT fire.
    before = list(paid_calls)
    replay = arm_a.run_arm(
        ["v1", "gap"], _ARM_MANIFEST,
        make_judge=arm_a.replay_make_judge(tmp_path),
        load_inputs=_arm_load_inputs)

    assert paid_calls == before  # no new paid calls during replay
    # Identical inputs + identical persisted responses ⇒ identical arm result.
    assert replay == paid


def test_replay_load_inputs_surfaces_a_vod_with_no_persisted_raw_as_a_gap(tmp_path) -> None:
    # A VOD that was never part of the paid run has no persisted raw — the replay loader
    # returns None (a surfaced coverage gap), never a re-priced call.
    load = arm_a.replay_load_inputs(tmp_path / "storage", tmp_path)
    assert load("never_judged") is None


# ═════════════════════════════════════════════════════════════════════════════
# Issue #98 — Arm B (Archetype-Rubric) + the combined-tier B-vs-A verdict.
# Arm B is the archetype arm: the SAME neutral base as Arm A + the 8 Jynxzi
# archetypes injected (the single isolated variable). The verdict reads combined
# (gold∪silver) floor recall@20 from both arms and gates on a pre-fixed +0.03 cut.
# ═════════════════════════════════════════════════════════════════════════════

# A small, deterministic archetype fixture — the prompt builder takes archetypes as
# a parameter, so the unit test is not coupled to the real 8-archetype profile.
_ARCHETYPES = [
    {"name": "BALRIGHT_VERBAL_TIC", "description": "his signature catchphrase moment",
     "transcript_signals": ["says balright", "rapid-fire verbal tic"], "weight": 0.3},
    {"name": "MOD_TROLL_BETRAYAL", "description": "a mod turns on him mid-stream",
     "transcript_signals": ["mod betrays him"], "weight": 0.2},
]


# ─────────────────────────────────────────────────────────────────────────────
# Slice 8 — Arm B's archetype prompt: the neutral base + the archetypes (name +
# description + transcript_signals via render_profile_rules) + the score-higher
# instruction; the block lives ONLY in Arm B, never in Arm A's neutral prompt.
# ─────────────────────────────────────────────────────────────────────────────

def test_archetype_prompt_injects_archetypes_and_score_higher_on_the_neutral_base() -> None:
    rows = [_row(start=100.0, end=120.0, text="he calmly explains the en passant trap")]

    prompt = arm_a.build_archetype_judge_prompt(rows, archetypes=_ARCHETYPES)

    # SAME neutral base: still transcript-fed, score-only, 0-1, the same response contract.
    assert "he calmly explains the en passant trap" in prompt
    assert '"scores"' in prompt and '"repost_virality"' in prompt
    assert "0.0" in prompt and "1.0" in prompt
    assert '"rationale"' not in prompt
    # The archetypes: name + description + transcript_signals, rendered via the SAME
    # render_profile_rules the serving finder uses ("listen for:" is its signal format).
    assert "BALRIGHT_VERBAL_TIC" in prompt and "his signature catchphrase moment" in prompt
    assert "says balright" in prompt  # a transcript_signal made it in
    assert "MOD_TROLL_BETRAYAL" in prompt
    assert "listen for:" in prompt
    # The single-variable instruction: reward a transcript that matches an archetype.
    assert "archetype" in prompt.lower()
    assert "higher" in prompt.lower()


def test_archetype_block_appears_only_in_arm_b_never_in_arm_a() -> None:
    rows = [_row(start=100.0, end=120.0, text="content")]

    arm_b_prompt = arm_a.build_archetype_judge_prompt(rows, archetypes=_ARCHETYPES)
    arm_a_prompt = arm_a.build_neutral_judge_prompt(rows)

    # The isolated variable: the archetype block is in B and absent from A — same base,
    # one difference. (Arm A's own neutrality is locked by its Slice-1 test above.)
    assert "BALRIGHT_VERBAL_TIC" in arm_b_prompt
    assert "BALRIGHT_VERBAL_TIC" not in arm_a_prompt
    assert "archetype" not in arm_a_prompt.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Slice 9 — the combined-tier verdict gate: B − A on combined (gold∪silver) floor
# recall@20, against the pre-fixed +0.03 cut; fail-closed when unmeasurable.
# ─────────────────────────────────────────────────────────────────────────────

def _arm_floor(floor20: float | None) -> dict:
    """The slice of a directional_eval.run result the combined-tier gate reads — the
    combined (gold∪silver) floor recall@20."""
    return {"floor": {"recall": {"20": floor20}}}


def test_combined_gate_passes_iff_floor_recall_clears_the_prefixed_cut() -> None:
    # The cut is pre-fixed IN CODE (+0.03) — read explicitly, never rationalised after.
    assert arm_a.COMBINED_GATE_CUT == 0.03

    # Exactly +0.03 (B 0.40 − A 0.37) clears the cut → PASS. The gate reads the
    # combined floor (run_arm scores over gold∪silver, equally weighted).
    gate = arm_a.evaluate_combined_gate(_arm_floor(0.40), _arm_floor(0.37))

    assert gate["passed"] is True
    assert gate["k"] == 20
    assert gate["cut"] == 0.03
    assert gate["combined_floor_recall_at_20"] == {"current": 0.40, "baseline": 0.37, "delta": 0.03}
    assert "PASS" in gate["reason"]


def test_combined_gate_is_no_go_when_the_rise_is_below_the_cut() -> None:
    # +0.02 (< +0.03) → NO-GO even though recall rose — the cut is the bar, not "> 0".
    gate = arm_a.evaluate_combined_gate(_arm_floor(0.39), _arm_floor(0.37))

    assert gate["passed"] is False
    assert gate["combined_floor_recall_at_20"]["delta"] == 0.02
    assert "NO-GO" in gate["reason"]


def test_combined_gate_fails_closed_when_a_recall_is_unmeasurable() -> None:
    # A missing recall on either side is never read as an improvement (fail-closed).
    gate = arm_a.evaluate_combined_gate(_arm_floor(0.50), _arm_floor(None))

    assert gate["passed"] is False
    assert gate["combined_floor_recall_at_20"]["delta"] is None
    assert "NO-GO" in gate["reason"]


# ─────────────────────────────────────────────────────────────────────────────
# Slice 10 — per-gold moves: exactly the golds served in B's top-20 but not A's.
# ─────────────────────────────────────────────────────────────────────────────

def _gt_hit(gt_index: int, desc: str, views: float, hit_rank: int | None) -> dict:
    """One row of a directional_eval per-VOD gt-hit table (the move-diff reads these)."""
    return {
        "gt_index": gt_index, "gt_start": 0.0, "gt_end": 0.0,
        "desc": desc, "views": views,
        "hit": hit_rank is not None, "hit_rank": hit_rank,
    }


def test_per_gold_moves_lists_exactly_golds_served_in_b_not_a_ordered_by_views() -> None:
    # Arm A served only "reveal"; "buried-low"/"buried-high"/"deep" were missed by A.
    arm_a_res = {"per_vod": [
        {"vod_id": "v1", "gt_hits": [
            _gt_hit(0, "reveal", 1_000_000.0, 5),       # served in A AND B → not a move
            _gt_hit(1, "buried-low", 200_000.0, None),  # missed in A, served in B → MOVE
            _gt_hit(2, "deep", 800_000.0, None),        # missed in A AND B → not a move
        ]},
        {"vod_id": "v2", "gt_hits": [
            _gt_hit(0, "buried-high", 900_000.0, None),  # missed in A, served in B → MOVE
        ]},
    ]}
    arm_b_res = {"per_vod": [
        {"vod_id": "v1", "gt_hits": [
            _gt_hit(0, "reveal", 1_000_000.0, 3),
            _gt_hit(1, "buried-low", 200_000.0, 8),
            _gt_hit(2, "deep", 800_000.0, None),
        ]},
        {"vod_id": "v2", "gt_hits": [
            _gt_hit(0, "buried-high", 900_000.0, 12),
        ]},
    ]}

    moves = arm_a.per_gold_moves(arm_b_res, arm_a_res)

    # Exactly the two golds B served in its top-20 that A did not — highest views first,
    # each carrying A's rank (None when A never served it) and B's new rank.
    assert [m["desc"] for m in moves] == ["buried-high", "buried-low"]
    assert moves[0] == {
        "vod_id": "v2", "gt_index": 0, "desc": "buried-high",
        "views": 900_000.0, "a_rank": None, "b_rank": 12,
    }
    assert moves[1]["vod_id"] == "v1"
    assert moves[1]["a_rank"] is None and moves[1]["b_rank"] == 8


# ─────────────────────────────────────────────────────────────────────────────
# Slice 11 — read-only breakdowns: gold-only, silver-only, combined-headline
# recall@20 (carried, not gated). Re-scored per tier from each arm's predictions.
# ─────────────────────────────────────────────────────────────────────────────

_BREAKDOWN_MANIFEST = {
    "v1": [
        {"gt_tier": "gold", "start": 100.0, "end": 120.0, "views": 1_400_000.0,
         "desc": "G", "source_url": "x"},
        {"gt_tier": "silver", "start": 300.0, "end": 330.0, "views": 50_000.0,
         "desc": "S", "source_url": "x"},
    ],
}


def _pred(start: float, end: float) -> dict:
    return {"start_time": start, "end_time": end}


def test_breakdowns_carry_gold_only_silver_only_and_combined_headline_recall() -> None:
    # Arm A serves only the gold; Arm B serves gold + silver (the archetype rubric
    # lifted the silver into the top-20). Predictions double as the raw pool, so each
    # VOD is scored (not a coverage gap).
    arm_a_res = {
        "final_by_vod": {"v1": [_pred(100.0, 120.0)]},
        "raw_by_vod": {"v1": [_pred(100.0, 120.0)]},
        "headline": {"recall": {"20": 1.0}},  # the 1.4M gold is in the served top-20
    }
    arm_b_res = {
        "final_by_vod": {"v1": [_pred(100.0, 120.0), _pred(300.0, 330.0)]},
        "raw_by_vod": {"v1": [_pred(100.0, 120.0), _pred(300.0, 330.0)]},
        "headline": {"recall": {"20": 1.0}},
    }

    breakdowns = arm_a.build_breakdowns(_BREAKDOWN_MANIFEST, ["v1"], arm_b_res, arm_a_res)

    # gold-only: both arms serve the gold → 1.0 each, no move.
    assert breakdowns["gold_only_recall_at_20"] == {"current": 1.0, "baseline": 1.0, "delta": 0.0}
    # silver-only: A misses the silver (0.0), B serves it (1.0) → +1.0. This is the
    # read-only colour the combined gate's headline number hides.
    assert breakdowns["silver_only_recall_at_20"] == {"current": 1.0, "baseline": 0.0, "delta": 1.0}
    # combined-headline: read straight from each arm's headline tier (views ≥ 100k).
    assert breakdowns["combined_headline_recall_at_20"] == {"current": 1.0, "baseline": 1.0, "delta": 0.0}


# ─────────────────────────────────────────────────────────────────────────────
# Slice 12 — run_verdict end-to-end: both arms at FULL depth (no top-N cut), the
# pre-fixed gate, breakdowns, and per-gold moves, all from real arm results.
# ─────────────────────────────────────────────────────────────────────────────

def _verdict_segments() -> list[RichSegment]:
    # Four well-separated candidate windows, each its own segment pair, so snap lands
    # cleanly and all four survive enforce_diversity.
    return [
        RichSegment(start=100.0, end=110.0, text="Gold one."),
        RichSegment(start=110.0, end=120.0, text="It lands."),
        RichSegment(start=300.0, end=315.0, text="Silver beat."),
        RichSegment(start=315.0, end=330.0, text="It continues."),
        RichSegment(start=900.0, end=910.0, text="Junk one."),
        RichSegment(start=910.0, end=920.0, text="More junk."),
        RichSegment(start=1100.0, end=1115.0, text="Junk two."),
        RichSegment(start=1115.0, end=1130.0, text="Yet more junk."),
    ]


def _verdict_raw_rows() -> list[dict]:
    # Composites give a clean rank order C1>C2>C3>C4, so the judge's 1-based ids map to
    # id1→C1[100-120], id2→C2[300-330], id3→C3[900-920], id4→C4[1100-1130].
    return [
        _row(start=100.0, end=120.0, composite=8.0),    # C1 — covers the gold G1
        _row(start=300.0, end=330.0, composite=6.0),    # C2 — covers the silver G2
        _row(start=900.0, end=920.0, composite=4.0),    # C3 — junk (no gold here)
        _row(start=1100.0, end=1130.0, composite=2.0),  # C4 — junk
    ]


# gold∪silver population = 2 (one gold G1 at 1.4M, one silver G2 at 50k).
_VERDICT_MANIFEST = {
    "v1": [
        {"gt_tier": "gold", "start": 100.0, "end": 120.0, "views": 1_400_000.0,
         "desc": "G1", "source_url": "x"},
        {"gt_tier": "silver", "start": 300.0, "end": 330.0, "views": 50_000.0,
         "desc": "G2", "source_url": "x"},
    ],
}

# Two canned judge verdicts addressed by the composite rank ids. SERVE_JUNK lifts the
# two junk candidates (ids 3,4) to the top; SERVE_GOLDS lifts the gold-covering ones.
_SERVE_JUNK = (
    '{"scores": [{"id": 1, "repost_virality": 0.1}, {"id": 2, "repost_virality": 0.1}, '
    '{"id": 3, "repost_virality": 0.9}, {"id": 4, "repost_virality": 0.8}]}'
)
_SERVE_GOLDS = (
    '{"scores": [{"id": 1, "repost_virality": 0.9}, {"id": 2, "repost_virality": 0.8}, '
    '{"id": 3, "repost_virality": 0.1}, {"id": 4, "repost_virality": 0.1}]}'
)


def _verdict_load_inputs(vod_id: str):
    return (_verdict_raw_rows(), _verdict_segments()) if vod_id == "v1" else None


def _make_judge_serving(raw: str):
    """A per-VOD make_judge whose canned judge verdict is ``raw`` (network-free)."""
    def _make(_vod_id: str, segments: list):
        return arm_a.neutral_repost_judge(
            call_provider=arm_a.fake_gemini_provider(raw),
            persist_raw=lambda _t: None,
            segments=segments,
        )
    return _make


def test_run_verdict_passes_when_arm_b_lifts_golds_into_the_served_top_20() -> None:
    # Arm A serves the two junk candidates (0 golds); Arm B serves the gold-covering
    # candidates (both golds). top_k=2 so the judge ranking actually decides the served set.
    verdict = arm_a.run_verdict(
        ["v1"], _VERDICT_MANIFEST,
        make_judge_a=_make_judge_serving(_SERVE_JUNK),
        make_judge_b=_make_judge_serving(_SERVE_GOLDS),
        load_inputs_a=_verdict_load_inputs,
        top_k=2,
    )

    # Both arms judged the FULL deduped pool — no top-N cut. The cut-removal is legible:
    # scored == pool_size, cut 0, depth None (#98 AC).
    assert verdict["depth"] is None
    cut_b = verdict["arm_b"]["shortlist_cuts"]["v1"]
    assert cut_b["pool_size"] == 4 and cut_b["scored"] == 4 and cut_b["cut"] == 0
    assert cut_b["depth"] is None

    # The gate: B combined floor recall@20 = 1.0, A = 0.0 → +1.0 ≥ +0.03 → PASS.
    assert verdict["gate"]["passed"] is True
    assert verdict["gate"]["combined_floor_recall_at_20"] == {"current": 1.0, "baseline": 0.0, "delta": 1.0}

    # Read-only breakdowns colour the move: both the gold and the silver tier rose.
    bd = verdict["breakdowns"]
    assert bd["gold_only_recall_at_20"]["delta"] == 1.0
    assert bd["silver_only_recall_at_20"]["delta"] == 1.0
    assert bd["combined_headline_recall_at_20"]["delta"] == 1.0

    # per-gold moves: exactly G1 and G2 entered B's top-20 (not A's), biggest views first.
    moves = verdict["per_gold_moves"]
    assert [m["desc"] for m in moves] == ["G1", "G2"]
    assert moves[0]["a_rank"] is None and moves[0]["b_rank"] == 1
    assert moves[1]["a_rank"] is None and moves[1]["b_rank"] == 2


def test_run_verdict_is_no_go_when_arm_b_lifts_nothing_over_arm_a() -> None:
    # Both arms serve junk → identical 0.0 combined floor recall → delta 0 < +0.03 → NO-GO.
    verdict = arm_a.run_verdict(
        ["v1"], _VERDICT_MANIFEST,
        make_judge_a=_make_judge_serving(_SERVE_JUNK),
        make_judge_b=_make_judge_serving(_SERVE_JUNK),
        load_inputs_a=_verdict_load_inputs,
        top_k=2,
    )

    assert verdict["gate"]["passed"] is False
    assert verdict["gate"]["combined_floor_recall_at_20"]["delta"] == 0.0
    assert "NO-GO" in verdict["gate"]["reason"]
    # No gold moved into B's top-20 that wasn't already in A's.
    assert verdict["per_gold_moves"] == []


# ─────────────────────────────────────────────────────────────────────────────
# Slice 13 — the Arm B make_judge wiring: the archetype prompt builder + the 8
# committed Jynxzi archetypes flow into the real judge.
# ─────────────────────────────────────────────────────────────────────────────

def test_load_jynxzi_archetypes_returns_the_eight_committed_archetypes() -> None:
    archetypes = arm_a.load_jynxzi_archetypes()

    # The 8 Jynxzi archetypes the AC pins, each with the fields render_profile_rules
    # renders (name + description + transcript_signals).
    assert len(archetypes) == 8
    assert all(
        a.get("name") and a.get("description") and a.get("transcript_signals")
        for a in archetypes
    )


def test_paid_make_judge_with_archetype_builder_feeds_archetypes_into_the_prompt(tmp_path) -> None:
    captured: dict[str, str] = {}

    def recording(prompt: str) -> str:
        captured["prompt"] = prompt
        return '{"scores": [{"id": 1, "repost_virality": 0.5}]}'

    make = arm_a.paid_make_judge(
        tmp_path, provider=recording,
        build_prompt=arm_a.archetype_prompt_builder(_ARCHETYPES))
    judge = make("v1", _arm_segments())
    judge([_row(start=100.0, end=120.0)])

    # Arm B's real make_judge injects the archetype block into the judge prompt; a
    # neutral make_judge (the default) would not (Arm A's neutrality is locked elsewhere).
    assert "BALRIGHT_VERBAL_TIC" in captured["prompt"]
    assert "listen for:" in captured["prompt"]


# ─────────────────────────────────────────────────────────────────────────────
# Slice 14 — the verdict replays bit-for-bit network-free from the persisted A and
# B raw responses (each arm's own artifact dir).
# ─────────────────────────────────────────────────────────────────────────────

def test_verdict_paid_persists_both_arms_then_replays_bit_for_bit_without_paying(tmp_path) -> None:
    arm_a_dir = tmp_path / "arm_a"
    arm_b_dir = tmp_path / "arm_b"
    a_calls: list[str] = []
    b_calls: list[str] = []

    def rec_a(_prompt: str) -> str:
        a_calls.append("a")
        return _SERVE_JUNK

    def rec_b(_prompt: str) -> str:
        b_calls.append("b")
        return _SERVE_GOLDS

    paid = arm_a.run_verdict(
        ["v1"], _VERDICT_MANIFEST,
        make_judge_a=arm_a.paid_make_judge(arm_a_dir, provider=rec_a),
        make_judge_b=arm_a.paid_make_judge(
            arm_b_dir, provider=rec_b,
            build_prompt=arm_a.archetype_prompt_builder(_ARCHETYPES)),
        load_inputs_a=_verdict_load_inputs,
        top_k=2,
    )

    # Each arm persisted its own unparsed raw to its OWN dir — the replayable artifacts.
    assert (arm_a_dir / "v1.raw.txt").read_text(encoding="utf-8") == _SERVE_JUNK
    assert (arm_b_dir / "v1.raw.txt").read_text(encoding="utf-8") == _SERVE_GOLDS
    assert a_calls == ["a"] and b_calls == ["b"]  # one paid call per arm per VOD
    assert paid["gate"]["passed"] is True

    # Replay: each arm reads ITS persisted raw; neither provider fires again.
    before_a, before_b = list(a_calls), list(b_calls)
    replay = arm_a.run_verdict(
        ["v1"], _VERDICT_MANIFEST,
        make_judge_a=arm_a.replay_make_judge(arm_a_dir),
        make_judge_b=arm_a.replay_make_judge(
            arm_b_dir, build_prompt=arm_a.archetype_prompt_builder(_ARCHETYPES)),
        load_inputs_a=_verdict_load_inputs,
        top_k=2,
    )

    assert a_calls == before_a and b_calls == before_b  # no new paid calls during replay
    # Identical inputs + identical persisted A and B responses ⇒ identical verdict.
    assert replay == paid


# ═════════════════════════════════════════════════════════════════════════════
# Issue #99 — Render-Step: each arm's gated top-20 as raw 16:9 cuts → frontend jobs.
# The render adapter cuts the EXACT replay_postprocess top-20 (no reframe/packaging,
# no video copy) into storage_dir/<arm_job_id>/clips/, registers one frontend job per
# (VOD, arm), and surfaces VODs with no local video.mp4 instead of dropping them. The
# real ffmpeg cutter / jobs.json registry are injected seams — these tests are
# ffmpeg-free and disk-light (a fake cutter, an in-memory store + register sink).
# ═════════════════════════════════════════════════════════════════════════════

import json

from models import ClipResult, JobStatus
from services.job_artifact_store import InMemoryJobArtifactStore


def _top20(*windows: tuple) -> list[dict]:
    """A gated top-20 the renderer consumes: post-processed rows in product-rank order."""
    return [_row(start=s, end=e, composite=c) for (s, e, c) in windows]


def _prebuilt_clips(job_id: str, *windows: tuple) -> list[ClipResult]:
    """ClipResults built WITHOUT a cut — so a registration test never renders (#99 AC)."""
    return [
        ClipResult(
            candidate_id=f"llm_moment_{rank:02d}", filename=f"clip_{rank:02d}.mp4",
            download_url=f"/static/{job_id}/clips/clip_{rank:02d}.mp4",
            start_time=s, end_time=e, viral_score=c, reason="")
        for rank, (s, e, c) in enumerate(windows, start=1)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Slice 1 — render_arm_clips cuts the EXACT top-20 windows to clip_NN.mp4 via the
# injected cutter, returning ClipResults with the existing /static/<job>/clips/ URLs.
# ─────────────────────────────────────────────────────────────────────────────

def test_render_arm_clips_cuts_the_exact_top20_windows_to_clip_nn_with_static_urls(tmp_path) -> None:
    clips_dir = tmp_path / "archetype_retest_v1_A" / "clips"
    video = tmp_path / "_download_cache" / "twitch" / "vod_v1.mp4"
    top20 = _top20((100.0, 120.0, 8.0), (300.0, 330.0, 6.0))

    cuts: list[tuple] = []

    def fake_cut_segment(video_path, start, end, output_path) -> None:
        cuts.append((video_path, start, end, output_path))

    clips = arm_a.render_arm_clips(
        video, top20, "archetype_retest_v1_A", clips_dir, cut_segment=fake_cut_segment)

    # Each top-20 moment is cut to clip_NN.mp4 (1-based rank), reading the cached video
    # IN PLACE (the same video_path every cut — no copy) at its EXACT gated window.
    assert [c[0] for c in cuts] == [video, video]
    assert [(c[1], c[2]) for c in cuts] == [(100.0, 120.0), (300.0, 330.0)]
    assert [c[3] for c in cuts] == [clips_dir / "clip_01.mp4", clips_dir / "clip_02.mp4"]

    # The ClipResults the existing frontend viewer renders: clip_NN filenames, the
    # /static/<job_id>/clips/ route URLs, and the exact gated windows in rank order.
    assert [c.filename for c in clips] == ["clip_01.mp4", "clip_02.mp4"]
    assert [c.download_url for c in clips] == [
        "/static/archetype_retest_v1_A/clips/clip_01.mp4",
        "/static/archetype_retest_v1_A/clips/clip_02.mp4",
    ]
    assert [(c.start_time, c.end_time) for c in clips] == [(100.0, 120.0), (300.0, 330.0)]
    # The candidate_id matches the serving pipeline's final-moment shape (llm_moment_NN).
    assert [c.candidate_id for c in clips] == ["llm_moment_01", "llm_moment_02"]


# ─────────────────────────────────────────────────────────────────────────────
# Slice 2 — register_arm_job writes final_moments = the gated top-20 + registers a
# done JobStatus carrying the ClipResults. Verifiable WITHOUT a render (#99 AC).
# ─────────────────────────────────────────────────────────────────────────────

def test_register_arm_job_writes_final_moments_and_a_done_job_without_rendering() -> None:
    store = InMemoryJobArtifactStore()
    registered: list = []
    top20 = _top20((100.0, 120.0, 8.0), (300.0, 330.0, 6.0))
    clips = _prebuilt_clips("archetype_retest_v1_A", (100.0, 120.0, 8.0), (300.0, 330.0, 6.0))

    job = arm_a.register_arm_job(
        "archetype_retest_v1_A", top20, clips,
        store=store, register_job=registered.append, profile="archetype_retest_A")

    # final_moments = the gated top-20 in the SAME serving shape ({moments, count}) the
    # diagnostics endpoint already reads — exactly what the recall measured.
    fm = store.read_final_moments("archetype_retest_v1_A")
    assert fm["count"] == 2
    assert [(m["start_time"], m["end_time"]) for m in fm["moments"]] == [(100.0, 120.0), (300.0, 330.0)]
    assert [m["candidate_id"] for m in fm["moments"]] == ["llm_moment_01", "llm_moment_02"]

    # A done job carrying the 20 ClipResults is registered through the injected sink —
    # no cut happened here (the clips were pre-built), proving registration needs no render.
    assert registered == [job]
    assert job.job_id == "archetype_retest_v1_A"
    assert job.status == "done" and job.progress_pct == 100
    assert job.profile == "archetype_retest_A"
    assert [c.download_url for c in job.clips] == [
        "/static/archetype_retest_v1_A/clips/clip_01.mp4",
        "/static/archetype_retest_v1_A/clips/clip_02.mp4",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Slice 3 — render_and_register_arm: A and B never collide (distinct job ids, clip
# dirs, /static URLs); a VOD with no local video.mp4 is a surfaced gap, not dropped.
# ─────────────────────────────────────────────────────────────────────────────

def test_render_and_register_arm_keeps_a_and_b_apart_and_surfaces_a_missing_video(tmp_path) -> None:
    storage = tmp_path / "storage"
    store = InMemoryJobArtifactStore()
    top20 = _top20((100.0, 120.0, 8.0), (300.0, 330.0, 6.0))

    cuts: list = []
    registered: list = []
    # v1 has a cached video; v2 does NOT — the surfaced render gap.
    videos = {"v1": tmp_path / "_download_cache" / "twitch" / "vod_v1.mp4"}

    def common() -> dict:
        return dict(
            storage_dir=storage, store=store, resolve_video=videos.get,
            cut_segment=lambda _v, _s, _e, out: cuts.append(out),
            register_job=registered.append)

    res_a = arm_a.render_and_register_arm("v1", "A", top20, **common())
    res_b = arm_a.render_and_register_arm("v1", "B", top20, **common())
    res_gap = arm_a.render_and_register_arm("v2", "A", top20, **common())

    # Collision-free: A and B get distinct archetype_retest_<vod>_<A|B> job ids…
    assert res_a["job_id"] == "archetype_retest_v1_A"
    assert res_b["job_id"] == "archetype_retest_v1_B"
    assert res_a["rendered"] and res_b["rendered"]
    assert {j.job_id for j in registered} == {"archetype_retest_v1_A", "archetype_retest_v1_B"}
    # …and each arm's cuts land under ITS OWN storage_dir/<job_id>/clips/ (no shared dir).
    assert {c.parent for c in cuts} == {
        storage / "archetype_retest_v1_A" / "clips",
        storage / "archetype_retest_v1_B" / "clips",
    }
    # A's registered clip URLs stay in A's namespace — they never point at B's.
    a_job = next(j for j in registered if j.job_id == "archetype_retest_v1_A")
    assert all("/static/archetype_retest_v1_A/clips/" in c.download_url for c in a_job.clips)
    assert store.read_final_moments("archetype_retest_v1_B")["count"] == 2

    # The VOD with no local video is a SURFACED gap — no cut, no job, named with the reason.
    assert res_gap == {
        "vod_id": "v2", "arm": "A", "job_id": "archetype_retest_v2_A",
        "rendered": False, "reason": "no_local_video", "clip_count": 0,
    }
    assert "archetype_retest_v2_A" not in {j.job_id for j in registered}
    assert store.read_final_moments("archetype_retest_v2_A") is None


# ─────────────────────────────────────────────────────────────────────────────
# Slice 4 — render_both_arms: one frontend job per (VOD, arm), A and B side by side,
# each rendering ITS OWN served top-20; VODs with no local video surfaced as gaps.
# ─────────────────────────────────────────────────────────────────────────────

def test_render_both_arms_registers_one_job_per_vod_arm_and_surfaces_gaps(tmp_path) -> None:
    storage = tmp_path / "storage"
    store = InMemoryJobArtifactStore()
    # Each arm serves its OWN top-20 — A's windows differ from B's (the whole point of the
    # side-by-side review: the archetype rubric picked a different served set).
    final_a = {"v1": _top20((100.0, 120.0, 8.0), (300.0, 330.0, 6.0)),
               "v2": _top20((10.0, 30.0, 5.0))}
    final_b = {"v1": _top20((500.0, 520.0, 9.0), (700.0, 720.0, 7.0)),
               "v2": _top20((10.0, 30.0, 5.0))}
    videos = {"v1": tmp_path / "vod_v1.mp4"}  # v2 has NO local video

    cuts: list = []
    registered: list = []
    summary = arm_a.render_both_arms(
        final_a, final_b,
        storage_dir=storage, store=store, resolve_video=videos.get,
        cut_segment=lambda _v, _s, _e, out: cuts.append(out),
        register_job=registered.append)

    # One frontend job per (VOD, arm) with a local video — A and B side by side for v1.
    assert set(summary["registered_job_ids"]) == {"archetype_retest_v1_A", "archetype_retest_v1_B"}
    assert {j.job_id for j in registered} == {"archetype_retest_v1_A", "archetype_retest_v1_B"}
    # Each arm rendered ITS OWN served top-20 into its own final_moments (A's ≠ B's).
    a_fm = store.read_final_moments("archetype_retest_v1_A")
    b_fm = store.read_final_moments("archetype_retest_v1_B")
    assert [(m["start_time"], m["end_time"]) for m in a_fm["moments"]] == [(100.0, 120.0), (300.0, 330.0)]
    assert [(m["start_time"], m["end_time"]) for m in b_fm["moments"]] == [(500.0, 520.0), (700.0, 720.0)]
    # v2 (no local video) is surfaced for BOTH arms — never silently dropped.
    assert set(summary["render_gap_jobs"]) == {"archetype_retest_v2_A", "archetype_retest_v2_B"}
    assert all(o["reason"] == "no_local_video"
               for o in summary["outcomes"] if not o["rendered"])
    assert not any(j.job_id.startswith("archetype_retest_v2") for j in registered)
    # Cuts only happened for the video that exists (v1, both arms), in arm-scoped dirs.
    assert {c.parent for c in cuts} == {
        storage / "archetype_retest_v1_A" / "clips",
        storage / "archetype_retest_v1_B" / "clips",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Slice 5a — the real registry + video-resolver adapters (no ffmpeg). upsert_jobs_registry
# merges an arm job into jobs.json without clobbering live jobs; cached_twitch_video
# resolves the download cache, surfacing absence as None.
# ─────────────────────────────────────────────────────────────────────────────

def test_upsert_jobs_registry_merges_without_clobbering_and_replaces_by_job_id(tmp_path) -> None:
    registry = tmp_path / "jobs.json"
    # A real serving job already in the registry — the offline render must NOT wipe it.
    registry.write_text(
        json.dumps([{"job_id": "real_job", "status": "done", "profile": "general"}]),
        encoding="utf-8")

    job = JobStatus(job_id="archetype_retest_v1_A", status="done",
                    progress_pct=100, profile="archetype_retest_A")
    arm_a.upsert_jobs_registry(registry, job)

    data = json.loads(registry.read_text(encoding="utf-8"))
    ids = [j["job_id"] for j in data]
    # Merged: the existing live job is preserved AND the arm job is appended.
    assert "real_job" in ids and "archetype_retest_v1_A" in ids
    arm_row = next(j for j in data if j["job_id"] == "archetype_retest_v1_A")
    assert arm_row["status"] == "done" and arm_row["profile"] == "archetype_retest_A"

    # Re-running the render upserts by job_id — the same arm job replaces, never duplicates.
    arm_a.upsert_jobs_registry(registry, job)
    data2 = json.loads(registry.read_text(encoding="utf-8"))
    assert [j["job_id"] for j in data2].count("archetype_retest_v1_A") == 1
    assert "real_job" in [j["job_id"] for j in data2]


def test_upsert_jobs_registry_starts_a_fresh_registry_when_absent(tmp_path) -> None:
    registry = tmp_path / "jobs.json"  # no file yet
    job = JobStatus(job_id="archetype_retest_v1_B", status="done",
                    progress_pct=100, profile="archetype_retest_B")

    arm_a.upsert_jobs_registry(registry, job)

    data = json.loads(registry.read_text(encoding="utf-8"))
    assert [j["job_id"] for j in data] == ["archetype_retest_v1_B"]


def test_cached_twitch_video_resolves_the_download_cache_or_surfaces_absence(tmp_path) -> None:
    cache = tmp_path / "_download_cache" / "twitch"
    cache.mkdir(parents=True)
    (cache / "vod_123.mp4").write_bytes(b"x")

    # A cached VOD resolves to its in-place video.mp4 (read by the renderer, never copied)…
    assert arm_a.cached_twitch_video("123", tmp_path) == cache / "vod_123.mp4"
    # …and a VOD with no local video surfaces as None — the renderer turns that into a gap.
    assert arm_a.cached_twitch_video("missing", tmp_path) is None


# ═════════════════════════════════════════════════════════════════════════════
# Issue #101 — the paid-run driver: read #100's fresh re-transcribed segments,
# gate on the consistency report, run paid → replay → render, read the verdict.
# The whole CLI is exercised on net-free / ffmpeg-free injected leaves (a fake
# Gemini provider, a recording cutter, a list register sink) — never a paid call.
# ═════════════════════════════════════════════════════════════════════════════

def _write_jsonl(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Slice 101-1 — fresh_segment_load_inputs pairs the original candidate pool (raw
# rows) with the FRESH re-transcribed segments (#100), NEVER the stale ones; a VOD
# missing either the pool or the fresh transcript is a surfaced gap (None).
# ─────────────────────────────────────────────────────────────────────────────

def test_fresh_segment_inputs_pair_the_pool_with_the_retranscribed_segments(tmp_path) -> None:
    storage = tmp_path / "storage"
    # The original replayable job dir: the candidate POOL + the STALE segments.
    pool_debug = storage / "pipeline_debug_run_v1" / "debug"
    _write_jsonl(pool_debug / "llm_moment_finder_moments_raw.jsonl",
                 [{"start_time": 100.0, "end_time": 120.0, "llm_virality": 0.8}])
    _write_jsonl(pool_debug / "segments.jsonl",
                 [{"start": 100.0, "end": 120.0, "text": "STALE transcript from the old run"}])
    # The fresh transcribe-only dir (#100): the re-transcribed segments the judge MUST read.
    fresh_debug = storage / "transcribe_only_v1" / "debug"
    _write_jsonl(fresh_debug / "segments.jsonl",
                 [{"start": 100.0, "end": 120.0, "text": "FRESH transcript from large-v3-turbo"}])

    load = arm_a.fresh_segment_load_inputs(storage, storage)
    loaded = load("v1")

    assert loaded is not None
    raw_rows, segments = loaded
    # The candidate pool comes from the original job dir…
    assert [r["start_time"] for r in raw_rows] == [100.0]
    # …but the transcript is the FRESH re-transcribed one (#100), never the stale dir's
    # — the whole point of Slice 5: the judge scores #100's consistent transcript basis.
    assert [s.text for s in segments] == ["FRESH transcript from large-v3-turbo"]
    assert all("STALE" not in s.text for s in segments)


def test_fresh_segment_inputs_surface_a_gap_when_pool_or_fresh_transcript_is_missing(tmp_path) -> None:
    storage = tmp_path / "storage"
    load = arm_a.fresh_segment_load_inputs(storage, storage)

    # A VOD with a fresh transcript but NO candidate pool — surfaced gap, never scored.
    _write_jsonl(storage / "transcribe_only_nopool" / "debug" / "segments.jsonl",
                 [{"start": 0.0, "end": 10.0, "text": "fresh but no pool"}])
    assert load("nopool") is None

    # A VOD with a candidate pool but NO fresh transcript — the re-transcription never
    # covered it, so it is a surfaced gap (we never judge it on a stale transcript).
    pool_debug = storage / "pipeline_debug_run_nofresh" / "debug"
    _write_jsonl(pool_debug / "llm_moment_finder_moments_raw.jsonl", [{"start_time": 5.0, "end_time": 15.0}])
    _write_jsonl(pool_debug / "segments.jsonl", [{"start": 5.0, "end": 15.0, "text": "stale only"}])
    assert load("nofresh") is None


# ─────────────────────────────────────────────────────────────────────────────
# Slice 101-2 — the hard consistency gate: refuse the paid run unless EVERY judged
# VOD passed the #100 report. Documented exclusions (no local video) never block; a
# flagged OR missing judged VOD aborts before any paid call (fail-closed).
# ─────────────────────────────────────────────────────────────────────────────

def _consistency_report(passed_ids, flagged=None, excluded=()):
    flagged = flagged or {}
    reports = [T.VodReport(vod_id=v, passed=True, flags=[]) for v in passed_ids]
    reports += [T.VodReport(vod_id=v, passed=False, flags=list(fl)) for v, fl in flagged.items()]
    clean = not excluded and not flagged
    return T.ConsistencyReport(clean=clean, vod_reports=reports, excluded=list(excluded))


def test_consistency_gate_passes_when_every_judged_vod_is_clean_despite_exclusions() -> None:
    # 2 documented exclusions (no local video) make report.clean False — but they are
    # NOT in the judged set, so they must NOT block the paid run (#100 fail-closed semantics).
    report = _consistency_report(["v1", "v2"], excluded=["g1", "g2"])
    assert report.clean is False

    # No raise: every judged VOD passed.
    arm_a.assert_transcripts_consistent(report, ["v1", "v2"])


def test_consistency_gate_aborts_when_a_judged_vod_is_flagged() -> None:
    report = _consistency_report(["v1"], flagged={"v2": ["over_run"]})

    with pytest.raises(arm_a.TranscriptConsistencyError) as exc:
        arm_a.assert_transcripts_consistent(report, ["v1", "v2"])

    # The abort names the offending VOD + its flag — surfaced, never silently judged.
    assert "v2" in str(exc.value) and "over_run" in str(exc.value)


def test_consistency_gate_aborts_when_a_judged_vod_has_no_fresh_transcript() -> None:
    # A judged VOD absent from the report has no fresh transcript — we have no consistency
    # verdict for it, so fail-closed: we never judge it on a stale / missing transcript.
    report = _consistency_report(["v1"])

    with pytest.raises(arm_a.TranscriptConsistencyError) as exc:
        arm_a.assert_transcripts_consistent(report, ["v1", "v_missing"])

    assert "v_missing" in str(exc.value)


# ─────────────────────────────────────────────────────────────────────────────
# Slice 101-3 — run_paid_verdict: the gate clears, both arms judge on the FRESH
# transcripts and persist to their OWN dirs (the archetype prompt routes to Arm B),
# and run_replay_verdict reproduces the verdict bit-for-bit, network-free.
# ─────────────────────────────────────────────────────────────────────────────

def _retest_storage(tmp_path):
    """A storage dir holding v1's candidate POOL + the fresh #100 transcripts on disk."""
    storage = tmp_path / "storage"
    pool = storage / "pipeline_debug_run_v1" / "debug"
    _write_jsonl(pool / "llm_moment_finder_moments_raw.jsonl", _verdict_raw_rows())
    # A deliberately STALE pool-dir transcript — the loader must ignore it for the fresh one.
    _write_jsonl(pool / "segments.jsonl", [{"start": 0.0, "end": 1.0, "text": "STALE"}])
    # Fresh #100 segments via the REAL serializer, so they carry the canonical writer schema
    # the consistency report checks (the CLI `run` mode builds that report off them).
    fresh = storage / "transcribe_only_v1" / "debug"
    fresh.mkdir(parents=True, exist_ok=True)
    (fresh / "segments.jsonl").write_text(format_jsonl(_verdict_segments()), encoding="utf-8")
    return storage


def test_run_paid_verdict_persists_both_arms_then_replays_bit_for_bit(tmp_path) -> None:
    storage = _retest_storage(tmp_path)
    artifacts = tmp_path / "artifacts"
    report = _consistency_report(["v1"])  # clean → the gate lets the paid run proceed

    calls: list[str] = []

    def provider(prompt: str) -> str:
        calls.append("call")
        # Prompt-sensitive: Arm B carries the archetype block, Arm A does not — proving
        # the prompt routing is real (and letting B lift the gold A's rubric buried).
        return _SERVE_GOLDS if "archetype" in prompt.lower() else _SERVE_JUNK

    paid = arm_a.run_paid_verdict(
        ["v1"], _VERDICT_MANIFEST,
        storage=storage, predictions_dir=storage, artifacts_root=artifacts,
        report=report, provider=provider, archetypes=_ARCHETYPES, top_k=2)

    # Each arm persisted its OWN unparsed raw to its OWN dir; the archetype prompt routed
    # to Arm B (gold-serving response), the neutral prompt to Arm A (junk-serving).
    assert (artifacts / "arm_a" / "v1.raw.txt").read_text(encoding="utf-8") == _SERVE_JUNK
    assert (artifacts / "arm_b" / "v1.raw.txt").read_text(encoding="utf-8") == _SERVE_GOLDS
    assert calls == ["call", "call"]  # one paid call per arm
    # The judge scored the FRESH-transcript candidate set and Arm B lifted the gold A
    # missed → the pre-fixed combined-floor gate reads PASS.
    assert paid["gate"]["passed"] is True

    # Replay: each arm reads ITS persisted raw; the provider must NOT fire again.
    before = list(calls)
    replay = arm_a.run_replay_verdict(
        ["v1"], _VERDICT_MANIFEST,
        storage=storage, predictions_dir=storage, artifacts_root=artifacts,
        archetypes=_ARCHETYPES, top_k=2)

    assert calls == before  # no new paid calls during the free replay
    # Identical inputs + identical persisted A and B responses ⇒ identical verdict.
    assert replay == paid


def test_run_paid_verdict_aborts_with_no_paid_calls_when_a_judged_vod_is_flagged(tmp_path) -> None:
    storage = _retest_storage(tmp_path)
    artifacts = tmp_path / "artifacts"
    report = _consistency_report([], flagged={"v1": ["over_run"]})  # v1 is flagged

    calls: list[str] = []

    def provider(prompt: str) -> str:
        calls.append("call")
        return _SERVE_GOLDS

    with pytest.raises(arm_a.TranscriptConsistencyError):
        arm_a.run_paid_verdict(
            ["v1"], _VERDICT_MANIFEST,
            storage=storage, predictions_dir=storage, artifacts_root=artifacts,
            report=report, provider=provider, archetypes=_ARCHETYPES, top_k=2)

    # The hard gate fired BEFORE any paid call — no money spent, no raw persisted.
    assert calls == []
    assert not (artifacts / "arm_a" / "v1.raw.txt").exists()


# ─────────────────────────────────────────────────────────────────────────────
# Slice 101-5 — render_verdict renders BOTH arms straight from a verdict's served
# top-20s (arm_a/arm_b final_by_vod) → one frontend job per (VOD, arm), so what the
# user reviews is byte-identical to what the recall measured.
# ─────────────────────────────────────────────────────────────────────────────

def test_render_verdict_renders_both_arms_from_the_verdicts_served_top20s(tmp_path) -> None:
    storage = tmp_path / "storage"
    store = InMemoryJobArtifactStore()
    # A verdict shape (the only fields render_verdict reads): each arm's served top-20.
    verdict = {
        "arm_a": {"final_by_vod": {"v1": _top20((100.0, 120.0, 8.0), (300.0, 330.0, 6.0))}},
        "arm_b": {"final_by_vod": {"v1": _top20((500.0, 520.0, 9.0))}},
    }
    videos = {"v1": tmp_path / "vod_v1.mp4"}

    cuts: list = []
    registered: list = []
    summary = arm_a.render_verdict(
        verdict, storage_dir=storage, store=store, resolve_video=videos.get,
        cut_segment=lambda _v, _s, _e, out: cuts.append(out),
        register_job=registered.append)

    # One frontend job per (VOD, arm) — A and B side by side.
    assert set(summary["registered_job_ids"]) == {"archetype_retest_v1_A", "archetype_retest_v1_B"}
    # Each arm rendered ITS OWN served top-20 (A's windows ≠ B's) — exactly the verdict's set.
    a_fm = store.read_final_moments("archetype_retest_v1_A")
    b_fm = store.read_final_moments("archetype_retest_v1_B")
    assert [(m["start_time"], m["end_time"]) for m in a_fm["moments"]] == [(100.0, 120.0), (300.0, 330.0)]
    assert [(m["start_time"], m["end_time"]) for m in b_fm["moments"]] == [(500.0, 520.0)]


# ─────────────────────────────────────────────────────────────────────────────
# Slice 101-6 — format_verdict_table: the pre-fixed verdict ABLESEN — the gated line
# (B−A combined floor r@20 vs the +0.03 cut), the read-only breakdowns, the per-gold
# moves, and the follow-on decision label fixed in code (PASS → ARCHETYPE_JUDGE_LIVES,
# NO-GO → RANKING_TRACK_CLOSED). No post-hoc interpretation.
# ─────────────────────────────────────────────────────────────────────────────

def _passing_verdict() -> dict:
    return arm_a.run_verdict(
        ["v1"], _VERDICT_MANIFEST,
        make_judge_a=_make_judge_serving(_SERVE_JUNK),
        make_judge_b=_make_judge_serving(_SERVE_GOLDS),
        load_inputs_a=_verdict_load_inputs, top_k=2)


def _no_go_verdict() -> dict:
    return arm_a.run_verdict(
        ["v1"], _VERDICT_MANIFEST,
        make_judge_a=_make_judge_serving(_SERVE_JUNK),
        make_judge_b=_make_judge_serving(_SERVE_JUNK),
        load_inputs_a=_verdict_load_inputs, top_k=2)


def test_verdict_table_reads_a_pass_as_archetype_judge_lives() -> None:
    table = arm_a.format_verdict_table(_passing_verdict())

    # The read: PASS → the pre-fixed ARCHETYPE_JUDGE_LIVES decision, never the NO-GO label.
    assert "PASS" in table
    assert arm_a.DECISION_ARCHETYPE_JUDGE_LIVES in table
    assert arm_a.DECISION_RANKING_TRACK_CLOSED not in table
    # The gated line is surfaced verbatim (combined floor r@20, the +0.03 cut)…
    assert "combined floor" in table.lower() and "0.03" in table
    # …and the per-gold moves the reader interprets the effect by (G1/G2 entered B's top-20).
    assert "G1" in table and "G2" in table


def test_verdict_table_reads_a_no_go_as_ranking_track_closed() -> None:
    table = arm_a.format_verdict_table(_no_go_verdict())

    assert "NO-GO" in table
    assert arm_a.DECISION_RANKING_TRACK_CLOSED in table
    assert arm_a.DECISION_ARCHETYPE_JUDGE_LIVES not in table


# ─────────────────────────────────────────────────────────────────────────────
# Slice 101-7 — the CLI: `run` gates + judges + persists + prints the verdict table
# (--only = the 1-VOD smoke); `render` re-reads the persisted raws free and registers
# one frontend job per (VOD, arm). Every heavy leaf (Gemini, ffmpeg, the registry) is
# injected, so the whole copy-paste surface is exercised net-free / ffmpeg-free.
# ─────────────────────────────────────────────────────────────────────────────

def _cli_fixture(tmp_path):
    storage = _retest_storage(tmp_path)
    # A meta sidecar so the consistency report can score v1 (span 100–1130 ⇒ clean at 1130).
    T.fresh_meta_path(storage, "v1").write_text(
        json.dumps({"duration_sec": 1130.0}), encoding="utf-8")
    # A cached local video so the render step produces clips (not a no-video gap).
    cache = storage / "_download_cache" / "twitch"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "vod_v1.mp4").write_bytes(b"x")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"vod_id": "v1", "gt_moments": _VERDICT_MANIFEST["v1"]}), encoding="utf-8")
    csv_path = tmp_path / "anchors.csv"
    csv_path.write_text("vod_id\nv1\n", encoding="utf-8")
    return storage, manifest, csv_path


def test_cli_run_then_render_routes_through_paid_replay_and_registers_both_arms(tmp_path, capsys) -> None:
    storage, manifest, csv_path = _cli_fixture(tmp_path)
    base = ["--storage", str(storage), "--manifest", str(manifest),
            "--anchor-csv", str(csv_path), "--top-k", "2"]

    calls: list[str] = []

    def provider(prompt: str) -> str:
        calls.append("call")
        return _SERVE_GOLDS if "archetype" in prompt.lower() else _SERVE_JUNK

    # `run --only v1` is the 1-VOD smoke: the gate clears, both arms judge + persist, and
    # the verdict table is printed with the pre-fixed decision read off (not rationalised).
    rc = arm_a.main(["run", "--only", "v1", *base], provider=provider)
    out = capsys.readouterr().out
    assert rc == 0
    assert "VERDICT:" in out and arm_a.DECISION_ARCHETYPE_JUDGE_LIVES in out
    assert calls == ["call", "call"]  # one paid call per arm, just the one smoke VOD
    assert (storage / "archetype_judge_retest" / "arm_a" / "v1.raw.txt").exists()
    assert (storage / "archetype_judge_retest" / "arm_b" / "v1.raw.txt").exists()

    # `render` re-reads the persisted raws FREE (no paid call) and registers one frontend
    # job per (VOD, arm) through the injected sink — A and B side by side in the viewer.
    cuts: list = []
    registered: list = []
    rc = arm_a.main(
        ["render", *base], store=InMemoryJobArtifactStore(),
        cut_segment=lambda _v, _s, _e, out: cuts.append(out),
        register_job=registered.append)
    out = capsys.readouterr().out
    assert rc == 0
    assert calls == ["call", "call"]  # render never pays — it replays the persisted raws
    assert {j.job_id for j in registered} == {"archetype_retest_v1_A", "archetype_retest_v1_B"}


def test_cli_run_aborts_nonzero_when_the_consistency_report_flags_a_judged_vod(tmp_path, capsys) -> None:
    storage, manifest, csv_path = _cli_fixture(tmp_path)
    # Break v1's transcript coverage: a tiny duration makes span_end overrun → `over_run` flag.
    T.fresh_meta_path(storage, "v1").write_text(
        json.dumps({"duration_sec": 10.0}), encoding="utf-8")

    calls: list[str] = []

    def provider(prompt: str) -> str:
        calls.append("call")
        return _SERVE_GOLDS

    rc = arm_a.main(
        ["run", "--storage", str(storage), "--manifest", str(manifest),
         "--anchor-csv", str(csv_path), "--top-k", "2"], provider=provider)
    out = capsys.readouterr().out

    # The paid run is refused before any paid call — surfaced, non-zero, documented.
    assert rc != 0
    assert "ABORT" in out.upper() and "v1" in out
    assert calls == []
