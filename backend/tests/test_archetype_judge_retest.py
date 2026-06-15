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

from ml import archetype_judge_retest as arm_a
from services.transcriber import RichSegment


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
