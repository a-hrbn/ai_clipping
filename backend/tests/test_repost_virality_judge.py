from __future__ import annotations

import json
import logging

from ml import repost_virality_judge as judge


def _row(rank: int, start: float, end: float, *, reason: str = "", composite: float = 5.0) -> dict:
    """A deduped serving-currency row, the shape llm_moments_to_post_processor_rows emits."""
    return {
        "rank": rank,
        "start_time": start,
        "end_time": end,
        "viral_score": composite,
        "composite_score": composite,
        "llm_virality": 0.5,
        "hook": 0.5,
        "flow": 0.5,
        "value": 0.5,
        "trend": 0.5,
        "reason": reason,
    }


def test_fake_adapter_overwrites_rank_score_deterministically() -> None:
    rows = [
        _row(1, 100.0, 120.0, reason="quiet quotable beat", composite=5.0),
        _row(2, 300.0, 330.0, reason="rage clutch", composite=9.0),
    ]
    rerank = judge.fake_repost_virality_judge({1: 0.9, 2: 0.1})

    out_a = rerank([dict(r) for r in rows])
    out_b = rerank([dict(r) for r in rows])

    # The judge score (0-1) is written onto composite_score on the pipeline 1-10 scale,
    # overriding the original composite — the quiet beat now outranks the rage clutch.
    assert [r["composite_score"] for r in out_a] == [9.0, 1.0]
    # Deterministic: same rows in, identical rows out.
    assert out_a == out_b


def test_real_adapter_persists_raw_response_before_parsing() -> None:
    raw = '{"scores": [{"id": 1, "repost_virality": 0.8}]}'
    events: list[str] = []
    persisted: list[str] = []

    def call_provider(_prompt: str) -> str:
        events.append("call")
        return raw

    def persist_raw(text: str) -> None:
        events.append("persist")
        persisted.append(text)

    rerank = judge.repost_virality_judge(call_provider=call_provider, persist_raw=persist_raw)
    rerank([_row(1, 100.0, 120.0)])

    # The exact, unparsed provider output is persisted — and the provider is called,
    # then the raw is persisted, before any structured output is read.
    assert persisted == [raw]
    assert events == ["call", "persist"]


def test_real_adapter_parses_compact_scores_and_overwrites_rows() -> None:
    raw = '{"scores": [{"id": 1, "repost_virality": 0.2}, {"id": 2, "repost_virality": 0.95}]}'
    rerank = judge.repost_virality_judge(
        call_provider=lambda _p: raw,
        persist_raw=lambda _t: None,
    )

    out = rerank([_row(1, 100.0, 120.0, composite=9.0), _row(2, 300.0, 330.0, composite=4.0)])

    assert [r["composite_score"] for r in out] == [2.0, 9.5]
    assert [r["start_time"] for r in out] == [100.0, 300.0]


def test_parse_judge_scores_salvages_complete_items_from_truncated_response() -> None:
    # The provider hit max_tokens mid-array: ids 1 and 2 are complete, id 3 is cut off.
    raw = (
        '{"scores": ['
        '{"id": 1, "repost_virality": 0.8}, '
        '{"id": 2, "repost_virality": 0.6}, '
        '{"id": 3, "repost_virality": 0.4'
    )

    scores = judge.parse_judge_scores(raw)

    # Balanced items are recovered; the unterminated tail is discarded, not guessed.
    assert scores == {1: 0.8, 2: 0.6}


def test_unscored_candidate_is_dropped_and_the_drop_is_logged(caplog) -> None:
    raw = '{"scores": [{"id": 1, "repost_virality": 0.7}, {"id": 3, "repost_virality": 0.9}]}'
    rerank = judge.repost_virality_judge(call_provider=lambda _p: raw, persist_raw=lambda _t: None)
    rows = [
        _row(1, 10.0, 20.0, composite=5.0),
        _row(2, 40.0, 50.0, composite=8.0),  # judge said nothing — must NOT survive on its old 8.0
        _row(3, 70.0, 80.0, composite=5.0),
    ]

    with caplog.at_level(logging.INFO, logger="ml.repost_virality_judge"):
        out = rerank(rows)

    # Fail-closed: the unscored candidate is dropped, not default-filled.
    assert [r["start_time"] for r in out] == [10.0, 70.0]
    assert [r["composite_score"] for r in out] == [7.0, 9.0]
    # The drop is accounted for — a truncated/partial judge run never reads as
    # "scored everything" (PRD #17: kein stilles Truncaten).
    events = [json.loads(r.message) for r in caplog.records if r.name == "ml.repost_virality_judge"]
    assert {"candidates": 3, "scored": 2, "dropped": 1}.items() <= events[-1].items()


def test_build_judge_prompt_requests_compact_per_candidate_repost_scores() -> None:
    rows = [
        _row(1, 100.0, 120.0, reason="quiet quotable Messi explanation"),
        _row(2, 300.0, 330.0, reason="rage clutch meltdown"),
    ]

    prompt = judge.build_judge_prompt(rows)

    # Each candidate is addressable by the same 1-based id the parser maps back.
    assert "[1]" in prompt and "[2]" in prompt
    assert "quiet quotable Messi explanation" in prompt and "rage clutch meltdown" in prompt
    # The response contract the parser and prompt must agree on.
    assert '"scores"' in prompt and '"repost_virality"' in prompt
    # Scored on REPOST virality, explicitly not live arousal.
    assert "repost" in prompt.lower()
    # Compact by default: scores only, no per-candidate rationale/freetext requested.
    assert '"rationale"' not in prompt


def test_anthropic_sonnet_provider_targets_configured_topk_model(monkeypatch) -> None:
    captured: dict = {}

    def fake_call(prompt, *, model, **kwargs):
        captured["prompt"] = prompt
        captured["model"] = model
        return '{"scores": []}', {"input_tokens": 1}

    monkeypatch.setattr(judge.settings, "anthropic_model_topk", "claude-sonnet-4-6")
    monkeypatch.setattr(judge.d0, "call_anthropic_moment_finder", fake_call)

    provider = judge.anthropic_sonnet_provider()
    raw = provider("PROMPT")

    # Routes through the strong model the config names, returning just the text — no
    # network is touched (the provider call is monkeypatched), so the test is free.
    assert captured["model"] == "claude-sonnet-4-6"
    assert captured["prompt"] == "PROMPT"
    assert raw == '{"scores": []}'
