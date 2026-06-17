"""Slice 2 (#104) — the Archetype-Judge wired into serving via the #103 seam.

find_moments re-ranks its deduped pool through the SAME RerankFn the offline
re-test validated (neutral_repost_judge + archetype_prompt_builder + the Gemini
provider, scoped to a pool-level RankScoreFn by judge_shortlist_rerank(depth=None)),
behind an opt-in flag that is default-OFF. Every test drives the judge through the
network-free fake_gemini_provider — no paid call, no ffmpeg.
"""
from __future__ import annotations

import json

from ml.archetype_judge_retest import fake_gemini_provider
from services import moment_finder
from services.transcriber import RichSegment


def _two_window_segments() -> list[RichSegment]:
    """Two well-separated windows so dedupe + diversity keep both candidates."""
    return [
        RichSegment(start=10.0, end=22.0, text="actual hook starts here now."),
        RichSegment(start=22.0, end=35.0, text="and the clear payoff lands."),
        RichSegment(start=200.0, end=212.0, text="a late beat begins quietly."),
        RichSegment(start=212.0, end=225.0, text="and it wraps up cleanly here."),
    ]


def _ab_pool() -> list[dict]:
    """The canonical 2-moment pool: an early high-virality beat (A) and a lower late
    one (B). Under the composite A leads; a judge that scores B higher flips it."""
    return [
        {"start": 10, "end": 35, "hook": 0.8, "flow": 0.7, "value": 0.9,
         "trend": 0.6, "virality": 0.95, "rationale": "early strong"},
        {"start": 200, "end": 225, "hook": 0.5, "flow": 0.5, "value": 0.5,
         "trend": 0.5, "virality": 0.5, "rationale": "mid late"},
    ]


def _fake_finder(raw_moments: list[dict]):
    """Monkeypatch target: the moment-finder provider returns a canned pool."""

    def fake_call(prompt: str, **kwargs):
        return (json.dumps({"moments": raw_moments}), {"provider": "fake"})

    return fake_call


def test_archetype_judge_full_override_replaces_composite_rank(monkeypatch) -> None:
    # Voll-Override (#104 AC): with the flag on and the profile carrying archetypes,
    # the judge score IS the rank — no blend. Composite would lead with the
    # high-virality early moment; the judge flips the order to the late one.
    raw = [
        {"start": 10, "end": 35, "hook": 0.8, "flow": 0.7, "value": 0.9,
         "trend": 0.6, "virality": 0.95, "rationale": "strong payoff"},
        {"start": 200, "end": 225, "hook": 0.5, "flow": 0.5, "value": 0.5,
         "trend": 0.5, "virality": 0.5, "rationale": "mid late"},
    ]
    monkeypatch.setattr(moment_finder.d0, "call_moment_finder_provider", _fake_finder(raw))

    # ids are the composite-rank positions: 1 = the high-virality early moment.
    canned = json.dumps({"scores": [
        {"id": 1, "repost_virality": 0.10},
        {"id": 2, "repost_virality": 0.95},
    ]})

    result = moment_finder.find_moments(
        _two_window_segments(), profile="jynxzi", vod_id="vod-1", max_tokens=1000,
        archetype_judge=True, judge_provider=fake_gemini_provider(canned),
    )

    assert [r["reason"] for r in result.post_processed_moments] == ["mid late", "strong payoff"]
    assert result.post_processed_moments[0]["composite_score"] == 9.5


def test_archetype_judge_serving_top20_matches_offline_replay(monkeypatch) -> None:
    # Parität (#104 AC 1): with the judge on, find_moments' delivered top-20 is
    # bit-equal to the offline replay_postprocess over the SAME pool + segments +
    # canned judge scores — the convergence the offline-validated judge relies on.
    from ml import replay_harness

    raw = _ab_pool()
    monkeypatch.setattr(moment_finder.d0, "call_moment_finder_provider", _fake_finder(raw))
    segments = _two_window_segments()
    canned = json.dumps({"scores": [
        {"id": 1, "repost_virality": 0.20},
        {"id": 2, "repost_virality": 0.90},
    ]})

    # Build the same pool the serving path builds internally, then replay the offline
    # post-processing over it under the same judge seam (default knobs find_moments uses).
    moments = [
        moment_finder.d0.LlmMoment(
            start=m["start"], end=m["end"], hook=m["hook"], flow=m["flow"],
            value=m["value"], trend=m["trend"], virality=m["virality"],
            rationale=m["rationale"],
        )
        for m in raw
    ]
    pool_rows = moment_finder.llm_moments_to_post_processor_rows(moments)
    profile_spec = moment_finder.d0.load_clip_profile("jynxzi")
    replay_rank = moment_finder.archetype_judge_rank_score_fn(
        profile_spec=profile_spec, segments=segments,
        provider=fake_gemini_provider(canned),
    )
    expected = replay_harness.replay_postprocess(
        pool_rows, segments, rank_score_fn=replay_rank, top_k=replay_harness.DEFAULT_TOP_K,
    )

    result = moment_finder.find_moments(
        segments, profile="jynxzi", vod_id="vod-1", max_tokens=1000,
        archetype_judge=True, judge_provider=fake_gemini_provider(canned),
    )

    assert result.post_processed_moments[:replay_harness.DEFAULT_TOP_K] == expected


def _boom_provider(_prompt: str) -> str:
    raise AssertionError("judge provider must not be called for a profile with no archetypes")


def test_archetype_judge_profile_gate_falls_back_to_composite(monkeypatch) -> None:
    # Profil-Gate (#104 AC 2): the flag is on, but the profile carries NO archetypes,
    # so find_moments ranks via the composite — output bit-equal to the composite path
    # and the judge provider is never called.
    raw = _ab_pool()
    monkeypatch.setattr(moment_finder.d0, "call_moment_finder_provider", _fake_finder(raw))
    segments = _two_window_segments()

    baseline = moment_finder.find_moments(
        segments, profile="general", vod_id="vod-1", max_tokens=1000,
    )
    gated = moment_finder.find_moments(
        segments, profile="general", vod_id="vod-1", max_tokens=1000,
        archetype_judge=True, judge_provider=_boom_provider,
    )

    assert gated.post_processed_moments == baseline.post_processed_moments
    assert "archetype_judge" not in gated.summary  # gate short-circuits before the judge runs


def test_archetype_judge_default_off_keeps_composite_even_with_archetypes(monkeypatch) -> None:
    # Flag default-OFF (#104 AC 7): the default find_moments call does NOT run the judge
    # even for an archetype profile (jynxzi) — production stays on the composite path until
    # a caller opts in. The provider is never built or called.
    raw = _ab_pool()
    monkeypatch.setattr(moment_finder.d0, "call_moment_finder_provider", _fake_finder(raw))
    segments = _two_window_segments()

    default_run = moment_finder.find_moments(
        segments, profile="jynxzi", vod_id="vod-1", max_tokens=1000,
    )
    explicit_off = moment_finder.find_moments(
        segments, profile="jynxzi", vod_id="vod-1", max_tokens=1000,
        archetype_judge=False, judge_provider=_boom_provider,
    )

    assert default_run.post_processed_moments == explicit_off.post_processed_moments
    assert "archetype_judge" not in default_run.summary
    # The early-strong moment leads on the composite path (no judge re-rank).
    assert default_run.post_processed_moments[0]["reason"] == "early strong"


def _recording_provider(prompts: list[str], canned: str):
    def _call(prompt: str) -> str:
        prompts.append(prompt)
        return canned

    return _call


def test_archetype_judge_prompt_carries_raw_12s_excerpt_no_markers(monkeypatch) -> None:
    # Excerpt (#104 AC): each candidate's judge line carries its ±12 s RAW transcript
    # excerpt built from the job segments — no chat=/spk= finder markers, and a segment
    # beyond the ±12 s pad is excluded.
    raw = [{"start": 100, "end": 110, "hook": 0.8, "flow": 0.8, "value": 0.8,
            "trend": 0.8, "virality": 0.9, "rationale": "the moment"}]
    monkeypatch.setattr(moment_finder.d0, "call_moment_finder_provider", _fake_finder(raw))
    segments = [
        RichSegment(start=98.0, end=104.0, text="hook builds toward the punchline",
                    speaker="STREAMER", chat_spike_score=9.0),
        RichSegment(start=104.0, end=112.0, text="and then the payoff finally lands",
                    speaker="GUEST_1"),
        RichSegment(start=200.0, end=210.0, text="FARAWAYBEACON an unrelated later topic"),
    ]
    prompts: list[str] = []
    canned = json.dumps({"scores": [{"id": 1, "repost_virality": 0.8}]})

    moment_finder.find_moments(
        segments, profile="jynxzi", vod_id="vod-1", max_tokens=1000,
        archetype_judge=True, judge_provider=_recording_provider(prompts, canned),
    )

    assert len(prompts) == 1
    prompt = prompts[0]
    assert "transcript:" in prompt
    assert "hook builds toward the punchline" in prompt
    assert "the payoff finally lands" in prompt
    assert "FARAWAYBEACON" not in prompt          # beyond the ±12 s pad -> excluded
    assert "chat=" not in prompt and "spk=" not in prompt
    assert moment_finder.EXCERPT_PAD_SEC == 12.0


def _scripted_provider(responses: list[str]):
    """A stateful provider returning each canned response in turn (per call)."""
    it = iter(responses)

    def _call(_prompt: str) -> str:
        return next(it)

    return _call


def test_archetype_judge_retries_underscored_batch_and_keeps_fullest(monkeypatch) -> None:
    # Chunk + per-batch retry (#104 AC): a batch that scores below the completeness
    # threshold is re-asked and the FULLEST response kept. Without retry the dropped
    # candidate is fail-closed out; with retry it is recovered and the fuller scoring wins.
    raw = _ab_pool()
    monkeypatch.setattr(moment_finder.d0, "call_moment_finder_provider", _fake_finder(raw))
    segments = _two_window_segments()

    incomplete = json.dumps({"scores": [{"id": 1, "repost_virality": 0.30}]})
    complete = json.dumps({"scores": [
        {"id": 1, "repost_virality": 0.30},
        {"id": 2, "repost_virality": 0.95},
    ]})

    # No retry: the under-scored batch drops the unscored candidate (fail-closed).
    no_retry = moment_finder.find_moments(
        segments, profile="jynxzi", vod_id="vod-1", max_tokens=1000,
        archetype_judge=True, judge_chunk_size=2, judge_max_retries=0,
        judge_provider=_scripted_provider([incomplete]),
    )
    assert [r["reason"] for r in no_retry.post_processed_moments] == ["early strong"]

    # With retry: the re-ask comes back complete, the fuller scoring is kept, and the
    # recovered candidate (higher judge score) now leads.
    with_retry = moment_finder.find_moments(
        segments, profile="jynxzi", vod_id="vod-1", max_tokens=1000,
        archetype_judge=True, judge_chunk_size=2, judge_max_retries=3,
        judge_provider=_scripted_provider([incomplete, complete]),
    )
    assert [r["reason"] for r in with_retry.post_processed_moments] == ["mid late", "early strong"]


def _raising_provider(_prompt: str) -> str:
    raise RuntimeError("provider timeout")


def test_archetype_judge_degrades_to_composite_on_empty_and_on_error(monkeypatch) -> None:
    # degrade-to-composite (#104 AC): an empty/over-truncated response (the offline LOUD
    # EmptyJudgeResponseError) AND a provider error/timeout both degrade — serving returns
    # the composite top-k (non-empty), the degrade is recorded, and nothing is raised.
    raw = _ab_pool()
    monkeypatch.setattr(moment_finder.d0, "call_moment_finder_provider", _fake_finder(raw))
    segments = _two_window_segments()

    baseline = moment_finder.find_moments(
        segments, profile="jynxzi", vod_id="vod-1", max_tokens=1000,
    )

    # Empty response -> EmptyJudgeResponseError inside the judge -> degrade, no raise.
    empty_run = moment_finder.find_moments(
        segments, profile="jynxzi", vod_id="vod-1", max_tokens=1000,
        archetype_judge=True, judge_provider=fake_gemini_provider(""),
    )
    assert empty_run.post_processed_moments  # never empty
    assert empty_run.post_processed_moments == baseline.post_processed_moments
    assert empty_run.summary["judge_degraded"] is True
    assert empty_run.summary["judge_degrade_reason"] == "EmptyJudgeResponseError"

    # Provider error/timeout -> degrade, no raise.
    error_run = moment_finder.find_moments(
        segments, profile="jynxzi", vod_id="vod-1", max_tokens=1000,
        archetype_judge=True, judge_provider=_raising_provider,
    )
    assert error_run.post_processed_moments == baseline.post_processed_moments
    assert error_run.summary["judge_degraded"] is True
    assert error_run.summary["judge_degrade_reason"] == "RuntimeError"


def test_archetype_judge_usage_is_reported_in_summary(monkeypatch) -> None:
    # Judge-Nutzung im Job-Summary (#104 AC): a successful judge run surfaces its
    # provider, model, archetype count, latency, and scored count — and the model
    # override flows through.
    raw = _ab_pool()
    monkeypatch.setattr(moment_finder.d0, "call_moment_finder_provider", _fake_finder(raw))
    segments = _two_window_segments()
    canned = json.dumps({"scores": [
        {"id": 1, "repost_virality": 0.40},
        {"id": 2, "repost_virality": 0.90},
    ]})

    result = moment_finder.find_moments(
        segments, profile="jynxzi", vod_id="vod-1", max_tokens=1000,
        archetype_judge=True, judge_model="gemini-custom",
        judge_provider=fake_gemini_provider(canned),
    )

    summary = result.summary
    assert summary["archetype_judge"] is True
    assert summary["judge_provider"] == "gemini"
    assert summary["judge_model"] == "gemini-custom"          # override flows through
    assert summary["judge_archetype_count"] == 8              # the 8 Jynxzi archetypes
    assert summary["judge_degraded"] is False
    assert summary["judge_scored_count"] == 2
    assert isinstance(summary["judge_latency_sec"], float)
    assert summary["judge_latency_sec"] >= 0.0


def test_archetype_judge_with_eval_recall_runs_both_branches(monkeypatch) -> None:
    # Regression: the pool-level judge rank_score_fn (rows -> rows) must stay distinct
    # from the per-LlmMoment cross-chunk scorer the eval-recall raw branch feeds to
    # llm_moments_to_post_processor_rows. Both flags on at once must run cleanly — the
    # delivered set is judged, while the raw-recall branch keeps the un-judged ceiling.
    raw = _ab_pool()
    monkeypatch.setattr(moment_finder.d0, "call_moment_finder_provider", _fake_finder(raw))
    segments = _two_window_segments()
    canned = json.dumps({"scores": [
        {"id": 1, "repost_virality": 0.20},
        {"id": 2, "repost_virality": 0.90},
    ]})

    result = moment_finder.find_moments(
        segments, profile="jynxzi", vod_id="vod-1", max_tokens=1000,
        archetype_judge=True, eval_recall=True,
        judge_provider=fake_gemini_provider(canned),
    )

    assert [r["reason"] for r in result.post_processed_moments] == ["mid late", "early strong"]
    assert result.ranked_candidates           # typed encoding of the judged delivered set
    assert result.raw_moments                 # un-judged raw-recall ceiling
    assert result.summary["judge_degraded"] is False
