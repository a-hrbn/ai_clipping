from __future__ import annotations

import json

from services import moment_finder
from services.transcriber import RichSegment


def test_find_moments_reuses_d0_prompt_filters_noise_and_builds_ranked_candidates(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_call(prompt: str, **kwargs):
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return (
            json.dumps(
                {
                    "moments": [
                        {
                            "start": 10,
                            "end": 35,
                            "hook": 0.8,
                            "flow": 0.7,
                            "value": 0.9,
                            "trend": 0.6,
                            "virality": 0.95,
                            "rationale": "strong payoff",
                        },
                        {
                            "start": 100,
                            "end": 250,
                            "hook": 1.0,
                            "flow": 1.0,
                            "value": 1.0,
                            "trend": 1.0,
                            "virality": 1.0,
                            "rationale": "blob",
                        },
                    ]
                }
            ),
            {"provider": "fake"},
        )

    monkeypatch.setattr(moment_finder.d0, "call_moment_finder_provider", fake_call)
    segments = [
        RichSegment(start=0.0, end=1.0, text="you"),
        RichSegment(start=10.0, end=20.0, text="actual hook starts."),
        RichSegment(start=20.0, end=35.0, text="clear payoff!"),
        RichSegment(start=100.0, end=250.0, text="oversized region."),
    ]

    result = moment_finder.find_moments(
        segments,
        profile="jynxzi",
        vod_id="vod-1",
        provider="gemini",
        model="gemini-test",
        max_moments=10,
        max_tokens=1000,
        eval_recall=True,
    )

    assert "actual hook starts" in str(captured["prompt"])
    assert "[00:00 - 00:01] you" not in str(captured["prompt"])
    assert "t=10s - t=20s" in str(captured["prompt"])
    assert captured["kwargs"]["provider"] == "gemini"
    assert captured["kwargs"]["model"] == "gemini-test"
    assert result.summary["raw_moment_count"] == 2
    assert result.summary["timestamp_repaired_moment_count"] == 0
    assert result.summary["capped_moment_count"] == 1
    assert result.summary["dropped_long_moment_count"] == 0
    assert result.summary["ranked_candidate_count"] == 2
    ranked = result.ranked_candidates[0]
    assert ranked.candidate.source == "llm_moment_finder"
    assert ranked.rank_score == 10.0
    assert ranked.candidate.end <= 190.0
    assert ranked.features is not None
    assert ranked.features.has_llm_semantics is True
    assert ranked.features.projected_hook_score == 1.0
    assert "llm_moment_finder" in ranked.rationale


def test_rich_segments_map_streamer_role_to_streamer_name() -> None:
    # WALL-Re-Test (#92): the diarizer's STREAMER role is rendered as the
    # streamer's name in the finder prompt; guests keep their stable GUEST_n
    # label and un-diarized segments stay UNKNOWN.
    segments = [
        RichSegment(start=0.0, end=2.0, text="streamer line", speaker="STREAMER"),
        RichSegment(start=2.0, end=4.0, text="guest line", speaker="GUEST_1"),
        RichSegment(start=4.0, end=6.0, text="unknown line", speaker="UNKNOWN"),
    ]

    out = moment_finder.rich_segments_to_transcript_segments(segments, streamer_name="Jynxzi")

    assert [s.speaker for s in out] == ["Jynxzi", "GUEST_1", "UNKNOWN"]


def test_rich_segments_keep_streamer_role_when_no_name_given() -> None:
    segments = [RichSegment(start=0.0, end=2.0, text="streamer line", speaker="STREAMER")]

    out = moment_finder.rich_segments_to_transcript_segments(segments)

    assert out[0].speaker == "STREAMER"


def test_find_moments_eval_recall_off_skips_raw_and_typed(monkeypatch) -> None:
    """Serving default (eval_recall=False): no raw/pre-diversity branch and no
    typed RankedCandidate are materialized; the post-processed dict-rows (the one
    serving currency) stay bit-identical to the eval_recall=True run."""

    def fake_call(prompt: str, **kwargs):
        return (
            json.dumps(
                {
                    "moments": [
                        {
                            "start": 10,
                            "end": 35,
                            "hook": 0.8,
                            "flow": 0.7,
                            "value": 0.9,
                            "trend": 0.6,
                            "virality": 0.95,
                            "rationale": "strong payoff",
                        }
                    ]
                }
            ),
            {"provider": "fake"},
        )

    monkeypatch.setattr(moment_finder.d0, "call_moment_finder_provider", fake_call)
    segments = [
        RichSegment(start=10.0, end=20.0, text="actual hook starts."),
        RichSegment(start=20.0, end=35.0, text="clear payoff!"),
    ]

    serving = moment_finder.find_moments(
        segments, profile="jynxzi", vod_id="vod-1", max_tokens=1000
    )
    eval_run = moment_finder.find_moments(
        segments, profile="jynxzi", vod_id="vod-1", max_tokens=1000, eval_recall=True
    )

    # The serving currency is present and bit-identical to the eval run.
    assert serving.post_processed_moments
    assert serving.post_processed_moments == eval_run.post_processed_moments
    # Eval-only encodings stay empty on the serving default.
    assert serving.ranked_candidates == []
    assert serving.raw_moments == []
    assert serving.summary["ranked_candidate_count"] == 0
    assert serving.summary["raw_moment_count"] == 0
    # eval_recall=True reproduces both encodings as today.
    assert eval_run.ranked_candidates
    assert eval_run.raw_moments


def test_post_processed_rows_to_ranked_candidates_sorts_by_pipeline_score() -> None:
    ranked = moment_finder.post_processed_rows_to_ranked_candidates(
        [
            {
                "start_time": 50.0,
                "end_time": 70.0,
                "viral_score": 4.0,
                "llm_virality": 0.4,
                "rationale": "weaker",
            },
            {
                "start_time": 10.0,
                "end_time": 30.0,
                "viral_score": 8.0,
                "llm_virality": 0.8,
                "rationale": "stronger",
            },
        ]
    )

    assert [item.candidate.start for item in ranked] == [10.0, 50.0]
    assert ranked[0].rank_score == 8.0
    assert ranked[0].features is not None
    assert ranked[0].features.source == "llm_moment_finder"


def test_cross_chunk_rank_scores_keep_chunk_tops_on_top_after_pipeline_scaling() -> None:
    """Regression: percentile 1.0 + additive tiebreak pushed chunk-top scores
    above 1.0, so _virality_to_pipeline_score skipped the x10 scaling and every
    chunk's best moment ranked near the bottom (all_v5: GT rank 1 -> 356)."""

    def moment(start: float, virality: float) -> moment_finder.d0.LlmMoment:
        return moment_finder.d0.LlmMoment(
            start=start, end=start + 20.0,
            hook=virality, flow=virality, value=virality, trend=virality,
            virality=virality, rationale="",
        )

    # Two 2700s chunks; chunk 0 has the globally strongest moment.
    chunk0 = [moment(100.0, 0.95), moment(400.0, 0.6), moment(900.0, 0.5)]
    chunk1 = [moment(2800.0, 0.7), moment(3200.0, 0.4)]
    moments = chunk0 + chunk1

    scores = moment_finder._cross_chunk_rank_scores(moments, 2700.0, 0.0)
    pipeline = {
        key: moment_finder._virality_to_pipeline_score(score)
        for key, score in scores.items()
    }

    # All rank scores stay in the 0-1 range so the x10 conversion is uniform.
    assert all(score <= 1.0 for score in scores.values())
    ranked = sorted(moments, key=lambda m: pipeline[moment_finder._moment_key(m)], reverse=True)
    # Chunk tops (100s and 2800s) must occupy the global top ranks.
    assert {ranked[0].start, ranked[1].start} == {100.0, 2800.0}
    # Tiebreak between equal percentiles: higher raw composite wins.
    assert ranked[0].start == 100.0
