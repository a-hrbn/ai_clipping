from __future__ import annotations

import json

from services import moment_finder
from services.transcriber import RichSegment


def _row(start, end, composite, *, reason) -> dict:
    """A post-processor input row (the dict currency snap_boundaries consumes)."""
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


def test_find_moments_delivers_exactly_what_the_shared_pool_produces(monkeypatch) -> None:
    # Convergence (#103, criterion 1): find_moments and the offline replay call
    # ONE shared post-processing function. Driving find_moments with a fake
    # provider and feeding the SAME moments straight through post_process_pool
    # must yield the identical delivered set — the parity the offline-validated
    # judge (Slice 2) relies on. (No length-discipline edits in this fixture, so
    # the pool find_moments builds equals the one constructed here.)
    raw = [
        {"start": 10, "end": 35, "hook": 0.8, "flow": 0.7, "value": 0.9,
         "trend": 0.6, "virality": 0.95, "rationale": "strong payoff"},
        {"start": 60, "end": 85, "hook": 0.5, "flow": 0.6, "value": 0.4,
         "trend": 0.3, "virality": 0.5, "rationale": "mid moment"},
        {"start": 200, "end": 225, "hook": 0.7, "flow": 0.7, "value": 0.7,
         "trend": 0.7, "virality": 0.7, "rationale": "late strong"},
    ]

    def fake_call(prompt: str, **kwargs):
        return (json.dumps({"moments": raw}), {"provider": "fake"})

    monkeypatch.setattr(moment_finder.d0, "call_moment_finder_provider", fake_call)
    segments = [
        RichSegment(start=10.0, end=20.0, text="actual hook starts."),
        RichSegment(start=20.0, end=35.0, text="clear payoff!"),
        RichSegment(start=60.0, end=75.0, text="a mid moment begins."),
        RichSegment(start=75.0, end=90.0, text="and it wraps up here!"),
        RichSegment(start=200.0, end=215.0, text="a late strong beat."),
        RichSegment(start=215.0, end=230.0, text="that finishes cleanly!"),
    ]

    moments = [
        moment_finder.d0.LlmMoment(
            start=m["start"], end=m["end"], hook=m["hook"], flow=m["flow"],
            value=m["value"], trend=m["trend"], virality=m["virality"],
            rationale=m["rationale"],
        )
        for m in raw
    ]
    pool_rows = moment_finder.llm_moments_to_post_processor_rows(moments)
    expected = moment_finder.post_process_pool(pool_rows, segments)

    result = moment_finder.find_moments(
        segments, profile="jynxzi", vod_id="vod-1", max_tokens=1000
    )

    assert result.post_processed_moments == expected


def test_post_process_pool_rank_score_fn_steers_the_delivered_order() -> None:
    # The pool-level seam (ADR 0001): a non-identity rank_score_fn receives the
    # deduped pool and overrides composite_score, flipping the delivered order.
    # This is the hook the Archetype-Judge plugs into (Slice 2).
    segments = [
        RichSegment(start=0.0, end=8.0, text="An opening statement that ends here."),
        RichSegment(start=100.0, end=108.0, text="A separate later topic ending here."),
    ]
    rows = [
        _row(0.0, 8.0, 8.0, reason="A"),     # higher composite -> leads under identity
        _row(100.0, 108.0, 7.0, reason="B"),
    ]

    def by_start_time(deduped: list[dict]) -> list[dict]:
        return [dict(r, composite_score=r["start_time"]) for r in deduped]

    baseline = moment_finder.post_process_pool(rows, segments)
    rescored = moment_finder.post_process_pool(rows, segments, rank_score_fn=by_start_time)

    assert baseline[0]["reason"] == "A"
    assert rescored[0]["reason"] == "B"  # later start now wins


def test_post_process_pool_top_k_truncates_to_the_k_highest_ranked() -> None:
    segments = [
        RichSegment(start=0.0, end=8.0, text="An opening statement that ends here."),
        RichSegment(start=100.0, end=108.0, text="A separate later topic ending here."),
    ]
    rows = [
        _row(0.0, 8.0, 8.0, reason="A"),
        _row(100.0, 108.0, 7.0, reason="B"),
    ]

    kept = moment_finder.post_process_pool(rows, segments, top_k=1)

    assert [r["reason"] for r in kept] == ["A"]  # the higher-composite moment


def test_post_process_pool_snaps_dedupes_diversifies_and_ranks() -> None:
    # The shared serving post-processing: snap -> dedupe -> rank_score_fn ->
    # diversity -> rank. Two rows snap onto the same window (a >=0.8-overlap
    # duplicate); the real dedupe drops the lower-composite one. A far-away row
    # survives. The delivered set is ranked by composite (keep above far).
    segments = [
        RichSegment(start=0.0, end=30.0, text="A long opening statement that ends here."),
        RichSegment(start=200.0, end=230.0, text="A far away separate topic ending here."),
    ]
    rows = [
        _row(2.0, 25.0, 8.0, reason="keep"),
        _row(5.0, 28.0, 7.0, reason="drop"),  # snaps onto "keep" -> duplicate
        _row(205.0, 225.0, 6.0, reason="far"),
    ]

    delivered = moment_finder.post_process_pool(rows, segments)

    assert [r["reason"] for r in delivered] == ["keep", "far"]
    assert "drop" not in {r["reason"] for r in delivered}


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


def test_find_moments_serving_output_is_bit_equal_to_today(monkeypatch) -> None:
    # Regression guard (#103): converging find_moments onto the shared
    # snap -> dedupe -> rank_score_fn -> diversity -> rank post-processing must
    # leave the delivered serving currency (post_processed_moments) byte-for-byte
    # unchanged on a fixed pool+segments fixture. The two mid moments overlap
    # after snapping (a real dedupe drop); the rest survive and rank by composite.
    def fake_call(prompt: str, **kwargs):
        return (
            json.dumps({"moments": [
                {"start": 10, "end": 35, "hook": 0.8, "flow": 0.7, "value": 0.9,
                 "trend": 0.6, "virality": 0.95, "rationale": "strong payoff"},
                {"start": 60, "end": 85, "hook": 0.5, "flow": 0.6, "value": 0.4,
                 "trend": 0.3, "virality": 0.5, "rationale": "mid moment"},
                {"start": 62, "end": 88, "hook": 0.4, "flow": 0.5, "value": 0.3,
                 "trend": 0.2, "virality": 0.45, "rationale": "near-dup of mid"},
                {"start": 200, "end": 225, "hook": 0.7, "flow": 0.7, "value": 0.7,
                 "trend": 0.7, "virality": 0.7, "rationale": "late strong"},
            ]}),
            {"provider": "fake"},
        )

    monkeypatch.setattr(moment_finder.d0, "call_moment_finder_provider", fake_call)
    segments = [
        RichSegment(start=0.0, end=5.0, text="Intro sentence one."),
        RichSegment(start=10.0, end=20.0, text="actual hook starts."),
        RichSegment(start=20.0, end=35.0, text="clear payoff!"),
        RichSegment(start=60.0, end=75.0, text="a mid moment begins."),
        RichSegment(start=75.0, end=90.0, text="and it wraps up here!"),
        RichSegment(start=200.0, end=215.0, text="a late strong beat."),
        RichSegment(start=215.0, end=230.0, text="that finishes cleanly!"),
    ]

    result = moment_finder.find_moments(
        segments, profile="jynxzi", vod_id="vod-1", max_tokens=1000
    )

    assert result.post_processed_moments == [
        {"rank": 1, "start_time": 10.0, "end_time": 35.0, "viral_score": 9.5,
         "composite_score": 8.625, "llm_virality": 0.95, "hook": 0.8, "flow": 0.7,
         "value": 0.9, "trend": 0.6, "reason": "strong payoff"},
        {"rank": 2, "start_time": 200.0, "end_time": 230.0, "viral_score": 7.0,
         "composite_score": 7.0, "llm_virality": 0.7, "hook": 0.7, "flow": 0.7,
         "value": 0.7, "trend": 0.7, "reason": "late strong"},
        {"rank": 3, "start_time": 60.0, "end_time": 90.0, "viral_score": 5.0,
         "composite_score": 4.85, "llm_virality": 0.5, "hook": 0.5, "flow": 0.6,
         "value": 0.4, "trend": 0.3, "reason": "mid moment"},
    ]


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
