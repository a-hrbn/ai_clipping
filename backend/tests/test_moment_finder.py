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
