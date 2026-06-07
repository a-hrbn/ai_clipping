from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from ml import run_llm_moment_finder_d0 as d0


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_transcript(root: Path, vod_id: str) -> None:
    vod_dir = root / vod_id
    vod_dir.mkdir(parents=True, exist_ok=True)
    (vod_dir / "transcript_fixture.json").write_text(
        json.dumps(
            {
                "metadata": {"vod_id": vod_id},
                "transcript": [
                    {"start": 100.0, "end": 102.0, "text": "setup"},
                    {"start": 110.0, "end": 112.0, "text": "payoff"},
                ],
            }
        ),
        encoding="utf-8",
    )


def test_format_transcript_filters_short_filler_noise() -> None:
    prompt_transcript = d0.format_transcript_for_prompt(
        [
            d0.TranscriptSegment(start=0.0, end=1.0, text="you"),
            d0.TranscriptSegment(start=2.0, end=4.0, text="you you you"),
            d0.TranscriptSegment(start=10.0, end=13.0, text="real reaction"),
        ],
        filter_noise=True,
    )

    assert "real reaction" in prompt_transcript.text
    assert "you you you" not in prompt_transcript.text
    assert prompt_transcript.kept_segments == 1
    assert prompt_transcript.dropped_segments == 2
    assert prompt_transcript.dropped_reasons == {
        "repeated_filler_token": 1,
        "single_filler_token": 1,
    }


def test_format_transcript_uses_absolute_seconds_as_authoritative_timestamp() -> None:
    prompt_transcript = d0.format_transcript_for_prompt(
        [
            d0.TranscriptSegment(start=1184.0, end=1186.0, text="Is this even legal?"),
        ],
        filter_noise=True,
    )

    assert "[t=1184s - t=1186s | clock=19:44 - 19:46] Is this even legal?" in prompt_transcript.text


def test_select_spike_vod_uses_train_vod_with_transcript_and_excludes_holdout(tmp_path: Path) -> None:
    positives = [
        {"vod_id": "holdout", "candidate_id": "h1", "window_start": 10.0, "window_end": 20.0, "label": 1},
        {"vod_id": "holdout", "candidate_id": "h2", "window_start": 30.0, "window_end": 40.0, "label": 1},
        {"vod_id": "train", "candidate_id": "t1", "window_start": 100.0, "window_end": 120.0, "label": 1},
    ]
    positive_rows = tmp_path / "positives.jsonl"
    recent_holdout = tmp_path / "recent_holdout.jsonl"
    training_holdout = tmp_path / "training_holdout.jsonl"
    cache_root = tmp_path / "vod_feature_cache"
    _write_jsonl(positive_rows, positives)
    _write_jsonl(recent_holdout, [{"vod_id": "holdout"}])
    _write_jsonl(training_holdout, [])
    _write_transcript(cache_root, "holdout")
    _write_transcript(cache_root, "train")

    grouped = d0.positive_windows_by_vod(d0.read_jsonl(positive_rows))
    holdout_ids = d0.load_holdout_vod_ids(
        recent_holdout_policy_path=recent_holdout,
        training_holdout_policy_path=training_holdout,
        positive_rows=d0.read_jsonl(positive_rows),
    )

    assert d0.select_spike_vod(
        grouped,
        vod_cache_root=cache_root,
        requested_vod_id="",
        holdout_vod_ids=holdout_ids,
        allow_holdout=False,
    ) == "train"


def test_select_spike_vods_can_return_multiple_train_vods(tmp_path: Path) -> None:
    positives = [
        {"vod_id": "a", "candidate_id": "a1", "window_start": 10.0, "window_end": 20.0, "label": 1, "views": 10},
        {"vod_id": "b", "candidate_id": "b1", "window_start": 30.0, "window_end": 40.0, "label": 1, "views": 20},
        {"vod_id": "c", "candidate_id": "c1", "window_start": 50.0, "window_end": 60.0, "label": 1, "views": 30},
    ]
    cache_root = tmp_path / "vod_feature_cache"
    for vod_id in ["a", "b", "c"]:
        _write_transcript(cache_root, vod_id)

    grouped = d0.positive_windows_by_vod(positives)

    assert d0.select_spike_vods(
        grouped,
        vod_cache_root=cache_root,
        requested_vod_id="",
        holdout_vod_ids=set(),
        allow_holdout=False,
        limit=2,
    ) == ["c", "b"]


def test_parse_moments_and_evaluate_top_n_overlap() -> None:
    moments = d0.parse_moments(
        """
        ```json
        {"moments": [
          {"start": 100, "end": 120, "hook": 9, "flow": 8, "value": 9, "trend": 7, "virality": 0.95, "rationale": "strong"},
          {"start": 300, "end": 330, "hook": 0.4, "flow": 0.5, "value": 0.5, "trend": 0.2, "virality": 0.5, "rationale": "weak"}
        ]}
        ```
        """
    )
    positives = [
        d0.PositiveWindow(positive_id="p1", vod_id="v", start=105.0, end=115.0),
        d0.PositiveWindow(positive_id="p2", vod_id="v", start=500.0, end=520.0),
    ]

    report = d0.evaluate_moments(moments, positives, top_n=1, overlap_ratio=0.45)

    assert report["matched_positive_windows"] == 1
    assert report["positive_windows"] == 2
    assert report["recall_at_top_n"] == 0.5
    assert report["details"][0]["best_moment"]["overlap_ratio"] == 1.0
    assert report["details"][1]["matched"] is False


def test_parse_moments_repairs_missing_commas_between_objects() -> None:
    moments = d0.parse_moments(
        """
        {
          "moments": [
            {"start": 100, "end": 120, "hook": 0.9, "flow": 0.8, "value": 0.9, "trend": 0.7, "virality": 0.95, "rationale": "first"}
            {"start": 300, "end": 330, "hook": 0.4, "flow": 0.5, "value": 0.5, "trend": 0.2, "virality": 0.5, "rationale": "second"}
          ]
        }
        """
    )

    assert [moment.start for moment in moments] == [100.0, 300.0]


def test_parse_moments_recovers_after_stray_character_between_objects() -> None:
    moments = d0.parse_moments(
        """
        {
          "moments": [
            {"start": 100, "end": 120, "hook": 0.9, "flow": 0.8, "value": 0.9, "trend": 0.7, "virality": 0.95, "rationale": "first"},
        s   {"start": 300, "end": 330, "hook": 0.4, "flow": 0.5, "value": 0.5, "trend": 0.2, "virality": 0.5, "rationale": "second"}
          ]
        }
        """
    )

    assert [moment.start for moment in moments] == [100.0, 300.0]


def test_parse_moments_recovers_complete_objects_from_truncated_response() -> None:
    moments = d0.parse_moments(
        """
        {
          "moments": [
            {"start": 100, "end": 120, "hook": 0.9, "flow": 0.8, "value": 0.9, "trend": 0.7, "virality": 0.95, "rationale": "first"},
            {"start": 300, "end": 330, "hook": 0.4, "flow": 0.5, "value": 0.5, "trend": 0.2, "virality": 0.5, "rationale": "second"},
            {"start": 500, "end": 530, "hook": 0.4, "flow": 0.5, "value": 0.5, "trend": 0.2, "virality": 0.5, "rationale": "unterminated
        """
    )

    assert [moment.start for moment in moments] == [100.0, 300.0]


def test_repair_clock_encoded_timestamps_converts_compact_mmss_without_rationale_matching() -> None:
    moments = [
        d0.LlmMoment(
            start=1938.0,
            end=2036.0,
            hook=0.8,
            flow=0.7,
            value=0.9,
            trend=0.7,
            virality=0.9,
            rationale="Streamer receives an airsoft shotgun as fan mail, initially mistaking it for a real firearm.",
        )
    ]

    repaired = d0.repair_clock_encoded_timestamps(moments)

    assert repaired.repaired_moment_count == 1
    assert repaired.moments[0].start == 1178.0
    assert repaired.moments[0].end == 1236.0


def test_repair_clock_encoded_timestamps_keeps_invalid_clock_values() -> None:
    moments = [
        d0.LlmMoment(
            start=1968.0,
            end=2036.0,
            hook=0.8,
            flow=0.7,
            value=0.9,
            trend=0.7,
            virality=0.9,
            rationale="Seconds part above 59 cannot be compact clock time.",
        )
    ]

    repaired = d0.repair_clock_encoded_timestamps(moments)

    assert repaired.repaired_moment_count == 0
    assert repaired.moments[0].start == 1968.0
    assert repaired.moments[0].end == 2036.0


def test_evaluate_moments_can_match_positive_center_inside_moment() -> None:
    moments = [d0.LlmMoment(start=109.0, end=200.0, hook=1, flow=1, value=1, trend=1, virality=1, rationale="broad")]
    positives = [d0.PositiveWindow(positive_id="p1", vod_id="v", start=100.0, end=120.0)]

    overlap_only = d0.evaluate_moments(moments, positives, top_n=1, overlap_ratio=0.8, match_mode="overlap")
    center_or_overlap = d0.evaluate_moments(
        moments,
        positives,
        top_n=1,
        overlap_ratio=0.8,
        match_mode="center_or_overlap",
    )

    assert overlap_only["matched_positive_windows"] == 0
    assert center_or_overlap["matched_positive_windows"] == 1
    assert center_or_overlap["details"][0]["best_moment"]["positive_center_inside"] is True


def test_enforce_moment_length_discipline_caps_windows_and_drops_blobs() -> None:
    moments = [
        d0.LlmMoment(start=0.0, end=30.0, hook=1, flow=1, value=1, trend=1, virality=0.5, rationale="tight"),
        d0.LlmMoment(start=100.0, end=205.0, hook=1, flow=1, value=1, trend=1, virality=0.9, rationale="cap"),
        d0.LlmMoment(start=300.0, end=500.0, hook=1, flow=1, value=1, trend=1, virality=1.0, rationale="blob"),
        d0.LlmMoment(start=600.0, end=600.0, hook=1, flow=1, value=1, trend=1, virality=0.1, rationale="zero"),
    ]

    result = d0.enforce_moment_length_discipline(moments, max_duration_sec=90.0, blob_drop_duration_sec=120.0)

    assert result.input_moment_count == 4
    assert len(result.moments) == 2
    assert result.capped_moment_count == 1
    assert len(result.dropped_blob_moments) == 1
    assert result.dropped_non_positive_duration_count == 1
    assert result.moments[1].start == 100.0
    assert result.moments[1].end == 190.0
    assert result.moments[1].duration == 90.0


def test_default_length_discipline_caps_moderate_blobs_before_dropping_marathon_windows() -> None:
    moments = [
        d0.LlmMoment(start=5006.0, end=5216.0, hook=1, flow=1, value=1, trend=1, virality=0.9, rationale="pokemon"),
        d0.LlmMoment(start=0.0, end=600.0, hook=1, flow=1, value=1, trend=1, virality=0.9, rationale="marathon"),
    ]

    result = d0.enforce_moment_length_discipline(moments)

    assert len(result.moments) == 1
    assert result.moments[0].start == 5006.0
    assert result.moments[0].end == 5096.0
    assert result.capped_moment_count == 1
    assert len(result.dropped_blob_moments) == 1


def test_dedupe_moments_by_overlap_removes_repeated_llm_windows() -> None:
    moments = [
        d0.LlmMoment(start=100.0, end=160.0, hook=1, flow=1, value=1, trend=1, virality=0.9, rationale="first"),
        d0.LlmMoment(start=100.0, end=160.0, hook=1, flow=1, value=1, trend=1, virality=0.9, rationale="repeat"),
        d0.LlmMoment(start=112.0, end=160.0, hook=1, flow=1, value=1, trend=1, virality=0.8, rationale="overlap"),
        d0.LlmMoment(start=300.0, end=330.0, hook=1, flow=1, value=1, trend=1, virality=0.7, rationale="distinct"),
    ]

    kept, duplicates = d0.dedupe_moments_by_overlap(moments, overlap_ratio=0.8)

    assert [(moment.start, moment.end) for moment in kept] == [(100.0, 160.0), (300.0, 330.0)]
    assert len(duplicates) == 2


def test_evaluate_moments_ignores_over_max_duration_matches() -> None:
    moments = [
        d0.LlmMoment(start=0.0, end=200.0, hook=1, flow=1, value=1, trend=1, virality=1, rationale="too broad"),
    ]
    positives = [d0.PositiveWindow(positive_id="p1", vod_id="v", start=95.0, end=105.0)]

    loose = d0.evaluate_moments(moments, positives, top_n=1, overlap_ratio=0.45, match_mode="center_or_overlap")
    strict = d0.evaluate_moments(
        moments,
        positives,
        top_n=1,
        overlap_ratio=0.45,
        match_mode="center_or_overlap",
        max_duration_sec=90.0,
    )

    assert loose["matched_positive_windows"] == 1
    assert strict["matched_positive_windows"] == 0
    assert strict["moment_count"] == 0
    assert strict["ignored_long_moment_count"] == 1
    assert strict["top_n"] == 0


def test_parse_rescore_response_and_apply_scores_reorders_moments() -> None:
    rows = [
        {"rank": 1, "start": 100, "end": 120, "virality": 0.9, "rationale": "base first"},
        {"rank": 2, "start": 300, "end": 330, "virality": 0.2, "rationale": "base second"},
    ]
    scores = d0.parse_rescore_response(
        json.dumps(
            {
                "scores": [
                    {"id": "m1", "score": 0.1, "rationale": "weak"},
                    {"id": "m2", "score": 0.95, "rationale": "clipper-like"},
                ]
            }
        )
    )

    rescored = d0.apply_rescore_scores(rows, scores)

    assert [row["base_rank"] for row in rescored] == [2, 1]
    assert rescored[0]["rescore_rank"] == 1
    assert rescored[0]["rescore_source"] == "llm"


def test_apply_rescore_scores_can_require_complete_scores() -> None:
    rows = [
        {"rank": 1, "start": 100, "end": 120, "virality": 0.9},
        {"rank": 2, "start": 300, "end": 330, "virality": 0.2},
    ]

    try:
        d0.apply_rescore_scores(rows, {"m1": {"score": 0.1}}, require_all_scores=True)
    except ValueError as exc:
        assert "Missing 1 rescore score" in str(exc)
        assert "m2" in str(exc)
    else:
        raise AssertionError("Expected missing rescore score to fail.")


def test_filter_rescore_moment_rows_drops_long_windows() -> None:
    kept, dropped = d0.filter_rescore_moment_rows(
        [
            {"rank": 1, "start": 0.0, "end": 30.0},
            {"rank": 2, "start": 0.0, "end": 121.0},
        ],
        max_duration_sec=120.0,
    )

    assert [row["rank"] for row in kept] == [1]
    assert [row["rank"] for row in dropped] == [2]


def test_parse_rescore_response_recovers_after_stray_character_between_scores() -> None:
    scores = d0.parse_rescore_response(
        """
        {
          "scores": [
            {"id": "m1", "score": 0.1, "rationale": "weak"},
        e   {"id": "m2", "score": 0.95, "rationale": "clipper-like"}
          ]
        }
        """
    )

    assert scores["m1"]["score"] == 0.1
    assert scores["m2"]["score"] == 0.95


def test_parse_rescore_response_recovers_complete_scores_from_truncated_response() -> None:
    scores = d0.parse_rescore_response(
        """
        {
          "scores": [
            {"id": "m1", "score": 0.1, "rationale": "weak"},
            {"id": "m2", "score": 0.95, "rationale": "clipper-like"},
            {"id": "m3", "score":
        """
    )

    assert sorted(scores) == ["m1", "m2"]
    assert scores["m2"]["score"] == 0.95


def test_rescore_few_shots_exclude_target_vod(tmp_path: Path) -> None:
    cache_root = tmp_path / "vod_feature_cache"
    _write_transcript(cache_root, "target")
    _write_transcript(cache_root, "example")
    positives = {
        "target": [d0.PositiveWindow(positive_id="t", vod_id="target", start=100.0, end=120.0, title="target")],
        "example": [d0.PositiveWindow(positive_id="e", vod_id="example", start=100.0, end=120.0, title="example")],
    }

    rows = d0.build_rescore_few_shots(
        target_vod_id="target",
        positives_by_vod=positives,
        holdout_vod_ids=set(),
        vod_cache_root=cache_root,
        limit=10,
        allow_holdout=False,
    )

    assert [row["vod_id"] for row in rows] == ["example"]
    assert "payoff" in rows[0]["excerpt"]


def test_run_d0_with_mock_response_writes_outputs(tmp_path: Path) -> None:
    positive_rows = tmp_path / "positives.jsonl"
    recent_holdout = tmp_path / "recent_holdout.jsonl"
    training_holdout = tmp_path / "training_holdout.jsonl"
    cache_root = tmp_path / "vod_feature_cache"
    output_prefix = tmp_path / "artifacts" / "d0"
    mock_response = tmp_path / "response.json"
    _write_jsonl(
        positive_rows,
        [{"vod_id": "train", "candidate_id": "p1", "window_start": 100.0, "window_end": 120.0, "label": 1}],
    )
    _write_jsonl(recent_holdout, [])
    _write_jsonl(training_holdout, [])
    _write_transcript(cache_root, "train")
    mock_response.write_text(
        json.dumps(
            {
                "moments": [
                    {
                        "start": 99.0,
                        "end": 121.0,
                        "hook": 0.8,
                        "flow": 0.8,
                        "value": 0.9,
                        "trend": 0.7,
                        "virality": 0.92,
                        "rationale": "matches",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = d0.run_d0(
        profile="jynxzi",
        vod_id="",
        positive_rows_path=positive_rows,
        recent_holdout_policy_path=recent_holdout,
        training_holdout_policy_path=training_holdout,
        vod_cache_root=cache_root,
        video_source_root=tmp_path / "video_sources",
        output_prefix=output_prefix,
        provider="gemini",
        model="mock-sonnet",
        top_n=30,
        max_moments=30,
        vod_limit=1,
        overlap_ratio=0.45,
        match_mode="center_or_overlap",
        min_duration_sec=8.0,
        max_duration_sec=90.0,
        temperature=0.0,
        max_tokens=1024,
        timeout_sec=1.0,
        api_retries=0,
        retry_backoff_sec=0.0,
        gate_min_recall=0.15,
        filter_transcript_noise=True,
        allow_holdout=False,
        dry_run=False,
        mock_response_path=mock_response,
    )

    assert report["event"] == "llm_moment_finder_d0_done"
    assert report["evaluation"]["matched_positive_windows"] == 1
    assert (output_prefix.with_suffix(".moments.json")).exists()
    assert (output_prefix.with_suffix(".eval.json")).exists()
    assert (output_prefix.with_suffix(".prompt.txt")).exists()
    assert (output_prefix.with_suffix(".response.txt")).read_text(encoding="utf-8") == mock_response.read_text(encoding="utf-8")


def test_run_d0_with_bad_mock_response_writes_raw_response_and_error(tmp_path: Path) -> None:
    positive_rows = tmp_path / "positives.jsonl"
    recent_holdout = tmp_path / "recent_holdout.jsonl"
    training_holdout = tmp_path / "training_holdout.jsonl"
    cache_root = tmp_path / "vod_feature_cache"
    output_prefix = tmp_path / "artifacts" / "d0"
    mock_response = tmp_path / "bad_response.json"
    _write_jsonl(
        positive_rows,
        [{"vod_id": "train", "candidate_id": "p1", "window_start": 100.0, "window_end": 120.0, "label": 1}],
    )
    _write_jsonl(recent_holdout, [])
    _write_jsonl(training_holdout, [])
    _write_transcript(cache_root, "train")
    mock_response.write_text('{"moments": [not-json]}', encoding="utf-8")

    report = d0.run_d0(
        profile="jynxzi",
        vod_id="",
        positive_rows_path=positive_rows,
        recent_holdout_policy_path=recent_holdout,
        training_holdout_policy_path=training_holdout,
        vod_cache_root=cache_root,
        video_source_root=tmp_path / "video_sources",
        output_prefix=output_prefix,
        provider="gemini",
        model="mock-gemini",
        top_n=30,
        max_moments=30,
        vod_limit=1,
        overlap_ratio=0.45,
        match_mode="center_or_overlap",
        min_duration_sec=8.0,
        max_duration_sec=90.0,
        temperature=0.0,
        max_tokens=1024,
        timeout_sec=1.0,
        api_retries=0,
        retry_backoff_sec=0.0,
        gate_min_recall=0.15,
        filter_transcript_noise=True,
        allow_holdout=False,
        dry_run=False,
        mock_response_path=mock_response,
    )

    assert report["event"] == "llm_moment_finder_d0_failed"
    assert report["error_type"] == "JSONDecodeError"
    assert (output_prefix.with_suffix(".response.txt")).read_text(encoding="utf-8") == '{"moments": [not-json]}'


def test_run_d0_can_reuse_existing_response_without_provider_call(tmp_path: Path) -> None:
    positive_rows = tmp_path / "positives.jsonl"
    recent_holdout = tmp_path / "recent_holdout.jsonl"
    training_holdout = tmp_path / "training_holdout.jsonl"
    cache_root = tmp_path / "vod_feature_cache"
    output_prefix = tmp_path / "artifacts" / "d0"
    _write_jsonl(
        positive_rows,
        [{"vod_id": "train", "candidate_id": "p1", "window_start": 100.0, "window_end": 120.0, "label": 1}],
    )
    _write_jsonl(recent_holdout, [])
    _write_jsonl(training_holdout, [])
    _write_transcript(cache_root, "train")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_prefix.with_suffix(".response.txt").write_text(
        json.dumps(
            {
                "moments": [
                    {
                        "start": 99.0,
                        "end": 121.0,
                        "hook": 0.8,
                        "flow": 0.8,
                        "value": 0.9,
                        "trend": 0.7,
                        "virality": 0.92,
                        "rationale": "matches",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = d0.run_d0(
        profile="jynxzi",
        vod_id="",
        positive_rows_path=positive_rows,
        recent_holdout_policy_path=recent_holdout,
        training_holdout_policy_path=training_holdout,
        vod_cache_root=cache_root,
        video_source_root=tmp_path / "video_sources",
        output_prefix=output_prefix,
        provider="gemini",
        model="mock-gemini",
        top_n=30,
        max_moments=30,
        vod_limit=1,
        overlap_ratio=0.45,
        match_mode="center_or_overlap",
        min_duration_sec=8.0,
        max_duration_sec=90.0,
        temperature=0.0,
        max_tokens=1024,
        timeout_sec=1.0,
        api_retries=0,
        retry_backoff_sec=0.0,
        gate_min_recall=0.15,
        filter_transcript_noise=True,
        allow_holdout=False,
        dry_run=False,
        reuse_responses=True,
        mock_response_path=None,
    )

    assert report["event"] == "llm_moment_finder_d0_done"
    assert report["usage"] == {"reused_response": str(output_prefix.with_suffix(".response.txt"))}
    assert report["evaluation"]["matched_positive_windows"] == 1


def test_run_rescore_existing_with_mock_response_writes_operating_points(tmp_path: Path) -> None:
    positive_rows = tmp_path / "positives.jsonl"
    recent_holdout = tmp_path / "recent_holdout.jsonl"
    training_holdout = tmp_path / "training_holdout.jsonl"
    cache_root = tmp_path / "vod_feature_cache"
    output_prefix = tmp_path / "artifacts" / "d0"
    mock_response = tmp_path / "rescore_response.json"
    _write_jsonl(
        positive_rows,
        [
            {"vod_id": "target", "candidate_id": "p1", "window_start": 100.0, "window_end": 120.0, "label": 1},
            {"vod_id": "example", "candidate_id": "e1", "window_start": 100.0, "window_end": 120.0, "label": 1},
        ],
    )
    _write_jsonl(recent_holdout, [])
    _write_jsonl(training_holdout, [])
    _write_transcript(cache_root, "target")
    _write_transcript(cache_root, "example")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_prefix.with_suffix(".moments.json").write_text(
        json.dumps(
            {
                "moments": [
                    {"rank": 1, "start": 300.0, "end": 330.0, "hook": 0.2, "flow": 0.2, "value": 0.2, "trend": 0.2, "virality": 0.2, "rationale": "miss"},
                    {"rank": 2, "start": 99.0, "end": 121.0, "hook": 0.9, "flow": 0.8, "value": 0.9, "trend": 0.7, "virality": 0.5, "rationale": "hit"},
                ]
            }
        ),
        encoding="utf-8",
    )
    mock_response.write_text(
        json.dumps({"scores": [{"id": "m1", "score": 0.1}, {"id": "m2", "score": 0.95, "rationale": "best"}]}),
        encoding="utf-8",
    )

    report = d0.run_d0_rescore_existing(
        profile="jynxzi",
        vod_id="target",
        positive_rows_path=positive_rows,
        recent_holdout_policy_path=recent_holdout,
        training_holdout_policy_path=training_holdout,
        vod_cache_root=cache_root,
        video_source_root=tmp_path / "video_sources",
        output_prefix=output_prefix,
        provider="gemini",
        model="gemini-2.5-pro",
        vod_limit=1,
        overlap_ratio=0.45,
        match_mode="center_or_overlap",
        timeout_sec=1.0,
        api_retries=0,
        retry_backoff_sec=0.0,
        temperature=0.0,
        max_tokens=1024,
        rescore_batch_size=40,
        rescore_max_duration_sec=120.0,
        few_shot_limit=10,
        allow_holdout=False,
        dry_run=False,
        mock_response_path=mock_response,
    )

    assert report["event"] == "llm_moment_finder_d0_rescore_batch_done"
    assert report["operating_points"]["12"]["matched_positive_windows"] == 1
    assert report["gate"]["passes"] is True
    prompt = output_prefix.with_suffix(".rescore_prompt.txt").read_text(encoding="utf-8")
    assert "VOD example" in prompt
    assert "VOD target" not in prompt.split("Few-shot Clipper picks:", 1)[1].split("Return raw JSON only:", 1)[0]


def test_run_rescore_existing_fails_when_mock_response_misses_scores(tmp_path: Path) -> None:
    positive_rows = tmp_path / "positives.jsonl"
    recent_holdout = tmp_path / "recent_holdout.jsonl"
    training_holdout = tmp_path / "training_holdout.jsonl"
    cache_root = tmp_path / "vod_feature_cache"
    output_prefix = tmp_path / "artifacts" / "d0"
    mock_response = tmp_path / "rescore_response.json"
    _write_jsonl(
        positive_rows,
        [{"vod_id": "target", "candidate_id": "p1", "window_start": 100.0, "window_end": 120.0, "label": 1}],
    )
    _write_jsonl(recent_holdout, [])
    _write_jsonl(training_holdout, [])
    _write_transcript(cache_root, "target")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_prefix.with_suffix(".moments.json").write_text(
        json.dumps(
            {
                "moments": [
                    {"rank": 1, "start": 99.0, "end": 121.0, "virality": 0.5},
                    {"rank": 2, "start": 300.0, "end": 330.0, "virality": 0.2},
                ]
            }
        ),
        encoding="utf-8",
    )
    mock_response.write_text(json.dumps({"scores": [{"id": "m1", "score": 0.9}]}), encoding="utf-8")

    report = d0.run_d0_rescore_existing(
        profile="jynxzi",
        vod_id="target",
        positive_rows_path=positive_rows,
        recent_holdout_policy_path=recent_holdout,
        training_holdout_policy_path=training_holdout,
        vod_cache_root=cache_root,
        video_source_root=tmp_path / "video_sources",
        output_prefix=output_prefix,
        provider="gemini",
        model="gemini-2.5-pro",
        vod_limit=1,
        overlap_ratio=0.45,
        match_mode="center_or_overlap",
        timeout_sec=1.0,
        api_retries=0,
        retry_backoff_sec=0.0,
        temperature=0.0,
        max_tokens=1024,
        rescore_batch_size=40,
        rescore_max_duration_sec=120.0,
        few_shot_limit=0,
        allow_holdout=False,
        dry_run=False,
        mock_response_path=mock_response,
    )

    assert report["event"] == "llm_moment_finder_d0_rescore_batch_failed"
    assert report["failed_vod_count"] == 1
    assert "missing 1 rescore score" in report["per_vod"][0]["error"]


def test_run_rescore_existing_dry_run_batches_and_filters_long_moments(tmp_path: Path) -> None:
    positive_rows = tmp_path / "positives.jsonl"
    recent_holdout = tmp_path / "recent_holdout.jsonl"
    training_holdout = tmp_path / "training_holdout.jsonl"
    cache_root = tmp_path / "vod_feature_cache"
    output_prefix = tmp_path / "artifacts" / "d0"
    _write_jsonl(
        positive_rows,
        [{"vod_id": "target", "candidate_id": "p1", "window_start": 100.0, "window_end": 120.0, "label": 1}],
    )
    _write_jsonl(recent_holdout, [])
    _write_jsonl(training_holdout, [])
    _write_transcript(cache_root, "target")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_prefix.with_suffix(".moments.json").write_text(
        json.dumps(
            {
                "moments": [
                    {"rank": 1, "start": 0.0, "end": 30.0, "virality": 0.5},
                    {"rank": 2, "start": 40.0, "end": 70.0, "virality": 0.5},
                    {"rank": 3, "start": 80.0, "end": 230.0, "virality": 0.5},
                ]
            }
        ),
        encoding="utf-8",
    )

    report = d0.run_d0_rescore_existing(
        profile="jynxzi",
        vod_id="target",
        positive_rows_path=positive_rows,
        recent_holdout_policy_path=recent_holdout,
        training_holdout_policy_path=training_holdout,
        vod_cache_root=cache_root,
        video_source_root=tmp_path / "video_sources",
        output_prefix=output_prefix,
        provider="gemini",
        model="gemini-2.5-pro",
        vod_limit=1,
        overlap_ratio=0.45,
        match_mode="center_or_overlap",
        timeout_sec=1.0,
        api_retries=0,
        retry_backoff_sec=0.0,
        temperature=0.0,
        max_tokens=1024,
        rescore_batch_size=1,
        rescore_max_duration_sec=120.0,
        few_shot_limit=0,
        allow_holdout=False,
        dry_run=True,
    )

    assert report["event"] == "llm_moment_finder_d0_rescore_batch_dry_run"
    per_vod = report["per_vod"][0]
    assert per_vod["moment_count"] == 2
    assert per_vod["dropped_long_moment_count"] == 1
    assert per_vod["rescore_batch_count"] == 2
    assert (output_prefix.parent / "d0.rescore_prompt_001.txt").exists()
    assert (output_prefix.parent / "d0.rescore_prompt_002.txt").exists()


def test_run_d0_batch_dry_run_writes_aggregate_eval(tmp_path: Path) -> None:
    positive_rows = tmp_path / "positives.jsonl"
    recent_holdout = tmp_path / "recent_holdout.jsonl"
    training_holdout = tmp_path / "training_holdout.jsonl"
    cache_root = tmp_path / "vod_feature_cache"
    output_prefix = tmp_path / "artifacts" / "d0"
    _write_jsonl(
        positive_rows,
        [
            {"vod_id": "a", "candidate_id": "a1", "window_start": 100.0, "window_end": 120.0, "label": 1},
            {"vod_id": "b", "candidate_id": "b1", "window_start": 100.0, "window_end": 120.0, "label": 1},
        ],
    )
    _write_jsonl(recent_holdout, [])
    _write_jsonl(training_holdout, [])
    _write_transcript(cache_root, "a")
    _write_transcript(cache_root, "b")

    report = d0.run_d0(
        profile="jynxzi",
        vod_id="",
        positive_rows_path=positive_rows,
        recent_holdout_policy_path=recent_holdout,
        training_holdout_policy_path=training_holdout,
        vod_cache_root=cache_root,
        video_source_root=tmp_path / "video_sources",
        output_prefix=output_prefix,
        provider="gemini",
        model="mock-sonnet",
        top_n=180,
        max_moments=180,
        vod_limit=2,
        overlap_ratio=0.45,
        match_mode="center_or_overlap",
        min_duration_sec=8.0,
        max_duration_sec=90.0,
        temperature=0.0,
        max_tokens=16000,
        timeout_sec=1.0,
        api_retries=0,
        retry_backoff_sec=0.0,
        gate_min_recall=0.15,
        filter_transcript_noise=True,
        allow_holdout=False,
        dry_run=True,
        mock_response_path=None,
    )

    assert report["event"] == "llm_moment_finder_d0_batch_dry_run"
    assert report["vod_count"] == 2
    assert report["positive_windows"] == 2
    assert (output_prefix.with_suffix(".eval.json")).exists()
    assert (tmp_path / "artifacts" / "d0_a.prompt.txt").exists()
    assert (tmp_path / "artifacts" / "d0_b.prompt.txt").exists()


def test_main_writes_default_generation_into_run_directory(tmp_path: Path) -> None:
    positive_rows = tmp_path / "positives.jsonl"
    recent_holdout = tmp_path / "recent_holdout.jsonl"
    training_holdout = tmp_path / "training_holdout.jsonl"
    cache_root = tmp_path / "vod_feature_cache"
    run_root = tmp_path / "runs"
    _write_jsonl(
        positive_rows,
        [{"vod_id": "train", "candidate_id": "p1", "window_start": 100.0, "window_end": 120.0, "label": 1}],
    )
    _write_jsonl(recent_holdout, [])
    _write_jsonl(training_holdout, [])
    _write_transcript(cache_root, "train")

    exit_code = d0.main(
        [
            "--dry-run",
            "--vod-id",
            "train",
            "--positive-rows",
            str(positive_rows),
            "--recent-holdout-policy",
            str(recent_holdout),
            "--training-holdout-policy",
            str(training_holdout),
            "--vod-cache-root",
            str(cache_root),
            "--run-root",
            str(run_root),
            "--run-id",
            "job 1",
            "--top-n",
            "1",
            "--max-moments",
            "1",
        ]
    )

    run_dir = run_root / "job_1"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert manifest["run_id"] == "job_1"
    assert manifest["moment_prefix"] == str(run_dir / "jynxzi_llm_moment_finder_d0")
    assert (run_dir / "jynxzi_llm_moment_finder_d0.prompt.txt").exists()
    assert (run_dir / "jynxzi_llm_moment_finder_d0.eval.json").exists()


def test_run_d0_preflight_failure_blocks_provider_call(tmp_path: Path, monkeypatch) -> None:
    positive_rows = tmp_path / "positives.jsonl"
    recent_holdout = tmp_path / "recent_holdout.jsonl"
    training_holdout = tmp_path / "training_holdout.jsonl"
    cache_root = tmp_path / "vod_feature_cache"
    output_prefix = tmp_path / "artifacts" / "d0"
    _write_jsonl(
        positive_rows,
        [{"vod_id": "train", "candidate_id": "p1", "window_start": 100.0, "window_end": 120.0, "label": 1}],
    )
    _write_jsonl(recent_holdout, [])
    _write_jsonl(training_holdout, [])
    _write_transcript(cache_root, "train")
    preflight_summary = {
        "event": "full_vod_download_preflight",
        "failed_vod_ids": ["train"],
        "all_available": False,
        "by_vod": {"train": {"status": "download_failed", "error": "gone"}},
    }

    def fake_preflight(*args, **kwargs):
        raise d0.vod_download_preflight.VodDownloadPreflightError(preflight_summary)

    def fail_provider(*args, **kwargs):
        raise AssertionError("provider should not be called after preflight failure")

    monkeypatch.setattr(d0.vod_download_preflight, "ensure_vod_sources", fake_preflight)
    monkeypatch.setattr(d0, "call_moment_finder_provider", fail_provider)

    report = d0.run_d0(
        profile="jynxzi",
        vod_id="",
        positive_rows_path=positive_rows,
        recent_holdout_policy_path=recent_holdout,
        training_holdout_policy_path=training_holdout,
        vod_cache_root=cache_root,
        video_source_root=tmp_path / "video_sources",
        output_prefix=output_prefix,
        provider="gemini",
        model="mock-gemini",
        top_n=180,
        max_moments=180,
        vod_limit=1,
        overlap_ratio=0.45,
        match_mode="center_or_overlap",
        min_duration_sec=8.0,
        max_duration_sec=90.0,
        temperature=0.0,
        max_tokens=16000,
        timeout_sec=1.0,
        api_retries=0,
        retry_backoff_sec=0.0,
        gate_min_recall=0.15,
        filter_transcript_noise=True,
        allow_holdout=False,
        dry_run=False,
        mock_response_path=None,
    )

    assert report["event"] == "llm_moment_finder_d0_preflight_failed"
    assert report["gate"]["blocked"] is True
    assert report["download_preflight"]["failed_vod_ids"] == ["train"]
    assert (output_prefix.with_suffix(".eval.json")).exists()


def test_run_d0_preflight_runs_before_provider_call(tmp_path: Path, monkeypatch) -> None:
    positive_rows = tmp_path / "positives.jsonl"
    recent_holdout = tmp_path / "recent_holdout.jsonl"
    training_holdout = tmp_path / "training_holdout.jsonl"
    cache_root = tmp_path / "vod_feature_cache"
    output_prefix = tmp_path / "artifacts" / "d0"
    calls: list[str] = []
    _write_jsonl(
        positive_rows,
        [{"vod_id": "train", "candidate_id": "p1", "window_start": 100.0, "window_end": 120.0, "label": 1}],
    )
    _write_jsonl(recent_holdout, [])
    _write_jsonl(training_holdout, [])
    _write_transcript(cache_root, "train")

    def fake_preflight(vod_ids, **kwargs):
        calls.append("preflight")
        assert list(vod_ids) == ["train"]
        assert kwargs["download_missing"] is True
        assert kwargs["require_all"] is True
        return {"all_available": True, "failed_vod_ids": [], "by_vod": {}}

    def fake_provider(*args, **kwargs):
        calls.append("provider")
        return (
            json.dumps(
                {
                    "moments": [
                        {
                            "start": 99.0,
                            "end": 121.0,
                            "hook": 0.8,
                            "flow": 0.8,
                            "value": 0.9,
                            "trend": 0.7,
                            "virality": 0.92,
                            "rationale": "matches",
                        }
                    ]
                }
            ),
            {"mock": True},
        )

    monkeypatch.setattr(d0.vod_download_preflight, "ensure_vod_sources", fake_preflight)
    monkeypatch.setattr(d0, "call_moment_finder_provider", fake_provider)

    report = d0.run_d0(
        profile="jynxzi",
        vod_id="",
        positive_rows_path=positive_rows,
        recent_holdout_policy_path=recent_holdout,
        training_holdout_policy_path=training_holdout,
        vod_cache_root=cache_root,
        video_source_root=tmp_path / "video_sources",
        output_prefix=output_prefix,
        provider="gemini",
        model="mock-gemini",
        top_n=180,
        max_moments=180,
        vod_limit=1,
        overlap_ratio=0.45,
        match_mode="center_or_overlap",
        min_duration_sec=8.0,
        max_duration_sec=90.0,
        temperature=0.0,
        max_tokens=16000,
        timeout_sec=1.0,
        api_retries=0,
        retry_backoff_sec=0.0,
        gate_min_recall=0.15,
        filter_transcript_noise=True,
        allow_holdout=False,
        dry_run=False,
        mock_response_path=None,
    )

    assert calls == ["preflight", "provider"]
    assert report["event"] == "llm_moment_finder_d0_done"
    assert report["download_preflight"]["all_available"] is True


def test_call_anthropic_retries_read_timeout(monkeypatch) -> None:
    calls = {"count": 0}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "content": [{"type": "text", "text": "{\"moments\": []}"}],
                    "usage": {"input_tokens": 10, "output_tokens": 2},
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        calls["count"] += 1
        if calls["count"] == 1:
            raise TimeoutError("The read operation timed out")
        return FakeResponse()

    monkeypatch.setattr(d0.settings, "anthropic_api_key", "test-key")
    monkeypatch.setattr(d0.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(d0.time, "sleep", lambda seconds: None)

    text, usage = d0.call_anthropic_moment_finder(
        "prompt",
        model="mock-sonnet",
        temperature=0.0,
        max_tokens=16000,
        timeout_sec=900.0,
        retries=1,
        retry_backoff_sec=0.0,
    )

    assert calls["count"] == 2
    assert text == "{\"moments\": []}"
    assert usage["input_tokens"] == 10


def test_call_gemini_uses_openai_compatible_endpoint(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            message = types.SimpleNamespace(content='{"moments": []}')
            choice = types.SimpleNamespace(message=message)
            usage = types.SimpleNamespace(prompt_tokens=20, completion_tokens=4, total_tokens=24)
            return types.SimpleNamespace(choices=[choice], usage=usage)

    class FakeClient:
        def __init__(self, *, api_key, base_url, timeout):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            captured["timeout"] = timeout
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    fake_openai = types.SimpleNamespace(OpenAI=FakeClient)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setattr(d0.settings, "gemini_api_key", "gem-key")

    text, usage = d0.call_gemini_moment_finder(
        "prompt",
        model="gemini-2.5-flash-lite",
        temperature=0.0,
        max_tokens=16000,
        timeout_sec=900.0,
        retries=0,
        retry_backoff_sec=0.0,
    )

    assert captured["base_url"] == d0.GEMINI_OPENAI_BASE_URL
    assert captured["model"] == "gemini-2.5-flash-lite"
    assert captured["response_format"] == {"type": "json_object"}
    assert text == '{"moments": []}'
    assert usage == {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24}


def test_provider_model_defaults_to_gemini_flash_lite() -> None:
    assert d0.normalize_provider("") == "gemini"
    assert d0.resolve_provider_model("gemini", "") == "gemini-2.5-flash-lite"
    assert d0.resolve_provider_model("anthropic", "") == d0.DEFAULT_ANTHROPIC_MODEL


def test_aggregate_blocks_gate_when_any_vod_failed(tmp_path: Path) -> None:
    report = d0.aggregate_d0_reports(
        [
            {
                "event": "llm_moment_finder_d0_done",
                "vod_id": "good",
                "evaluation": {"positive_windows": 1, "matched_positive_windows": 1, "recall_at_top_n": 1.0},
            },
            {
                "event": "llm_moment_finder_d0_failed",
                "vod_id": "bad",
                "positive_windows": 1,
                "gate": {"blocked": True},
                "error_type": "RuntimeError",
                "error": "timeout",
            },
        ],
        output_prefix=tmp_path / "d0",
        top_n=180,
        max_moments=180,
        overlap_ratio=0.45,
        match_mode="center_or_overlap",
        gate_min_recall=0.15,
        dry_run=False,
    )

    assert report["event"] == "llm_moment_finder_d0_batch_partial_failed"
    assert report["failed_vod_count"] == 1
    assert report["gate"]["blocked"] is True
    assert report["gate"]["passes"] is False
