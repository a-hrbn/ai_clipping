import json
from pathlib import Path

from ml import export_serving_hard_negatives


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_export_serving_hard_negatives_keeps_high_ranked_non_positive_windows(tmp_path: Path) -> None:
    positives = tmp_path / "positives.jsonl"
    _write_jsonl(
        positives,
        [
            {
                "label": 1,
                "vod_id": "123",
                "window_start": 100.0,
                "window_end": 120.0,
            }
        ],
    )
    job_dir = tmp_path / "storage" / "job-a"
    debug_dir = job_dir / "debug"
    job_dir.mkdir(parents=True)
    (job_dir / "request.json").write_text(
        json.dumps({"url": "https://www.twitch.tv/videos/123"}),
        encoding="utf-8",
    )
    _write_jsonl(
        debug_dir / "ranked_candidates_pre_judge.jsonl",
        [
            {
                "rank_score": 9.0,
                "candidate": {"id": "good-miss", "start": 10.0, "end": 30.0, "source": "dense_inventory"},
                "features": {"hook_strength": 0.6},
                "rationale": [],
            },
            {
                "rank_score": 8.0,
                "candidate": {"id": "positive-overlap", "start": 101.0, "end": 116.0, "source": "dense_inventory"},
                "features": {"hook_strength": 0.8},
                "rationale": [],
            },
            {
                "rank_score": 7.0,
                "candidate": {"id": "selected", "start": 200.0, "end": 220.0, "source": "dense_inventory"},
                "features": {"hook_strength": 0.7},
                "rationale": [],
            },
        ],
    )
    _write_jsonl(
        debug_dir / "ranked_candidates_v2.jsonl",
        [
            {
                "candidate_id": "selected",
                "selected_after_diversity": True,
            }
        ],
    )

    payload = export_serving_hard_negatives.export_serving_hard_negatives(
        storage_dir=tmp_path / "storage",
        positive_rows_path=positives,
        pre_judge_top_k=10,
        max_per_vod=10,
        max_rows=10,
        positive_exclusion_overlap=0.25,
    )

    rows = payload["rows"]
    assert [row["candidate_id"] for row in rows] == ["good-miss"]
    assert rows[0]["dataset_source"] == "serving_trace_hard_negative_train"
    assert rows[0]["label_class"] == "serving_trace_hard_negative"
    assert rows[0]["best_positive_overlap_ratio"] == 0.0
    assert payload["summary"]["exported_rows"] == 1
    assert payload["summary"]["skipped"]["positive_window_overlap"] == 1
    assert payload["summary"]["skipped"]["selected_or_judged_good"] == 1


def test_export_serving_hard_negatives_prioritizes_same_vod_lookalikes(tmp_path: Path) -> None:
    positives = tmp_path / "positives.jsonl"
    _write_jsonl(positives, [{"label": 1, "vod_id": "123", "window_start": 100.0, "window_end": 120.0}])
    job_dir = tmp_path / "storage" / "job-a"
    debug_dir = job_dir / "debug"
    job_dir.mkdir(parents=True)
    (job_dir / "request.json").write_text(json.dumps({"url": "https://www.twitch.tv/videos/123"}), encoding="utf-8")
    _write_jsonl(
        debug_dir / "ranked_candidates_pre_judge.jsonl",
        [
            {
                "rank_score": 20.0,
                "candidate": {"id": "positive-ref", "start": 100.0, "end": 120.0, "source": "chat_velocity"},
                "features": {"chat_velocity_score": 0.9, "chat_peak": 1.0, "payoff_score": 8.0},
                "rationale": [],
            },
            {
                "rank_score": 19.0,
                "candidate": {"id": "generic-high-rank", "start": 200.0, "end": 220.0, "source": "quote"},
                "features": {"chat_velocity_score": 0.0, "chat_peak": 0.0, "payoff_score": 2.0},
                "rationale": [],
            },
            {
                "rank_score": 12.0,
                "candidate": {"id": "same-vod-lookalike", "start": 300.0, "end": 320.0, "source": "chat_velocity"},
                "features": {"chat_velocity_score": 0.88, "chat_peak": 0.95, "payoff_score": 7.8},
                "rationale": [],
            },
        ],
    )

    payload = export_serving_hard_negatives.export_serving_hard_negatives(
        storage_dir=tmp_path / "storage",
        positive_rows_path=positives,
        pre_judge_top_k=10,
        max_per_vod=1,
        max_rows=10,
        positive_exclusion_overlap=0.25,
    )

    assert [row["candidate_id"] for row in payload["rows"]] == ["same-vod-lookalike"]
    assert payload["rows"][0]["same_vod_positive_lookalike_score"] > 0.95
    assert payload["summary"]["selection_strategy"] == "same_vod_positive_lookalike"


def test_export_serving_hard_negatives_global_cap_is_fair_across_vods(tmp_path: Path) -> None:
    positives = tmp_path / "positives.jsonl"
    _write_jsonl(
        positives,
        [
            {"label": 1, "vod_id": "123", "window_start": 100.0, "window_end": 120.0},
            {"label": 1, "vod_id": "456", "window_start": 100.0, "window_end": 120.0},
        ],
    )
    for vod_id in ("123", "456"):
        job_dir = tmp_path / "storage" / f"job-{vod_id}"
        debug_dir = job_dir / "debug"
        job_dir.mkdir(parents=True)
        (job_dir / "request.json").write_text(
            json.dumps({"url": f"https://www.twitch.tv/videos/{vod_id}"}),
            encoding="utf-8",
        )
        _write_jsonl(
            debug_dir / "ranked_candidates_pre_judge.jsonl",
            [
                {
                    "rank_score": 20.0,
                    "candidate": {"id": f"positive-{vod_id}", "start": 100.0, "end": 120.0},
                    "features": {"chat_peak": 1.0},
                    "rationale": [],
                },
                {
                    "rank_score": 10.0,
                    "candidate": {"id": f"miss-{vod_id}", "start": 200.0, "end": 220.0},
                    "features": {"chat_peak": 1.0},
                    "rationale": [],
                },
            ],
        )

    payload = export_serving_hard_negatives.export_serving_hard_negatives(
        storage_dir=tmp_path / "storage",
        positive_rows_path=positives,
        pre_judge_top_k=10,
        max_per_vod=2,
        max_rows=2,
    )

    assert {row["vod_id"] for row in payload["rows"]} == {"123", "456"}


def test_export_serving_hard_negatives_skips_holdout_vods_by_default(tmp_path: Path) -> None:
    positives = tmp_path / "positives.jsonl"
    holdout = tmp_path / "holdout.jsonl"
    _write_jsonl(positives, [{"label": 1, "vod_id": "123", "window_start": 100.0, "window_end": 120.0}])
    _write_jsonl(holdout, [{"vod_id": "123", "dataset_role": "serving_recall_holdout"}])
    job_dir = tmp_path / "storage" / "job-a"
    debug_dir = job_dir / "debug"
    job_dir.mkdir(parents=True)
    (job_dir / "request.json").write_text(json.dumps({"url": "https://www.twitch.tv/videos/123"}), encoding="utf-8")
    _write_jsonl(
        debug_dir / "ranked_candidates_pre_judge.jsonl",
        [{"rank_score": 9.0, "candidate": {"id": "miss", "start": 10.0, "end": 30.0}}],
    )

    payload = export_serving_hard_negatives.export_serving_hard_negatives(
        storage_dir=tmp_path / "storage",
        positive_rows_path=positives,
        holdout_policy_rows_path=holdout,
        pre_judge_top_k=10,
        max_per_vod=10,
        max_rows=10,
    )

    assert payload["rows"] == []
    assert payload["summary"]["holdout_vods"] == 1
    assert payload["summary"]["skipped"]["holdout_vod_id"] == 1


def test_export_serving_hard_negatives_can_include_holdout_vods_for_diagnostics(tmp_path: Path) -> None:
    positives = tmp_path / "positives.jsonl"
    holdout = tmp_path / "holdout.jsonl"
    _write_jsonl(positives, [{"label": 1, "vod_id": "123", "window_start": 100.0, "window_end": 120.0}])
    _write_jsonl(holdout, [{"vod_id": "123", "dataset_role": "serving_recall_holdout"}])
    job_dir = tmp_path / "storage" / "job-a"
    debug_dir = job_dir / "debug"
    job_dir.mkdir(parents=True)
    (job_dir / "request.json").write_text(json.dumps({"url": "https://www.twitch.tv/videos/123"}), encoding="utf-8")
    _write_jsonl(
        debug_dir / "ranked_candidates_pre_judge.jsonl",
        [{"rank_score": 9.0, "candidate": {"id": "miss", "start": 10.0, "end": 30.0}}],
    )

    payload = export_serving_hard_negatives.export_serving_hard_negatives(
        storage_dir=tmp_path / "storage",
        positive_rows_path=positives,
        holdout_policy_rows_path=holdout,
        include_holdout_vods=True,
        pre_judge_top_k=10,
        max_per_vod=10,
        max_rows=10,
    )

    assert [row["candidate_id"] for row in payload["rows"]] == ["miss"]
    assert payload["summary"]["include_holdout_vods"] is True


def test_export_serving_hard_negatives_filters_to_requested_vod_ids(tmp_path: Path) -> None:
    positives = tmp_path / "positives.jsonl"
    _write_jsonl(
        positives,
        [
            {"label": 1, "vod_id": "123", "window_start": 100.0, "window_end": 120.0},
            {"label": 1, "vod_id": "456", "window_start": 100.0, "window_end": 120.0},
        ],
    )
    for vod_id in ("123", "456"):
        job_dir = tmp_path / "storage" / f"job-{vod_id}"
        debug_dir = job_dir / "debug"
        job_dir.mkdir(parents=True)
        (job_dir / "request.json").write_text(
            json.dumps({"url": f"https://www.twitch.tv/videos/{vod_id}"}),
            encoding="utf-8",
        )
        _write_jsonl(
            debug_dir / "ranked_candidates_pre_judge.jsonl",
            [
                {
                    "rank_score": 9.0,
                    "candidate": {"id": f"miss-{vod_id}", "start": 10.0, "end": 30.0},
                    "rationale": [],
                }
            ],
        )

    payload = export_serving_hard_negatives.export_serving_hard_negatives(
        storage_dir=tmp_path / "storage",
        positive_rows_path=positives,
        vod_ids={"456"},
        pre_judge_top_k=10,
        max_per_vod=10,
        max_rows=10,
    )

    assert [row["candidate_id"] for row in payload["rows"]] == ["miss-456"]
    assert payload["summary"]["vod_filter_ids"] == ["456"]
    assert payload["summary"]["skipped"]["outside_vod_filter"] == 1
