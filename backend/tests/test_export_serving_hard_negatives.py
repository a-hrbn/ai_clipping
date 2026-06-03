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
