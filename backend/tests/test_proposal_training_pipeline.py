import json
import sys
import types

sys.modules.setdefault("whisper", types.SimpleNamespace(load_model=lambda *_args, **_kwargs: None))

from ml import materialize_proposal_training_debug, train_proposal_pipeline
from services.transcriber import RichSegment


def test_materialize_proposal_debug_from_cached_transcript(tmp_path, monkeypatch) -> None:
    positives = tmp_path / "cross_platform.jsonl"
    positives.write_text(
        json.dumps({"vod_id": "123", "label": 1, "window_start": 12.0, "window_end": 24.0}) + "\n",
        encoding="utf-8",
    )
    transcript = [
        RichSegment(start=0.0, end=8.0, text="quiet intro"),
        RichSegment(start=8.0, end=16.0, text="wait chat watch this"),
        RichSegment(start=16.0, end=24.0, text="no way that was insane", emotions=["exclamation"]),
        RichSegment(start=24.0, end=32.0, text="that was crazy"),
    ]

    def fake_load(_cache_dir, _vod_id):
        return tmp_path / "transcript.json", transcript

    monkeypatch.setattr(materialize_proposal_training_debug.vod_training, "_load_compatible_transcript_cache", fake_load)

    summary = materialize_proposal_training_debug.materialize_proposal_debug(
        profile="jynxzi",
        cross_platform_rows=positives,
        storage_dir=tmp_path / "materialized",
        vod_cache_root=tmp_path / "vod_cache",
        max_vods=5,
        durations=(12.0,),
        stride_sec=4.0,
        max_candidates_per_vod=20,
        refresh=False,
    )

    debug_dir = tmp_path / "materialized" / "proposal_training_123" / "debug"
    assert summary["materialized_vods"] == 1
    assert (debug_dir / "segments.jsonl").exists()
    assert (debug_dir / "candidate_proposals.jsonl").exists()
    proposal_rows = [json.loads(line) for line in (debug_dir / "candidate_proposals.jsonl").read_text().splitlines()]
    assert proposal_rows
    assert all("features" in row for row in proposal_rows)


def test_train_proposal_pipeline_runs_materialize_build_and_train(tmp_path, monkeypatch) -> None:
    calls = {}
    positive_rows = tmp_path / "cross_platform.jsonl"
    training_rows = tmp_path / "proposal_rows.jsonl"
    false_negative_rows = tmp_path / "proposal_false_negatives.jsonl"
    output = tmp_path / "proposal_scorer.json"
    summary_output = tmp_path / "proposal_training_pipeline_summary.json"
    storage_dir = tmp_path / "storage"
    repost_rows = tmp_path / "jynxzi_cross_platform_reposts.jsonl"
    diagnostic_rows = tmp_path / "jynxzi_cross_platform_fingerprint_diagnostics.jsonl"
    positive_rows.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "vod_id": "123",
                        "label": 1,
                        "window_start": 10.0,
                        "window_end": 25.0,
                        "repost_title": "Jynxzi opens fanmail box",
                        "transcript_repost_excerpt": "fanmail letter inside the box",
                    }
                ),
                json.dumps(
                    {
                        "vod_id": "456",
                        "label": 1,
                        "window_start": 20.0,
                        "window_end": 40.0,
                        "repost_title": "Jynxzi reacts to fanmail package",
                        "transcript_repost_excerpt": "package and fanmail gift in the box",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    repost_rows.write_text(
        json.dumps(
            {
                "source_url": "https://www.youtube.com/watch?v=unmatched",
                "title": "Jynxzi starts a Minecraft hardcore reset",
                "views": 123_000,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    diagnostic_rows.write_text(
        json.dumps(
            {
                "accepted": False,
                "repost_source_url": "https://www.youtube.com/watch?v=unmatched",
                "transcript_repost_excerpt": "minecraft hardcore world reset after a bad spawn",
                "repost_views": 123_000,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_materialize(**kwargs):
        calls["materialize"] = kwargs
        return {"materialized_vods": 1, "cached_debug_vods": 0}

    def fake_build(argv):
        calls["build"] = list(argv)
        training_rows.write_text(
            "\n".join(
                [
                    json.dumps({"candidate_id": "pos", "label": 1, "features": {"hook_strength": 0.9}}),
                    json.dumps({"candidate_id": "neg", "label": 0, "features": {"hook_strength": 0.1}}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return 0

    def fake_collect_debug(_storage_dir, *, positive_rows_path=None):
        calls["collect_debug"] = (str(_storage_dir), str(positive_rows_path))
        return [
            {
                "candidate_id": "missed",
                "label": 1,
                "sample_weight": 5.0,
                "label_source": "cross_platform_proposal_false_negative",
                "features": {"hook_strength": 0.7},
            }
        ]

    monkeypatch.setattr(train_proposal_pipeline.materialize_proposal_training_debug, "materialize_proposal_debug", fake_materialize)
    monkeypatch.setattr(train_proposal_pipeline.build_proposal_training_rows, "main", fake_build)
    monkeypatch.setattr(train_proposal_pipeline.train_proposal_scorer, "collect_training_rows", fake_collect_debug)

    rc = train_proposal_pipeline.main(
        [
            "--profile",
            "jynxzi",
            "--storage-dir",
            str(storage_dir),
            "--positive-rows",
            str(positive_rows),
            "--training-rows",
            str(training_rows),
            "--output",
            str(output),
            "--summary-output",
            str(summary_output),
            "--false-negative-output",
            str(false_negative_rows),
            "--min-rows",
            "2",
            "--materialize-max-vods",
            "3",
            "--date-from",
            "2026-03-01",
            "--date-to",
            "2026-05-23",
        ]
    )

    assert rc == 0
    assert calls["materialize"]["storage_dir"] == storage_dir / "_proposal_training_materialized"
    assert "--storage-dir" in calls["build"]
    assert str(storage_dir / "_proposal_training_materialized") in calls["build"]
    assert calls["collect_debug"] == (str(storage_dir), str(positive_rows))
    artifact = json.loads(output.read_text(encoding="utf-8"))
    summary = json.loads(summary_output.read_text(encoding="utf-8"))
    exported_false_negatives = [json.loads(line) for line in false_negative_rows.read_text(encoding="utf-8").splitlines()]
    assert artifact["training_summary"]["status"] == "trained"
    assert summary["false_negative_training_rows"] == 1
    assert summary["context_pattern_summary"]["patterns"] == 1
    assert summary["context_pattern_summary"]["input_sources"]["deduped_rows"] == 3
    source_paths = [
        source["path"]
        for source in summary["context_pattern_summary"]["input_sources"]["sources"]
    ]
    assert str(repost_rows) in source_paths
    assert str(diagnostic_rows) in source_paths
    assert artifact["training_summary"]["rows"] == 3
    assert (tmp_path / "jynxzi_context_patterns.json").exists()
    assert exported_false_negatives[0]["candidate_id"] == "missed"


def test_train_proposal_pipeline_filters_holdout_vods(tmp_path, monkeypatch) -> None:
    calls = {}
    positive_rows = tmp_path / "cross_platform.jsonl"
    holdout_rows = tmp_path / "holdout.jsonl"
    training_rows = tmp_path / "proposal_rows.jsonl"
    output = tmp_path / "proposal_scorer.json"
    summary_output = tmp_path / "summary.json"
    storage_dir = tmp_path / "storage"
    positive_rows.write_text(
        "\n".join(
            [
                json.dumps({"vod_id": "train", "label": 1, "window_start": 10.0, "window_end": 20.0}),
                json.dumps({"vod_id": "holdout", "label": 1, "window_start": 30.0, "window_end": 40.0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    holdout_rows.write_text(
        json.dumps(
            {
                "dataset_source": "recent_vod_holdout_policy",
                "gold_holdout_split": "holdout",
                "vod_id": "holdout",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_materialize(**kwargs):
        calls["materialize_rows"] = kwargs["cross_platform_rows"]
        return {"materialized_vods": 1}

    def fake_build(argv):
        calls["build"] = list(argv)
        training_rows.write_text(
            "\n".join(
                [
                    json.dumps({"candidate_id": "pos", "label": 1, "features": {"hook_strength": 0.9}}),
                    json.dumps({"candidate_id": "neg", "label": 0, "features": {"hook_strength": 0.1}}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return 0

    def fake_collect_debug(_storage_dir, *, positive_rows_path=None):
        calls["collect_debug"] = str(positive_rows_path)
        return [
            {"vod_id": "holdout", "candidate_id": "h", "label": 1, "features": {"hook_strength": 0.9}},
            {"vod_id": "train", "candidate_id": "t", "label": 1, "features": {"hook_strength": 0.8}},
        ]

    monkeypatch.setattr(train_proposal_pipeline.context_patterns, "write_pattern_store", lambda *args, **kwargs: {"patterns": [], "usable_rows": 0, "source_rows": 0})
    monkeypatch.setattr(train_proposal_pipeline.materialize_proposal_training_debug, "materialize_proposal_debug", fake_materialize)
    monkeypatch.setattr(train_proposal_pipeline.build_proposal_training_rows, "main", fake_build)
    monkeypatch.setattr(train_proposal_pipeline.train_proposal_scorer, "collect_training_rows", fake_collect_debug)

    rc = train_proposal_pipeline.main(
        [
            "--profile",
            "jynxzi",
            "--storage-dir",
            str(storage_dir),
            "--positive-rows",
            str(positive_rows),
            "--holdout-policy-rows",
            str(holdout_rows),
            "--training-rows",
            str(training_rows),
            "--output",
            str(output),
            "--summary-output",
            str(summary_output),
            "--min-rows",
            "2",
        ]
    )

    assert rc == 0
    filtered_rows = [json.loads(line) for line in calls["materialize_rows"].read_text(encoding="utf-8").splitlines()]
    assert [row["vod_id"] for row in filtered_rows] == ["train"]
    assert str(calls["materialize_rows"]) == calls["build"][calls["build"].index("--cross-platform-rows") + 1]
    assert calls["collect_debug"] == str(calls["materialize_rows"])
    summary = json.loads(summary_output.read_text(encoding="utf-8"))
    assert summary["holdout_filter_summary"]["positive_rows_removed_by_holdout"] == 1
    assert summary["holdout_filter_summary"]["debug_rows_after_holdout_filter"] == 1
