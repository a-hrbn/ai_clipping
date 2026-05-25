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
    output = tmp_path / "proposal_scorer.json"
    storage_dir = tmp_path / "storage"

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
        return []

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
            "--min-rows",
            "2",
            "--materialize-max-vods",
            "3",
        ]
    )

    assert rc == 0
    assert calls["materialize"]["storage_dir"] == storage_dir / "_proposal_training_materialized"
    assert "--storage-dir" in calls["build"]
    assert str(storage_dir / "_proposal_training_materialized") in calls["build"]
    assert calls["collect_debug"] == (str(storage_dir), str(positive_rows))
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["training_summary"]["status"] == "trained"
