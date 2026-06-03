import json

from ml import run_serving_recall_holdout


def test_load_holdout_vods_uses_holdout_rows_once_in_rank_order(tmp_path) -> None:
    policy = tmp_path / "holdout.jsonl"
    policy.write_text(
        "\n".join(
            [
                json.dumps({"vod_id": "b", "vod_url": "https://www.twitch.tv/videos/b", "gold_holdout_split": "holdout", "holdout_rank": 2}),
                json.dumps({"vod_id": "a", "vod_url": "https://www.twitch.tv/videos/a", "dataset_source": "recent_vod_holdout_policy", "holdout_rank": 1}),
                json.dumps({"vod_id": "a", "vod_url": "https://www.twitch.tv/videos/a", "gold_holdout_split": "holdout", "holdout_rank": 3}),
                json.dumps({"vod_id": "train", "dataset_source": "outcome_train"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = run_serving_recall_holdout.load_holdout_vods(policy)

    assert [row["vod_id"] for row in rows] == ["a", "b"]


def test_main_dry_run_plans_holdout_jobs_and_skips_pipeline(tmp_path) -> None:
    policy = tmp_path / "holdout.jsonl"
    positives = tmp_path / "positives.jsonl"
    output = tmp_path / "eval.json"
    summary = tmp_path / "summary.json"
    policy.write_text(
        "\n".join(
            [
                json.dumps({"vod_id": "123", "vod_url": "https://www.twitch.tv/videos/123", "gold_holdout_split": "holdout", "holdout_rank": 1}),
                json.dumps({"vod_id": "456", "vod_url": "https://www.twitch.tv/videos/456", "gold_holdout_split": "holdout", "holdout_rank": 2}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    positives.write_text("", encoding="utf-8")

    rc = run_serving_recall_holdout.main(
        [
            "--holdout-policy-rows",
            str(policy),
            "--positive-rows",
            str(positives),
            "--storage-dir",
            str(tmp_path / "storage"),
            "--output",
            str(output),
            "--run-summary-output",
            str(summary),
            "--run-id",
            "test",
            "--dry-run",
        ]
    )

    assert rc == 0
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["dry_run"] is True
    assert payload["requested_vods"] == 2
    assert payload["jobs_by_status"] == {"dry_run": 2}
    assert [row["job_id"] for row in payload["jobs"]] == [
        "holdout_eval_test_123",
        "holdout_eval_test_456",
    ]
    assert not output.exists()


def test_pipeline_env_disables_oracle_pushes(tmp_path) -> None:
    positives = tmp_path / "positive_rows.jsonl"
    env = run_serving_recall_holdout.pipeline_env(positives)

    assert env["SERVING_CROSS_PLATFORM_RECALL_ENABLED"] == "false"
    assert env["SERVING_CROSS_PLATFORM_FEATURE_BOOSTS_ENABLED"] == "false"
    assert env["SERVING_CROSS_PLATFORM_POSITIVE_ROWS_PATH"] == str(positives)


def test_stage_label_matches_frontend_pipeline_steps() -> None:
    assert run_serving_recall_holdout.stage_label("downloading") == "download VOD"
    assert run_serving_recall_holdout.stage_label("transcribing") == "transcribe"
    assert run_serving_recall_holdout.stage_label("analyzing") == "candidate generation / ranker"
    assert run_serving_recall_holdout.stage_label("judging") == "clip judge"
