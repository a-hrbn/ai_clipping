import json
import threading
import time

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
    assert payload["fail_fast"] is True
    assert payload["progress_interval_sec"] == 3.0
    assert payload["enable_clip_judge"] is False
    assert payload["judge_policy"]["default"] == "judge_off"
    assert payload["phase_a_pipeline_env"] == {
        "DENSE_CANDIDATE_INVENTORY_SOURCE_CAP": "700",
        "ENABLE_DENSE_CANDIDATE_INVENTORY": "true",
    }
    assert payload["requested_vods"] == 2
    assert payload["jobs_by_status"] == {"dry_run": 2}
    assert [row["job_id"] for row in payload["jobs"]] == [
        "holdout_eval_test_123",
        "holdout_eval_test_456",
    ]
    assert not output.exists()


def test_pipeline_env_disables_oracle_pushes(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_DENSE_CANDIDATE_INVENTORY", "false")
    monkeypatch.setenv("DENSE_CANDIDATE_INVENTORY_SOURCE_CAP", "12")
    monkeypatch.delenv("CLIP_JUDGE_SAMPLE_COUNT", raising=False)
    monkeypatch.delenv("CLIP_JUDGE_UNCERTAIN_SAMPLE_COUNT", raising=False)
    positives = tmp_path / "positive_rows.jsonl"
    env = run_serving_recall_holdout.pipeline_env(positives)

    assert env["SERVING_CROSS_PLATFORM_RECALL_ENABLED"] == "false"
    assert env["SERVING_CROSS_PLATFORM_FEATURE_BOOSTS_ENABLED"] == "false"
    assert env["SERVING_CROSS_PLATFORM_POSITIVE_ROWS_PATH"] == str(positives)
    assert env["TWITCH_DOWNLOAD_CACHE_TTL_SEC"] == "-1"
    assert env["ENABLE_CLIP_JUDGE"] == "false"
    assert env["ENABLE_DENSE_CANDIDATE_INVENTORY"] == "true"
    assert env["DENSE_CANDIDATE_INVENTORY_SOURCE_CAP"] == "700"
    assert sorted(key for key in env if key.startswith("DENSE_CANDIDATE_INVENTORY")) == [
        "DENSE_CANDIDATE_INVENTORY_SOURCE_CAP",
    ]
    assert "CLIP_JUDGE_SAMPLE_COUNT" not in env
    assert "CLIP_JUDGE_UNCERTAIN_SAMPLE_COUNT" not in env


def test_pipeline_env_can_enable_judge_reference_mode(tmp_path) -> None:
    positives = tmp_path / "positive_rows.jsonl"
    env = run_serving_recall_holdout.pipeline_env(
        positives,
        enable_clip_judge=True,
        judge_sample_count=3,
    )

    assert env["ENABLE_CLIP_JUDGE"] == "true"
    assert env["CLIP_JUDGE_SAMPLE_COUNT"] == "3"
    assert env["CLIP_JUDGE_UNCERTAIN_SAMPLE_COUNT"] == "3"


def test_stage_label_matches_frontend_pipeline_steps() -> None:
    assert run_serving_recall_holdout.stage_label("downloading") == "download VOD"
    assert run_serving_recall_holdout.stage_label("transcribing") == "transcribe"
    assert run_serving_recall_holdout.stage_label("analyzing") == "candidate generation / ranker"
    assert run_serving_recall_holdout.stage_label("judging") == "clip judge"


def test_main_runs_holdout_jobs_with_two_workers_and_keeps_summary_order(tmp_path, monkeypatch) -> None:
    policy = tmp_path / "holdout.jsonl"
    positives = tmp_path / "positives.jsonl"
    output = tmp_path / "eval.json"
    summary = tmp_path / "summary.json"
    policy.write_text(
        "\n".join(
            [
                json.dumps({"vod_id": "111", "vod_url": "https://www.twitch.tv/videos/111", "gold_holdout_split": "holdout", "holdout_rank": 1}),
                json.dumps({"vod_id": "222", "vod_url": "https://www.twitch.tv/videos/222", "gold_holdout_split": "holdout", "holdout_rank": 2}),
                json.dumps({"vod_id": "333", "vod_url": "https://www.twitch.tv/videos/333", "gold_holdout_split": "holdout", "holdout_rank": 3}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    positives.write_text("", encoding="utf-8")

    lock = threading.Lock()
    active = 0
    max_active = 0

    def fake_run_pipeline_job(**kwargs):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        vod_id = kwargs["url"].rsplit("/", 1)[-1]
        return {
            "vod_id": vod_id,
            "url": kwargs["url"],
            "job_id": kwargs["job_id"],
            "status": "done",
            "debug_dir": str(tmp_path / "storage" / kwargs["job_id"] / "debug"),
        }

    monkeypatch.setattr(run_serving_recall_holdout, "run_pipeline_job", fake_run_pipeline_job)
    monkeypatch.setattr(
        run_serving_recall_holdout.evaluate_serving_recall_holdout,
        "evaluate_holdout",
        lambda **_: {"aggregate": {"positive_windows": 0}, "jobs_evaluated": 0, "jobs_skipped": {}},
    )

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
            "parallel",
            "--workers",
            "2",
        ]
    )

    assert rc == 0
    assert max_active == 2
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["workers"] == 2
    assert payload["jobs_by_status"] == {"done": 3}
    assert [row["job_id"] for row in payload["jobs"]] == [
        "holdout_eval_parallel_111",
        "holdout_eval_parallel_222",
        "holdout_eval_parallel_333",
    ]
