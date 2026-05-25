import json

from ml import build_proposal_training_rows
from ml import train_proposal_scorer


def test_collect_training_rows_uses_feedback_and_quality_gate_labels(tmp_path) -> None:
    debug = tmp_path / "job-1" / "debug"
    debug.mkdir(parents=True)
    (debug / "candidate_proposals.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"candidate_id": "good", "features": {"hook_strength": 0.8, "payoff_strength": 0.9}}),
                json.dumps({"candidate_id": "bad", "features": {"hook_strength": 0.1, "payoff_strength": 0.1}}),
            ]
        ),
        encoding="utf-8",
    )
    (debug / "feedback.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"candidate_id": "good", "feedback_rating": "good", "selected": True}),
                json.dumps({"candidate_id": "bad", "feedback_rating": "bad", "selected": False}),
            ]
        ),
        encoding="utf-8",
    )

    rows = train_proposal_scorer.collect_training_rows(tmp_path)
    artifact = train_proposal_scorer.train_mean_difference_model(rows, profile="jynxzi", min_rows=2)

    assert len(rows) == 2
    assert artifact["training_summary"]["status"] == "trained"
    assert artifact["training_summary"]["positive_rows"] == 1
    assert artifact["training_summary"]["negative_rows"] == 1
    assert artifact["weights"]["payoff_strength"] > 0


def test_collect_training_rows_uses_only_judged_quality_gate_labels(tmp_path) -> None:
    debug = tmp_path / "job-judge" / "debug"
    debug.mkdir(parents=True)
    (debug / "candidate_proposals.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"candidate_id": "judged-good", "features": {"hook_strength": 0.8}}),
                json.dumps({"candidate_id": "judged-bad", "features": {"hook_strength": 0.1}}),
                json.dumps({"candidate_id": "unjudged-good", "features": {"hook_strength": 0.7}}),
            ]
        ),
        encoding="utf-8",
    )
    (debug / "serving_quality_gate.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"candidate_id": "judged-good", "accepted": True, "has_judge_result": True, "judge_score": 7.2}),
                json.dumps({"candidate_id": "judged-bad", "accepted": False, "has_judge_result": True, "judge_score": 4.0}),
                json.dumps({"candidate_id": "unjudged-good", "accepted": True, "has_judge_result": False}),
            ]
        ),
        encoding="utf-8",
    )

    rows = train_proposal_scorer.collect_training_rows(tmp_path)

    labels = {row["candidate_id"]: row["label"] for row in rows}
    weights = {row["candidate_id"]: row["sample_weight"] for row in rows}
    assert labels == {"judged-good": 1, "judged-bad": 0}
    assert weights["judged-good"] > 1.6
    assert weights["judged-bad"] > 1.3


def test_collect_training_rows_from_files_accepts_generated_rows(tmp_path) -> None:
    rows_path = tmp_path / "proposal_rows.jsonl"
    rows_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "candidate_id": "positive",
                        "label": 1,
                        "sample_weight": 2.0,
                        "label_source": "proposal_existing_feature_backfill",
                        "features": {"hook_strength": 0.8, "payoff_strength": 0.9},
                    }
                ),
                json.dumps(
                    {
                        "candidate_id": "negative",
                        "label": 0,
                        "features": {"hook_strength": 0.1, "payoff_strength": 0.1},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    rows = train_proposal_scorer.collect_training_rows_from_files([rows_path])
    artifact = train_proposal_scorer.train_logistic_regression_model(rows, profile="jynxzi", min_rows=2)

    assert len(rows) == 2
    assert artifact["training_summary"]["status"] == "trained"
    assert artifact["model_type"] == "logistic_regression_proposal_scorer"
    assert "metrics" in artifact["training_summary"]


def test_collect_training_rows_from_files_filters_gold_eval_rows(tmp_path) -> None:
    rows_path = tmp_path / "proposal_rows.jsonl"
    rows_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "candidate_id": "gold",
                        "label": 1,
                        "dataset_source": "gold_eval",
                        "features": {"hook_strength": 1.0},
                    }
                ),
                json.dumps(
                    {
                        "candidate_id": "kept",
                        "label": 0,
                        "dataset_source": "proposal_cross_platform_backfill",
                        "features": {"hook_strength": 0.1},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    rows = train_proposal_scorer.collect_training_rows_from_files([rows_path])

    assert [row["candidate_id"] for row in rows] == ["kept"]


def test_existing_feature_row_backfill_excludes_gold_eval_rows(tmp_path) -> None:
    rows_path = tmp_path / "existing_rows.jsonl"
    rows_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "candidate_id": "cross-platform-positive",
                        "vod_id": "1",
                        "label": 1,
                        "dataset_source": "cross_platform_transcript_match_train",
                        "candidate_source": "quote",
                        "window_start": 10.0,
                        "window_end": 30.0,
                        "features": {
                            "duration": 20.0,
                            "hook_strength": 0.8,
                            "payoff_score": 6.0,
                            "chat_peak": 1.0,
                        },
                    }
                ),
                json.dumps(
                    {
                        "candidate_id": "gold-row",
                        "vod_id": "1",
                        "label": 1,
                        "dataset_source": "gold_eval",
                        "candidate_source": "quote",
                        "features": {"duration": 20.0},
                    }
                ),
                json.dumps(
                    {
                        "candidate_id": "auto-negative",
                        "vod_id": "2",
                        "label": 0,
                        "dataset_source": "auto_outcome_negative_train",
                        "candidate_source": "background_negative",
                        "features": {"duration": 30.0, "hook_strength": 0.1, "payoff_score": 0.0},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    rows, summary = build_proposal_training_rows.build_rows_from_existing_feature_rows(
        [rows_path],
        profile="jynxzi",
    )

    assert summary["input_rows"] == 3
    assert summary["skipped_gold_rows"] == 1
    assert summary["positive_rows"] == 1
    assert summary["negative_rows"] == 1
    assert [row["candidate_id"] for row in rows] == ["cross-platform-positive", "auto-negative"]
    assert rows[0]["dataset_source"] == "proposal_existing_feature_backfill"
    assert rows[0]["features"]["late_payoff_score"] == 0.75
    assert rows[0]["features"]["source_prior"] == build_proposal_training_rows.proposal_scorer.source_prior_for_candidate("quote")


def test_build_rows_from_existing_feature_rows_is_not_default(tmp_path) -> None:
    output = tmp_path / "rows.jsonl"
    summary = tmp_path / "summary.json"
    existing = tmp_path / "existing.jsonl"
    existing.write_text(
        json.dumps(
            {
                "candidate_id": "legacy",
                "label": 1,
                "dataset_source": "cross_platform_transcript_match_train",
                "candidate_source": "quote",
                "features": {"duration": 20.0, "hook_strength": 0.8},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    build_proposal_training_rows.main(
        [
            "--cross-platform-rows",
            str(tmp_path / "missing_cross_platform.jsonl"),
            "--storage-dir",
            str(tmp_path / "storage"),
            "--output",
            str(output),
            "--summary-output",
            str(summary),
        ]
    )

    assert output.read_text(encoding="utf-8") == ""
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["existing_feature_backfill"]["rows"] == 0
