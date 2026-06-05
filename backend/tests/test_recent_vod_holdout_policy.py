import json

from ml import build_recent_vod_holdout, build_training_holdout_policy


def test_recent_vod_holdout_selects_latest_unique_vods(tmp_path) -> None:
    inventory = tmp_path / "inventory.jsonl"
    rows = []
    for index in range(12):
        rows.append(
            {
                "vod_id": str(1000 + index),
                "created_at": f"2026-05-{index + 1:02d}T12:00:00+00:00",
                "title": f"vod {index}",
            }
        )
    rows.append({"vod_id": "1001", "created_at": "2026-06-01T12:00:00+00:00", "title": "duplicate newer"})
    inventory.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    output = tmp_path / "recent_holdout.jsonl"
    summary = tmp_path / "recent_holdout.json"

    rc = build_recent_vod_holdout.main(
        [
            "--profile",
            "jynxzi",
            "--vod-inventory",
            str(inventory),
            "--output",
            str(output),
            "--summary-output",
            str(summary),
            "--count",
            "10",
        ]
    )

    assert rc == 0
    exported = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert len(exported) == 10
    assert exported[0]["vod_id"] == "1001"
    assert exported[0]["gold_holdout_split"] == "holdout"
    assert exported[0]["dataset_source"] == "recent_vod_holdout_policy"
    assert payload["selected_vod_ids"][0] == "1001"
    assert "1000" not in payload["selected_vod_ids"]


def test_recent_vod_holdout_preserves_existing_policy_by_default(tmp_path) -> None:
    inventory = tmp_path / "inventory.jsonl"
    inventory.write_text(
        "\n".join(
            json.dumps(
                {
                    "vod_id": str(2000 + index),
                    "created_at": f"2026-06-{index + 1:02d}T12:00:00+00:00",
                    "title": f"new vod {index}",
                }
            )
            for index in range(3)
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "recent_holdout.jsonl"
    summary = tmp_path / "recent_holdout.json"
    existing = {
        "dataset_source": "recent_vod_holdout_policy",
        "gold_holdout_split": "holdout",
        "vod_id": "1234",
        "created_at": "2026-05-01T12:00:00+00:00",
    }
    output.write_text(json.dumps(existing) + "\n", encoding="utf-8")

    rc = build_recent_vod_holdout.main(
        [
            "--vod-inventory",
            str(inventory),
            "--output",
            str(output),
            "--summary-output",
            str(summary),
            "--count",
            "2",
        ]
    )

    assert rc == 0
    exported = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert [row["vod_id"] for row in exported] == ["1234"]
    assert payload["event"] == "recent_vod_holdout_policy_preserved"
    assert payload["preserved_existing_output"] is True
    assert payload["selected_vod_ids"] == ["1234"]


def test_recent_vod_holdout_refresh_replaces_existing_policy(tmp_path) -> None:
    inventory = tmp_path / "inventory.jsonl"
    rows = [
        {"vod_id": "2001", "created_at": "2026-06-01T12:00:00+00:00", "title": "newer"},
        {"vod_id": "2000", "created_at": "2026-05-31T12:00:00+00:00", "title": "older"},
    ]
    inventory.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    output = tmp_path / "recent_holdout.jsonl"
    summary = tmp_path / "recent_holdout.json"
    output.write_text(json.dumps({"vod_id": "1234", "gold_holdout_split": "holdout"}) + "\n", encoding="utf-8")

    rc = build_recent_vod_holdout.main(
        [
            "--vod-inventory",
            str(inventory),
            "--output",
            str(output),
            "--summary-output",
            str(summary),
            "--count",
            "1",
            "--refresh",
        ]
    )

    assert rc == 0
    exported = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert [row["vod_id"] for row in exported] == ["2001"]
    assert payload["event"] == "recent_vod_holdout_policy_built"
    assert payload["preserved_existing_output"] is False
    assert payload["refresh"] is True


def test_training_holdout_policy_merges_vod_and_window_rows(tmp_path) -> None:
    recent = tmp_path / "recent.jsonl"
    outcome = tmp_path / "outcome.jsonl"
    recent.write_text(
        json.dumps(
            {
                "dataset_source": "recent_vod_holdout_policy",
                "gold_holdout_split": "holdout",
                "vod_id": "123",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    outcome.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "dataset_source": "cross_platform_outcome_holdout",
                        "gold_holdout_split": "holdout",
                        "vod_id": "456",
                        "window_start": 10.0,
                        "window_end": 20.0,
                    }
                ),
                json.dumps(
                    {
                        "dataset_source": "cross_platform_outcome_holdout",
                        "gold_holdout_split": "holdout",
                        "vod_id": "456",
                        "window_start": 10.0,
                        "window_end": 20.0,
                    }
                ),
                json.dumps({"dataset_source": "outcome_train", "vod_id": "789"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "merged.jsonl"
    summary = tmp_path / "merged.json"

    rc = build_training_holdout_policy.main(
        [
            "--input",
            str(outcome),
            "--input",
            str(recent),
            "--output",
            str(output),
            "--summary-output",
            str(summary),
        ]
    )

    assert rc == 0
    exported = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert len(exported) == 2
    assert payload["unique_holdout_vods"] == 2
    assert payload["holdout_vod_ids"] == ["123", "456"]
