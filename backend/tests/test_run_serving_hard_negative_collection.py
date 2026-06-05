import json
from pathlib import Path

from ml import run_serving_hard_negative_collection as hardneg_collection


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_select_vods_for_date_range_filters_dates_and_holdouts(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.jsonl"
    holdout = tmp_path / "holdout.jsonl"
    _write_jsonl(
        inventory,
        [
            {"vod_id": "1", "created_at": "2026-03-01T01:00:00Z", "vod_url": "https://www.twitch.tv/videos/1"},
            {"vod_id": "2", "created_at": "2026-03-10T23:00:00Z", "vod_url": "https://www.twitch.tv/videos/2"},
            {"vod_id": "3", "created_at": "2026-03-11T00:00:00Z", "vod_url": "https://www.twitch.tv/videos/3"},
            {"vod_id": "4", "created_at": "", "vod_url": "https://www.twitch.tv/videos/4"},
        ],
    )
    _write_jsonl(holdout, [{"vod_id": "2"}])

    result = hardneg_collection.select_vods_for_date_range(
        inventory_path=inventory,
        holdout_policy_rows_path=holdout,
        profile="jynxzi",
        date_from="2026-03-01",
        date_to="2026-03-10",
        refresh_twitch=False,
        include_holdout_vods=False,
    )

    assert [row["vod_id"] for row in result["selected_vods"]] == ["1"]
    assert result["summary"]["skipped"]["holdout_vod_id"] == 1
    assert result["summary"]["skipped"]["outside_date_range"] == 1
    assert result["summary"]["skipped"]["missing_created_at"] == 1


def test_select_vods_merges_twitch_api_rows_with_cached_inventory(tmp_path: Path, monkeypatch) -> None:
    inventory = tmp_path / "inventory.jsonl"
    holdout = tmp_path / "holdout.jsonl"
    _write_jsonl(
        inventory,
        [
            {
                "vod_id": "10",
                "created_at": "",
                "vod_url": "https://www.twitch.tv/videos/10",
                "vod_duration_seconds": 100.0,
                "inventory_source": "vod_feature_cache",
            }
        ],
    )
    _write_jsonl(holdout, [])

    monkeypatch.setattr(
        hardneg_collection,
        "fetch_twitch_vods",
        lambda **_: [
            {
                "vod_id": "10",
                "created_at": "2026-03-03T12:00:00Z",
                "vod_url": "https://www.twitch.tv/videos/10",
                "title": "api title",
                "inventory_source": "twitch_vod_api",
            }
        ],
    )

    result = hardneg_collection.select_vods_for_date_range(
        inventory_path=inventory,
        holdout_policy_rows_path=holdout,
        profile="jynxzi",
        date_from="2026-03-01",
        date_to="2026-03-10",
        refresh_twitch=True,
        include_holdout_vods=False,
    )

    assert [row["vod_id"] for row in result["selected_vods"]] == ["10"]
    selected = result["selected_vods"][0]
    assert selected["created_at"] == "2026-03-03T12:00:00Z"
    assert selected["vod_duration_seconds"] == 100.0
    assert result["summary"]["twitch_api_rows"] == 1


def test_planned_job_id_is_stable_and_distinguishable() -> None:
    assert hardneg_collection.planned_job_id("run:1", "123") == "hardneg_collect_run_1_123"
