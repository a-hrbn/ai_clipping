from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import datetime, timezone

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml.row_identity import read_jsonl

DEFAULT_ARTIFACT_DIR = BACKEND_ROOT / "ml" / "artifacts"
DEFAULT_VOD_INVENTORY = DEFAULT_ARTIFACT_DIR / "jynxzi_twitch_vod_inventory.jsonl"
DEFAULT_OUTPUT = DEFAULT_ARTIFACT_DIR / "jynxzi_recent_vod_holdout_policy.jsonl"
DEFAULT_SUMMARY = DEFAULT_ARTIFACT_DIR / "jynxzi_recent_vod_holdout_summary.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export the most recent Twitch VODs as holdout-policy rows for training exclusion."
    )
    parser.add_argument("--profile", default="jynxzi")
    parser.add_argument("--vod-inventory", type=pathlib.Path, default=DEFAULT_VOD_INVENTORY)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=pathlib.Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args(argv)

    inventory_rows = read_jsonl(args.vod_inventory)
    selected = select_recent_vods(inventory_rows, count=max(int(args.count or 0), 0))
    policy_rows = [policy_row(row, profile=args.profile, rank=index + 1) for index, row in enumerate(selected)]
    _write_jsonl_if_changed(args.output, policy_rows)

    created_values = [str(row.get("created_at") or "") for row in policy_rows if str(row.get("created_at") or "")]
    summary = {
        "event": "recent_vod_holdout_policy_built",
        "profile": args.profile,
        "vod_inventory": str(args.vod_inventory),
        "input_rows": len(inventory_rows),
        "output": str(args.output),
        "rows": len(policy_rows),
        "requested_count": max(int(args.count or 0), 0),
        "selected_vod_ids": [str(row.get("vod_id") or "") for row in policy_rows],
        "latest_created_at": created_values[0] if created_values else "",
        "oldest_created_at": created_values[-1] if created_values else "",
        "missing_created_at_rows": sum(1 for row in inventory_rows if not str(row.get("created_at") or "").strip()),
    }
    _write_json_if_changed(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return 0


def select_recent_vods(rows: list[dict], *, count: int) -> list[dict]:
    if count <= 0:
        return []
    deduped: dict[str, dict] = {}
    for index, row in enumerate(rows):
        vod_id = str(row.get("vod_id") or row.get("id") or "").strip()
        if not vod_id:
            continue
        copied = dict(row)
        copied["_input_index"] = index
        existing = deduped.get(vod_id)
        if existing is None or _sort_key(copied) > _sort_key(existing):
            deduped[vod_id] = copied
    return sorted(deduped.values(), key=_sort_key, reverse=True)[:count]


def policy_row(row: dict, *, profile: str, rank: int) -> dict:
    vod_id = str(row.get("vod_id") or row.get("id") or "").strip()
    created_at = str(row.get("created_at") or "").strip()
    vod_url = str(row.get("vod_url") or row.get("url") or f"https://www.twitch.tv/videos/{vod_id}").strip()
    return {
        "dataset_source": "recent_vod_holdout_policy",
        "supervision_tier": "recent_vod_holdout",
        "gold_holdout_split": "holdout",
        "gold_holdout_policy": "recent_vod_fixed_v1",
        "holdout_reason": "latest_vod_serving_recall_holdout",
        "holdout_rank": int(rank),
        "profile": profile,
        "streamer": str(row.get("streamer") or row.get("broadcaster") or profile),
        "broadcaster": str(row.get("broadcaster") or row.get("streamer") or profile),
        "vod_id": vod_id,
        "vod_url": vod_url,
        "source_url": vod_url,
        "created_at": created_at,
        "vod_created_at": created_at,
        "title": str(row.get("title") or ""),
        "vod_duration_seconds": _optional_float(row.get("vod_duration_seconds") or row.get("duration_seconds")),
        "label": 0,
        "selected": False,
    }


def _sort_key(row: dict) -> tuple[float, int, int]:
    timestamp = _created_timestamp(row.get("created_at"))
    try:
        vod_id_number = int(str(row.get("vod_id") or row.get("id") or "0"))
    except ValueError:
        vod_id_number = 0
    try:
        input_index = int(row.get("_input_index") or 0)
    except (TypeError, ValueError):
        input_index = 0
    return (timestamp, vod_id_number, -input_index)


def _created_timestamp(value) -> float:
    raw = str(value or "").strip()
    if not raw:
        return 0.0
    normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _optional_float(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_jsonl_if_changed(path: pathlib.Path, rows: list[dict]) -> None:
    payload = "\n".join(json.dumps(row, ensure_ascii=True) for row in rows)
    if payload:
        payload += "\n"
    _write_text_if_changed(path, payload)


def _write_json_if_changed(path: pathlib.Path, payload: dict) -> None:
    _write_text_if_changed(path, json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


def _write_text_if_changed(path: pathlib.Path, payload: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") == payload:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(payload, encoding="utf-8")
    tmp_path.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
