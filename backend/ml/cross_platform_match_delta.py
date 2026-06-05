"""Report cross-platform match deltas during ranker training."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def build_delta_payload(
    *,
    phase: str,
    before_path: Path,
    after_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    before_rows = _read_jsonl(before_path)
    after_rows = _read_jsonl(after_path)
    summary = _read_json(summary_path)

    before_keys = Counter(_row_key(row) for row in before_rows)
    after_keys = Counter(_row_key(row) for row in after_rows)
    before_set = set(before_keys)
    after_set = set(after_keys)
    added_keys = after_set - before_set
    removed_keys = before_set - after_set
    retained_keys = after_set & before_set

    added_rows = [row for row in after_rows if _row_key(row) in added_keys]
    removed_rows = [row for row in before_rows if _row_key(row) in removed_keys]

    payload: dict[str, Any] = {
        "event": "ranker_train_cross_platform_match_delta",
        "phase": phase,
        "previous_rows": len(before_rows),
        "current_rows": len(after_rows),
        "added_rows": len(added_rows),
        "removed_rows": len(removed_rows),
        "retained_rows": sum(after_keys[key] for key in retained_keys),
        "previous_unique_matches": len(before_set),
        "current_unique_matches": len(after_set),
        "added_unique_matches": len(added_keys),
        "removed_unique_matches": len(removed_keys),
        "retained_unique_matches": len(retained_keys),
        "added_vods": len({_vod_id(row) for row in added_rows if _vod_id(row)}),
        "removed_vods": len({_vod_id(row) for row in removed_rows if _vod_id(row)}),
        "current_vods": len({_vod_id(row) for row in after_rows if _vod_id(row)}),
        "summary_status": str(summary.get("status") or ""),
        "summary_matched_rows": _safe_int(summary.get("matched_rows")),
        "summary_output_rows": _safe_int(summary.get("output_rows")),
        "summary_repost_rows": _safe_int(summary.get("repost_rows")),
    }
    if added_rows:
        payload["added_match_samples"] = [_sample_row(row) for row in added_rows[:5]]
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _row_key(row: dict[str, Any]) -> str:
    return json.dumps(
        {
            "match_kind": _match_kind(row),
            "source_url": _source_url(row),
            "vod_id": _vod_id(row),
            "window_end": _clean_float(row.get("window_end") or row.get("end") or row.get("end_sec")),
            "window_start": _clean_float(row.get("window_start") or row.get("start") or row.get("start_sec")),
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sample_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_url": _source_url(row),
        "vod_id": _vod_id(row),
        "window_start": _clean_float(row.get("window_start") or row.get("start") or row.get("start_sec")),
        "window_end": _clean_float(row.get("window_end") or row.get("end") or row.get("end_sec")),
        "similarity": row.get("transcript_similarity") or row.get("confidence") or row.get("label_score"),
        "views": row.get("repost_views") or row.get("view_count") or row.get("views"),
    }


def _source_url(row: dict[str, Any]) -> str:
    return str(
        row.get("repost_source_url")
        or row.get("source_url")
        or row.get("url")
        or row.get("repost_url")
        or ""
    ).strip()


def _vod_id(row: dict[str, Any]) -> str:
    return str(row.get("vod_id") or "").strip()


def _match_kind(row: dict[str, Any]) -> str:
    return str(
        row.get("match_kind")
        or row.get("dataset_source")
        or row.get("source")
        or row.get("label_source")
        or ""
    ).strip()


def _clean_float(value: object) -> float | None:
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None


def _safe_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Log cross-platform match deltas as a JSON event.")
    parser.add_argument("--phase", required=True)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args(argv)
    payload = build_delta_payload(
        phase=args.phase,
        before_path=Path(args.before),
        after_path=Path(args.after),
        summary_path=Path(args.summary),
    )
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
