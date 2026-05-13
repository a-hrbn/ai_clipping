from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import date

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml.collect_twitch_clip_labels import _label_row_key
from ml.row_identity import read_jsonl, write_json, write_jsonl


DEFAULT_LABELS_PATH = BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_twitch_clip_labels.jsonl"
DEFAULT_SUMMARY_PATH = BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_twitch_clip_label_prune_summary.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prune cumulative Twitch proxy labels to the requested training date range."
    )
    parser.add_argument("--labels", default=str(DEFAULT_LABELS_PATH))
    parser.add_argument("--output", default="")
    parser.add_argument("--summary-out", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--date-from", required=True)
    parser.add_argument("--date-to", required=True)
    parser.add_argument("--preserve-source", action="append", default=[])
    args = parser.parse_args()

    labels_path = pathlib.Path(args.labels)
    output_path = pathlib.Path(args.output) if args.output else labels_path
    payload = prune_labels(
        read_jsonl(labels_path),
        date_from=date.fromisoformat(args.date_from),
        date_to=date.fromisoformat(args.date_to),
        preserve_sources={str(item).strip() for item in args.preserve_source if str(item).strip()},
    )
    write_jsonl(output_path, payload["rows"])
    write_json(pathlib.Path(args.summary_out), payload["summary"])
    print(json.dumps(payload["summary"], ensure_ascii=True))
    return 0


def prune_labels(
    labels: list[dict],
    *,
    date_from: date,
    date_to: date,
    preserve_sources: set[str],
) -> dict[str, object]:
    kept_by_key: dict[tuple, dict] = {}
    kept_in_range = 0
    kept_preserved = 0
    removed = 0
    for row in labels:
        in_range = _row_in_date_range(row, date_from=date_from, date_to=date_to)
        preserved = str(row.get("manual_gold_review_source") or "").strip() in preserve_sources
        if not in_range and not preserved:
            removed += 1
            continue
        if in_range:
            kept_in_range += 1
        elif preserved:
            kept_preserved += 1
        kept_by_key[_label_row_key(row)] = dict(row)

    rows = sorted(
        kept_by_key.values(),
        key=lambda row: (
            str(row.get("vod_id") or ""),
            float(row.get("window_start") or 0.0),
            str(row.get("label_class") or ""),
            str(row.get("negative_source") or ""),
        ),
    )
    summary = {
        "input_rows": len(labels),
        "output_rows": len(rows),
        "removed_rows": removed,
        "kept_in_date_range_rows": kept_in_range,
        "kept_preserved_source_rows": kept_preserved,
        "date_from": date_from.isoformat(),
        "date_to": date_to.isoformat(),
        "preserve_sources": sorted(preserve_sources),
    }
    return {"rows": rows, "summary": summary}


def _row_in_date_range(row: dict, *, date_from: date, date_to: date) -> bool:
    row_date = _row_date(row)
    return row_date is not None and date_from <= row_date <= date_to


def _row_date(row: dict) -> date | None:
    raw = str(row.get("created_at") or row.get("cluster_created_at") or "").strip()
    if len(raw) < 10:
        return None
    try:
        return date.fromisoformat(raw[:10])
    except ValueError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
