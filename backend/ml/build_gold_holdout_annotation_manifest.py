from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from collections import Counter

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml.row_identity import read_jsonl
from services.gold_eval_policy import GOLD_HOLDOUT_SPLIT_FIELD


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a prioritized manifest of gold-holdout VODs for more annotation.")
    parser.add_argument(
        "--gold-rows",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_gold_eval_rows.jsonl"),
    )
    parser.add_argument(
        "--diagnosis-csv",
        default=None,
    )
    parser.add_argument(
        "--out",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_gold_holdout_annotation_manifest.csv"),
    )
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    rows = build_manifest(
        gold_rows_path=pathlib.Path(args.gold_rows),
        diagnosis_csv=pathlib.Path(args.diagnosis_csv) if args.diagnosis_csv else None,
        limit=max(int(args.limit), 0),
    )
    _write_csv(pathlib.Path(args.out), rows)
    print(json.dumps({"rows": len(rows), "output": args.out}, ensure_ascii=True))
    return 0 if rows else 1


def build_manifest(*, gold_rows_path: pathlib.Path, diagnosis_csv: pathlib.Path | None, limit: int = 10) -> list[dict[str, object]]:
    gold_rows = [
        row
        for row in read_jsonl(gold_rows_path, strict=False)
        if isinstance(row, dict)
        and str(row.get(GOLD_HOLDOUT_SPLIT_FIELD) or "").strip().lower() == "holdout"
    ]
    diagnosis_errors = _diagnosis_error_counts(_read_csv(diagnosis_csv) if diagnosis_csv and diagnosis_csv.exists() else [])
    grouped: dict[str, list[dict]] = {}
    for row in gold_rows:
        vod_id = str(row.get("vod_id") or "").strip()
        if not vod_id:
            continue
        grouped.setdefault(vod_id, []).append(row)

    manifest: list[dict[str, object]] = []
    for vod_id, rows in grouped.items():
        first = rows[0]
        labels = Counter(int(row.get("label", 0) or 0) for row in rows)
        manifest.append(
            {
                "job_id": str(first.get("job_id") or f"twitch_vod_{vod_id}"),
                "vod_url": str(first.get("source_url") or f"https://www.twitch.tv/videos/{vod_id}"),
                "creator_id": str(first.get("creator_id") or first.get("streamer") or "jynxzi"),
                "platform": str(first.get("platform") or "twitch"),
                "vod_id": vod_id,
                "current_gold_rows": len(rows),
                "current_gold_positives": labels.get(1, 0),
                "current_gold_negatives": labels.get(0, 0),
                "diagnosis_error_rows": diagnosis_errors.get(vod_id, 0),
                "annotation_priority": "",
                "notes": "gold_holdout_candidate_batch",
            }
        )

    manifest.sort(
        key=lambda row: (
            -int(row["diagnosis_error_rows"]),
            int(row["current_gold_rows"]),
            int(row["current_gold_positives"]),
            str(row["vod_id"]),
        )
    )
    return manifest[:limit] if limit else manifest


def _diagnosis_error_counts(rows: list[dict]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        error_type = str(row.get("error_type") or "").strip()
        if error_type not in {"false_positive", "false_negative"}:
            continue
        vod_id = str(row.get("group") or row.get("vod_id") or "").strip()
        if vod_id:
            counts[vod_id] += 1
    return counts


def _read_csv(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
