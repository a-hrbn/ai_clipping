from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import build_eval_dataset
from ml.row_identity import read_jsonl, write_jsonl, write_json
from services.gold_eval_policy import annotate_gold_row, validate_gold_notes


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import labeled gold-eval CSV rows and rebuild a gold eval dataset."
    )
    parser.add_argument("--labeled-csv", required=True)
    parser.add_argument("--features-jsonl", default="")
    parser.add_argument(
        "--rows-out",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "gold_eval_rows.jsonl"),
    )
    parser.add_argument(
        "--output",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "gold_eval_dataset.jsonl"),
    )
    parser.add_argument(
        "--summary-out",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "gold_eval_summary.json"),
    )
    parser.add_argument("--positive-grade-threshold", type=int, default=2)
    args = parser.parse_args()

    labeled_csv = pathlib.Path(args.labeled_csv)
    features_jsonl = (
        pathlib.Path(args.features_jsonl)
        if args.features_jsonl
        else labeled_csv.with_suffix(".features.jsonl")
    )
    try:
        payload = import_gold_eval_labels(
            labeled_csv,
            features_jsonl,
            positive_grade_threshold=int(args.positive_grade_threshold),
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    rows_out = pathlib.Path(args.rows_out)
    rows_out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(rows_out, payload["rows"])

    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, payload["eval_rows"])

    summary_out = pathlib.Path(args.summary_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    write_json(summary_out, payload["summary"])
    print(json.dumps(payload["summary"], ensure_ascii=True))
    return 0


def import_gold_eval_labels(
    labeled_csv: pathlib.Path,
    features_jsonl: pathlib.Path,
    *,
    positive_grade_threshold: int = 2,
) -> dict[str, object]:
    queue_rows = _read_csv(labeled_csv)
    if not queue_rows:
        raise ValueError("Labeled gold-eval CSV is empty.")
    feature_rows = read_jsonl(features_jsonl)
    if not feature_rows:
        raise ValueError("Gold-eval features sidecar is empty.")

    threshold = max(0, min(int(positive_grade_threshold), 3))
    features_by_queue = {
        str(row.get("queue_id") or "").strip(): row
        for row in feature_rows
        if str(row.get("queue_id") or "").strip()
    }
    features_by_candidate: dict[str, list[dict]] = {}
    for row in feature_rows:
        candidate_id = str(row.get("candidate_id") or "").strip()
        if candidate_id:
            features_by_candidate.setdefault(candidate_id, []).append(row)

    labeled_rows: list[dict] = []
    grade_counts = {str(grade): 0 for grade in range(4)}
    skipped_unlabeled = 0

    for row in queue_rows:
        grade = _parse_gold_grade(row.get("gold_grade"))
        if grade is None:
            skipped_unlabeled += 1
            continue
        sidecar = _matching_sidecar(row, features_by_queue, features_by_candidate)
        notes = str(row.get("notes") or "").strip()
        heuristic_rank_score = _as_optional_float(sidecar.get("heuristic_rank_score"))
        validate_gold_notes(
            grade,
            notes,
            heuristic_rank_score=heuristic_rank_score,
        )
        label = 1 if grade >= threshold else 0
        feature_dict = dict(sidecar.get("features") or {})
        feature_dict.setdefault("candidate_id", str(sidecar.get("candidate_id") or row.get("candidate_id") or "unknown"))
        labeled_rows.append(
            annotate_gold_row(
                {
                "queue_id": str(row.get("queue_id") or sidecar.get("queue_id") or "").strip(),
                "job_id": str(sidecar.get("job_id") or row.get("job_id") or "").strip(),
                "candidate_id": str(sidecar.get("candidate_id") or row.get("candidate_id") or "").strip(),
                "vod_id": str(sidecar.get("vod_id") or row.get("vod_id") or "").strip(),
                "creator_id": str(sidecar.get("creator_id") or row.get("creator_id") or "").strip(),
                "publish_platform": str(sidecar.get("platform") or row.get("platform") or "").strip().lower(),
                "source_url": str(sidecar.get("source_url") or row.get("source_url") or "").strip(),
                "window_start": float(sidecar.get("window_start") or 0.0),
                "window_end": float(sidecar.get("window_end") or 0.0),
                "label": label,
                "ordinal_label": grade,
                "gold_grade": grade,
                "gold_notes": notes,
                "label_source": "gold_eval",
                "label_class": "gold_positive" if label == 1 else "gold_negative",
                "sample_weight": 1.0,
                "supervision_tier": "gold_eval",
                "supervision_reason": "manual_gold_eval",
                "dataset_source": "gold_eval_manual",
                "selected": bool(label),
                "exported": False,
                "candidate_source": str(feature_dict.get("source") or "").strip(),
                "heuristic_rank_score": heuristic_rank_score,
                "features": feature_dict,
                }
            )
        )
        grade_counts[str(grade)] += 1

    if not labeled_rows:
        raise ValueError("No labeled gold-eval rows found after filtering empty grades.")

    eval_rows = build_eval_dataset.build_eval_dataset(labeled_rows)
    eval_summary = build_eval_dataset.eval_dataset_summary(eval_rows)
    summary = {
        "queue_rows": len(queue_rows),
        "feature_rows": len(feature_rows),
        "labeled_rows": len(labeled_rows),
        "skipped_unlabeled_rows": skipped_unlabeled,
        "positive_grade_threshold": threshold,
        "grade_distribution": grade_counts,
        "eval_dataset": eval_summary,
    }
    return {
        "rows": labeled_rows,
        "eval_rows": eval_rows,
        "summary": summary,
    }


def _read_csv(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_gold_grade(value) -> int | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        grade = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid gold_grade value: {raw}") from exc
    if grade < 0 or grade > 3:
        raise ValueError(f"gold_grade must stay in 0..3, got {grade}")
    return grade


def _as_optional_float(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _matching_sidecar(
    queue_row: dict,
    features_by_queue: dict[str, dict],
    features_by_candidate: dict[str, list[dict]],
) -> dict:
    queue_id = str(queue_row.get("queue_id") or "").strip()
    if queue_id and queue_id in features_by_queue:
        return features_by_queue[queue_id]

    candidate_id = str(queue_row.get("candidate_id") or "").strip()
    matches = features_by_candidate.get(candidate_id, [])
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"No gold-eval sidecar row found for candidate_id={candidate_id!r}")
    raise ValueError(
        f"Ambiguous gold-eval sidecar rows for candidate_id={candidate_id!r}; queue_id is required."
    )


if __name__ == "__main__":
    raise SystemExit(main())
