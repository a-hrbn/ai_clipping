from __future__ import annotations

import argparse
import json
import pathlib
import sys

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml.row_identity import read_jsonl, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert locally exported retention metrics into outcome feedback rows.")
    parser.add_argument("--metrics", required=True, help="JSONL export with retention_3s/retention_50pct/completion/views fields.")
    parser.add_argument("--output", default=str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_outcome_feedback_rows.jsonl"))
    args = parser.parse_args()
    rows = import_retention_rows(pathlib.Path(args.metrics))
    output = pathlib.Path(args.output)
    write_jsonl(output, rows)
    print(json.dumps({"metrics": args.metrics, "output": str(output), "rows": len(rows)}, ensure_ascii=True))
    return 0


def import_retention_rows(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for row in read_jsonl(path, strict=False):
        retention_3s = _safe_float(row.get("retention_3s"))
        retention_50pct = _safe_float(row.get("retention_50pct"))
        completion = _safe_float(row.get("retention_complete") or row.get("completion_rate"))
        views = _safe_float(row.get("views") or row.get("view_count"))
        baseline = _safe_float(row.get("baseline_views") or row.get("baseline_view_rate"))
        label = None
        label_class = ""
        if retention_3s and retention_3s < 0.5:
            label = 0
            label_class = "auto_outcome_hard_negative"
        elif retention_50pct and retention_50pct < 0.4:
            label = 0
            label_class = "auto_outcome_hard_negative"
        elif completion > 0.6 and (baseline <= 0 or views > baseline):
            label = 1
            label_class = "outcome_positive"
        if label is None:
            continue
        copied = dict(row)
        copied.update(
            {
                "label": label,
                "selected": label == 1,
                "label_class": label_class,
                "dataset_source": "outcome_feedback",
                "label_source": "production_retention",
                "supervision_tier": "outcome_feedback",
                "sample_weight": 1.5 if int(row.get("age_days", 0) or 0) >= 14 else 1.35,
            }
        )
        rows.append(copied)
    return rows


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    raise SystemExit(main())
