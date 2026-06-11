from __future__ import annotations

import argparse
import pathlib
import sys

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml.row_identity import read_jsonl, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Export ambiguous or review-needed rows for manual annotation.")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = read_jsonl(pathlib.Path(args.labels))
    queue = []
    for row in rows:
        if _should_queue_row(row):
            queue.append(
                {
                    "sample_id": f"{row.get('vod_id','')}:{row.get('window_start','')}:{row.get('window_end','')}",
                    "vod_id": row.get("vod_id"),
                    "vod_url": row.get("vod_url"),
                    "start_time": row.get("window_start"),
                    "end_time": row.get("window_end"),
                    "existing_label_class": row.get("label_class"),
                    "existing_supervision_tier": row.get("supervision_tier"),
                    "queue_reason": _queue_reason(row),
                    "is_good_clip": None,
                    "quality_score": None,
                    "standalone": None,
                    "hook_strength": None,
                    "notes": "",
                }
            )
    output_path = pathlib.Path(args.output)
    write_jsonl(output_path, queue)
    print(f"Wrote {len(queue)} annotation rows to {output_path}")
    return 0


def _should_queue_row(row: dict) -> bool:
    label_class = str(row.get("label_class") or "").strip().lower()
    negative_source = str(row.get("negative_source") or "").strip().lower()
    supervision_tier = str(row.get("supervision_tier") or "").strip().lower()
    return (
        label_class == "ambiguous"
        or label_class == "weak_positive"
        or negative_source in {"boundary_mined", "between_positive_clusters"}
        or supervision_tier == "pipeline_feedback"
    )


def _queue_reason(row: dict) -> str:
    label_class = str(row.get("label_class") or "").strip().lower()
    negative_source = str(row.get("negative_source") or "").strip().lower()
    supervision_tier = str(row.get("supervision_tier") or "").strip().lower()
    if label_class == "ambiguous":
        return "ambiguous_window"
    if label_class == "weak_positive":
        return "weak_positive_review"
    if negative_source == "boundary_mined":
        return "boundary_hard_negative_review"
    if negative_source == "between_positive_clusters":
        return "interstitial_hard_negative_review"
    if supervision_tier == "pipeline_feedback":
        return "weak_pipeline_label_review"
    return "manual_review"


if __name__ == "__main__":
    raise SystemExit(main())
