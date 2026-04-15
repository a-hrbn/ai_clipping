from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate boundary-near-miss rows in a training dataset export.")
    parser.add_argument(
        "--dataset",
        default=str(BACKEND_ROOT / "tmp" / "storage" / "feedback" / "training_rows.jsonl"),
    )
    parser.add_argument("--top-kinds", type=int, default=8)
    args = parser.parse_args()

    rows = _read_jsonl(pathlib.Path(args.dataset))
    if not rows:
        print("No dataset rows found.")
        return 1

    boundary_rows = [row for row in rows if _has_boundary_reference(row)]
    summary = {
        "rows": len(rows),
        "boundary_rows": len(boundary_rows),
        "boundary_reference_rows": sum(1 for row in rows if int(row.get("label", 0)) == 1 and row.get("window_origin") != "boundary_perturbed"),
        "boundary_metrics": _boundary_metrics(boundary_rows),
        "by_perturbation_kind": _bucket_by_kind(boundary_rows, args.top_kinds),
    }
    print(json.dumps(summary, ensure_ascii=True))
    return 0


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _has_boundary_reference(row: dict) -> bool:
    return row.get("boundary_reference_start") not in (None, "") and row.get("boundary_reference_end") not in (None, "")


def _boundary_metrics(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {
            "mean_abs_start_error_sec": 0.0,
            "mean_abs_end_error_sec": 0.0,
            "mean_abs_total_error_sec": 0.0,
            "median_abs_total_error_sec": 0.0,
            "within_1s_total_rate": 0.0,
            "within_2s_total_rate": 0.0,
            "severe_error_rate": 0.0,
        }

    start_errors = [float(row.get("boundary_abs_start_error_sec") or 0.0) for row in rows]
    end_errors = [float(row.get("boundary_abs_end_error_sec") or 0.0) for row in rows]
    total_errors = [float(row.get("boundary_abs_total_error_sec") or 0.0) for row in rows]
    return {
        "mean_abs_start_error_sec": round(sum(start_errors) / len(start_errors), 3),
        "mean_abs_end_error_sec": round(sum(end_errors) / len(end_errors), 3),
        "mean_abs_total_error_sec": round(sum(total_errors) / len(total_errors), 3),
        "median_abs_total_error_sec": round(statistics.median(total_errors), 3),
        "within_1s_total_rate": round(sum(1 for value in total_errors if value <= 1.0) / len(total_errors), 3),
        "within_2s_total_rate": round(sum(1 for value in total_errors if value <= 2.0) / len(total_errors), 3),
        "severe_error_rate": round(sum(1 for value in total_errors if value >= 3.0) / len(total_errors), 3),
    }


def _bucket_by_kind(rows: list[dict], top_kinds: int) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        kind = str(row.get("boundary_perturbation_kind") or "unknown")
        grouped.setdefault(kind, []).append(row)

    summaries: list[dict] = []
    for kind, subset in grouped.items():
        summaries.append(
            {
                "kind": kind,
                "rows": len(subset),
                "metrics": _boundary_metrics(subset),
            }
        )
    summaries.sort(key=lambda item: (item["rows"], item["kind"]), reverse=True)
    return summaries[: max(int(top_kinds), 0)]


if __name__ == "__main__":
    raise SystemExit(main())
