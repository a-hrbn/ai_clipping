from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


DEFAULT_COMPARE_FEATURES = [
    "chapter_local_density",
    "duration",
    "payoff_score",
    "setup_depth_score",
    "profile_fit_score",
    "context_dependency_score",
    "hook_delay_sec",
    "hook_strength",
    "chat_activity_coverage",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze suspicious ranker feature correlations.")
    parser.add_argument(
        "--dataset",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_vod_training_rows.jsonl"),
    )
    parser.add_argument("--feature", default="slow_open_penalty")
    parser.add_argument(
        "--compare-features",
        default=",".join(DEFAULT_COMPARE_FEATURES),
        help="Comma-separated feature names to correlate against --feature.",
    )
    parser.add_argument(
        "--json-out",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_ranker_feature_confounds.json"),
    )
    parser.add_argument(
        "--csv-out",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_ranker_feature_confounds.csv"),
    )
    args = parser.parse_args()

    report = analyze_confounds(
        dataset_path=pathlib.Path(args.dataset),
        feature=args.feature,
        compare_features=[part.strip() for part in args.compare_features.split(",") if part.strip()],
    )
    _write_json(pathlib.Path(args.json_out), report)
    _write_csv(pathlib.Path(args.csv_out), list(report["correlations"]))
    print(json.dumps(report, ensure_ascii=True))
    return 0


def analyze_confounds(*, dataset_path: pathlib.Path, feature: str, compare_features: list[str]) -> dict[str, object]:
    rows = _read_jsonl(dataset_path)
    feature_values = [_feature_value(row, feature) for row in rows]
    correlations = [
        {
            "feature": other,
            "correlation": round(_pearson(feature_values, [_feature_value(row, other) for row in rows]), 6),
        }
        for other in compare_features
        if other != feature
    ]
    correlations.sort(key=lambda item: abs(float(item["correlation"])), reverse=True)
    label_summary = _label_summary(rows, feature)
    positive_mean = float((label_summary.get("1") or {}).get("mean", 0.0))
    negative_mean = float((label_summary.get("0") or {}).get("mean", 0.0))
    recommendation = "keep_feature"
    reason = "feature_direction_matches_label_means"
    if positive_mean >= negative_mean:
        recommendation = "exclude_from_supervised_ranker_training"
        reason = "penalty_feature_is_not_lower_for_positive_rows"

    return {
        "dataset": str(dataset_path),
        "rows": len(rows),
        "feature": feature,
        "label_summary": label_summary,
        "correlations": correlations,
        "recommendation": recommendation,
        "reason": reason,
    }


def _label_summary(rows: list[dict], feature: str) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for label in (0, 1):
        values = [_feature_value(row, feature) for row in rows if int(row.get("label", -1)) == label]
        if not values:
            continue
        ordered = sorted(values)
        summary[str(label)] = {
            "rows": len(values),
            "mean": round(sum(values) / len(values), 6),
            "median": round(_percentile(ordered, 0.5), 6),
            "p25": round(_percentile(ordered, 0.25), 6),
            "p75": round(_percentile(ordered, 0.75), 6),
        }
    return summary


def _feature_value(row: dict, feature: str) -> float:
    value = (row.get("features") or {}).get(feature, 0.0)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def _pearson(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=False))
    left_var = sum((x - left_mean) ** 2 for x in left)
    right_var = sum((y - right_mean) ** 2 for y in right)
    if left_var <= 0 or right_var <= 0:
        return 0.0
    return numerator / math.sqrt(left_var * right_var)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    index = max(0, min(len(values) - 1, int(math.floor((len(values) - 1) * q))))
    return values[index]


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _write_json(path: pathlib.Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


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
