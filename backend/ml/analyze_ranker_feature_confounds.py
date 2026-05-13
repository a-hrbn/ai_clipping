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
DEFAULT_PENALTY_FEATURES = ["slow_open_penalty", "weak_resolution_penalty", "context_dependency_score"]
DEFAULT_SCORE_FEATURE_SUFFIXES = ("_score", "_strength", "_coverage", "_peak", "_mean", "_density")
DEFAULT_NON_TRAINING_FEATURES = frozenset({"candidate_id", "source", "heuristic_score"})


def _default_excluded_training_features() -> frozenset[str]:
    try:
        from ml import train_ranker

        return DEFAULT_NON_TRAINING_FEATURES | train_ranker.excluded_training_feature_names()
    except Exception:
        return DEFAULT_NON_TRAINING_FEATURES


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
        "--audit-json-out",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_feature_correlation_audit.json"),
    )
    parser.add_argument(
        "--csv-out",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_ranker_feature_confounds.csv"),
    )
    parser.add_argument("--fail-threshold", type=float, default=0.85)
    args = parser.parse_args()

    report = analyze_confounds(
        dataset_path=pathlib.Path(args.dataset),
        feature=args.feature,
        compare_features=[part.strip() for part in args.compare_features.split(",") if part.strip()],
    )
    audit = analyze_feature_correlation_audit(
        dataset_path=pathlib.Path(args.dataset),
        fail_threshold=float(args.fail_threshold),
    )
    _write_json(pathlib.Path(args.json_out), report)
    _write_json(pathlib.Path(args.audit_json_out), audit)
    _write_csv(pathlib.Path(args.csv_out), list(report["correlations"]))
    print(json.dumps(report, ensure_ascii=True))
    return 1 if audit.get("failed") else 0


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


def analyze_feature_correlation_audit(
    *,
    dataset_path: pathlib.Path,
    fail_threshold: float = 0.85,
    excluded_feature_names: set[str] | frozenset[str] | None = None,
) -> dict[str, object]:
    excluded_feature_names = (
        _default_excluded_training_features()
        if excluded_feature_names is None
        else frozenset(str(name) for name in excluded_feature_names)
    )
    rows = _read_jsonl(dataset_path)
    feature_names = sorted({
        str(name)
        for row in rows
        for name, value in (row.get("features") or {}).items()
        if name not in excluded_feature_names and isinstance(value, (int, float, bool))
    })
    feature_values = {name: [_feature_value(row, name) for row in rows] for name in feature_names}
    correlations: list[dict[str, object]] = []
    for index, left in enumerate(feature_names):
        for right in feature_names[index + 1:]:
            pearson = _pearson(feature_values[left], feature_values[right])
            spearman = _spearman(feature_values[left], feature_values[right])
            max_abs = max(abs(pearson), abs(spearman))
            if max_abs <= 0:
                continue
            correlations.append(
                {
                    "left": left,
                    "right": right,
                    "pearson": round(pearson, 6),
                    "spearman": round(spearman, 6),
                    "max_abs_correlation": round(max_abs, 6),
                    "penalty_score_pair": _is_penalty_score_pair(left, right),
                }
            )
    correlations.sort(key=lambda item: (float(item["max_abs_correlation"]), str(item["left"]), str(item["right"])), reverse=True)
    failing = [
        item for item in correlations
        if bool(item["penalty_score_pair"]) and float(item["max_abs_correlation"]) > float(fail_threshold)
    ]
    return {
        "dataset": str(dataset_path),
        "rows": len(rows),
        "features": len(feature_names),
        "excluded_features": sorted(excluded_feature_names),
        "fail_threshold": float(fail_threshold),
        "failed": bool(failing),
        "failing_penalty_score_pairs": failing[:20],
        "top_correlations": correlations[:20],
    }


def _is_penalty_score_pair(left: str, right: str) -> bool:
    left_is_penalty = left in DEFAULT_PENALTY_FEATURES or left.endswith("_penalty")
    right_is_penalty = right in DEFAULT_PENALTY_FEATURES or right.endswith("_penalty")
    left_is_score = left.endswith(DEFAULT_SCORE_FEATURE_SUFFIXES)
    right_is_score = right.endswith(DEFAULT_SCORE_FEATURE_SUFFIXES)
    return (left_is_penalty and right_is_score) or (right_is_penalty and left_is_score)


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


def _spearman(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    return _pearson(_ranks(left), _ranks(right))


def _ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0 for _ in values]
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        rank = (index + end - 1) / 2.0 + 1.0
        for original_index, _ in indexed[index:end]:
            ranks[original_index] = rank
        index = end
    return ranks


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
