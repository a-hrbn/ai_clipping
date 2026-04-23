from __future__ import annotations

import argparse
import json
import pathlib
import sys

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.ranker import Ranker
from services.timeline_types import CandidateFeatures
from ml import train_ranker


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the current ranker artifact on exported training rows.")
    parser.add_argument(
        "--dataset",
        default=str(BACKEND_ROOT / "tmp" / "storage" / "feedback" / "training_rows.jsonl"),
    )
    parser.add_argument("--profile", default="gaming")
    parser.add_argument("--artifact-dir", default=str(BACKEND_ROOT / "ml" / "artifacts"))
    parser.add_argument("--top-groups", type=int, default=5)
    parser.add_argument("--report-out", default="")
    args = parser.parse_args()

    dataset_path = pathlib.Path(args.dataset)
    rows = _read_jsonl(dataset_path)
    if not rows:
        report = _empty_report(args.profile)
        payload = json.dumps(report, ensure_ascii=True)
        if args.report_out:
            report_path = pathlib.Path(args.report_out)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(payload, encoding="utf-8")
        print(payload)
        return 0

    baseline_ranker = Ranker(artifact_dir=pathlib.Path(args.artifact_dir), enable_trained_model=False)
    trained_ranker = Ranker(artifact_dir=pathlib.Path(args.artifact_dir), enable_trained_model=True)
    artifact_inspection = trained_ranker.inspect_artifact(args.profile)
    raw_artifact = artifact_inspection.get("artifact") if isinstance(artifact_inspection.get("artifact"), dict) else None
    artifact = raw_artifact if artifact_inspection.get("status") == "usable" else None
    features = [CandidateFeatures.from_dict(row["features"]) for row in rows]
    baseline_scores = {candidate_id: score for candidate_id, score in baseline_ranker.score(features, args.profile)}
    trained_scores = {candidate_id: score for candidate_id, score in trained_ranker.score(features, args.profile)}

    split = train_ranker._split_rows(rows)
    groups = train_ranker._ranking_group_labels(rows, split.get("unit", ""))
    labels = train_ranker.np.array([int(row["label"]) for row in rows], dtype=int)
    baseline_score_array = train_ranker.np.array([float(baseline_scores.get(row["candidate_id"], 0.0)) for row in rows], dtype=float)
    trained_score_array = train_ranker.np.array([float(trained_scores.get(row["candidate_id"], 0.0)) for row in rows], dtype=float)
    trained_threshold = train_ranker._artifact_serving_threshold(artifact)
    baseline_metrics = train_ranker._evaluate_scores(labels, baseline_score_array, score_kind="margin", groups=groups)
    trained_metrics = train_ranker._evaluate_scores(
        labels,
        trained_score_array,
        score_kind="margin",
        groups=groups,
        threshold=trained_threshold,
    )
    report = {
        "artifact": {
            "available": artifact is not None,
            "status": artifact_inspection.get("status", "missing"),
            "reason": artifact_inspection.get("reason", "artifact_file_missing"),
            "requested_profile": artifact_inspection.get("requested_profile", args.profile),
            "resolved_profile": artifact_inspection.get("resolved_profile", ""),
            "source": artifact_inspection.get("source", ""),
            "selected_model": raw_artifact.get("selected_model") if raw_artifact else "",
            "model_type": raw_artifact.get("model_type") if raw_artifact else "",
            "serving_threshold": raw_artifact.get("serving_threshold") if raw_artifact else None,
            "serving_threshold_calibration": raw_artifact.get("serving_threshold_calibration", {}) if raw_artifact else {},
            "training_stats": raw_artifact.get("training_stats", {}) if raw_artifact else {},
            "validation": raw_artifact.get("validation", {}) if raw_artifact else {},
        },
        "metrics": {
            "baseline_heuristic": baseline_metrics,
            "trained_runtime": trained_metrics,
            "delta": _metric_delta(trained_metrics, baseline_metrics),
        },
        "trained_runtime_uses_artifact": artifact is not None,
        "rows": len(rows),
        "positive_rows": sum(1 for row in rows if int(row["label"]) == 1),
        "negative_rows": sum(1 for row in rows if int(row["label"]) == 0),
        "split_strategy": split["strategy"],
        "split_unit": split.get("unit", ""),
        "leakage_risk": split.get("leakage_risk", "unknown"),
        "by_label_class": _bucket_metrics(rows, baseline_scores, trained_scores, "label_class", trained_threshold),
        "by_supervision_tier": _bucket_metrics(rows, baseline_scores, trained_scores, "supervision_tier", trained_threshold),
        "by_platform": _bucket_metrics(rows, baseline_scores, trained_scores, _bucket_key(rows, "platform_bucket", "publish_platform"), trained_threshold),
        "by_gold_holdout_split": _bucket_metrics(rows, baseline_scores, trained_scores, "gold_holdout_split", trained_threshold),
        "by_content_bucket": _bucket_metrics(rows, baseline_scores, trained_scores, "content_bucket", trained_threshold),
        "by_source_bucket": _bucket_metrics(rows, baseline_scores, trained_scores, _bucket_key(rows, "source_bucket", "candidate_source"), trained_threshold),
        "by_variant": _bucket_metrics(rows, baseline_scores, trained_scores, _bucket_key(rows, "variant_bucket", "final_variant_key"), trained_threshold),
        "top_groups": _group_metrics(rows, baseline_scores, trained_scores, split.get("unit", ""), args.top_groups, trained_threshold),
    }
    payload = json.dumps(report, ensure_ascii=True)
    if args.report_out:
        report_path = pathlib.Path(args.report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(payload, encoding="utf-8")
    print(payload)
    return 0


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            row = json.loads(line)
            if row.get("label") in {0, 1}:
                rows.append(row)
    return rows


def _empty_report(profile: str) -> dict[str, object]:
    return {
        "artifact": {
            "available": False,
            "status": "skipped",
            "reason": "no_dataset_rows",
            "requested_profile": profile,
            "resolved_profile": "",
            "source": "",
            "selected_model": "",
            "model_type": "",
            "serving_threshold": None,
            "serving_threshold_calibration": {},
            "training_stats": {},
            "validation": {},
        },
        "metrics": {},
        "trained_runtime_uses_artifact": False,
        "rows": 0,
        "positive_rows": 0,
        "negative_rows": 0,
        "split_strategy": "",
        "split_unit": "",
        "leakage_risk": "unknown",
        "by_label_class": {},
        "by_supervision_tier": {},
        "by_platform": {},
        "by_gold_holdout_split": {},
        "by_content_bucket": {},
        "by_source_bucket": {},
        "by_variant": {},
        "top_groups": [],
    }


def _bucket_key(rows: list[dict], preferred: str, fallback: str) -> str:
    if any(str(row.get(preferred) or "").strip() for row in rows):
        return preferred
    return fallback


def _bucket_metrics(
    rows: list[dict],
    baseline_scores: dict[str, float],
    trained_scores: dict[str, float],
    key: str,
    trained_threshold: float | None = None,
) -> dict[str, dict]:
    bucketed: dict[str, dict] = {}
    values = sorted({str(row.get(key) or "").strip() for row in rows if str(row.get(key) or "").strip()})
    for value in values:
        subset = [row for row in rows if str(row.get(key) or "").strip() == value]
        labels = train_ranker.np.array([int(row["label"]) for row in subset], dtype=int)
        subset_baseline = train_ranker.np.array([float(baseline_scores.get(row["candidate_id"], 0.0)) for row in subset], dtype=float)
        subset_trained = train_ranker.np.array([float(trained_scores.get(row["candidate_id"], 0.0)) for row in subset], dtype=float)
        groups = train_ranker._ranking_group_labels(subset, key if key in {"creator_id", "vod_id", "source_url"} else "")
        baseline_metrics = train_ranker._evaluate_scores(labels, subset_baseline, score_kind="margin", groups=groups)
        trained_metrics = train_ranker._evaluate_scores(
            labels,
            subset_trained,
            score_kind="margin",
            groups=groups,
            threshold=trained_threshold,
        )
        bucketed[value] = {
            "rows": len(subset),
            "baseline_heuristic": baseline_metrics,
            "trained_runtime": trained_metrics,
            "delta": _metric_delta(trained_metrics, baseline_metrics),
        }
    return bucketed


def _group_metrics(
    rows: list[dict],
    baseline_scores: dict[str, float],
    trained_scores: dict[str, float],
    split_unit: str,
    top_k: int,
    trained_threshold: float | None = None,
) -> list[dict]:
    unit = split_unit or "creator_id"
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        group = train_ranker._ranking_group_key(row, unit)
        grouped.setdefault(group, []).append(row)

    summaries: list[dict] = []
    for group, subset in grouped.items():
        labels = train_ranker.np.array([int(row["label"]) for row in subset], dtype=int)
        subset_baseline = train_ranker.np.array([float(baseline_scores.get(row["candidate_id"], 0.0)) for row in subset], dtype=float)
        subset_trained = train_ranker.np.array([float(trained_scores.get(row["candidate_id"], 0.0)) for row in subset], dtype=float)
        groups = train_ranker._ranking_group_labels(subset, unit)
        baseline_metrics = train_ranker._evaluate_scores(labels, subset_baseline, score_kind="margin", groups=groups)
        trained_metrics = train_ranker._evaluate_scores(
            labels,
            subset_trained,
            score_kind="margin",
            groups=groups,
            threshold=trained_threshold,
        )
        summaries.append(
            {
                "group": group,
                "rows": len(subset),
                "baseline_heuristic": baseline_metrics,
                "trained_runtime": trained_metrics,
                "delta": _metric_delta(trained_metrics, baseline_metrics),
            }
        )
    summaries.sort(key=lambda item: (item["rows"], item["group"]), reverse=True)
    return summaries[: max(int(top_k), 0)]


def _metric_delta(current: dict, baseline: dict) -> dict[str, float]:
    tracked = (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "precision_at_5",
        "recall_at_5",
        "ndcg_at_5",
        "ndcg_at_10",
        "mrr",
        "ranking_metric_value",
    )
    delta: dict[str, float] = {}
    for key in tracked:
        delta[key] = round(float(current.get(key, 0.0)) - float(baseline.get(key, 0.0)), 3)
    return delta


if __name__ == "__main__":
    raise SystemExit(main())
