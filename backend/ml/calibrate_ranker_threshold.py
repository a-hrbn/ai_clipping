from __future__ import annotations

import argparse
import json
import pathlib
import sys

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import train_ranker
from services.ranker import Ranker
from services.timeline_types import CandidateFeatures


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate a ranker artifact serving threshold on labeled eval rows.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--profile", default="gaming")
    parser.add_argument("--artifact-dir", default=str(BACKEND_ROOT / "ml" / "artifacts"))
    parser.add_argument("--split-field", default="")
    parser.add_argument("--split-value", default="")
    parser.add_argument("--source", default="eval_calibration")
    parser.add_argument(
        "--selection-mode",
        choices=("balanced_filter", "recall_precision"),
        default="balanced_filter",
        help="Use recall_precision to require min recall on all reference datasets, then maximize precision.",
    )
    parser.add_argument(
        "--recall-reference-dataset",
        action="append",
        default=[],
        help="Additional JSONL dataset that must satisfy --min-recall in recall_precision mode.",
    )
    parser.add_argument("--min-recall", type=float, default=train_ranker._SERVING_THRESHOLD_MIN_RECALL)
    parser.add_argument("--max-selected-rate", type=float, default=train_ranker._SERVING_THRESHOLD_MAX_SELECTED_RATE)
    parser.add_argument("--target-precision", type=float, default=train_ranker._SERVING_THRESHOLD_TARGET_PRECISION)
    parser.add_argument("--min-rows", type=int, default=10)
    parser.add_argument("--min-positive-rows", type=int, default=3)
    parser.add_argument("--report-out", default="")
    parser.add_argument("--write-artifact", action="store_true")
    args = parser.parse_args()

    report = calibrate_threshold(
        dataset_path=pathlib.Path(args.dataset),
        profile=args.profile,
        artifact_dir=pathlib.Path(args.artifact_dir),
        split_field=args.split_field,
        split_value=args.split_value,
        source=args.source,
        selection_mode=args.selection_mode,
        recall_reference_datasets=[pathlib.Path(path) for path in args.recall_reference_dataset],
        min_recall=args.min_recall,
        max_selected_rate=args.max_selected_rate,
        target_precision=args.target_precision,
        min_rows=args.min_rows,
        min_positive_rows=args.min_positive_rows,
        write_artifact=args.write_artifact,
    )
    payload = json.dumps(report, ensure_ascii=True)
    if args.report_out:
        report_path = pathlib.Path(args.report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(payload, encoding="utf-8")
    print(payload)
    return 0


def calibrate_threshold(
    *,
    dataset_path: pathlib.Path,
    profile: str,
    artifact_dir: pathlib.Path,
    split_field: str = "",
    split_value: str = "",
    source: str = "eval_calibration",
    selection_mode: str = "balanced_filter",
    recall_reference_datasets: list[pathlib.Path] | None = None,
    min_recall: float = train_ranker._SERVING_THRESHOLD_MIN_RECALL,
    max_selected_rate: float = train_ranker._SERVING_THRESHOLD_MAX_SELECTED_RATE,
    target_precision: float = train_ranker._SERVING_THRESHOLD_TARGET_PRECISION,
    min_rows: int = 10,
    min_positive_rows: int = 3,
    write_artifact: bool = False,
) -> dict[str, object]:
    rows = _read_jsonl(dataset_path)
    calibration_rows = _calibration_rows(rows, split_field, split_value)
    positive_rows = sum(1 for row in calibration_rows if int(row.get("label", 0)) == 1)
    negative_rows = sum(1 for row in calibration_rows if int(row.get("label", 0)) == 0)
    base_report: dict[str, object] = {
        "status": "skipped",
        "reason": "",
        "profile": profile,
        "dataset": str(dataset_path),
        "split_field": split_field,
        "split_value": split_value,
        "source": source,
        "selection_mode": str(selection_mode or "balanced_filter"),
        "recall_reference_datasets": [str(path) for path in (recall_reference_datasets or [])],
        "rows": len(rows),
        "calibration_rows": len(calibration_rows),
        "positive_rows": positive_rows,
        "negative_rows": negative_rows,
        "write_artifact": bool(write_artifact),
    }
    if len(calibration_rows) < int(min_rows):
        return {**base_report, "reason": "calibration_rows_below_minimum"}
    if positive_rows < int(min_positive_rows) or negative_rows < 1:
        return {**base_report, "reason": "calibration_class_rows_below_minimum"}

    ranker = Ranker(artifact_dir=artifact_dir, enable_trained_model=True)
    inspection = ranker.inspect_artifact(profile)
    artifact = inspection.get("artifact") if isinstance(inspection.get("artifact"), dict) else None
    artifact_status = str(inspection.get("status") or "missing")
    artifact_reason = str(inspection.get("reason") or "")
    can_calibrate_rejected = artifact_status == "rejected" and artifact_reason.startswith("validation_filter_")
    if (artifact_status != "usable" and not can_calibrate_rejected) or artifact is None:
        return {
            **base_report,
            "reason": str(inspection.get("reason") or "artifact_not_usable"),
            "artifact_status": str(inspection.get("status") or "missing"),
        }

    primary_set = _scored_calibration_set(
        calibration_rows,
        profile=profile,
        ranker=ranker,
        artifact=artifact,
        source=source,
        use_runtime_scoring=artifact_status == "usable",
    )
    split = primary_set["split"]
    reference_sets = [primary_set]
    for index, reference_path in enumerate(recall_reference_datasets or [], start=1):
        reference_rows = _read_jsonl(reference_path)
        if not reference_rows:
            continue
        reference_sets.append(
            _scored_calibration_set(
                reference_rows,
                profile=profile,
                ranker=ranker,
                artifact=artifact,
                source=f"recall_reference_{index}:{reference_path.name}",
                use_runtime_scoring=artifact_status == "usable",
            )
        )
    if str(selection_mode or "").strip().lower() == "recall_precision" or len(reference_sets) > 1:
        calibration = _calibrate_recall_precision_threshold(
            reference_sets,
            source=source,
            min_recall=float(min_recall),
            max_selected_rate=float(max_selected_rate),
            target_precision=float(target_precision),
        )
    else:
        calibration = train_ranker._calibrate_serving_threshold(
            primary_set["labels"],
            primary_set["scores"],
            groups=primary_set["groups"],
            source=source,
            min_recall=float(min_recall),
            max_selected_rate=float(max_selected_rate),
            target_precision=float(target_precision),
        )
    calibration["calibration_rows"] = len(calibration_rows)
    calibration["positive_rows"] = positive_rows
    calibration["negative_rows"] = negative_rows
    calibration["reference_sets"] = [
        {
            "source": str(reference_set["source"]),
            "rows": int(reference_set["rows"]),
            "positive_rows": int(reference_set["positive_rows"]),
            "negative_rows": int(reference_set["negative_rows"]),
        }
        for reference_set in reference_sets
    ]
    calibration["split_field"] = split_field
    calibration["split_value"] = split_value
    calibration["split_strategy"] = split.get("strategy", "")
    calibration["split_unit"] = split.get("unit", "")

    report = {
        **base_report,
        "status": "updated" if write_artifact else "computed",
        "reason": "ok",
        "artifact_status": inspection.get("status", ""),
        "artifact_reason": artifact_reason,
        "old_threshold": artifact.get("serving_threshold"),
        "new_threshold": calibration["threshold"],
        "calibration": calibration,
    }
    if write_artifact:
        previous_filter_gate = artifact.get("filter_gate")
        artifact["serving_threshold"] = float(calibration["threshold"])
        artifact["serving_threshold_calibration"] = calibration
        artifact["filter_gate"] = _filter_gate_from_calibration_preserving_bucket_routing(
            calibration,
            previous_filter_gate,
        )
        _write_artifact(artifact_dir, profile, artifact)
        _write_artifact_health(artifact_dir, profile, artifact)
    return report


def _filter_gate_from_calibration_preserving_bucket_routing(
    calibration: dict[str, object],
    previous_filter_gate: object,
) -> dict[str, object]:
    filter_gate = train_ranker._filter_gate_from_calibration(calibration)
    if not isinstance(previous_filter_gate, dict):
        return filter_gate
    global_failures = list(filter_gate.get("failures") or [])
    for key in (
        "bucket_gates",
        "bucket_gate_policy",
        "passed_bucket_count",
        "serving_threshold_per_bucket",
        "selected_model_per_bucket",
    ):
        value = previous_filter_gate.get(key)
        if value not in (None, {}, []):
            filter_gate[key] = value
    if isinstance(filter_gate.get("bucket_gates"), dict) and filter_gate.get("bucket_gates"):
        policy = filter_gate.get("bucket_gate_policy") if isinstance(filter_gate.get("bucket_gate_policy"), dict) else {}
        min_passed_buckets = int(policy.get("min_passed_buckets") or 0)
        passed_bucket_count = int(filter_gate.get("passed_bucket_count") or 0)
        bucket_failures: list[str] = []
        if min_passed_buckets and passed_bucket_count < min_passed_buckets:
            bucket_failures.append("bucket_gate_passed_count_below_minimum")
        previous_failures = [str(failure) for failure in (previous_filter_gate.get("failures") or [])]
        if "bucket_gate_ranking_metric_below_minimum" in previous_failures:
            bucket_failures.append("bucket_gate_ranking_metric_below_minimum")
        filter_gate["global_failures"] = global_failures
        filter_gate["failures"] = bucket_failures
        filter_gate["passed"] = not bucket_failures
    elif "global_failures" not in filter_gate and previous_filter_gate.get("global_failures") is not None:
        filter_gate["global_failures"] = previous_filter_gate.get("global_failures")
    return filter_gate


def _scored_calibration_set(
    rows: list[dict],
    *,
    profile: str,
    ranker: Ranker,
    artifact: dict,
    source: str,
    use_runtime_scoring: bool = True,
) -> dict[str, object]:
    features = [CandidateFeatures.from_dict(row["features"]) for row in rows]
    score_map = (
        {candidate_id: score for candidate_id, score in ranker.score(features, profile)}
        if use_runtime_scoring
        else _score_with_artifact(ranker, features, profile, artifact)
    )
    labels = train_ranker.np.array([int(row["label"]) for row in rows], dtype=int)
    scores = train_ranker.np.array(
        [float(score_map.get(_score_lookup_key(row), score_map.get(str(row.get("candidate_id") or ""), 0.0))) for row in rows],
        dtype=float,
    )
    split = train_ranker._split_rows(rows)
    groups = train_ranker._ranking_group_labels(rows, split.get("unit", ""))
    return {
        "source": source,
        "rows": len(rows),
        "positive_rows": int(sum(1 for row in rows if int(row.get("label", 0)) == 1)),
        "negative_rows": int(sum(1 for row in rows if int(row.get("label", 0)) == 0)),
        "labels": labels,
        "scores": scores,
        "groups": groups,
        "split": split,
    }


def _calibrate_recall_precision_threshold(
    reference_sets: list[dict[str, object]],
    *,
    source: str,
    min_recall: float,
    max_selected_rate: float,
    target_precision: float,
) -> dict[str, object]:
    default_threshold = train_ranker._default_classification_threshold("margin")
    if not reference_sets:
        return train_ranker._calibrate_serving_threshold(
            train_ranker.np.array([], dtype=int),
            train_ranker.np.array([], dtype=float),
            source=source,
            min_recall=min_recall,
            max_selected_rate=max_selected_rate,
            target_precision=target_precision,
        )

    finite_scores: list[float] = []
    for reference_set in reference_sets:
        scores = reference_set["scores"]
        finite_scores.extend(float(score) for score in scores.tolist() if train_ranker.math.isfinite(float(score)))
    if not finite_scores:
        primary = reference_sets[0]
        return train_ranker._calibrate_serving_threshold(
            primary["labels"],
            primary["scores"],
            groups=primary["groups"],
            source=source,
            min_recall=min_recall,
            max_selected_rate=max_selected_rate,
            target_precision=target_precision,
        )

    epsilon = 1e-6
    candidates = {float(default_threshold), min(finite_scores) - epsilon, max(finite_scores) + epsilon}
    candidates.update(finite_scores)
    best_threshold = float(default_threshold)
    best_metrics_by_source: dict[str, dict[str, float]] = {}
    best_aggregate_metrics: dict[str, float] = {}
    best_key: tuple = ()
    for threshold in sorted(candidates):
        metrics_by_source = _reference_metrics(reference_sets, float(threshold))
        aggregate_metrics = _aggregate_reference_metrics(reference_sets, float(threshold))
        key = _recall_precision_threshold_key(
            metrics_by_source,
            aggregate_metrics,
            float(threshold),
            min_recall=min_recall,
            max_selected_rate=max_selected_rate,
            target_precision=target_precision,
        )
        if not best_key or key > best_key:
            best_threshold = float(threshold)
            best_metrics_by_source = metrics_by_source
            best_aggregate_metrics = aggregate_metrics
            best_key = key

    min_reference_recall = min(float(metrics.get("recall", 0.0)) for metrics in best_metrics_by_source.values())
    selected_threshold_reason = _recall_precision_selection_reason(
        best_metrics_by_source,
        best_aggregate_metrics,
        min_recall=min_recall,
        max_selected_rate=max_selected_rate,
        target_precision=target_precision,
    )
    return {
        "source": source,
        "metric": "max_precision_under_recall_target",
        "threshold_policy": "recall_first_multi_reference",
        "selected_threshold_reason": selected_threshold_reason,
        "score_kind": "margin",
        "min_recall": float(min_recall),
        "max_selected_rate": float(max_selected_rate),
        "target_precision": float(target_precision),
        "threshold": round(float(best_threshold), 6),
        "metrics": best_aggregate_metrics,
        "reference_metrics": best_metrics_by_source,
        "min_reference_recall": round(float(min_reference_recall), 3),
        "default_threshold": float(default_threshold),
        "candidate_count": len(candidates),
    }


def _score_lookup_key(row: dict) -> str:
    features = row.get("features")
    if isinstance(features, dict):
        feature_candidate_id = str(features.get("candidate_id") or "").strip()
        if feature_candidate_id:
            return feature_candidate_id
    return str(row.get("candidate_id") or "").strip()


def _reference_metrics(reference_sets: list[dict[str, object]], threshold: float) -> dict[str, dict[str, float]]:
    metrics_by_source: dict[str, dict[str, float]] = {}
    for reference_set in reference_sets:
        source = str(reference_set.get("source") or "reference")
        metrics_by_source[source] = train_ranker._evaluate_scores(
            reference_set["labels"],
            reference_set["scores"],
            score_kind="margin",
            groups=reference_set.get("groups"),
            threshold=threshold,
        )
    return metrics_by_source


def _aggregate_reference_metrics(reference_sets: list[dict[str, object]], threshold: float) -> dict[str, float]:
    labels = train_ranker.np.concatenate([reference_set["labels"] for reference_set in reference_sets])
    scores = train_ranker.np.concatenate([reference_set["scores"] for reference_set in reference_sets])
    groups: list[str] = []
    for reference_set in reference_sets:
        source = str(reference_set.get("source") or "reference")
        groups.extend(f"{source}:{group}" for group in reference_set.get("groups") or [])
    return train_ranker._evaluate_scores(labels, scores, score_kind="margin", groups=groups, threshold=threshold)


def _recall_precision_threshold_key(
    metrics_by_source: dict[str, dict[str, float]],
    aggregate_metrics: dict[str, float],
    threshold: float,
    *,
    min_recall: float,
    max_selected_rate: float,
    target_precision: float,
) -> tuple:
    recalls = [float(metrics.get("recall", 0.0)) for metrics in metrics_by_source.values()]
    precisions = [float(metrics.get("precision", 0.0)) for metrics in metrics_by_source.values()]
    recall_gap = sum(max(0.0, float(min_recall) - recall) for recall in recalls)
    recall_passed = recall_gap <= 1e-12
    selected_rate = float(aggregate_metrics.get("selected_rate", 1.0))
    selected_rate_passed = selected_rate <= float(max_selected_rate)
    min_precision = min(precisions) if precisions else 0.0
    aggregate_precision = float(aggregate_metrics.get("precision", 0.0))
    aggregate_recall = float(aggregate_metrics.get("recall", 0.0))
    f0_75 = train_ranker._fbeta(aggregate_precision, aggregate_recall, train_ranker._FILTER_GATE_BETA)
    filter_recall_passed = aggregate_recall >= train_ranker._FILTER_GATE_MIN_RECALL
    filter_precision_passed = aggregate_precision >= train_ranker._FILTER_GATE_MIN_PRECISION
    filter_f1_passed = float(aggregate_metrics.get("f1", 0.0)) >= train_ranker._FILTER_GATE_MIN_F1
    filter_gate_passed = selected_rate_passed and filter_recall_passed and filter_precision_passed and filter_f1_passed
    return (
        1 if selected_rate_passed else 0,
        1 if recall_passed else 0,
        (1 if min_precision >= float(target_precision) else 0) if recall_passed else 0,
        min_precision if recall_passed else 0.0,
        aggregate_precision if recall_passed else 0.0,
        f0_75 if recall_passed else 0.0,
        aggregate_recall,
        0.0 if recall_passed else -recall_gap,
        1 if filter_gate_passed else 0,
        float(aggregate_metrics.get("f1", 0.0)),
        -selected_rate,
        float(threshold),
    )


def _recall_precision_selection_reason(
    metrics_by_source: dict[str, dict[str, float]],
    aggregate_metrics: dict[str, float],
    *,
    min_recall: float,
    max_selected_rate: float,
    target_precision: float,
) -> str:
    recalls = [float(metrics.get("recall", 0.0)) for metrics in metrics_by_source.values()]
    precisions = [float(metrics.get("precision", 0.0)) for metrics in metrics_by_source.values()]
    recall_passed = bool(recalls) and min(recalls) >= float(min_recall)
    selected_rate_passed = float(aggregate_metrics.get("selected_rate", 1.0)) <= float(max_selected_rate)
    precision_passed = bool(precisions) and min(precisions) >= float(target_precision)
    if selected_rate_passed and recall_passed and precision_passed:
        return "max_precision_under_recall_target"
    if selected_rate_passed and recall_passed:
        return "max_precision_under_recall_target_below_precision_floor"
    if selected_rate_passed:
        return "best_filter_gate_under_selected_rate_below_recall_target"
    if recall_passed:
        return "recall_target_above_selected_rate_rejected"
    return "best_degraded_below_recall_target"


def _score_with_artifact(
    ranker: Ranker,
    features: list[CandidateFeatures],
    profile: str,
    artifact: dict,
) -> dict[str, float]:
    heuristic = ranker._heuristic_scores(features, profile)
    for feature, (heuristic_score, _rationale) in zip(features, heuristic):
        feature.heuristic_score = float(heuristic_score)
    trained_scores = ranker._trained_scores(features, artifact)
    blend = ranker._safe_blend(artifact)
    return {
        feature.candidate_id: round(
            ranker._clamp_score(float(heuristic_score) * (1.0 - blend) + float(trained_score) * blend),
            3,
        )
        for feature, (heuristic_score, _rationale), trained_score in zip(features, heuristic, trained_scores)
    }


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("label") in {0, 1} and isinstance(row.get("features"), dict):
            rows.append(row)
    return rows


def _calibration_rows(rows: list[dict], split_field: str, split_value: str) -> list[dict]:
    field = str(split_field or "").strip()
    value = str(split_value or "").strip()
    if not field or not value:
        return list(rows)
    return [row for row in rows if str(row.get(field) or "").strip() == value]


def _write_artifact(artifact_dir: pathlib.Path, profile: str, artifact: dict) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / f"{profile}_ranker.json"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(artifact, ensure_ascii=True, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _write_artifact_health(artifact_dir: pathlib.Path, profile: str, artifact: dict) -> None:
    health = train_ranker._artifact_health_summary(artifact)
    path = artifact_dir / f"{profile}_artifact_health.json"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(health, ensure_ascii=True, indent=2), encoding="utf-8")
    tmp_path.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
