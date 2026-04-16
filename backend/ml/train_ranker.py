from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

try:
    from lightgbm import LGBMRanker
except Exception:  # pragma: no cover - optional dependency
    LGBMRanker = None

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.training_export import export_training_rows
from services.ranker import OFFLINE_ONLY_FEATURE_NAMES


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a simple linear ranker from candidate feedback.")
    parser.add_argument("--storage-dir", default=str(BACKEND_ROOT / "tmp" / "storage"))
    parser.add_argument("--profile", default="gaming")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--artifact-dir", default=str(BACKEND_ROOT / "ml" / "artifacts"))
    parser.add_argument("--report-out", default="")
    args = parser.parse_args()

    storage_dir = pathlib.Path(args.storage_dir)
    dataset_path = pathlib.Path(args.dataset) if args.dataset else storage_dir / "feedback" / "training_rows.jsonl"
    if not args.dataset:
        export_training_rows(storage_dir, dataset_path)

    rows = _read_jsonl(dataset_path)
    report_path = pathlib.Path(args.report_out) if args.report_out else None
    return train_from_rows(rows, args.profile, pathlib.Path(args.artifact_dir), report_path=report_path)


def train_from_dataset(
    dataset_path: pathlib.Path,
    profile: str,
    artifact_dir: pathlib.Path,
    report_path: pathlib.Path | None = None,
) -> int:
    rows = _read_jsonl(dataset_path)
    return train_from_rows(rows, profile, artifact_dir, report_path=report_path)


def train_from_rows(
    rows: list[dict],
    profile: str,
    artifact_dir: pathlib.Path,
    *,
    report_path: pathlib.Path | None = None,
) -> int:
    if not rows:
        print("No training rows found.")
        return 1

    positive = [row for row in rows if row["label"] == 1]
    negative = [row for row in rows if row["label"] == 0]
    if not positive or not negative:
        print("Need both positive and negative examples.")
        return 1

    split = _split_rows(rows)
    train_rows = split["train"]
    validation_rows = split["validation"]
    if not train_rows or not validation_rows:
        print("Need both train and validation rows.")
        return 1

    train_positive = [row["features"] for row in train_rows if row["label"] == 1]
    train_negative = [row["features"] for row in train_rows if row["label"] == 0]
    if not train_positive or not train_negative:
        print("Need both positive and negative training examples.")
        return 1

    feature_names = _numeric_feature_names(train_positive[0])
    scales = _compute_scales(train_rows, feature_names)
    train_matrix = _feature_matrix(train_rows, feature_names, scales)
    train_labels = np.array([int(row["label"]) for row in train_rows], dtype=int)
    train_weights = np.array([_sample_weight(row) for row in train_rows], dtype=float)
    validation_matrix = _feature_matrix(validation_rows, feature_names, scales)
    validation_labels = np.array([int(row["label"]) for row in validation_rows], dtype=int)
    validation_groups = _ranking_group_labels(validation_rows, split.get("unit", ""))

    linear_weights = _mean_difference_weights(train_positive, train_negative, feature_names, scales)
    linear_raw_scores = _linear_raw_scores(validation_rows, feature_names, linear_weights, scales, 0.0)
    linear_metrics = _evaluate_scores(validation_labels, linear_raw_scores, groups=validation_groups)

    logistic = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=0,
        solver="liblinear",
    )
    logistic.fit(train_matrix, train_labels, sample_weight=train_weights)
    logistic_raw_scores = logistic.decision_function(validation_matrix)
    logistic_metrics = _evaluate_scores(validation_labels, logistic_raw_scores, groups=validation_groups)

    pairwise_candidate = _fit_pairwise_logistic(
        train_rows,
        validation_rows,
        feature_names,
        scales,
        split.get("unit", ""),
    )

    gradient_boost = GradientBoostingClassifier(random_state=0)
    gradient_boost.fit(train_matrix, train_labels, sample_weight=train_weights)
    gradient_boost_scores = gradient_boost.predict_proba(validation_matrix)[:, 1]
    gradient_metrics = _evaluate_scores(
        validation_labels,
        gradient_boost_scores,
        score_kind="probability",
        groups=validation_groups,
    )

    lightgbm_ranker_candidate = _fit_lightgbm_ranker(
        train_rows,
        validation_rows,
        feature_names,
        scales,
        split.get("unit", ""),
    )

    model_candidates = [
        {
            "name": "mean_difference_linear",
            "artifact_model_type": "linear",
            "weights": linear_weights,
            "intercept": 0.0,
            "metrics": linear_metrics,
        },
        {
            "name": "logistic_regression",
            "artifact_model_type": "logistic",
            "weights": {
                name: float(value)
                for name, value in zip(feature_names, logistic.coef_[0], strict=False)
            },
            "intercept": float(logistic.intercept_[0]),
            "metrics": logistic_metrics,
        },
        *([pairwise_candidate] if pairwise_candidate is not None else []),
        {
            "name": "gradient_boosting",
            "artifact_model_type": "gradient_boosting",
            "feature_importances": {
                name: float(value)
                for name, value in zip(feature_names, gradient_boost.feature_importances_, strict=False)
            },
            "fitted_model": gradient_boost,
            "metrics": gradient_metrics,
        },
        *([lightgbm_ranker_candidate] if lightgbm_ranker_candidate is not None else []),
    ]
    selected_model = _select_model(model_candidates)

    model_path = None
    if selected_model["artifact_model_type"] in {"gradient_boosting", "lightgbm_ranker"}:
        model_path = artifact_dir / f"{profile}_ranker_model.joblib"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        tmp_model_path = model_path.with_suffix(model_path.suffix + ".tmp")
        joblib.dump(selected_model["fitted_model"], tmp_model_path)
        tmp_model_path.replace(model_path)

    artifact = {
        "version": 3,
        "artifact_family": "ranker_v2",
        "profile": profile,
        "model_type": selected_model["artifact_model_type"],
        "selected_model": selected_model["name"],
        "selection_metric": "composite_validation_score",
        "label_schema_version": 3,
        "feature_schema_version": 3,
        "feature_names": feature_names,
        "weights": selected_model.get("weights", {}),
        "scales": scales,
        "intercept": selected_model.get("intercept", 0.0),
        "model_path": model_path.name if model_path else "",
        "blend": 0.45,
        "top_feature_importances": _feature_importance_map(selected_model, feature_names),
        "training_stats": {
            "rows": len(rows),
            "positive_rows": len(positive),
            "negative_rows": len(negative),
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
            "split_strategy": split["strategy"],
            "split_unit": split.get("unit", ""),
            "leakage_risk": split.get("leakage_risk", "unknown"),
            "supervision_tiers": _count_values(rows, "supervision_tier"),
            "dataset_sources": _count_values(rows, "dataset_source"),
            "platforms": _count_values(rows, "publish_platform"),
            "variants": _count_values(rows, "final_variant_key"),
            "content_buckets": _count_values(rows, "content_bucket"),
        },
        "validation": selected_model["metrics"],
        "model_comparison": {
            candidate["name"]: candidate["metrics"]
            for candidate in model_candidates
        },
        "bucketed_validation": _bucketed_validation(
            selected_model["artifact_model_type"],
            validation_rows,
            feature_names,
            scales,
            selected_model,
            split.get("unit", ""),
        ),
    }

    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"{profile}_ranker.json"
    tmp_artifact_path = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
    tmp_artifact_path.write_text(json.dumps(artifact, ensure_ascii=True, indent=2), encoding="utf-8")
    tmp_artifact_path.replace(artifact_path)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(artifact, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Wrote artifact to {artifact_path}")
    print(
        json.dumps(
            {
                "selected_model": selected_model["name"],
                "validation": artifact["validation"],
                "training_stats": artifact["training_stats"],
                "model_comparison": artifact["model_comparison"],
            },
            ensure_ascii=True,
        )
    )
    return 0


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            row = json.loads(line)
            if row.get("label") in {0, 1}:
                rows.append(row)
    return rows


def _numeric_feature_names(feature_row: dict) -> list[str]:
    names: list[str] = []
    for key, value in feature_row.items():
        if key in {"candidate_id", "source"} or key in OFFLINE_ONLY_FEATURE_NAMES:
            continue
        if isinstance(value, (int, float, bool)):
            names.append(key)
    return sorted(names)


def _split_rows(rows: list[dict]) -> dict[str, list[dict] | str]:
    grouped_split = _split_rows_by_group(rows)
    if grouped_split is not None:
        return _validated_split(rows, grouped_split)

    time_split = _split_rows_by_time(rows)
    if time_split is not None:
        return _validated_split(rows, time_split)

    ordered = sorted(rows, key=lambda row: (_group_key(row), row.get("job_id", ""), row.get("candidate_id", "")))
    if len(ordered) < 4:
        return _stratified_fallback_split(ordered, strategy="fallback_half_split", unit="row")

    grouped: dict[str, list[dict]] = {}
    for row in ordered:
        grouped.setdefault(_group_key(row), []).append(row)

    real_group_keys = [key for key in grouped if key != "unknown"]
    if len(real_group_keys) <= 1:
        return _stratified_fallback_split(ordered, strategy="stratified_fallback", unit="creator_id")

    group_keys = sorted(grouped, key=lambda key: (len(grouped[key]), key), reverse=True)
    target_validation_size = max(2, len(ordered) // 5)

    train: list[dict] = []
    validation: list[dict] = []
    for key in group_keys:
        bucket = validation if len(validation) < target_validation_size else train
        bucket.extend(grouped[key])

    if not train or not validation:
        return _stratified_fallback_split(ordered, strategy="fallback_half_split", unit="row")
    return _validated_split(rows, {
        "train": train,
        "validation": validation,
        "strategy": "group_holdout",
        "unit": "creator_id",
        "leakage_risk": "medium",
    })


def _split_rows_by_group(rows: list[dict]) -> dict[str, list[dict] | str] | None:
    candidate_units = [
        ("vod_id", 2, "low"),
        ("source_url", 2, "low"),
        ("upload_date", 4, "low"),
    ]
    for unit, min_groups, leakage_risk in candidate_units:
        split = _group_holdout_by_unit(rows, unit, min_groups=min_groups, leakage_risk=leakage_risk)
        if split is not None:
            return split
    return None


def _split_rows_by_time(rows: list[dict]) -> dict[str, list[dict] | str] | None:
    creators = {value for value in (_group_key(row) for row in rows) if value and value != "unknown"}
    if len(creators) > 1:
        return None

    stamped = [(row, _row_timestamp(row)) for row in rows]
    stamped = [(row, ts) for row, ts in stamped if ts > 0]
    if len(stamped) < max(4, len(rows) // 2):
        return None

    ordered = [row for row, _ in sorted(stamped, key=lambda item: (item[1], item[0]["candidate_id"]))]
    split_at = max(1, int(math.floor(len(ordered) * 0.8)))
    train = ordered[:split_at]
    validation = ordered[split_at:]
    if not train or not validation:
        return None
    return {
        "train": train,
        "validation": validation,
        "strategy": "time_holdout",
        "unit": "timestamp",
        "leakage_risk": "medium",
    }


def _group_holdout_by_unit(
    rows: list[dict],
    unit: str,
    *,
    min_groups: int,
    leakage_risk: str,
) -> dict[str, list[dict] | str] | None:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        value = str(row.get(unit, "")).strip()
        if not value:
            continue
        grouped.setdefault(value, []).append(row)

    if len(grouped) < min_groups:
        return None

    ordered_groups = sorted(grouped.items(), key=lambda item: (len(item[1]), item[0]), reverse=True)
    target_validation_size = max(2, len(rows) // 5)
    train: list[dict] = []
    validation: list[dict] = []
    validation_positive = 0
    validation_negative = 0

    for _, group_rows in ordered_groups:
        group_positive = sum(1 for row in group_rows if int(row["label"]) == 1)
        group_negative = len(group_rows) - group_positive
        put_in_validation = (
            len(validation) < target_validation_size
            or validation_positive == 0
            or validation_negative == 0
        )
        bucket = validation if put_in_validation else train
        bucket.extend(group_rows)
        if put_in_validation:
            validation_positive += group_positive
            validation_negative += group_negative

    if not train or not validation:
        return None
    split = {
        "train": train,
        "validation": validation,
        "strategy": "group_holdout",
        "unit": unit,
        "leakage_risk": leakage_risk,
    }
    return split if _split_has_both_classes(train) and _split_has_both_classes(validation) else None


def _stratified_fallback_split(rows: list[dict], strategy: str, unit: str) -> dict[str, list[dict] | str]:
    positives = [row for row in rows if int(row["label"]) == 1]
    negatives = [row for row in rows if int(row["label"]) == 0]
    if not positives or not negatives:
        midpoint = max(1, len(rows) // 2)
        return {
            "train": rows[:midpoint],
            "validation": rows[midpoint:],
            "strategy": strategy,
            "unit": unit,
            "leakage_risk": "high",
        }

    def split_bucket(bucket: list[dict]) -> tuple[list[dict], list[dict]]:
        split_at = max(1, int(math.floor(len(bucket) * 0.8)))
        split_at = min(split_at, len(bucket) - 1) if len(bucket) > 1 else 1
        return bucket[:split_at], bucket[split_at:]

    pos_train, pos_validation = split_bucket(positives)
    neg_train, neg_validation = split_bucket(negatives)
    train = sorted(pos_train + neg_train, key=lambda row: (row.get("job_id", ""), row.get("candidate_id", "")))
    validation = sorted(pos_validation + neg_validation, key=lambda row: (row.get("job_id", ""), row.get("candidate_id", "")))
    return {
        "train": train,
        "validation": validation,
        "strategy": strategy,
        "unit": unit,
        "leakage_risk": "high",
    }


def _split_has_both_classes(rows: list[dict]) -> bool:
    labels = {int(row["label"]) for row in rows if row.get("label") in {0, 1}}
    return labels == {0, 1}


def _validated_split(rows: list[dict], split: dict[str, list[dict] | str]) -> dict[str, list[dict] | str]:
    train_rows = list(split.get("train") or [])
    validation_rows = list(split.get("validation") or [])
    if _split_has_both_classes(train_rows) and _split_has_both_classes(validation_rows):
        return split

    fallback_strategy = f"{split.get('strategy', 'split')}_fallback_stratified"
    fallback = _stratified_fallback_split(rows, strategy=fallback_strategy, unit="row")
    if _split_has_both_classes(list(fallback["train"])) and _split_has_both_classes(list(fallback["validation"])):
        return fallback
    return fallback


def _group_key(row: dict) -> str:
    for key in ("creator_id", "streamer", "job_id"):
        value = str(row.get(key, "")).strip().lower()
        if value:
            return value
    return "unknown"


def _evaluate(rows: list[dict], feature_names: list[str], weights: dict[str, float], scales: dict[str, dict[str, float]]) -> dict[str, float]:
    labels = np.array([int(row["label"]) for row in rows], dtype=int)
    scores = _linear_raw_scores(rows, feature_names, weights, scales, 0.0)
    return _evaluate_scores(labels, scores)


def _compute_scales(rows: list[dict], feature_names: list[str]) -> dict[str, dict[str, float]]:
    scales: dict[str, dict[str, float]] = {}
    for name in feature_names:
        combined = [float(row["features"].get(name, 0.0)) for row in rows]
        mean = sum(combined) / len(combined)
        variance = sum((value - mean) ** 2 for value in combined) / max(len(combined), 1)
        scales[name] = {"mean": mean, "std": variance ** 0.5 or 1.0}
    return scales


def _mean_difference_weights(
    train_positive: list[dict],
    train_negative: list[dict],
    feature_names: list[str],
    scales: dict[str, dict[str, float]],
) -> dict[str, float]:
    weights: dict[str, float] = {}
    for name in feature_names:
        pos_vals = [float(item.get(name, 0.0)) for item in train_positive]
        neg_vals = [float(item.get(name, 0.0)) for item in train_negative]
        std = float(scales.get(name, {}).get("std", 1.0)) or 1.0
        weights[name] = ((sum(pos_vals) / len(pos_vals)) - (sum(neg_vals) / len(neg_vals))) / std
    return weights


def _fit_pairwise_logistic(
    train_rows: list[dict],
    validation_rows: list[dict],
    feature_names: list[str],
    scales: dict[str, dict[str, float]],
    split_unit: str,
) -> dict | None:
    pair_matrix, pair_labels, pair_weights = _pairwise_training_matrix(train_rows, feature_names, scales)
    if len(pair_labels) < 8 or len(set(pair_labels.tolist())) < 2:
        return None

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=0,
        solver="liblinear",
    )
    model.fit(pair_matrix, pair_labels, sample_weight=pair_weights)
    weights = {
        name: float(value)
        for name, value in zip(feature_names, model.coef_[0], strict=False)
    }
    intercept = float(model.intercept_[0])
    validation_scores = _linear_raw_scores(validation_rows, feature_names, weights, scales, intercept)
    metrics = _evaluate_scores(
        np.array([int(row["label"]) for row in validation_rows], dtype=int),
        validation_scores,
        groups=_ranking_group_labels(validation_rows, split_unit),
    )
    return {
        "name": "pairwise_logistic_regression",
        "artifact_model_type": "logistic",
        "weights": weights,
        "intercept": intercept,
        "metrics": metrics,
    }


def _fit_lightgbm_ranker(
    train_rows: list[dict],
    validation_rows: list[dict],
    feature_names: list[str],
    scales: dict[str, dict[str, float]],
    split_unit: str,
) -> dict | None:
    if LGBMRanker is None:
        return None

    train_groups = _ranking_group_sizes(train_rows, split_unit)
    validation_groups = _ranking_group_sizes(validation_rows, split_unit)
    if not train_groups or not validation_groups:
        return None
    if sum(size >= 2 for size in train_groups) < 2:
        return None

    train_matrix = _feature_matrix(train_rows, feature_names, scales)
    validation_matrix = _feature_matrix(validation_rows, feature_names, scales)
    train_targets = np.array([_ranking_target(row) for row in train_rows], dtype=int)
    validation_targets = np.array([_ranking_target(row) for row in validation_rows], dtype=int)
    validation_binary = np.array([int(row["label"]) for row in validation_rows], dtype=int)
    train_weights = np.array([_sample_weight(row) for row in train_rows], dtype=float)

    model = LGBMRanker(
        objective="lambdarank",
        n_estimators=80,
        learning_rate=0.08,
        num_leaves=15,
        min_child_samples=5,
        random_state=0,
        verbosity=-1,
    )
    try:
        model.fit(
            train_matrix,
            train_targets,
            group=train_groups,
            sample_weight=train_weights,
            eval_set=[(validation_matrix, validation_targets)],
            eval_group=[validation_groups],
            eval_at=[5],
        )
    except Exception:
        return None

    validation_scores = np.array(model.predict(validation_matrix), dtype=float)
    metrics = _evaluate_scores(
        validation_binary,
        validation_scores,
        groups=_ranking_group_labels(validation_rows, split_unit),
    )
    return {
        "name": "lightgbm_ranker",
        "artifact_model_type": "lightgbm_ranker",
        "feature_importances": {
            name: float(value)
            for name, value in zip(feature_names, model.feature_importances_, strict=False)
        },
        "fitted_model": model,
        "metrics": metrics,
    }


def _pairwise_training_matrix(
    rows: list[dict],
    feature_names: list[str],
    scales: dict[str, dict[str, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(_group_key(row), []).append(row)

    pairs: list[list[float]] = []
    labels: list[int] = []
    weights: list[float] = []
    for group_rows in grouped.values():
        positives = [row for row in group_rows if int(row["label"]) == 1]
        negatives = [row for row in group_rows if int(row["label"]) == 0]
        if not positives or not negatives:
            continue
        max_pairs = min(len(positives) * len(negatives), 64)
        pair_count = 0
        for positive in positives:
            pos_vec = [
                _normalized_feature_value(positive["features"], name, scales)
                for name in feature_names
            ]
            for negative in negatives:
                if pair_count >= max_pairs:
                    break
                neg_vec = [
                    _normalized_feature_value(negative["features"], name, scales)
                    for name in feature_names
                ]
                diff = [p - n for p, n in zip(pos_vec, neg_vec, strict=False)]
                weight = (_sample_weight(positive) + _sample_weight(negative)) / 2.0
                pairs.append(diff)
                labels.append(1)
                weights.append(weight)
                pairs.append([-value for value in diff])
                labels.append(0)
                weights.append(weight)
                pair_count += 1
    return (
        np.array(pairs, dtype=float) if pairs else np.empty((0, len(feature_names)), dtype=float),
        np.array(labels, dtype=int),
        np.array(weights, dtype=float),
    )


def _feature_matrix(rows: list[dict], feature_names: list[str], scales: dict[str, dict[str, float]]) -> np.ndarray:
    return np.array([
        [
            _normalized_feature_value(row["features"], name, scales)
            for name in feature_names
        ]
        for row in rows
    ], dtype=float)


def _ranking_group_sizes(rows: list[dict], split_unit: str) -> list[int]:
    if not rows:
        return []
    grouped: dict[str, int] = {}
    order: list[str] = []
    for row in rows:
        key = _ranking_group_key(row, split_unit)
        if key not in grouped:
            grouped[key] = 0
            order.append(key)
        grouped[key] += 1
    return [grouped[key] for key in order if grouped[key] > 0]


def _ranking_group_labels(rows: list[dict], split_unit: str) -> list[str]:
    return [_ranking_group_key(row, split_unit) for row in rows]


def _ranking_group_key(row: dict, split_unit: str) -> str:
    if split_unit:
        value = str(row.get(split_unit, "")).strip()
        if value:
            return value
    for key in ("vod_id", "source_url", "creator_id", "job_id"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return str(row.get("candidate_id", "unknown"))


def _ranking_target(row: dict) -> int:
    value = row.get("ordinal_label")
    if value not in (None, ""):
        try:
            return max(int(value), 0)
        except (TypeError, ValueError):
            pass
    return max(int(row.get("label", 0)), 0)


def _linear_raw_scores(
    rows: list[dict],
    feature_names: list[str],
    weights: dict[str, float],
    scales: dict[str, dict[str, float]],
    intercept: float,
) -> np.ndarray:
    scores: list[float] = []
    for row in rows:
        score = float(intercept)
        for name in feature_names:
            score += _normalized_feature_value(row["features"], name, scales) * float(weights.get(name, 0.0))
        scores.append(score)
    return np.array(scores, dtype=float)


def _normalized_feature_value(features: dict, name: str, scales: dict[str, dict[str, float]]) -> float:
    value = float(features.get(name, 0.0))
    scale = scales.get(name, {"mean": 0.0, "std": 1.0})
    std = float(scale.get("std", 1.0)) or 1.0
    return (value - float(scale.get("mean", 0.0))) / std


def _evaluate_scores(
    labels: np.ndarray,
    scores: np.ndarray,
    score_kind: str = "margin",
    groups: list[str] | None = None,
) -> dict[str, float]:
    total = int(len(labels)) or 1
    if score_kind == "probability":
        predicted = (scores >= 0.5).astype(int)
        ranking_scores = scores
    else:
        predicted = (scores >= 0.0).astype(int)
        ranking_scores = 1.0 / (1.0 + np.exp(-np.clip(scores, -20.0, 20.0)))

    true_positive = int(np.sum((predicted == 1) & (labels == 1)))
    false_positive = int(np.sum((predicted == 1) & (labels == 0)))
    false_negative = int(np.sum((predicted == 0) & (labels == 1)))
    true_negative = int(np.sum((predicted == 0) & (labels == 0)))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-9)

    metrics = {
        "accuracy": round(float(np.mean(predicted == labels)), 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "positive_rate": round(float(np.mean(labels == 1)), 3),
    }
    for k in (3, 5):
        precision_k, recall_k = _grouped_precision_recall_at_k(labels, ranking_scores, k, groups)
        metrics[f"precision_at_{k}"] = round(precision_k, 3)
        metrics[f"recall_at_{k}"] = round(recall_k, 3)
    metrics["ndcg_at_5"] = round(_grouped_ndcg_at_k(labels, ranking_scores, 5, groups), 3)
    if groups:
        metrics["ranking_groups"] = len({group for group in groups if str(group).strip()})
        for k in (3, 5):
            precision_k, recall_k = _precision_recall_at_k(labels, ranking_scores, k)
            metrics[f"global_precision_at_{k}"] = round(precision_k, 3)
            metrics[f"global_recall_at_{k}"] = round(recall_k, 3)
        metrics["global_ndcg_at_5"] = round(_ndcg_at_k(labels, ranking_scores, 5), 3)
    metrics["rows"] = total
    return metrics


def _precision_recall_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> tuple[float, float]:
    if len(labels) == 0:
        return 0.0, 0.0
    top_idx = np.argsort(scores)[::-1][: min(k, len(labels))]
    hits = int(np.sum(labels[top_idx] == 1))
    return hits / max(len(top_idx), 1), hits / max(int(np.sum(labels == 1)), 1)


def _ndcg_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    if len(labels) == 0:
        return 0.0
    top_idx = np.argsort(scores)[::-1][: min(k, len(labels))]
    gains = labels[top_idx]
    dcg = sum((2 ** int(label) - 1) / math.log2(index + 2) for index, label in enumerate(gains))
    ideal = sorted(labels.tolist(), reverse=True)[: min(k, len(labels))]
    idcg = sum((2 ** int(label) - 1) / math.log2(index + 2) for index, label in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def _grouped_precision_recall_at_k(
    labels: np.ndarray,
    scores: np.ndarray,
    k: int,
    groups: list[str] | None,
) -> tuple[float, float]:
    if not groups:
        return _precision_recall_at_k(labels, scores, k)
    if len(groups) != len(labels):
        return _precision_recall_at_k(labels, scores, k)

    grouped_indices: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        grouped_indices.setdefault(str(group), []).append(index)

    precisions: list[float] = []
    recalls: list[float] = []
    for indices in grouped_indices.values():
        if not indices:
            continue
        precision_k, recall_k = _precision_recall_at_k(labels[indices], scores[indices], k)
        precisions.append(float(precision_k))
        recalls.append(float(recall_k))
    if not precisions:
        return _precision_recall_at_k(labels, scores, k)
    return sum(precisions) / len(precisions), sum(recalls) / len(recalls)


def _grouped_ndcg_at_k(
    labels: np.ndarray,
    scores: np.ndarray,
    k: int,
    groups: list[str] | None,
) -> float:
    if not groups:
        return _ndcg_at_k(labels, scores, k)
    if len(groups) != len(labels):
        return _ndcg_at_k(labels, scores, k)

    grouped_indices: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        grouped_indices.setdefault(str(group), []).append(index)

    values: list[float] = []
    for indices in grouped_indices.values():
        if not indices:
            continue
        values.append(float(_ndcg_at_k(labels[indices], scores[indices], k)))
    return sum(values) / len(values) if values else _ndcg_at_k(labels, scores, k)


def _select_model(candidates: list[dict]) -> dict:
    viable = [candidate for candidate in candidates if _model_meets_minimums(candidate["metrics"])]
    pool = viable or candidates

    def sort_key(candidate: dict) -> tuple[float, float, float, float, str]:
        metrics = candidate["metrics"]
        return (
            _selection_score(metrics),
            float(metrics.get("precision_at_5", 0.0)),
            float(metrics.get("ndcg_at_5", 0.0)),
            float(metrics.get("f1", 0.0)),
            float(metrics.get("accuracy", 0.0)),
            candidate["name"],
        )

    return max(pool, key=sort_key)


def _selection_score(metrics: dict[str, float]) -> float:
    return (
        float(metrics.get("ndcg_at_5", 0.0)) * 0.4
        + float(metrics.get("precision_at_5", 0.0)) * 0.25
        + float(metrics.get("f1", 0.0)) * 0.2
        + float(metrics.get("accuracy", 0.0)) * 0.15
    )


def _model_meets_minimums(metrics: dict[str, float]) -> bool:
    accuracy = float(metrics.get("accuracy", 0.0))
    ranking_groups = int(metrics.get("ranking_groups", 0) or 0)
    ranking = max(
        float(metrics.get("ndcg_at_5", 0.0)),
        float(metrics.get("precision_at_5", 0.0)),
    )
    if ranking_groups and ranking_groups < 2:
        return False
    return accuracy >= 0.58 and ranking >= 0.6


def _sample_weight(row: dict) -> float:
    supervision_multiplier = {
        "outcome_feedback": 1.35,
        "manual_annotation": 1.5,
        "twitch_clip_proxy": 1.15,
        "short_form_discovered": 1.0,
        "pipeline_feedback": 0.85,
        "synthetic": 0.6,
    }.get(str(row.get("supervision_tier", "")).strip().lower(), 1.0)

    if row.get("sample_weight") not in (None, ""):
        try:
            return float(row["sample_weight"]) * supervision_multiplier
        except (TypeError, ValueError):
            pass
    label_class = str(row.get("label_class", "")).strip().lower()
    base_weight = {
        "viral_positive": 2.5,
        "positive": 1.5,
        "weak_positive": 0.8,
        "hard_negative": 1.8,
        "background_negative": 1.0,
        "short_form_positive": 1.4,
        "short_form_negative": 1.0,
        "outcome_positive": 2.2,
        "outcome_negative": 1.8,
        "pipeline_positive": 1.0,
        "pipeline_negative": 0.8,
    }.get(label_class, 1.0)
    return base_weight * supervision_multiplier * _outcome_weight_multiplier(row)


def _outcome_weight_multiplier(row: dict) -> float:
    multiplier = 1.0
    if str(row.get("supervision_tier", "")).strip().lower() == "outcome_feedback":
        retention_3s = _safe_float(row.get("retention_3s"))
        completion_rate = _safe_float(row.get("completion_rate"))
        baseline = _safe_float(row.get("baseline_view_rate"))
        observed = _safe_float(row.get("observed_view_rate"))
        if retention_3s >= 0.8:
            multiplier += 0.15
        elif 0.0 < retention_3s <= 0.35:
            multiplier -= 0.1
        if completion_rate >= 0.55:
            multiplier += 0.1
        elif 0.0 < completion_rate <= 0.2:
            multiplier -= 0.08
        if baseline > 0 and observed > 0:
            ratio = observed / baseline
            if ratio >= 1.25:
                multiplier += 0.12
            elif ratio <= 0.8:
                multiplier -= 0.08
    if str(row.get("publish_platform") or "").strip().lower() in {"tiktok", "youtube_shorts", "instagram_reels"}:
        multiplier += 0.03
    return max(0.5, min(1.5, multiplier))


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _bucketed_validation(
    model_type: str,
    rows: list[dict],
    feature_names: list[str],
    scales: dict[str, dict[str, float]],
    selected_model: dict,
    split_unit: str,
) -> dict[str, dict[str, float]]:
    buckets: dict[str, dict[str, float]] = {}
    for bucket in sorted({str(row.get("label_class", "")) for row in rows if row.get("label_class")}):
        subset = [row for row in rows if str(row.get("label_class", "")) == bucket]
        if not subset:
            continue
        subset_groups = _ranking_group_labels(subset, split_unit)
        if model_type == "linear":
            scores = _linear_raw_scores(subset, feature_names, selected_model["weights"], scales, float(selected_model.get("intercept", 0.0)))
            labels = np.array([int(row["label"]) for row in subset], dtype=int)
            buckets[bucket] = _evaluate_scores(labels, scores, groups=subset_groups)
        elif model_type == "logistic":
            scores = _linear_raw_scores(subset, feature_names, selected_model["weights"], scales, float(selected_model.get("intercept", 0.0)))
            labels = np.array([int(row["label"]) for row in subset], dtype=int)
            buckets[bucket] = _evaluate_scores(labels, scores, groups=subset_groups)
        elif model_type in {"gradient_boosting", "lightgbm_ranker"}:
            fitted_model = selected_model.get("fitted_model")
            if fitted_model is None:
                continue
            if model_type == "gradient_boosting":
                scores = fitted_model.predict_proba(_feature_matrix(subset, feature_names, scales))[:, 1]
                labels = np.array([int(row["label"]) for row in subset], dtype=int)
                buckets[bucket] = _evaluate_scores(
                    labels,
                    scores,
                    score_kind="probability",
                    groups=subset_groups,
                )
                continue
            scores = np.array(fitted_model.predict(_feature_matrix(subset, feature_names, scales)), dtype=float)
            labels = np.array([int(row["label"]) for row in subset], dtype=int)
            buckets[bucket] = _evaluate_scores(labels, scores, groups=subset_groups)
    return buckets


def _feature_importance_map(selected_model: dict, feature_names: list[str]) -> dict[str, float]:
    if selected_model.get("feature_importances"):
        values = {
            str(name): abs(float(value))
            for name, value in dict(selected_model["feature_importances"]).items()
        }
    elif selected_model.get("weights"):
        values = {
            str(name): abs(float(value))
            for name, value in dict(selected_model["weights"]).items()
        }
    else:
        values = {name: 0.0 for name in feature_names}
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]), reverse=True)
    return {name: round(value, 6) for name, value in ordered[:25]}


def _row_timestamp(row: dict) -> int:
    for key in ("publish_timestamp", "timestamp"):
        value = row.get(key)
        if value not in (None, ""):
            try:
                return int(value)
            except (TypeError, ValueError):
                pass

    upload_date = str(row.get("upload_date") or "").strip()
    if len(upload_date) == 8 and upload_date.isdigit():
        try:
            dt = datetime.strptime(upload_date, "%Y%m%d").replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except ValueError:
            return 0
    return 0


def _count_values(rows: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "").strip()
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    raise SystemExit(main())
