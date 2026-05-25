from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pathlib
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

try:
    from lightgbm import LGBMRanker, early_stopping
except Exception:  # pragma: no cover - optional dependency
    LGBMRanker = None
    early_stopping = None

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config import settings
from ml import mlflow_tracking
from services.gold_eval_policy import GOLD_HOLDOUT_SPLIT_FIELD
from services import content_buckets, semantic_service
from services.training_export import export_training_rows
from services.ranker import OFFLINE_ONLY_FEATURE_NAMES, Ranker
from services.timeline_types import CandidateFeatures


logger = logging.getLogger(__name__)

_LIGHTGBM_PROMOTION_MIN_ROWS = 400
_GRADIENT_BOOSTING_PROMOTION_MIN_ROWS = 400
_ADVANCED_MODEL_PROMOTION_MIN_ROWS = min(_LIGHTGBM_PROMOTION_MIN_ROWS, _GRADIENT_BOOSTING_PROMOTION_MIN_ROWS)
_LIGHTGBM_DEFAULT_PARAMS = {
    "n_estimators": 80,
    "learning_rate": 0.08,
    "num_leaves": 15,
    "min_child_samples": 5,
    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "reg_alpha": 0.1,
}
_LIGHTGBM_HPO_PARAM_KEYS = frozenset(_LIGHTGBM_DEFAULT_PARAMS)
_BLEND_CANDIDATES = tuple(round(value, 2) for value in np.arange(0.15, 1.05, 0.05))
_HEURISTIC_FEATURE_NAME = "heuristic_score"
# Keep chat features trainable. Older runs had all-zero chat category columns,
# but the replay alignment now emits numeric category and velocity signals.
_SPARSE_TRAINING_FEATURE_NAMES: frozenset[str] = frozenset()
_CAPABILITY_FEATURE_NAMES: frozenset[str] = frozenset({
    "has_chat_features",
    "has_llm_semantics",
    "has_vector_similarity",
})
_CONFOUND_TRAINING_FEATURE_NAMES: frozenset[str] = frozenset({
    # Current Jynxzi eval analysis showed this penalty receiving a positive
    # model weight. Keep serving-time heuristic semantics, but keep it out of
    # supervised ranker fitting until the label set is large enough to
    # disambiguate slow starts from long/high-density clips.
    "slow_open_penalty",
})
_POSITIVE_SIGN_FEATURES: frozenset[str] = frozenset({
    "arc_payoff_match",
    "chat_spike_coverage",
    "hook_strength",
    "payoff_score",
})
_TOP_CHAT_INTERACTION_FEATURES = content_buckets.TOP_CHAT_INTERACTION_FEATURES
_CONTENT_BUCKET_FEATURE_PREFIX = content_buckets.CONTENT_BUCKET_FEATURE_PREFIX
_CONTENT_BUCKET_CHAT_INTERACTION_PREFIX = content_buckets.CONTENT_BUCKET_CHAT_INTERACTION_PREFIX
_MAX_CANDIDATE_SOURCE_SHARE = 0.25
_TRAINED_SCORE_CLAMP_ABS = 12.0
_BACKGROUND_NEGATIVE_SOURCE = "background_negative"
_BOUNDARY_PERTURBATION_SOURCE = "boundary_perturbation"
_BACKGROUND_NEGATIVE_TARGET_MIN_ROWS = 1000
_DYNAMIC_SOURCE_CAP_NUMERATOR = 1.5
_DYNAMIC_SOURCE_CAP_MAX_SHARE = 0.20
_DYNAMIC_SOURCE_CAP_MIN_SHARE = 0.08
_BUCKET_GATE_MIN_PASSED_BUCKETS = 3
_BUCKET_GATE_MIN_RANKING_METRIC = 0.70
_BUCKET_GATE_MIN_ROWS = 6
_MIN_JYNXZI_VALIDATION_GROUPS = 40
_MIN_JYNXZI_ROLLOUT_ROWS = 1200
_MIN_JYNXZI_ROLLOUT_POSITIVES = 400
_MIN_JYNXZI_INVENTORY_ROLLOUT_ROWS = 300
_MIN_JYNXZI_INVENTORY_ROLLOUT_POSITIVES = 120
_MIN_JYNXZI_INVENTORY_VALIDATION_GROUPS = 8
_JYNXZI_INVENTORY_ROWS_PER_GROUP = 8
_JYNXZI_INVENTORY_POSITIVES_PER_GROUP = 4
_JYNXZI_INVENTORY_VALIDATION_GROUP_SHARE = 0.40
_TWITCH_CLIP_PROXY_SUPERVISION_MULTIPLIER = 0.5
_SPLIT_RANDOM_SEED = int(settings.ranker_hpo_seed)
_ARTIFACT_VERSION = 4
_FEATURE_SCHEMA_VERSION = 6
_DEFAULT_MAX_TRAINING_FEATURES = 40
_LIGHTGBM_PREFERENCE_TOLERANCE = 0.02
_JUDGE_TEACHER_FEATURE_NAMES = frozenset(
    {
        "judge_score",
        "judge_confidence",
        "judge_scroll_stop_power",
        "judge_standalone_clarity",
        "judge_payoff_strength",
        "judge_narrative_completeness",
        "judge_platform_fit_shortform",
        "has_judge_score",
        "judge_llm_used",
    }
)


def excluded_training_feature_names() -> frozenset[str]:
    return (
        OFFLINE_ONLY_FEATURE_NAMES
        | _SPARSE_TRAINING_FEATURE_NAMES
        | _CONFOUND_TRAINING_FEATURE_NAMES
    )
_MIN_GROUP_NDCG_AT_5_P25 = 0.45
_MIN_GROUP_NDCG_AT_5_MIN = 0.2
_MIN_GROUP_RANKING_METRIC_P25 = 0.45
_MIN_GROUP_RANKING_METRIC_MIN = 0.2
_DEFAULT_MARGIN_CLASSIFICATION_THRESHOLD = 0.0
_DEFAULT_PROBABILITY_CLASSIFICATION_THRESHOLD = 0.5
_SERVING_THRESHOLD_MIN_RECALL = 0.70
_SERVING_THRESHOLD_MAX_SELECTED_RATE = 0.50
_SERVING_THRESHOLD_TARGET_PRECISION = 0.55
_SERVING_THRESHOLD_FALLBACK_MAX_SELECTED_RATE = 0.60
_FILTER_GATE_MIN_PRECISION = 0.50
_FILTER_GATE_MIN_RECALL = 0.65
_FILTER_GATE_MIN_F1 = 0.55
_FILTER_GATE_MAX_SELECTED_RATE = 0.75
_FILTER_GATE_MIN_ACCURACY_FLOOR = 0.55
_FILTER_GATE_BETA = 0.75
_SERVING_THRESHOLD_HARD_MAX_SELECTED_RATE = _FILTER_GATE_MAX_SELECTED_RATE


def _inject_heuristic_features(rows: list[dict], profile: str) -> None:
    """Compute the rule-based heuristic score for each row and add it as a feature.

    Lets the trained ranker treat the heuristic as just another signal instead of
    relying on a post-hoc linear blend at serving time. Mutates rows in place.
    """
    if not rows:
        return
    ranker = Ranker(enable_trained_model=False)
    feature_rows = [
        CandidateFeatures.from_dict(
            {
                "candidate_id": str(row.get("candidate_id") or "unknown"),
                **dict(row.get("features") or {}),
            }
        )
        for row in rows
    ]
    heuristic_scores = ranker._heuristic_scores(feature_rows, profile)
    for row, (score, _rationale) in zip(rows, heuristic_scores):
        features = dict(row.get("features") or {})
        features[_HEURISTIC_FEATURE_NAME] = float(score)
        row["features"] = features


def _prepare_ranker_training_rows(rows: list[dict]) -> tuple[list[dict], dict[str, object]]:
    filtered: list[dict] = []
    excluded_twitch_proxy = 0
    for row in rows:
        if _is_twitch_proxy_training_row(row):
            excluded_twitch_proxy += 1
            continue
        copied = dict(row)
        copied["features"] = _serving_training_features(copied)
        filtered.append(copied)

    has_background_negatives = any(_is_background_negative_training_row(row) for row in filtered)
    source_cap = _dynamic_candidate_source_share(filtered) if has_background_negatives else _MAX_CANDIDATE_SOURCE_SHARE
    target_min_rows = _BACKGROUND_NEGATIVE_TARGET_MIN_ROWS if has_background_negatives else _ADVANCED_MODEL_PROMOTION_MIN_ROWS
    capped, cap_report = _cap_candidate_source_share(
        filtered,
        source_cap,
        min_rows=min(target_min_rows, len(filtered)),
        refill_excluded_sources={_BOUNDARY_PERTURBATION_SOURCE} if has_background_negatives else frozenset(),
    )
    return capped, {
        "input_rows": len(rows),
        "after_twitch_proxy_exclusion": len(filtered),
        "excluded_twitch_clip_proxy_rows": excluded_twitch_proxy,
        "candidate_source_cap": cap_report,
        "capability_features_enabled": sorted(_CAPABILITY_FEATURE_NAMES),
        "content_bucket_interactions_enabled": True,
    }


def _is_twitch_proxy_training_row(row: dict) -> bool:
    if _is_gold_eval_row(row):
        return False
    supervision_tier = str(row.get("supervision_tier") or "").strip().lower()
    dataset_source = str(row.get("dataset_source") or "").strip().lower()
    return supervision_tier == "twitch_clip_proxy" or dataset_source == "twitch_vod_primary"


def _is_background_negative_training_row(row: dict) -> bool:
    return (
        str(row.get("supervision_tier") or "").strip().lower() == "background_negative"
        or str(row.get("label_class") or "").strip().lower() == "background_negative"
        or str(row.get("candidate_source") or "").strip().lower() == _BACKGROUND_NEGATIVE_SOURCE
    )


def _dynamic_candidate_source_share(rows: list[dict]) -> float:
    sources = {
        str(row.get("candidate_source") or (row.get("features") or {}).get("source") or "unknown").strip().lower() or "unknown"
        for row in rows
    }
    if not sources:
        return _MAX_CANDIDATE_SOURCE_SHARE
    dynamic = _DYNAMIC_SOURCE_CAP_NUMERATOR / max(len(sources), 1)
    return max(_DYNAMIC_SOURCE_CAP_MIN_SHARE, min(_DYNAMIC_SOURCE_CAP_MAX_SHARE, dynamic))


def _serving_training_features(row: dict) -> dict[str, object]:
    features = dict(row.get("features") or {})
    bucket = str(row.get("content_bucket") or "").strip().lower() or _content_bucket_from_row(row, features)
    if bucket:
        features[f"{_CONTENT_BUCKET_FEATURE_PREFIX}{bucket}"] = 1.0
        for name in _TOP_CHAT_INTERACTION_FEATURES:
            value = _safe_float(features.get(name))
            if value:
                features[f"{_CONTENT_BUCKET_CHAT_INTERACTION_PREFIX}{bucket}__{name}"] = value
    return features


def _content_bucket_from_row(row: dict, features: dict) -> str:
    return content_buckets.bucket_from_row(row, features)


def _cap_candidate_source_share(
    rows: list[dict],
    max_share: float,
    *,
    min_rows: int = 0,
    refill_excluded_sources: set[str] | frozenset[str] = frozenset(),
) -> tuple[list[dict], dict[str, object]]:
    if not rows:
        return [], {
            "enabled": True,
            "max_share": float(max_share),
            "target_min_rows": 0,
            "removed_rows": 0,
            "by_source_before": {},
        }
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        source = str(row.get("candidate_source") or (row.get("features") or {}).get("source") or "unknown").strip().lower()
        grouped.setdefault(source or "unknown", []).append(row)
    target_min_rows = max(0, min(int(min_rows), len(rows)))
    if len(grouped) <= 1:
        return list(rows), {
            "enabled": True,
            "max_share": float(max_share),
            "target_min_rows": target_min_rows,
            "max_rows_per_source": len(rows),
            "rows_before": len(rows),
            "rows_after": len(rows),
            "removed_rows": 0,
            "adaptive_refill_rows": 0,
            "refilled_by_source": {},
            "removed_by_source": {},
            "by_source_before": {source: len(source_rows) for source, source_rows in sorted(grouped.items())},
            "by_source_after": _count_values(rows, "candidate_source"),
        }
    max_rows_per_source = max(1, int(math.ceil(len(rows) * float(max_share))))
    kept: list[dict] = []
    kept_by_source: dict[str, list[dict]] = {}
    remaining_by_source: dict[str, list[dict]] = {}
    for source, source_rows in grouped.items():
        ordered = _seeded_row_order(source_rows)
        selected = ordered[:max_rows_per_source]
        kept.extend(selected)
        kept_by_source[source] = list(selected)
        remaining = ordered[max_rows_per_source:]
        if remaining:
            remaining_by_source[source] = list(remaining)

    refilled_by_source: dict[str, int] = {}
    excluded_refill_sources = {str(source).strip().lower() for source in refill_excluded_sources}
    while target_min_rows and len(kept) < target_min_rows:
        refill_sources = [
            source
            for source, source_rows in remaining_by_source.items()
            if source_rows and source not in excluded_refill_sources
        ]
        if not refill_sources:
            break
        source = sorted(refill_sources, key=lambda value: (_refill_source_priority(value), len(kept_by_source.get(value, [])), value))[0]
        row = remaining_by_source[source].pop(0)
        kept.append(row)
        kept_by_source.setdefault(source, []).append(row)
        refilled_by_source[source] = refilled_by_source.get(source, 0) + 1

    removed_by_source = {
        source: len(source_rows)
        for source, source_rows in sorted(remaining_by_source.items())
        if source_rows
    }
    kept.sort(key=lambda row: (str(row.get("job_id", "")), str(row.get("candidate_id", ""))))
    before = {source: len(source_rows) for source, source_rows in sorted(grouped.items())}
    after = _count_values(kept, "candidate_source")
    return kept, {
        "enabled": True,
        "max_share": float(max_share),
        "target_min_rows": target_min_rows,
        "max_rows_per_source": max_rows_per_source,
        "rows_before": len(rows),
        "rows_after": len(kept),
        "removed_rows": len(rows) - len(kept),
        "adaptive_refill_rows": sum(refilled_by_source.values()),
        "refilled_by_source": refilled_by_source,
        "refill_excluded_sources": sorted(excluded_refill_sources),
        "removed_by_source": removed_by_source,
        "by_source_before": before,
        "by_source_after": after,
    }


def _refill_source_priority(source: str) -> int:
    if source == _BACKGROUND_NEGATIVE_SOURCE:
        return 0
    if source == _BOUNDARY_PERTURBATION_SOURCE:
        return 2
    return 1


def _min_validation_groups_required(profile: str) -> int:
    return _MIN_JYNXZI_VALIDATION_GROUPS if str(profile).strip().lower() == "jynxzi" else 0


def rollout_readiness_targets(profile: str, *, available_groups: int | None = None) -> dict[str, object]:
    profile_key = str(profile).strip().lower()
    raw_targets = {
        "training_rows": _MIN_JYNXZI_ROLLOUT_ROWS if profile_key == "jynxzi" else 0,
        "positive_rows": _MIN_JYNXZI_ROLLOUT_POSITIVES if profile_key == "jynxzi" else 0,
        "validation_groups": _MIN_JYNXZI_VALIDATION_GROUPS if profile_key == "jynxzi" else 0,
    }
    if profile_key != "jynxzi":
        return {
            **raw_targets,
            "mode": "profile_default",
            "raw_targets": dict(raw_targets),
            "available_groups": int(available_groups or 0),
        }

    group_count = max(0, int(available_groups or 0))
    inventory_limited = (
        bool(group_count)
        and (
            group_count < _MIN_JYNXZI_VALIDATION_GROUPS
            or group_count * _JYNXZI_INVENTORY_ROWS_PER_GROUP < _MIN_JYNXZI_ROLLOUT_ROWS
            or group_count * _JYNXZI_INVENTORY_POSITIVES_PER_GROUP < _MIN_JYNXZI_ROLLOUT_POSITIVES
        )
    )
    if not inventory_limited:
        return {
            **raw_targets,
            "mode": "absolute",
            "raw_targets": dict(raw_targets),
            "available_groups": group_count,
        }

    validation_target = int(math.floor(group_count * _JYNXZI_INVENTORY_VALIDATION_GROUP_SHARE))
    validation_target = min(
        group_count,
        _MIN_JYNXZI_VALIDATION_GROUPS,
        max(_MIN_JYNXZI_INVENTORY_VALIDATION_GROUPS, validation_target),
    )
    if group_count < _MIN_JYNXZI_INVENTORY_VALIDATION_GROUPS:
        validation_target = group_count
    training_target = min(
        _MIN_JYNXZI_ROLLOUT_ROWS,
        max(_MIN_JYNXZI_INVENTORY_ROLLOUT_ROWS, group_count * _JYNXZI_INVENTORY_ROWS_PER_GROUP),
    )
    positive_target = min(
        _MIN_JYNXZI_ROLLOUT_POSITIVES,
        max(_MIN_JYNXZI_INVENTORY_ROLLOUT_POSITIVES, group_count * _JYNXZI_INVENTORY_POSITIVES_PER_GROUP),
    )
    return {
        "training_rows": training_target,
        "positive_rows": positive_target,
        "validation_groups": validation_target,
        "mode": "inventory_calibrated",
        "raw_targets": dict(raw_targets),
        "available_groups": group_count,
        "calibration": {
            "rows_per_available_group": _JYNXZI_INVENTORY_ROWS_PER_GROUP,
            "positives_per_available_group": _JYNXZI_INVENTORY_POSITIVES_PER_GROUP,
            "validation_group_share": _JYNXZI_INVENTORY_VALIDATION_GROUP_SHARE,
            "minimum_training_rows": _MIN_JYNXZI_INVENTORY_ROLLOUT_ROWS,
            "minimum_positive_rows": _MIN_JYNXZI_INVENTORY_ROLLOUT_POSITIVES,
            "minimum_validation_groups": _MIN_JYNXZI_INVENTORY_VALIDATION_GROUPS,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a simple linear ranker from candidate feedback.")
    parser.add_argument("--storage-dir", default=str(BACKEND_ROOT / "tmp" / "storage"))
    parser.add_argument("--profile", default="gaming")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--artifact-dir", default=str(BACKEND_ROOT / "ml" / "artifacts"))
    parser.add_argument("--report-out", default="")
    parser.add_argument(
        "--allow-high-leakage-risk",
        action="store_true",
        help="Proceed with training even when the data split is flagged as high leakage risk (stratified fallback, no group holdout).",
    )
    args = parser.parse_args()

    storage_dir = pathlib.Path(args.storage_dir)
    dataset_path = pathlib.Path(args.dataset) if args.dataset else storage_dir / "feedback" / "training_rows.jsonl"
    if not args.dataset:
        export_training_rows(storage_dir, dataset_path)

    rows = _read_jsonl(dataset_path)
    report_path = pathlib.Path(args.report_out) if args.report_out else None
    return train_from_rows(
        rows,
        args.profile,
        pathlib.Path(args.artifact_dir),
        report_path=report_path,
        allow_high_leakage_risk=args.allow_high_leakage_risk,
    )


def train_from_dataset(
    dataset_path: pathlib.Path,
    profile: str,
    artifact_dir: pathlib.Path,
    report_path: pathlib.Path | None = None,
    *,
    allow_high_leakage_risk: bool = False,
) -> int:
    rows = _read_jsonl(dataset_path)
    return train_from_rows(
        rows,
        profile,
        artifact_dir,
        report_path=report_path,
        allow_high_leakage_risk=allow_high_leakage_risk,
    )


def train_from_rows(
    rows: list[dict],
    profile: str,
    artifact_dir: pathlib.Path,
    *,
    report_path: pathlib.Path | None = None,
    allow_high_leakage_risk: bool = False,
) -> int:
    if not rows:
        print("No training rows found.")
        return 1
    original_row_count = len(rows)
    rows, training_row_policy = _prepare_ranker_training_rows(rows)
    if not rows:
        print("No training rows remain after ranker training policy filters.")
        return 1

    with mlflow_tracking.start_training_session(profile=profile, rows=rows) as tracking:
        gold_guard = _gold_eval_training_guard(rows)
        if gold_guard is not None:
            print(
                "Training aborted: "
                f"{gold_guard['reason']} "
                f"(gold_rows={gold_guard['gold_rows']}, affected_rows={gold_guard['affected_rows']})."
            )
            tracking.set_tags({"training_status": "aborted", "abort_reason": gold_guard["reason"]})
            return 1

        positive = [row for row in rows if row["label"] == 1]
        negative = [row for row in rows if row["label"] == 0]
        if not positive or not negative:
            print("Need both positive and negative examples.")
            tracking.set_tags({"training_status": "aborted", "abort_reason": "missing_positive_or_negative_rows"})
            return 1

        split = _split_rows(rows)
        train_rows = split["train"]
        validation_rows = split["validation"]
        if not train_rows or not validation_rows:
            print("Need both train and validation rows.")
            tracking.set_tags({"training_status": "aborted", "abort_reason": "missing_train_or_validation_rows"})
            return 1
        split_unit = str(split.get("unit", ""))
        validation_groups_count = len(set(_ranking_group_labels(validation_rows, split_unit)))
        available_groups_count = len(set(_ranking_group_labels(rows, split_unit)))
        min_validation_groups = int(
            rollout_readiness_targets(profile, available_groups=available_groups_count).get("validation_groups") or 0
        )
        if min_validation_groups and validation_groups_count < min_validation_groups:
            print(
                "Training warning: validation group count below rollout target "
                f"(groups={validation_groups_count}, required={min_validation_groups}, profile={profile})."
            )

        leakage_risk = str(split.get("leakage_risk", "unknown")).lower()
        split_warning = str(split.get("warning", "")).strip()
        if split_warning:
            print(
                json.dumps(
                    {
                        "event": "train_ranker_split_warning",
                        "warning": split_warning,
                        "strategy": split.get("strategy", ""),
                        "unit": split.get("unit", ""),
                        "seed": _SPLIT_RANDOM_SEED,
                    },
                    ensure_ascii=True,
                )
            )
        if leakage_risk == "high" and not allow_high_leakage_risk:
            print(
                "Training aborted: data split has leakage_risk=high "
                f"(strategy={split.get('strategy', 'unknown')}, unit={split.get('unit', 'unknown')}). "
                "Train/validation examples likely come from the same creator/VOD, which produces "
                "over-optimistic validation metrics. Pass --allow-high-leakage-risk to override."
            )
            tracking.set_tags({"training_status": "aborted", "abort_reason": "leakage_risk_high"})
            return 1

        train_positive = [row["features"] for row in train_rows if row["label"] == 1]
        train_negative = [row["features"] for row in train_rows if row["label"] == 0]
        if not train_positive or not train_negative:
            print("Need both positive and negative training examples.")
            tracking.set_tags({"training_status": "aborted", "abort_reason": "missing_positive_or_negative_training_rows"})
            return 1

        available_training_feature_names = _numeric_feature_names_from_rows([row["features"] for row in train_rows])
        feature_names, feature_selection = _select_training_feature_names(
            train_rows,
            available_training_feature_names,
            max_features=_ranker_max_training_features(),
        )
        available_numeric_feature_names = _numeric_feature_names_from_rows([row["features"] for row in rows])
        ignored_numeric_feature_names = sorted(set(available_numeric_feature_names) - set(feature_names))
        if ignored_numeric_feature_names:
            logger.warning(
                "ranker_numeric_features_not_in_training_vector count=%s names=%s",
                len(ignored_numeric_feature_names),
                ",".join(ignored_numeric_feature_names[:50]),
            )
        scales = _compute_scales(train_rows, feature_names)
        train_matrix = _feature_matrix(train_rows, feature_names, scales)
        train_labels = np.array([int(row["label"]) for row in train_rows], dtype=int)
        train_weights = np.array([_sample_weight(row) for row in train_rows], dtype=float)
        validation_matrix = _feature_matrix(validation_rows, feature_names, scales)
        validation_labels = np.array([int(row["label"]) for row in validation_rows], dtype=int)
        validation_groups = _ranking_group_labels(validation_rows, split.get("unit", ""))

        linear_weights = _apply_sign_constraints(_mean_difference_weights(train_rows, feature_names, scales))
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

        hpo_artifact = _load_hpo_artifact(profile, artifact_dir)
        lightgbm_ranker_candidate = _fit_lightgbm_ranker(
            train_rows,
            validation_rows,
            feature_names,
            scales,
            split.get("unit", ""),
            hpo_params=_best_hpo_params(hpo_artifact),
            random_state=int(_safe_float((hpo_artifact or {}).get("seed")) or 0),
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
                "weights": _apply_sign_constraints({
                    name: float(value)
                    for name, value in zip(feature_names, logistic.coef_[0], strict=False)
                }),
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
        selected_model = _select_model(model_candidates, training_row_count=len(rows))
        blend_calibration = _calibrate_blend(
            validation_rows,
            profile,
            feature_names,
            scales,
            selected_model,
            split.get("unit", ""),
        )
        serving_score_array = _runtime_blended_scores(
            validation_rows,
            profile,
            feature_names,
            scales,
            selected_model,
            float(blend_calibration["best_blend"]),
        )
        training_score_array = _runtime_blended_scores(
            train_rows,
            profile,
            feature_names,
            scales,
            selected_model,
            float(blend_calibration["best_blend"]),
        )
        serving_threshold_calibration = _calibrate_serving_threshold(
            validation_labels,
            serving_score_array,
            groups=validation_groups,
            source="validation",
        )
        filter_gate = _filter_gate_from_calibration(serving_threshold_calibration)
        _apply_bucket_filter_gates(
            filter_gate,
            validation_rows,
            serving_score_array,
            ranking_metric_value=float(
                selected_model["metrics"].get(
                    "ranking_metric_value",
                    selected_model["metrics"].get("ndcg_at_5", 0.0),
                )
                or 0.0
            ),
        )

        model_path = None
        if selected_model["artifact_model_type"] in {"gradient_boosting", "lightgbm_ranker"}:
            model_path = artifact_dir / f"{profile}_ranker_model.joblib"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            tmp_model_path = model_path.with_suffix(model_path.suffix + ".tmp")
            joblib.dump(selected_model["fitted_model"], tmp_model_path)
            tmp_model_path.replace(model_path)

        candidate_source_mix = _candidate_source_mix(rows, "candidate_source")
        artifact = {
            "version": _ARTIFACT_VERSION,
            "artifact_family": "ranker_v2",
            "profile": profile,
            "model_type": selected_model["artifact_model_type"],
            "selected_model": selected_model["name"],
            "score_mode": str(selected_model.get("score_mode") or "absolute"),
            "selection_metric": "composite_validation_score",
            "label_schema_version": 3,
            "feature_schema_version": _FEATURE_SCHEMA_VERSION,
            "feature_names": feature_names,
            "available_numeric_feature_names": available_numeric_feature_names,
            "ignored_numeric_feature_names": ignored_numeric_feature_names,
            "feature_selection": feature_selection,
            "excluded_feature_names": sorted(
                OFFLINE_ONLY_FEATURE_NAMES
                | _SPARSE_TRAINING_FEATURE_NAMES
                | _CONFOUND_TRAINING_FEATURE_NAMES
                | {_HEURISTIC_FEATURE_NAME}
            ),
            "semantic_score_config": semantic_service.semantic_score_config(),
            "scope": "single_streamer" if profile == "jynxzi" else "profile",
            "valid_for": [profile],
            "weights": selected_model.get("weights", {}),
            "scales": scales,
            "intercept": selected_model.get("intercept", 0.0),
            "model_path": model_path.name if model_path else "",
            "blend": float(blend_calibration["best_blend"]),
            "blend_calibration": blend_calibration,
            "serving_threshold": float(serving_threshold_calibration["threshold"]),
            "serving_threshold_calibration": serving_threshold_calibration,
            "training_score_distribution": _score_distribution_summary(training_score_array),
            "validation_score_distribution": _score_distribution_summary(serving_score_array),
            "filter_gate": filter_gate,
            "hpo": _artifact_hpo_summary(hpo_artifact),
            "top_feature_importances": _feature_importance_map(selected_model, feature_names),
            "training_stats": {
                "rows": len(rows),
                "input_rows": original_row_count,
                "positive_rows": len(positive),
                "negative_rows": len(negative),
                "train_rows": len(train_rows),
                "validation_rows": len(validation_rows),
                "split_strategy": split["strategy"],
                "split_unit": split.get("unit", ""),
                "leakage_risk": split.get("leakage_risk", "unknown"),
                "split_strategy_warning": split_warning,
                "split_seed": _SPLIT_RANDOM_SEED,
                "supervision_tiers": _count_values(rows, "supervision_tier"),
                "dataset_sources": _count_values(rows, "dataset_source"),
                "platforms": _count_values(rows, "publish_platform"),
                "variants": _count_values(rows, "final_variant_key"),
                "content_buckets": _count_values(rows, "content_bucket"),
                "candidate_sources": candidate_source_mix["by_value"],
                "candidate_source_peak_share": candidate_source_mix["peak_share"],
                "training_row_policy": training_row_policy,
            },
            "split_strategy_warning": split_warning,
            "validation": selected_model["metrics"],
            "model_comparison": {
                candidate["name"]: candidate["metrics"]
                for candidate in model_candidates
            },
            "model_selection": _model_selection_report(model_candidates, selected_model, len(rows)),
            "rollout_readiness": _rollout_readiness_summary(
                profile,
                rows,
                validation_rows,
                selected_model,
                filter_gate,
                split.get("unit", ""),
            ),
            "bucketed_validation": _bucketed_validation(
                selected_model["artifact_model_type"],
                validation_rows,
                feature_names,
                scales,
                selected_model,
                split.get("unit", ""),
            ),
            "mlflow": {
                "enabled": tracking.enabled,
                "tracking_uri": mlflow_tracking.tracking_uri(),
                "experiment_name": mlflow_tracking.experiment_name() if tracking.enabled else "",
                "run_id": tracking.run_id,
            },
            "mlflow_run_id": tracking.run_id,
        }
        artifact_health = _artifact_health_summary(artifact)

        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / f"{profile}_ranker.json"
        tmp_artifact_path = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
        tmp_artifact_path.write_text(json.dumps(artifact, ensure_ascii=True, indent=2), encoding="utf-8")
        tmp_artifact_path.replace(artifact_path)
        artifact_health_path = artifact_dir / f"{profile}_artifact_health.json"
        tmp_artifact_health_path = artifact_health_path.with_suffix(artifact_health_path.suffix + ".tmp")
        tmp_artifact_health_path.write_text(json.dumps(artifact_health, ensure_ascii=True, indent=2), encoding="utf-8")
        tmp_artifact_health_path.replace(artifact_health_path)

        tracking.log_metrics(
            {
                "ndcg_at_5": artifact["validation"].get("ndcg_at_5"),
                "f1": artifact["validation"].get("f1"),
                "accuracy": artifact["validation"].get("accuracy"),
                "precision_at_5": artifact["validation"].get("precision_at_5"),
                "recall_at_5": artifact["validation"].get("recall_at_5"),
            }
        )
        tracking.set_tags(
            {
                "training_status": "succeeded",
                "selected_model": selected_model["name"],
                "dataset_rows": len(rows),
                "positive_rows": len(positive),
            }
        )
        tracking.log_artifact(artifact_path)
        tracking.log_artifact(artifact_health_path)
        registry_result = {"enabled": False}
        if str(os.environ.get("MLFLOW_REGISTER_ON_TRAIN") or "1").strip() != "0":
            registry_result = tracking.register_ranker_bundle(
                profile=profile,
                artifact_path=artifact_path,
                artifact_health_path=artifact_health_path,
                model_path=model_path,
                positive_rows=len(positive),
                candidate_source_peak_share=float(candidate_source_mix["peak_share"]),
                validation_ndcg_at_5=float(artifact["validation"].get("ndcg_at_5", 0.0) or 0.0),
            )
        if tracking.enabled and registry_result.get("enabled"):
            artifact["mlflow"]["model_name"] = registry_result.get("model_name", "")
            artifact["mlflow"]["registered_version"] = registry_result.get("version", "")
            artifact["mlflow"]["registered_stage"] = registry_result.get("stage", "")
            artifact["mlflow"]["promotion_gates"] = registry_result.get("promotion_gates", {})
            tmp_artifact_path = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
            tmp_artifact_path.write_text(json.dumps(artifact, ensure_ascii=True, indent=2), encoding="utf-8")
            tmp_artifact_path.replace(artifact_path)

        if report_path is not None:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(artifact, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"Wrote artifact to {artifact_path}")
        print(f"Wrote artifact health to {artifact_health_path}")
        print(
            json.dumps(
                {
                    "selected_model": selected_model["name"],
                    "validation": artifact["validation"],
                    "training_stats": artifact["training_stats"],
                    "model_comparison": artifact["model_comparison"],
                    "mlflow": artifact.get("mlflow", {}),
                },
                ensure_ascii=True,
            )
        )
        return 0


def _artifact_health_summary(artifact: dict[str, object]) -> dict[str, object]:
    training_stats = dict(artifact.get("training_stats") or {})
    validation = dict(artifact.get("validation") or {})
    split_warning = str(training_stats.get("split_strategy_warning") or artifact.get("split_strategy_warning") or "").strip()
    warnings = [split_warning] if split_warning else []
    ignored_numeric_feature_names = sorted(
        str(name)
        for name in (artifact.get("ignored_numeric_feature_names") or [])
        if str(name).strip()
    )
    if ignored_numeric_feature_names:
        warnings.append("ranker_ignored_numeric_features")
    return {
        "version": int(artifact.get("version") or 0),
        "profile": str(artifact.get("profile") or ""),
        "artifact_family": str(artifact.get("artifact_family") or ""),
        "selected_model": str(artifact.get("selected_model") or ""),
        "model_type": str(artifact.get("model_type") or ""),
        "feature_schema_version": int(artifact.get("feature_schema_version") or 0),
        "feature_names_count": len(list(artifact.get("feature_names") or [])),
        "available_numeric_feature_names_count": len(list(artifact.get("available_numeric_feature_names") or [])),
        "ignored_numeric_feature_names": ignored_numeric_feature_names,
        "label_schema_version": int(artifact.get("label_schema_version") or 0),
        "semantic_score_config": artifact.get("semantic_score_config") or {},
        "scope": str(artifact.get("scope") or ""),
        "valid_for": list(artifact.get("valid_for") or []),
        "serving_threshold": float(artifact.get("serving_threshold") or _DEFAULT_MARGIN_CLASSIFICATION_THRESHOLD),
        "serving_threshold_calibration": artifact.get("serving_threshold_calibration") or {},
        "training_score_distribution": artifact.get("training_score_distribution") or {},
        "validation_score_distribution": artifact.get("validation_score_distribution") or {},
        "score_distribution_drift": _score_distribution_drift_summary(artifact),
        "filter_gate": artifact.get("filter_gate") or {},
        "excluded_feature_names": sorted(
            str(name)
            for name in (artifact.get("excluded_feature_names") or [])
            if str(name).strip()
        ),
        "warnings": warnings,
        "gates": {
            "split_strategy": str(training_stats.get("split_strategy") or ""),
            "split_unit": str(training_stats.get("split_unit") or ""),
            "leakage_risk": str(training_stats.get("leakage_risk") or ""),
            "ranking_groups": int(validation.get("ranking_groups") or 0),
            "ranking_metric": str(validation.get("ranking_metric") or ""),
            "ranking_metric_value": float(validation.get("ranking_metric_value") or 0.0),
            "ranking_metric_min": float(validation.get("ranking_metric_min") or 0.0),
            "ranking_metric_p25": float(validation.get("ranking_metric_p25") or 0.0),
            "min_group_positive_rows": int(validation.get("min_group_positive_rows") or 0),
            "ndcg_at_5": float(validation.get("ndcg_at_5") or 0.0),
            "ndcg_at_5_min": float(validation.get("ndcg_at_5_min") or 0.0),
            "ndcg_at_5_p25": float(validation.get("ndcg_at_5_p25") or 0.0),
            "ndcg_at_10": float(validation.get("ndcg_at_10") or 0.0),
            "mrr": float(validation.get("mrr") or 0.0),
            "precision_at_5": float(validation.get("precision_at_5") or 0.0),
            "accuracy": float(validation.get("accuracy") or 0.0),
            "f1": float(validation.get("f1") or 0.0),
            "filter_precision": float((artifact.get("filter_gate") or {}).get("precision") or 0.0),
            "filter_recall": float((artifact.get("filter_gate") or {}).get("recall") or 0.0),
            "filter_f1": float((artifact.get("filter_gate") or {}).get("f1") or 0.0),
            "filter_selected_rate": float((artifact.get("filter_gate") or {}).get("selected_rate") or 0.0),
            "filter_gate_passed": bool((artifact.get("filter_gate") or {}).get("passed")),
        },
        "training_stats": training_stats,
        "validation": validation,
        "rollout_readiness": artifact.get("rollout_readiness") or {},
        "model_comparison": artifact.get("model_comparison") or {},
    }


def _score_distribution_summary(scores: np.ndarray | list[float]) -> dict[str, float | int]:
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "rows": 0,
            "min": 0.0,
            "p01": 0.0,
            "p05": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "clamp_fraction": 0.0,
        }
    abs_values = np.abs(values)
    return {
        "rows": int(values.size),
        "min": round(float(np.min(values)), 6),
        "p01": round(float(np.percentile(values, 1)), 6),
        "p05": round(float(np.percentile(values, 5)), 6),
        "p25": round(float(np.percentile(values, 25)), 6),
        "p50": round(float(np.percentile(values, 50)), 6),
        "p75": round(float(np.percentile(values, 75)), 6),
        "p95": round(float(np.percentile(values, 95)), 6),
        "p99": round(float(np.percentile(values, 99)), 6),
        "max": round(float(np.max(values)), 6),
        "mean": round(float(np.mean(values)), 6),
        "std": round(float(np.std(values)), 6),
        "clamp_fraction": round(float(np.mean(abs_values >= _TRAINED_SCORE_CLAMP_ABS * 0.95)), 6),
    }


def _score_distribution_drift_summary(artifact: dict[str, object]) -> dict[str, object]:
    reference = dict(artifact.get("training_score_distribution") or {})
    distribution = dict(artifact.get("validation_score_distribution") or {})
    filter_gate = dict(artifact.get("filter_gate") or {})
    warnings: list[str] = []
    clamp_fraction = float(distribution.get("clamp_fraction") or 0.0)
    selected_rate = float(filter_gate.get("selected_rate") or 0.0)
    ks_statistic = _distribution_ks_from_quantiles(reference, distribution)
    psi = _distribution_psi_from_quantiles(reference, distribution)
    if clamp_fraction >= 0.05:
        warnings.append("score_clamp_fraction_high")
    if selected_rate >= _FILTER_GATE_MAX_SELECTED_RATE:
        warnings.append("selected_rate_at_or_above_gate")
    if ks_statistic >= 0.25:
        warnings.append("score_distribution_ks_drift")
    if psi >= 0.20:
        warnings.append("score_distribution_psi_drift")
    return {
        "reference": reference,
        "baseline": distribution,
        "ks_test_available": bool(reference and distribution),
        "ks_statistic": round(float(ks_statistic), 6),
        "psi": round(float(psi), 6),
        "warnings": warnings,
    }


def _distribution_ks_from_quantiles(reference: dict[str, object], current: dict[str, object]) -> float:
    if not reference or not current:
        return 0.0
    max_delta = 0.0
    for key in ("p01", "p05", "p25", "p50", "p75", "p95", "p99"):
        ref = float(reference.get(key) or 0.0)
        cur = float(current.get(key) or 0.0)
        scale = max(float(reference.get("std") or 0.0), float(current.get("std") or 0.0), 1.0)
        max_delta = max(max_delta, abs(cur - ref) / scale)
    return min(max_delta, 1.0)


def _distribution_psi_from_quantiles(reference: dict[str, object], current: dict[str, object]) -> float:
    if not reference or not current:
        return 0.0
    ref_std = max(float(reference.get("std") or 0.0), 1.0)
    cur_std = max(float(current.get("std") or 0.0), 1.0)
    mean_delta = abs(float(current.get("mean") or 0.0) - float(reference.get("mean") or 0.0)) / max(ref_std, cur_std)
    std_delta = abs(cur_std - ref_std) / max(ref_std, cur_std)
    clamp_delta = abs(float(current.get("clamp_fraction") or 0.0) - float(reference.get("clamp_fraction") or 0.0))
    return min(mean_delta * 0.5 + std_delta * 0.35 + clamp_delta * 0.15, 5.0)


def _rollout_readiness_summary(
    profile: str,
    rows: list[dict],
    validation_rows: list[dict],
    selected_model: dict,
    filter_gate: dict,
    split_unit: str,
) -> dict[str, object]:
    validation_groups = len(set(_ranking_group_labels(validation_rows, str(split_unit))))
    available_groups = len(set(_ranking_group_labels(rows, str(split_unit))))
    positive_rows = sum(1 for row in rows if int(row.get("label", 0)) == 1)
    profile_key = str(profile).strip().lower()
    targets = rollout_readiness_targets(profile, available_groups=available_groups)
    training_target = int(targets.get("training_rows") or 0)
    positive_target = int(targets.get("positive_rows") or 0)
    validation_group_target = int(targets.get("validation_groups") or 0)
    failures: list[str] = []
    if str(selected_model.get("name") or "") == "fallback_heuristic":
        failures.append("selected_model_is_fallback_heuristic")
    if not bool(filter_gate.get("passed")):
        failures.append("filter_gate_failed")
    if profile_key == "jynxzi":
        if len(rows) < training_target:
            failures.append("training_rows_below_rollout_target")
        if positive_rows < positive_target:
            failures.append("positive_rows_below_rollout_target")
        if validation_groups < validation_group_target:
            failures.append("validation_groups_below_rollout_target")
    return {
        "passed": not failures,
        "failures": failures,
        "scope": "single_streamer" if profile_key == "jynxzi" else "profile",
        "valid_for": [profile],
        "training_rows": len(rows),
        "positive_rows": positive_rows,
        "validation_groups": validation_groups,
        "available_groups": available_groups,
        "targets": targets,
    }


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


def _gold_eval_training_guard(rows: list[dict]) -> dict[str, int | str] | None:
    gold_rows = [
        row
        for row in rows
        if str(row.get("supervision_tier") or "").strip().lower() == "gold_eval"
        or str(row.get("label_source") or "").strip().lower() == "gold_eval"
    ]
    if not gold_rows:
        return None

    missing_split = [
        row for row in gold_rows
        if str(row.get(GOLD_HOLDOUT_SPLIT_FIELD) or "").strip().lower() not in {"train", "holdout"}
    ]
    if missing_split:
        return {
            "reason": "gold_eval_rows_missing_frozen_holdout_split",
            "gold_rows": len(gold_rows),
            "affected_rows": len(missing_split),
        }

    holdout_rows = [
        row for row in gold_rows
        if str(row.get(GOLD_HOLDOUT_SPLIT_FIELD) or "").strip().lower() == "holdout"
    ]
    if holdout_rows:
        return {
            "reason": "gold_eval_holdout_rows_present_in_training_dataset",
            "gold_rows": len(gold_rows),
            "affected_rows": len(holdout_rows),
        }
    return None


def _numeric_feature_names(feature_row: dict) -> list[str]:
    _excluded = (
        {"candidate_id", "source", _HEURISTIC_FEATURE_NAME}
        | excluded_training_feature_names()
    )
    names: list[str] = []
    for key, value in feature_row.items():
        if key in _excluded:
            continue
        if isinstance(value, (int, float, bool)):
            names.append(key)
    return sorted(names)


def _numeric_feature_names_from_rows(feature_rows: list[dict]) -> list[str]:
    names: set[str] = set()
    for feature_row in feature_rows:
        if isinstance(feature_row, dict):
            names.update(_numeric_feature_names(feature_row))
    return sorted(names)


def _ranker_max_training_features() -> int:
    raw = (os.environ.get("RANKER_MAX_FEATURES") or str(_DEFAULT_MAX_TRAINING_FEATURES)).strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_MAX_TRAINING_FEATURES


def _select_training_feature_names(
    rows: list[dict],
    feature_names: list[str],
    *,
    max_features: int,
) -> tuple[list[str], dict[str, object]]:
    if max_features <= 0 or len(feature_names) <= max_features:
        return list(feature_names), {
            "enabled": False,
            "max_features": int(max_features),
            "method": "supervised_univariate_effect_size",
            "input_feature_count": len(feature_names),
            "selected_feature_count": len(feature_names),
            "dropped_feature_count": 0,
            "dropped_feature_names": [],
        }

    scored = [
        (score, name)
        for name in feature_names
        if (score := _feature_selection_score(rows, name)) > 0.0
    ]
    scored_by_name = {name: score for score, name in scored}
    preferred = [
        name
        for name in sorted(_JUDGE_TEACHER_FEATURE_NAMES & set(feature_names))
        if scored_by_name.get(name, 0.0) > 0.0
    ]
    if len(preferred) > max_features:
        preferred = [
            name
            for _score, name in sorted(
                ((scored_by_name[name], name) for name in preferred),
                key=lambda item: (-item[0], item[1]),
            )[:max_features]
        ]
    if not scored:
        selected = list(feature_names[:max_features])
    else:
        selected = list(preferred)
        remaining_slots = max(0, int(max_features) - len(selected))
        preferred_set = set(preferred)
        selected.extend([
            name
            for _score, name in sorted(scored, key=lambda item: (-item[0], item[1]))
            if name not in preferred_set
        ][:remaining_slots])
    selected_set = set(selected)
    dropped = sorted(name for name in feature_names if name not in selected_set)
    return sorted(selected), {
        "enabled": True,
        "max_features": int(max_features),
        "method": "supervised_univariate_effect_size_with_judge_teacher_priority",
        "input_feature_count": len(feature_names),
        "selected_feature_count": len(selected),
        "dropped_feature_count": len(dropped),
        "dropped_feature_names": dropped,
        "priority_feature_names": preferred,
        "top_feature_scores": [
            {"name": name, "score": round(float(score), 6)}
            for score, name in sorted(scored, key=lambda item: (-item[0], item[1]))[: min(20, len(scored))]
        ],
    }


def _feature_selection_score(rows: list[dict], name: str) -> float:
    positive: list[float] = []
    negative: list[float] = []
    for row in rows:
        value = _safe_float((row.get("features") or {}).get(name))
        if int(row.get("label", 0)) == 1:
            positive.append(value)
        else:
            negative.append(value)
    if not positive or not negative:
        return 0.0
    pos_mean = float(np.mean(positive))
    neg_mean = float(np.mean(negative))
    pooled_std = float(np.std(positive + negative)) or 1.0
    prevalence = sum(1 for value in positive + negative if abs(float(value)) > 1e-9) / max(len(positive) + len(negative), 1)
    return abs(pos_mean - neg_mean) / pooled_std * math.sqrt(max(prevalence, 0.05))


def _split_rows(rows: list[dict]) -> dict[str, list[dict] | str]:
    gold_train_rows = [
        row
        for row in rows
        if _is_gold_eval_row(row)
        and str(row.get(GOLD_HOLDOUT_SPLIT_FIELD) or "").strip().lower() == "train"
    ]
    if gold_train_rows and len(gold_train_rows) < len(rows):
        non_gold_rows = [row for row in rows if row not in gold_train_rows]
        split = _split_rows_without_gold_override(non_gold_rows)
        validation_rows = list(split.get("validation") or [])
        if _split_has_both_classes(validation_rows):
            train_rows = list(split.get("train") or [])
            train_rows.extend(gold_train_rows)
            split["train"] = train_rows
            split["gold_train_rows_forced_to_train"] = len(gold_train_rows)
            return split

    return _split_rows_without_gold_override(rows)


def _split_rows_without_gold_override(rows: list[dict]) -> dict[str, list[dict] | str]:
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

    ordered_groups = sorted(grouped.items(), key=_holdout_group_order_key)
    target_validation_size = max(2, len(rows) // 5)
    target_validation_groups = 2
    train: list[dict] = []
    validation: list[dict] = []
    validation_positive = 0
    validation_negative = 0
    validation_group_count = 0

    for _, group_rows in ordered_groups:
        group_positive = sum(1 for row in group_rows if int(row["label"]) == 1)
        group_negative = len(group_rows) - group_positive
        put_in_validation = (
            len(validation) < target_validation_size
            or validation_positive == 0
            or validation_negative == 0
            or validation_group_count < target_validation_groups
        )
        bucket = validation if put_in_validation else train
        bucket.extend(group_rows)
        if put_in_validation:
            validation_positive += group_positive
            validation_negative += group_negative
            validation_group_count += 1

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


def _holdout_group_order_key(item: tuple[str, list[dict]]) -> tuple[int, int, str]:
    key, group_rows = item
    positive = sum(1 for row in group_rows if int(row["label"]) == 1)
    negative = len(group_rows) - positive
    mixed_rank = 0 if positive > 0 and negative > 0 else 1
    return mixed_rank, len(group_rows), key


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
            "warning": "stratified_fallback_without_both_classes",
            "seed": _SPLIT_RANDOM_SEED,
        }

    def split_bucket(bucket: list[dict]) -> tuple[list[dict], list[dict]]:
        bucket = _seeded_row_order(bucket)
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
        "warning": "stratified_fallback_leakage_risk_high",
        "seed": _SPLIT_RANDOM_SEED,
    }


def _split_has_both_classes(rows: list[dict]) -> bool:
    labels = {int(row["label"]) for row in rows if row.get("label") in {0, 1}}
    return labels == {0, 1}


def _is_gold_eval_row(row: dict) -> bool:
    return (
        str(row.get("supervision_tier") or "").strip().lower() == "gold_eval"
        or str(row.get("label_source") or "").strip().lower() == "gold_eval"
    )


def _validated_split(rows: list[dict], split: dict[str, list[dict] | str]) -> dict[str, list[dict] | str]:
    train_rows = list(split.get("train") or [])
    validation_rows = list(split.get("validation") or [])
    if _split_has_both_classes(train_rows) and _split_has_both_classes(validation_rows):
        return split

    fallback_strategy = f"{split.get('strategy', 'split')}_fallback_stratified"
    fallback = _stratified_fallback_split(rows, strategy=fallback_strategy, unit="row")
    fallback["warning"] = (
        f"stratified fallback from {split.get('strategy', 'split')} "
        f"(unit={split.get('unit', '')}) because one split lost a class"
    )
    fallback["fallback_from_strategy"] = str(split.get("strategy", ""))
    fallback["fallback_from_unit"] = str(split.get("unit", ""))
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
    train_rows: list[dict],
    feature_names: list[str],
    scales: dict[str, dict[str, float]],
) -> dict[str, float]:
    positive_rows = [row for row in train_rows if int(row["label"]) == 1]
    negative_rows = [row for row in train_rows if int(row["label"]) == 0]
    weights: dict[str, float] = {}
    for name in feature_names:
        pos_vals = [float(row["features"].get(name, 0.0)) for row in positive_rows]
        neg_vals = [float(row["features"].get(name, 0.0)) for row in negative_rows]
        pos_weights = [_sample_weight(row) for row in positive_rows]
        neg_weights = [_sample_weight(row) for row in negative_rows]
        std = float(scales.get(name, {}).get("std", 1.0)) or 1.0
        positive_mean = _weighted_mean(pos_vals, pos_weights)
        negative_mean = _weighted_mean(neg_vals, neg_weights)
        weights[name] = (positive_mean - negative_mean) / std
    return weights


def _apply_sign_constraints(weights: dict[str, float]) -> dict[str, float]:
    constrained = dict(weights)
    for name in _POSITIVE_SIGN_FEATURES:
        if name in constrained and constrained[name] < 0.0:
            constrained[name] = 0.0
    return constrained


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
    weights = _apply_sign_constraints(weights)
    intercept = float(model.intercept_[0])
    validation_scores = _pairwise_median_raw_scores(
        validation_rows,
        feature_names,
        weights,
        scales,
        intercept,
        split_unit,
    )
    metrics = _evaluate_scores(
        np.array([int(row["label"]) for row in validation_rows], dtype=int),
        validation_scores,
        groups=_ranking_group_labels(validation_rows, split_unit),
    )
    return {
        "name": "pairwise_logistic_regression",
        "artifact_model_type": "pairwise_logistic",
        "score_mode": "pairwise_median",
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
    *,
    hpo_params: dict | None = None,
    random_state: int = 0,
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

    resolved_params = _resolve_lightgbm_params(hpo_params, feature_names=feature_names, random_state=random_state)
    model = LGBMRanker(**resolved_params)
    try:
        callbacks = [early_stopping(10, verbose=False)] if early_stopping is not None else []
        model.fit(
            train_matrix,
            train_targets,
            group=train_groups,
            sample_weight=train_weights,
            eval_set=[(validation_matrix, validation_targets)],
            eval_group=[validation_groups],
            eval_at=[3, 5, 10],
            callbacks=callbacks,
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
        "params": resolved_params,
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


def _pairwise_median_raw_scores(
    rows: list[dict],
    feature_names: list[str],
    weights: dict[str, float],
    scales: dict[str, dict[str, float]],
    intercept: float,
    split_unit: str,
) -> np.ndarray:
    if not rows:
        return np.array([], dtype=float)
    matrix = _feature_matrix(rows, feature_names, scales)
    groups = _ranking_group_labels(rows, split_unit)
    if not groups or len(groups) != len(rows):
        median = np.median(matrix, axis=0)
        deltas = matrix - median
    else:
        deltas = np.zeros_like(matrix)
        by_group: dict[str, list[int]] = {}
        for index, group in enumerate(groups):
            by_group.setdefault(str(group), []).append(index)
        for indices in by_group.values():
            group_matrix = matrix[indices]
            deltas[indices] = group_matrix - np.median(group_matrix, axis=0)
    vector = np.array([float(weights.get(name, 0.0)) for name in feature_names], dtype=float)
    return deltas @ vector + float(intercept)


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
    threshold: float | None = None,
) -> dict[str, float]:
    total = int(len(labels)) or 1
    classification_threshold = _default_classification_threshold(score_kind) if threshold is None else float(threshold)
    if score_kind == "probability":
        predicted = (scores >= classification_threshold).astype(int)
        ranking_scores = scores
    else:
        predicted = (scores >= classification_threshold).astype(int)
        ranking_scores = 1.0 / (1.0 + np.exp(-np.clip(scores, -20.0, 20.0)))

    true_positive = int(np.sum((predicted == 1) & (labels == 1)))
    false_positive = int(np.sum((predicted == 1) & (labels == 0)))
    false_negative = int(np.sum((predicted == 0) & (labels == 1)))
    true_negative = int(np.sum((predicted == 0) & (labels == 0)))
    selected_count = true_positive + false_positive
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
        "selected_count": selected_count,
        "selected_rate": round(float(selected_count / total), 3),
        "classification_threshold": round(float(classification_threshold), 6),
    }
    for k in (3, 5):
        precision_k, recall_k = _grouped_precision_recall_at_k(labels, ranking_scores, k, groups)
        metrics[f"precision_at_{k}"] = round(precision_k, 3)
        metrics[f"recall_at_{k}"] = round(recall_k, 3)
    ndcg_summary = _grouped_ndcg_summary_at_k(labels, ranking_scores, 5, groups)
    metrics["ndcg_at_5"] = round(ndcg_summary["mean"], 3)
    ndcg_10_summary = _grouped_ndcg_summary_at_k(labels, ranking_scores, 10, groups)
    metrics["ndcg_at_10"] = round(ndcg_10_summary["mean"], 3)
    mrr_summary = _grouped_mrr_summary(labels, ranking_scores, groups)
    metrics["mrr"] = round(mrr_summary["mean"], 3)
    if groups:
        metrics["ranking_groups"] = len({group for group in groups if str(group).strip()})
        metrics["ndcg_at_5_min"] = round(ndcg_summary["min"], 3)
        metrics["ndcg_at_5_p25"] = round(ndcg_summary["p25"], 3)
        metrics["ndcg_at_10_min"] = round(ndcg_10_summary["min"], 3)
        metrics["ndcg_at_10_p25"] = round(ndcg_10_summary["p25"], 3)
        metrics["mrr_min"] = round(mrr_summary["min"], 3)
        metrics["mrr_p25"] = round(mrr_summary["p25"], 3)
        for k in (3, 5):
            precision_k, recall_k = _precision_recall_at_k(labels, ranking_scores, k)
            metrics[f"global_precision_at_{k}"] = round(precision_k, 3)
            metrics[f"global_recall_at_{k}"] = round(recall_k, 3)
        metrics["global_ndcg_at_5"] = round(_ndcg_at_k(labels, ranking_scores, 5), 3)
        metrics["global_ndcg_at_10"] = round(_ndcg_at_k(labels, ranking_scores, 10), 3)
        metrics["global_mrr"] = round(_mean_reciprocal_rank(labels, ranking_scores), 3)
    ranking_metric = _select_ranking_metric(labels, groups)
    metrics["ranking_metric"] = ranking_metric
    metrics["ranking_metric_value"] = round(float(metrics.get(ranking_metric, 0.0)), 3)
    metrics["ranking_metric_min"] = round(float(metrics.get(f"{ranking_metric}_min", metrics["ranking_metric_value"])), 3)
    metrics["ranking_metric_p25"] = round(float(metrics.get(f"{ranking_metric}_p25", metrics["ranking_metric_value"])), 3)
    metrics["min_group_positive_rows"] = _min_group_positive_rows(labels, groups)
    metrics["rows"] = total
    return metrics


def _default_classification_threshold(score_kind: str) -> float:
    if str(score_kind).strip().lower() == "probability":
        return _DEFAULT_PROBABILITY_CLASSIFICATION_THRESHOLD
    return _DEFAULT_MARGIN_CLASSIFICATION_THRESHOLD


def _artifact_serving_threshold(artifact: dict | None) -> float | None:
    if not isinstance(artifact, dict) or "serving_threshold" not in artifact:
        return None
    try:
        threshold = float(artifact.get("serving_threshold"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(threshold):
        return None
    return threshold


def _calibrate_serving_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    groups: list[str] | None = None,
    source: str = "validation",
    score_kind: str = "margin",
    min_recall: float = _SERVING_THRESHOLD_MIN_RECALL,
    max_selected_rate: float = _SERVING_THRESHOLD_MAX_SELECTED_RATE,
    target_precision: float = _SERVING_THRESHOLD_TARGET_PRECISION,
) -> dict[str, object]:
    default_threshold = _default_classification_threshold(score_kind)
    default_metrics = _evaluate_scores(
        labels,
        scores,
        score_kind=score_kind,
        groups=groups,
        threshold=default_threshold,
    )
    if len(labels) == 0 or len(scores) == 0:
        return {
            "source": source,
            "metric": "max_recall_under_precision_target",
            "threshold_policy": "balanced_filter",
            "selected_threshold_reason": "empty_scores",
            "score_kind": score_kind,
            "min_recall": float(min_recall),
            "max_selected_rate": float(max_selected_rate),
            "target_precision": float(target_precision),
            "threshold": float(default_threshold),
            "metrics": default_metrics,
            "default_threshold": float(default_threshold),
            "default_metrics": default_metrics,
            "candidate_count": 0,
        }

    finite_scores = [float(score) for score in scores.tolist() if math.isfinite(float(score))]
    if not finite_scores:
        return {
            "source": source,
            "metric": "max_recall_under_precision_target",
            "threshold_policy": "balanced_filter",
            "selected_threshold_reason": "no_finite_scores",
            "score_kind": score_kind,
            "min_recall": float(min_recall),
            "max_selected_rate": float(max_selected_rate),
            "target_precision": float(target_precision),
            "threshold": float(default_threshold),
            "metrics": default_metrics,
            "default_threshold": float(default_threshold),
            "default_metrics": default_metrics,
            "candidate_count": 0,
        }

    epsilon = 1e-6
    candidates = {float(default_threshold), min(finite_scores) - epsilon, max(finite_scores) + epsilon}
    candidates.update(finite_scores)
    best_threshold = float(default_threshold)
    best_metrics = default_metrics
    best_key = _serving_threshold_key(best_metrics, best_threshold, min_recall, max_selected_rate, target_precision)
    for threshold_candidate in sorted(candidates):
        metrics = _evaluate_scores(
            labels,
            scores,
            score_kind=score_kind,
            groups=groups,
            threshold=float(threshold_candidate),
        )
        key = _serving_threshold_key(
            metrics,
            float(threshold_candidate),
            min_recall,
            max_selected_rate,
            target_precision,
        )
        if key > best_key:
            best_threshold = float(threshold_candidate)
            best_metrics = metrics
            best_key = key

    return {
        "source": source,
        "metric": "max_recall_under_precision_target",
        "threshold_policy": "balanced_filter",
        "selected_threshold_reason": _threshold_selection_reason(
            best_metrics,
            min_recall=min_recall,
            max_selected_rate=max_selected_rate,
            target_precision=target_precision,
        ),
        "score_kind": score_kind,
        "min_recall": float(min_recall),
        "max_selected_rate": float(max_selected_rate),
        "target_precision": float(target_precision),
        "threshold": round(float(best_threshold), 6),
        "metrics": best_metrics,
        "default_threshold": float(default_threshold),
        "default_metrics": default_metrics,
        "candidate_count": len(candidates),
    }


def _serving_threshold_key(
    metrics: dict[str, float],
    threshold: float,
    min_recall: float,
    max_selected_rate: float,
    target_precision: float,
) -> tuple:
    recall = float(metrics.get("recall", 0.0))
    precision = float(metrics.get("precision", 0.0))
    selected_rate = float(metrics.get("selected_rate", 1.0))
    f1 = float(metrics.get("f1", 0.0))
    f0_75 = _fbeta(precision, recall, _FILTER_GATE_BETA)
    hard_max_selected_rate = max(float(max_selected_rate), _SERVING_THRESHOLD_HARD_MAX_SELECTED_RATE)
    meets_recall = recall >= float(min_recall)
    meets_precision = precision >= float(target_precision)
    meets_gate_precision = precision >= _FILTER_GATE_MIN_PRECISION
    meets_target_selected_rate = selected_rate <= float(max_selected_rate)
    meets_hard_selected_rate = selected_rate <= hard_max_selected_rate
    fallback_max_selected_rate = min(
        hard_max_selected_rate,
        max(float(max_selected_rate), _SERVING_THRESHOLD_FALLBACK_MAX_SELECTED_RATE),
    )
    meets_fallback_selected_rate = selected_rate <= fallback_max_selected_rate
    meets_filter_constraints = 1 if meets_precision and meets_recall and meets_target_selected_rate else 0
    meets_recall_fallback = 1 if meets_recall and meets_fallback_selected_rate else 0
    meets_gate_filter_fallback = 1 if meets_recall and meets_gate_precision and meets_fallback_selected_rate else 0
    meets_gate_filter_hard_cap = (
        1
        if meets_gate_precision
        and recall >= _FILTER_GATE_MIN_RECALL
        and f1 >= _FILTER_GATE_MIN_F1
        and meets_hard_selected_rate
        else 0
    )
    precision_gap = max(0.0, float(target_precision) - precision)
    recall_gap = max(0.0, float(min_recall) - recall)
    hard_selected_rate_gap = max(0.0, selected_rate - hard_max_selected_rate)
    fallback_selected_rate_gap = max(0.0, selected_rate - fallback_max_selected_rate)
    return (
        1 if meets_hard_selected_rate else 0,
        meets_filter_constraints,
        recall if meets_filter_constraints else 0.0,
        meets_gate_filter_hard_cap,
        meets_gate_filter_fallback,
        meets_recall_fallback,
        f0_75,
        recall,
        f1,
        precision,
        -precision_gap,
        -recall_gap,
        -hard_selected_rate_gap,
        -fallback_selected_rate_gap,
        -selected_rate,
        -float(threshold),
    )


def _threshold_selection_reason(
    metrics: dict[str, float],
    *,
    min_recall: float,
    max_selected_rate: float,
    target_precision: float,
) -> str:
    precision = float(metrics.get("precision", 0.0))
    recall = float(metrics.get("recall", 0.0))
    selected_rate = float(metrics.get("selected_rate", 1.0))
    hard_max_selected_rate = max(float(max_selected_rate), _SERVING_THRESHOLD_HARD_MAX_SELECTED_RATE)
    if (
        precision >= float(target_precision)
        and recall >= float(min_recall)
        and selected_rate <= float(max_selected_rate)
    ):
        return "max_recall_under_precision_target"
    fallback_max_selected_rate = min(
        hard_max_selected_rate,
        max(float(max_selected_rate), _SERVING_THRESHOLD_FALLBACK_MAX_SELECTED_RATE),
    )
    if (
        precision >= _FILTER_GATE_MIN_PRECISION
        and recall >= float(min_recall)
        and selected_rate <= fallback_max_selected_rate
    ):
        return "f_score_fallback_under_gate_precision"
    if recall >= float(min_recall) and selected_rate <= fallback_max_selected_rate:
        return "f_score_fallback_below_precision_floor"
    if (
        precision >= _FILTER_GATE_MIN_PRECISION
        and recall >= _FILTER_GATE_MIN_RECALL
        and float(metrics.get("f1", 0.0)) >= _FILTER_GATE_MIN_F1
        and selected_rate <= hard_max_selected_rate
    ):
        return "best_filter_gate_under_hard_selected_rate"
    if recall >= float(min_recall) and selected_rate > hard_max_selected_rate:
        return "f_score_fallback_above_selected_rate"
    if precision >= float(target_precision) and selected_rate <= float(max_selected_rate):
        return "precision_target_below_min_recall"
    if selected_rate <= hard_max_selected_rate:
        return "best_degraded_under_selected_rate"
    return "best_degraded_filter_point"


def _fbeta(precision: float, recall: float, beta: float) -> float:
    beta_squared = float(beta) ** 2
    denominator = (beta_squared * float(precision)) + float(recall)
    if denominator <= 0:
        return 0.0
    return (1.0 + beta_squared) * float(precision) * float(recall) / denominator


def _filter_gate_from_calibration(calibration: dict[str, object]) -> dict[str, object]:
    metrics = dict(calibration.get("metrics") or {})
    precision = float(metrics.get("precision", 0.0) or 0.0)
    recall = float(metrics.get("recall", 0.0) or 0.0)
    f1 = float(metrics.get("f1", 0.0) or 0.0)
    selected_count = int(
        metrics.get(
            "selected_count",
            int(metrics.get("true_positive", 0) or 0) + int(metrics.get("false_positive", 0) or 0),
        )
        or 0
    )
    rows = int(metrics.get("rows") or 0)
    selected_rate = float(metrics.get("selected_rate", selected_count / rows if rows else 1.0) or 0.0)
    accuracy = float(metrics.get("accuracy", 0.0) or 0.0)
    thresholds = {
        "min_precision": _FILTER_GATE_MIN_PRECISION,
        "min_recall": _FILTER_GATE_MIN_RECALL,
        "min_f1": _FILTER_GATE_MIN_F1,
        "max_selected_rate": _FILTER_GATE_MAX_SELECTED_RATE,
        "min_accuracy_floor": _FILTER_GATE_MIN_ACCURACY_FLOOR,
        "target_precision": float(calibration.get("target_precision") or _SERVING_THRESHOLD_TARGET_PRECISION),
    }
    failures: list[str] = []
    if precision < _FILTER_GATE_MIN_PRECISION:
        failures.append("filter_precision_below_minimum")
    if recall < _FILTER_GATE_MIN_RECALL:
        failures.append("filter_recall_below_minimum")
    if f1 < _FILTER_GATE_MIN_F1:
        failures.append("filter_f1_below_minimum")
    if selected_rate > _FILTER_GATE_MAX_SELECTED_RATE:
        failures.append("filter_selected_rate_above_maximum")
    return {
        "source": str(calibration.get("source") or ""),
        "threshold": float(calibration.get("threshold") or 0.0),
        "metric": str(calibration.get("metric") or ""),
        "threshold_policy": str(calibration.get("threshold_policy") or ""),
        "selected_threshold_reason": str(calibration.get("selected_threshold_reason") or ""),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "accuracy": round(accuracy, 3),
        "selected_rate": round(selected_rate, 3),
        "selected_count": selected_count,
        "rows": rows,
        "f0_75": round(_fbeta(precision, recall, _FILTER_GATE_BETA), 3),
        "accuracy_gate_is_diagnostic": True,
        "thresholds": thresholds,
        "passed": not failures,
        "failures": failures,
    }


def _apply_bucket_filter_gates(
    filter_gate: dict[str, object],
    rows: list[dict],
    scores: np.ndarray,
    *,
    ranking_metric_value: float,
) -> None:
    if len(rows) == 0 or len(scores) == 0:
        return
    by_bucket: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        bucket = str(row.get("content_bucket") or "").strip().lower() or _content_bucket_from_row(row, dict(row.get("features") or {}))
        by_bucket.setdefault(bucket or "general", []).append(index)
    bucket_gates: dict[str, dict[str, object]] = {}
    threshold = float(filter_gate.get("threshold") or 0.0)
    serving_threshold_per_bucket: dict[str, float] = {}
    selected_model_per_bucket: dict[str, str] = {}
    passed_bucket_count = 0
    for bucket, indices in sorted(by_bucket.items()):
        labels = np.array([int(rows[index]["label"]) for index in indices], dtype=int)
        bucket_scores = np.array([float(scores[index]) for index in indices], dtype=float)
        label_values = set(labels.tolist())
        if len(label_values) == 1 and len(indices) >= _BUCKET_GATE_MIN_ROWS:
            if int(labels[0]) == 1:
                bucket_threshold = float(np.min(bucket_scores))
            else:
                bucket_threshold = float(np.max(bucket_scores) + 1e-6)
            metrics = _evaluate_scores(labels, bucket_scores, threshold=bucket_threshold)
        elif len(label_values) >= 2 and len(indices) >= _BUCKET_GATE_MIN_ROWS:
            calibration = _calibrate_serving_threshold(
                labels,
                bucket_scores,
                groups=None,
                source=f"validation_bucket:{bucket}",
            )
            bucket_threshold = float(calibration.get("threshold") or threshold)
            metrics = dict(calibration.get("metrics") or _evaluate_scores(labels, bucket_scores, threshold=bucket_threshold))
        else:
            bucket_threshold = threshold
            metrics = _evaluate_scores(labels, bucket_scores, threshold=bucket_threshold)
        passed = True
        bucket_failures: list[str] = []
        one_class_positive = len(label_values) == 1 and int(labels[0]) == 1
        one_class_negative = len(label_values) == 1 and int(labels[0]) == 0
        if len(indices) < _BUCKET_GATE_MIN_ROWS:
            passed = False
            bucket_failures.append("bucket_validation_rows_below_minimum")
        if one_class_negative:
            passed = False
            bucket_failures.append("bucket_validation_one_class_negative")
        elif one_class_positive:
            bucket_failures.append("bucket_validation_one_class_positive")
        if not one_class_positive and float(metrics.get("precision", 0.0)) < _FILTER_GATE_MIN_PRECISION:
            passed = False
            bucket_failures.append("bucket_precision_below_minimum")
        if float(metrics.get("recall", 0.0)) < _FILTER_GATE_MIN_RECALL:
            passed = False
            bucket_failures.append("bucket_recall_below_minimum")
        if not one_class_positive and float(metrics.get("f1", 0.0)) < _FILTER_GATE_MIN_F1:
            passed = False
            bucket_failures.append("bucket_f1_below_minimum")
        if not one_class_positive and float(metrics.get("selected_rate", 0.0)) > _FILTER_GATE_MAX_SELECTED_RATE:
            passed = False
            bucket_failures.append("bucket_selected_rate_above_maximum")
        if passed:
            passed_bucket_count += 1
        bucket_gates[bucket] = {
            "rows": len(indices),
            "threshold": float(bucket_threshold),
            "f1": float(metrics.get("f1", 0.0)),
            "precision": float(metrics.get("precision", 0.0)),
            "recall": float(metrics.get("recall", 0.0)),
            "selected_rate": float(metrics.get("selected_rate", 0.0)),
            "passed": passed,
            "failures": bucket_failures,
        }
        serving_threshold_per_bucket[bucket] = float(bucket_threshold)
        selected_model_per_bucket[bucket] = "trained" if passed else "fallback_heuristic"
    filter_gate["bucket_gates"] = bucket_gates
    filter_gate["bucket_gate_policy"] = {
        "min_passed_buckets": _BUCKET_GATE_MIN_PASSED_BUCKETS,
        "min_ranking_metric_value": _BUCKET_GATE_MIN_RANKING_METRIC,
        "min_bucket_rows": _BUCKET_GATE_MIN_ROWS,
    }
    filter_gate["passed_bucket_count"] = passed_bucket_count
    filter_gate["serving_threshold_per_bucket"] = serving_threshold_per_bucket
    filter_gate["selected_model_per_bucket"] = selected_model_per_bucket
    aggregate_failures = []
    if passed_bucket_count < _BUCKET_GATE_MIN_PASSED_BUCKETS:
        aggregate_failures.append("bucket_gate_passed_count_below_minimum")
    if float(ranking_metric_value) < _BUCKET_GATE_MIN_RANKING_METRIC:
        aggregate_failures.append("bucket_gate_ranking_metric_below_minimum")
    filter_gate["global_failures"] = list(filter_gate.get("failures") or [])
    filter_gate["failures"] = aggregate_failures
    filter_gate["passed"] = not aggregate_failures


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


def _mean_reciprocal_rank(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(labels) == 0:
        return 0.0
    ranked_idx = np.argsort(scores)[::-1]
    for rank, index in enumerate(ranked_idx, start=1):
        if int(labels[index]) == 1:
            return 1.0 / rank
    return 0.0


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


def _grouped_ndcg_summary_at_k(
    labels: np.ndarray,
    scores: np.ndarray,
    k: int,
    groups: list[str] | None,
) -> dict[str, float]:
    if not groups or len(groups) != len(labels):
        value = float(_ndcg_at_k(labels, scores, k))
        return {"mean": value, "min": value, "p25": value}

    grouped_indices: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        grouped_indices.setdefault(str(group), []).append(index)

    values = [
        float(_ndcg_at_k(labels[indices], scores[indices], k))
        for indices in grouped_indices.values()
        if indices
    ]
    if not values:
        value = float(_ndcg_at_k(labels, scores, k))
        return {"mean": value, "min": value, "p25": value}
    ordered = sorted(values)
    percentile_index = max(0, int(math.floor((len(ordered) - 1) * 0.25)))
    return {
        "mean": sum(ordered) / len(ordered),
        "min": ordered[0],
        "p25": ordered[percentile_index],
    }


def _grouped_mrr_summary(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: list[str] | None,
) -> dict[str, float]:
    if not groups or len(groups) != len(labels):
        value = float(_mean_reciprocal_rank(labels, scores))
        return {"mean": value, "min": value, "p25": value}

    grouped_indices: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        grouped_indices.setdefault(str(group), []).append(index)

    values = [
        float(_mean_reciprocal_rank(labels[indices], scores[indices]))
        for indices in grouped_indices.values()
        if indices
    ]
    if not values:
        value = float(_mean_reciprocal_rank(labels, scores))
        return {"mean": value, "min": value, "p25": value}
    ordered = sorted(values)
    percentile_index = max(0, int(math.floor((len(ordered) - 1) * 0.25)))
    return {
        "mean": sum(ordered) / len(ordered),
        "min": ordered[0],
        "p25": ordered[percentile_index],
    }


def _min_group_positive_rows(labels: np.ndarray, groups: list[str] | None) -> int:
    if len(labels) == 0:
        return 0
    if not groups or len(groups) != len(labels):
        return int(np.sum(labels == 1))
    counts: dict[str, int] = {}
    for index, group in enumerate(groups):
        key = str(group)
        counts[key] = counts.get(key, 0) + int(labels[index] == 1)
    return min(counts.values()) if counts else 0


def _select_ranking_metric(labels: np.ndarray, groups: list[str] | None) -> str:
    min_positive_rows = _min_group_positive_rows(labels, groups)
    if min_positive_rows >= 10:
        return "ndcg_at_5"
    if min_positive_rows >= 5:
        return "ndcg_at_10"
    return "mrr"


def _select_model(candidates: list[dict], training_row_count: int) -> dict:
    eligible = [
        candidate
        for candidate in candidates
        if _candidate_is_promoted(candidate, training_row_count)
    ]
    pool_candidates = eligible or candidates
    gate_viable = [candidate for candidate in pool_candidates if _candidate_passes_selection_gate(candidate)]
    if not gate_viable:
        return _fallback_heuristic_candidate(pool_candidates, reason="no_candidate_passed_filter_gate")
    viable = [candidate for candidate in gate_viable if _model_meets_minimums(candidate["metrics"])]
    pool = viable or gate_viable
    preferred_tree = _preferred_lightgbm_candidate(pool)
    if preferred_tree is not None:
        return preferred_tree

    def sort_key(candidate: dict) -> tuple[float, float, float, float, str]:
        metrics = candidate["metrics"]
        return (
            _candidate_selection_score(candidate),
            float(metrics.get("precision_at_5", 0.0)),
            float(metrics.get("ranking_metric_value", metrics.get("ndcg_at_5", 0.0))),
            float(metrics.get("f1", 0.0)),
            float(metrics.get("accuracy", 0.0)),
            candidate["name"],
        )

    return max(pool, key=sort_key)


def _preferred_lightgbm_candidate(candidates: list[dict]) -> dict | None:
    if (os.environ.get("RANKER_PREFER_LIGHTGBM") or "1").strip().lower() in {"0", "false", "no"}:
        return None
    lightgbm = [
        candidate for candidate in candidates
        if str(candidate.get("artifact_model_type") or "").strip().lower() == "lightgbm_ranker"
    ]
    if not lightgbm:
        return None
    best = max(candidates, key=_candidate_selection_score)
    best_score = _candidate_selection_score(best)
    best_metrics = dict(best.get("metrics") or {})
    tolerance = _lightgbm_preference_tolerance()
    viable = []
    for candidate in lightgbm:
        score = _candidate_selection_score(candidate)
        if score + tolerance < best_score:
            continue
        metrics = dict(candidate.get("metrics") or {})
        recall_gain = float(metrics.get("recall", 0.0) or 0.0) - float(best_metrics.get("recall", 0.0) or 0.0)
        ranking_gain = float(metrics.get("ranking_metric_value", metrics.get("ndcg_at_5", 0.0)) or 0.0) - float(
            best_metrics.get("ranking_metric_value", best_metrics.get("ndcg_at_5", 0.0)) or 0.0
        )
        precision_at_5_gap = float(best_metrics.get("precision_at_5", 0.0) or 0.0) - float(metrics.get("precision_at_5", 0.0) or 0.0)
        if recall_gain >= 0.0 or ranking_gain >= 0.0 or precision_at_5_gap <= tolerance:
            viable.append(candidate)
    if not viable:
        return None
    return max(viable, key=lambda candidate: (_candidate_selection_score(candidate), candidate["name"]))


def _lightgbm_preference_tolerance() -> float:
    raw = (os.environ.get("RANKER_LIGHTGBM_SELECTION_TOLERANCE") or str(_LIGHTGBM_PREFERENCE_TOLERANCE)).strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return _LIGHTGBM_PREFERENCE_TOLERANCE


def _fallback_heuristic_candidate(candidates: list[dict], *, reason: str) -> dict:
    reference = max(candidates, key=_candidate_selection_score) if candidates else {}
    metrics = dict(reference.get("metrics") or {})
    return {
        "name": "fallback_heuristic",
        "artifact_model_type": "fallback_heuristic",
        "weights": {},
        "intercept": 0.0,
        "metrics": metrics,
        "selection_blocked_reason": reason,
    }


def _model_selection_report(candidates: list[dict], selected_model: dict, training_row_count: int) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        metrics = dict(candidate.get("metrics") or {})
        promoted = _candidate_is_promoted(candidate, training_row_count)
        meets_minimums = _model_meets_minimums(metrics)
        filter_gate_passed = _candidate_passes_selection_gate(candidate)
        rows.append(
            {
                "name": str(candidate.get("name") or ""),
                "artifact_model_type": str(candidate.get("artifact_model_type") or ""),
                "promoted": promoted,
                "filter_gate_passed": filter_gate_passed,
                "meets_minimums": meets_minimums,
                "selection_score": round(_candidate_selection_score(candidate), 6),
                "ranking_metric": str(metrics.get("ranking_metric") or "ndcg_at_5"),
                "ranking_metric_value": float(metrics.get("ranking_metric_value", metrics.get("ndcg_at_5", 0.0)) or 0.0),
                "ranking_metric_min": float(metrics.get("ranking_metric_min", 0.0) or 0.0),
                "ranking_metric_p25": float(metrics.get("ranking_metric_p25", 0.0) or 0.0),
                "precision_at_5": float(metrics.get("precision_at_5", 0.0) or 0.0),
                "f1": float(metrics.get("f1", 0.0) or 0.0),
                "accuracy": float(metrics.get("accuracy", 0.0) or 0.0),
                "blocked_reason": _model_selection_blocked_reason(candidate, training_row_count),
            }
        )
    return {
        "training_row_count": int(training_row_count),
        "metric": "composite_validation_score",
        "selected_model": str(selected_model.get("name") or ""),
        "prefer_lightgbm_when_close": (os.environ.get("RANKER_PREFER_LIGHTGBM") or "1").strip().lower() not in {"0", "false", "no"},
        "lightgbm_selection_tolerance": _lightgbm_preference_tolerance(),
        "promotion_min_rows": {
            "gradient_boosting": _GRADIENT_BOOSTING_PROMOTION_MIN_ROWS,
            "lightgbm_ranker": _LIGHTGBM_PROMOTION_MIN_ROWS,
            "pairwise_logistic": {
                "min_precision": 0.90,
                "min_recall": 0.50,
            },
        },
        "candidates": rows,
    }


def _model_selection_blocked_reason(candidate: dict, training_row_count: int) -> str:
    artifact_model_type = str(candidate.get("artifact_model_type") or "").strip().lower()
    if artifact_model_type == "fallback_heuristic":
        return str(candidate.get("selection_blocked_reason") or "fallback_heuristic_selected")
    if artifact_model_type == "lightgbm_ranker" and training_row_count < _LIGHTGBM_PROMOTION_MIN_ROWS:
        return f"requires_at_least_{_LIGHTGBM_PROMOTION_MIN_ROWS}_training_rows"
    if artifact_model_type == "gradient_boosting" and training_row_count < _GRADIENT_BOOSTING_PROMOTION_MIN_ROWS:
        return f"requires_at_least_{_GRADIENT_BOOSTING_PROMOTION_MIN_ROWS}_training_rows"
    if artifact_model_type == "pairwise_logistic" and not _pairwise_candidate_meets_promotion_gate(candidate):
        return "pairwise_requires_precision_0_90_recall_0_50"
    if not _candidate_passes_selection_gate(candidate):
        return "failed_filter_gate"
    if not _model_meets_minimums(dict(candidate.get("metrics") or {})):
        return "failed_minimum_validation_gates"
    return ""


def _candidate_is_promoted(candidate: dict, training_row_count: int) -> bool:
    artifact_model_type = str(candidate.get("artifact_model_type") or "").strip().lower()
    if artifact_model_type == "pairwise_logistic":
        return _pairwise_candidate_meets_promotion_gate(candidate)
    if artifact_model_type == "lightgbm_ranker":
        return training_row_count >= _LIGHTGBM_PROMOTION_MIN_ROWS
    if artifact_model_type == "gradient_boosting":
        return training_row_count >= _GRADIENT_BOOSTING_PROMOTION_MIN_ROWS
    return True


def _candidate_passes_selection_gate(candidate: dict) -> bool:
    if str(candidate.get("artifact_model_type") or "").strip().lower() == "pairwise_logistic":
        return _pairwise_candidate_meets_promotion_gate(candidate)
    return _model_passes_filter_gate(dict(candidate.get("metrics") or {}))


def _candidate_selection_score(candidate: dict) -> float:
    metrics = dict(candidate.get("metrics") or {})
    if str(candidate.get("artifact_model_type") or "").strip().lower() == "pairwise_logistic":
        ranking = float(metrics.get("ranking_metric_value", metrics.get("ndcg_at_5", 0.0)) or 0.0)
        precision = float(metrics.get("precision", metrics.get("precision_at_5", 0.0)) or 0.0)
        f1 = float(metrics.get("f1", 0.0) or 0.0)
        selected_rate = float(metrics.get("selected_rate", 0.0) or 0.0)
        selected_rate_score = max(0.0, 1.0 - max(0.0, selected_rate - 0.40))
        return ranking * 0.35 + precision * 0.35 + f1 * 0.20 + selected_rate_score * 0.10
    return _selection_score(metrics)


def _pairwise_candidate_meets_promotion_gate(candidate: dict) -> bool:
    metrics = dict(candidate.get("metrics") or {})
    return (
        float(metrics.get("precision", metrics.get("precision_at_5", 0.0)) or 0.0) >= 0.90
        and float(metrics.get("recall", 0.0) or 0.0) >= 0.50
    )


def _resolve_lightgbm_params(
    hpo_params: dict | None,
    *,
    feature_names: list[str] | None = None,
    random_state: int,
) -> dict[str, float | int | str | list[int]]:
    params: dict[str, float | int | str] = {
        "objective": "lambdarank",
        "verbosity": -1,
        "random_state": int(random_state),
        "n_jobs": -1,
    }
    params.update(_LIGHTGBM_DEFAULT_PARAMS)
    for key, value in (hpo_params or {}).items():
        if key not in _LIGHTGBM_HPO_PARAM_KEYS:
            continue
        if key in {"n_estimators", "num_leaves", "min_child_samples"}:
            params[key] = max(1, int(_safe_float(value)))
        else:
            params[key] = float(value)
    if float(params.get("bagging_fraction", 1.0)) < 0.999:
        params["bagging_freq"] = 1
    if feature_names:
        params["monotone_constraints"] = [
            1 if name in _POSITIVE_SIGN_FEATURES else 0
            for name in feature_names
        ]
    return params


def _load_hpo_artifact(profile: str, artifact_dir: pathlib.Path) -> dict | None:
    path = artifact_dir / f"{profile}_ranker_hpo.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if str(payload.get("profile") or "").strip() != profile:
        return None
    if not isinstance(payload.get("best_params"), dict):
        return None
    return payload


def _best_hpo_params(hpo_artifact: dict | None) -> dict | None:
    best_params = (hpo_artifact or {}).get("best_params")
    return dict(best_params) if isinstance(best_params, dict) else None


def _artifact_hpo_summary(hpo_artifact: dict | None) -> dict[str, object]:
    if not hpo_artifact:
        return {
            "loaded": False,
            "best_params": {},
            "best_ndcg_at_5": 0.0,
            "seed": int(settings.ranker_hpo_seed),
        }
    return {
        "loaded": True,
        "best_params": dict(hpo_artifact.get("best_params") or {}),
        "best_ndcg_at_5": round(float(hpo_artifact.get("best_value", 0.0)), 3),
        "seed": int(_safe_float(hpo_artifact.get("seed"))),
        "trials": max(0, int(_safe_float(hpo_artifact.get("trials")))),
    }


def _calibrate_blend(
    validation_rows: list[dict],
    profile: str,
    feature_names: list[str],
    scales: dict[str, dict[str, float]],
    selected_model: dict,
    split_unit: str,
) -> dict[str, object]:
    ranker = Ranker(enable_trained_model=False)
    feature_rows = [
        CandidateFeatures.from_dict(
            {
                "candidate_id": str(row.get("candidate_id") or "unknown"),
                **dict(row.get("features") or {}),
            }
        )
        for row in validation_rows
    ]
    heuristic_scores = np.array(
        [score for score, _ in ranker._heuristic_scores(feature_rows, profile)],
        dtype=float,
    )
    trained_scores = _runtime_trained_scores(
        validation_rows,
        feature_names,
        scales,
        selected_model,
        ranker,
    )
    labels = np.array([int(row["label"]) for row in validation_rows], dtype=int)
    groups = _ranking_group_labels(validation_rows, split_unit)
    results: dict[str, dict[str, float]] = {}
    if str(selected_model.get("artifact_model_type") or "") == "fallback_heuristic":
        metrics = _evaluate_scores(labels, heuristic_scores, groups=groups)
        return {
            "selection_metric": "heuristic_fallback",
            "best_blend": 0.0,
            "requested_best_blend": 0.0,
            "runtime_max_blend": round(float(_effective_serving_blend(1.0, ranker=ranker)), 2),
            "best_metrics": metrics,
            "grid_results": {"0.00": {**metrics, "effective_blend": 0.0}},
        }
    if str(selected_model.get("score_mode") or "") == "pairwise_median":
        metrics = _evaluate_scores(labels, trained_scores, groups=groups)
        return {
            "selection_metric": "pairwise_median_no_heuristic_blend",
            "best_blend": 0.0,
            "requested_best_blend": 0.0,
            "runtime_max_blend": 0.0,
            "best_metrics": metrics,
            "grid_results": {"pairwise_median": {**metrics, "effective_blend": 0.0}},
        }

    def _metric_key(metrics: dict[str, float], blend_value: float) -> tuple:
        # Break ties toward lower blend: when metrics are equal, prefer leaning
        # on the heuristic, which is auditable and does not depend on the
        # trained artifact's stability.
        return (
            _selection_score(metrics),
            float(metrics.get("precision_at_5", 0.0)),
            float(metrics.get("ndcg_at_5", 0.0)),
            float(metrics.get("f1", 0.0)),
            -blend_value,
        )

    best_blend = 0.0
    best_metrics = _evaluate_scores(labels, heuristic_scores, groups=groups)
    best_requested_blend = 0.0
    best_effective_blend = _effective_serving_blend(best_requested_blend, ranker=ranker)
    results[f"{best_requested_blend:.2f}"] = {
        **best_metrics,
        "effective_blend": round(float(best_effective_blend), 2),
    }
    best_key = _metric_key(best_metrics, best_effective_blend)
    for requested_blend in _BLEND_CANDIDATES:
        effective_blend = _effective_serving_blend(requested_blend, ranker=ranker)
        merged = heuristic_scores * (1.0 - effective_blend) + trained_scores * effective_blend
        metrics = _evaluate_scores(labels, merged, groups=groups)
        results[f"{requested_blend:.2f}"] = {
            **metrics,
            "effective_blend": round(float(effective_blend), 2),
        }
        key = _metric_key(metrics, effective_blend)
        if key > best_key:
            best_requested_blend = requested_blend
            best_effective_blend = effective_blend
            best_metrics = metrics
            best_key = key
    return {
        "selection_metric": "validation_primary_ranking_metric",
        "best_blend": round(float(best_effective_blend), 2),
        "requested_best_blend": round(float(best_requested_blend), 2),
        "runtime_max_blend": round(float(_effective_serving_blend(1.0, ranker=ranker)), 2),
        "best_metrics": best_metrics,
        "grid_results": results,
    }


def _effective_serving_blend(blend: float, *, ranker: Ranker | None = None) -> float:
    runtime_ranker = ranker or Ranker(enable_trained_model=False)
    return runtime_ranker._safe_blend({"blend": float(blend)})


def _runtime_blended_scores(
    rows: list[dict],
    profile: str,
    feature_names: list[str],
    scales: dict[str, dict[str, float]],
    selected_model: dict,
    blend: float,
) -> np.ndarray:
    ranker = Ranker(enable_trained_model=False)
    feature_rows = [
        CandidateFeatures.from_dict(
            {
                "candidate_id": str(row.get("candidate_id") or "unknown"),
                **dict(row.get("features") or {}),
            }
        )
        for row in rows
    ]
    heuristic_scores = np.array(
        [score for score, _ in ranker._heuristic_scores(feature_rows, profile)],
        dtype=float,
    )
    trained_scores = _runtime_trained_scores(
        rows,
        feature_names,
        scales,
        selected_model,
        ranker,
    )
    safe_blend = _effective_serving_blend(blend, ranker=ranker)
    if str(selected_model.get("score_mode") or "") == "pairwise_median":
        return trained_scores
    return heuristic_scores * (1.0 - safe_blend) + trained_scores * safe_blend


def _runtime_trained_scores(
    rows: list[dict],
    feature_names: list[str],
    scales: dict[str, dict[str, float]],
    selected_model: dict,
    ranker: Ranker,
) -> np.ndarray:
    model_type = str(selected_model.get("artifact_model_type") or "linear").strip().lower()
    if model_type == "fallback_heuristic":
        return np.zeros(len(rows), dtype=float)
    if model_type == "pairwise_logistic" or str(selected_model.get("score_mode") or "") == "pairwise_median":
        raw_scores = _pairwise_median_raw_scores(
            rows,
            feature_names,
            selected_model.get("weights", {}),
            scales,
            float(selected_model.get("intercept", 0.0)),
            "vod_id",
        )
        return np.array(
            [ranker._normalize_trained_score(float(score), "logistic") for score in raw_scores],
            dtype=float,
        )
    if model_type in {"linear", "logistic"}:
        raw_scores = _linear_raw_scores(
            rows,
            feature_names,
            selected_model.get("weights", {}),
            scales,
            float(selected_model.get("intercept", 0.0)),
        )
        return np.array(
            [ranker._normalize_trained_score(float(score), model_type) for score in raw_scores],
            dtype=float,
        )
    fitted_model = selected_model.get("fitted_model")
    if fitted_model is None:
        return np.zeros(len(rows), dtype=float)
    matrix = _feature_matrix(rows, feature_names, scales)
    if model_type == "gradient_boosting":
        probabilities = fitted_model.predict_proba(matrix)[:, 1]
        return np.array(
            [ranker._normalize_trained_score(float(score), model_type) for score in probabilities],
            dtype=float,
        )
    raw_scores = fitted_model.predict(matrix)
    return np.array(
        [ranker._normalize_trained_score(float(score), model_type) for score in raw_scores],
        dtype=float,
    )


def _selection_score(metrics: dict[str, float]) -> float:
    ranking = float(metrics.get("ndcg_at_5", metrics.get("ranking_metric_value", 0.0)) or 0.0)
    f1 = float(metrics.get("f1", 0.0) or 0.0)
    gate_passed = 1.0 if _model_passes_filter_gate(metrics) else 0.0
    selected_rate = float(metrics.get("selected_rate", 1.0) or 1.0)
    selected_rate_excess = max(0.0, selected_rate - _FILTER_GATE_MAX_SELECTED_RATE)
    selected_rate_score = max(0.0, 1.0 - selected_rate_excess)
    score = ranking * 0.4 + f1 * 0.3 + gate_passed * 0.2 + selected_rate_score * 0.1
    if not gate_passed:
        score *= 0.5
    return score


def _model_passes_filter_gate(metrics: dict[str, float]) -> bool:
    precision = float(metrics.get("precision", metrics.get("precision_at_5", 0.0)) or 0.0)
    recall = float(metrics.get("recall", 1.0) or 0.0)
    selected_rate = float(metrics.get("selected_rate", 0.0) or 0.0)
    return (
        precision >= _FILTER_GATE_MIN_PRECISION
        and recall >= _FILTER_GATE_MIN_RECALL
        and float(metrics.get("f1", 0.0) or 0.0) >= _FILTER_GATE_MIN_F1
        and selected_rate <= _FILTER_GATE_MAX_SELECTED_RATE
    )


def _model_meets_minimums(metrics: dict[str, float]) -> bool:
    f1 = float(metrics.get("f1", 0.0))
    ranking_groups = int(metrics.get("ranking_groups", 0) or 0)
    ranking = max(
        float(metrics.get("ranking_metric_value", metrics.get("ndcg_at_5", 0.0))),
        float(metrics.get("precision_at_5", 0.0)),
    )
    if ranking_groups and ranking_groups < 2:
        return False
    if ranking_groups >= 2:
        group_ranking_min = float(
            metrics.get(
                "ranking_metric_min",
                metrics.get(
                    "ndcg_at_5_min",
                    metrics.get("ranking_metric_value", metrics.get("ndcg_at_5", 0.0)),
                ),
            )
        )
        group_ranking_p25 = float(
            metrics.get(
                "ranking_metric_p25",
                metrics.get(
                    "ndcg_at_5_p25",
                    metrics.get("ranking_metric_value", metrics.get("ndcg_at_5", 0.0)),
                ),
            )
        )
        if group_ranking_min < _MIN_GROUP_RANKING_METRIC_MIN or group_ranking_p25 < _MIN_GROUP_RANKING_METRIC_P25:
            return False
    return f1 >= 0.55 and ranking >= 0.6


def _seeded_row_order(rows: list[dict]) -> list[dict]:
    if len(rows) <= 1:
        return list(rows)
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row.get("job_id", "")),
            str(row.get("candidate_id", "")),
            str(row.get("vod_id", "")),
        ),
    )
    rng = np.random.default_rng(_SPLIT_RANDOM_SEED)
    indices = rng.permutation(len(ordered)).tolist()
    return [ordered[index] for index in indices]


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    if not values:
        return 0.0
    total_weight = sum(float(weight) for weight in weights)
    if total_weight <= 0:
        return sum(values) / len(values)
    return sum(value * float(weight) for value, weight in zip(values, weights, strict=False)) / total_weight


def _sample_weight(row: dict) -> float:
    supervision_multiplier = {
        "outcome_feedback": 1.35,
        "judge_feedback": 1.22,
        "manual_annotation": 1.5,
        "twitch_clip_proxy": _TWITCH_CLIP_PROXY_SUPERVISION_MULTIPLIER,
        "twitch_top_viewed_weak_positive": 1.0,
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
        "outcome_top_viewed_positive": 0.55,
        "outcome_negative": 1.8,
        "pipeline_positive": 1.0,
        "pipeline_negative": 0.8,
        "judge_quality_positive": 1.35,
        "judge_quality_negative": 1.25,
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


def _candidate_source_mix(rows: list[dict], key: str) -> dict[str, object]:
    counts = _count_values(rows, key)
    total = sum(counts.values()) or 1
    peak_share = max((count / total for count in counts.values()), default=0.0)
    return {
        "rows": len(rows),
        "by_value": counts,
        "peak_share": round(float(peak_share), 3),
    }


if __name__ == "__main__":
    raise SystemExit(main())
