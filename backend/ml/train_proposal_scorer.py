from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

DEFAULT_FEATURES = (
    "duration",
    "segment_count",
    "token_count",
    "max_hype",
    "mean_hype",
    "arc_peak",
    "chat_peak",
    "motion_peak",
    "face_peak",
    "laughter_count",
    "exclamation_count",
    "question_count",
    "speaker_turns",
    "hook_strength",
    "payoff_strength",
    "hype_delta",
    "early_hook_score",
    "late_payoff_score",
    "standalone_text_score",
    "source_prior",
    "heuristic_hit_count",
    "strategy_relevance",
    "dense_inventory",
    "multimodal_proxy_score",
    "rage_score",
    "clutch_score",
    "funny_reaction_score",
    "trash_talk_payoff_score",
    "chat_shock_score",
    "ranked_tension_score",
    "failed_setup_score",
    "streamer_persona_beat_score",
    "context_pattern_score",
    "context_pattern_support",
    "context_pattern_views_prior",
    "context_pattern_confidence",
    "context_pattern_similarity_margin",
    "setup_present_score",
    "payoff_present_score",
    "context_strength_score",
    "context_completeness_score",
)
DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
_CROSS_PLATFORM_POSITIVE_WEIGHT = 2.4
_CROSS_PLATFORM_PROPOSAL_FALSE_NEGATIVE_WEIGHT = 5.0
_CROSS_PLATFORM_NEAR_MISS_WEIGHT = 0.8


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the cheap candidate proposal scorer.")
    parser.add_argument("--storage-dir", type=Path, default=Path("tmp/storage"))
    parser.add_argument("--profile", default="general")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--training-rows", action="append", type=Path, default=[])
    parser.add_argument("--positive-rows", type=Path, default=None)
    parser.add_argument("--min-rows", type=int, default=8)
    parser.add_argument("--l2", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--epochs", type=int, default=800)
    args = parser.parse_args(argv)

    rows = collect_training_rows(args.storage_dir, positive_rows_path=args.positive_rows)
    rows.extend(collect_training_rows_from_files(args.training_rows))
    artifact = train_logistic_regression_model(
        rows,
        profile=args.profile,
        min_rows=args.min_rows,
        l2=args.l2,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
    )
    output = args.output or DEFAULT_ARTIFACT_DIR / f"{args.profile}_proposal_scorer.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "proposal_scorer_trained",
                "output": str(output),
                **artifact["training_summary"],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


def collect_training_rows(storage_dir: Path, positive_rows_path: Path | None = None) -> list[dict]:
    rows: list[dict] = []
    positive_index = _positive_window_index(positive_rows_path)
    for debug_dir in sorted(storage_dir.glob("*/debug")):
        proposal_rows = _read_jsonl(debug_dir / "candidate_proposals.jsonl")
        if not proposal_rows:
            continue
        labels = _job_labels(debug_dir)
        for candidate_id, label in _cross_platform_labels(debug_dir, proposal_rows, positive_index).items():
            existing = labels.get(candidate_id)
            if existing is not None and _is_manual_feedback_label(existing[2]):
                continue
            if int(label[0]) == 1 or existing is None:
                labels[candidate_id] = label
        for proposal in proposal_rows:
            candidate_id = str(proposal.get("candidate_id") or "").strip()
            if not candidate_id or candidate_id not in labels:
                continue
            label, sample_weight, label_source = labels[candidate_id]
            features = proposal.get("features") or {}
            rows.append(
                {
                    "job_id": debug_dir.parent.name,
                    "vod_id": _job_vod_id(debug_dir.parent),
                    "candidate_id": candidate_id,
                    "window_start": proposal.get("start"),
                    "window_end": proposal.get("end"),
                    "label": label,
                    "sample_weight": sample_weight,
                    "label_source": label_source,
                    "dataset_source": "proposal_debug_feedback",
                    "features": features,
                }
            )
    return rows


def collect_training_rows_from_files(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        for row in _read_jsonl(path):
            if _is_gold_eval_row(row):
                continue
            features = row.get("features") or {}
            candidate_id = str(row.get("candidate_id") or "").strip()
            if not candidate_id or not isinstance(features, dict):
                continue
            try:
                label = int(row.get("label"))
            except (TypeError, ValueError):
                continue
            rows.append(
                {
                    "job_id": row.get("job_id") or "",
                    "vod_id": row.get("vod_id") or "",
                    "candidate_id": candidate_id,
                    "window_start": row.get("window_start") or row.get("start"),
                    "window_end": row.get("window_end") or row.get("end"),
                    "label": 1 if label == 1 else 0,
                    "sample_weight": float(row.get("sample_weight") or 1.0),
                    "label_source": row.get("label_source") or row.get("dataset_source") or "proposal_training_rows",
                    "dataset_source": row.get("dataset_source") or "proposal_training_rows",
                    "features": features,
                }
            )
    return rows


def train_logistic_regression_model(
    rows: list[dict],
    *,
    profile: str,
    min_rows: int = 8,
    l2: float = 0.05,
    learning_rate: float = 0.08,
    epochs: int = 800,
) -> dict:
    filtered_gold_rows = sum(1 for row in rows if _is_gold_eval_row(row))
    rows = [row for row in rows if not _is_gold_eval_row(row)]
    positives = [row for row in rows if int(row.get("label", 0)) == 1]
    negatives = [row for row in rows if int(row.get("label", 0)) == 0]
    usable = len(positives) + len(negatives)
    if usable < min_rows or not positives or not negatives:
        return {
            "model_type": "logistic_regression_proposal_scorer",
            "profile": profile,
            "link": "sigmoid",
            "bias": 0.0,
            "weights": {},
            "feature_means": {},
            "feature_scales": {},
            "feature_names": list(DEFAULT_FEATURES),
            "training_summary": {
                "status": "insufficient_rows",
                "rows": usable,
                "positive_rows": len(positives),
                "negative_rows": len(negatives),
                "min_rows": min_rows,
                "filtered_gold_rows": filtered_gold_rows,
            },
        }

    train_rows, eval_rows, split_summary = _grouped_train_eval_split(rows)
    means, scales = _weighted_feature_stats(train_rows, DEFAULT_FEATURES)
    weights, bias = _fit_logistic_regression(
        train_rows,
        DEFAULT_FEATURES,
        means,
        scales,
        l2=max(float(l2), 0.0),
        learning_rate=max(float(learning_rate), 1e-6),
        epochs=max(int(epochs), 1),
    )
    eval_source = "heldout" if _has_two_classes(eval_rows) else "train_fallback"
    metric_rows = eval_rows if eval_source == "heldout" else train_rows
    metrics = _classification_metrics(metric_rows, weights, bias, means, scales, DEFAULT_FEATURES)
    return {
        "model_type": "logistic_regression_proposal_scorer",
        "profile": profile,
        "link": "sigmoid",
        "bias": round(bias, 6),
        "weights": {name: round(value, 6) for name, value in weights.items()},
        "feature_means": {name: round(value, 6) for name, value in means.items()},
        "feature_scales": {name: round(value, 6) for name, value in scales.items()},
        "feature_names": list(DEFAULT_FEATURES),
        "training_summary": {
            "status": "trained",
            "rows": usable,
            "positive_rows": len(positives),
            "negative_rows": len(negatives),
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "eval_source": eval_source,
            "split": split_summary,
            "label_sources": _counts(str(row.get("label_source") or "unknown") for row in rows),
            "metrics": metrics,
            "filtered_gold_rows": filtered_gold_rows,
            "l2": round(max(float(l2), 0.0), 6),
            "learning_rate": round(max(float(learning_rate), 1e-6), 6),
            "epochs": max(int(epochs), 1),
            "weight_l1_norm": round(sum(abs(value) for value in weights.values()), 6),
        },
    }


def train_mean_difference_model(rows: list[dict], *, profile: str, min_rows: int = 8) -> dict:
    return train_logistic_regression_model(rows, profile=profile, min_rows=min_rows)


def _job_labels(debug_dir: Path) -> dict[str, tuple[int, float, str]]:
    labels: dict[str, tuple[int, float, str]] = {}
    for row in _read_jsonl(debug_dir / "feedback.jsonl"):
        candidate_id = str(row.get("candidate_id") or "").strip()
        if not candidate_id:
            continue
        rating = str(row.get("feedback_rating") or "").strip().lower()
        tags = {str(tag).strip().lower() for tag in row.get("feedback_tags") or []}
        if rating == "excellent":
            labels[candidate_id] = (1, 3.0, "frontend_excellent")
        elif rating == "good":
            labels[candidate_id] = (1, 2.4 if "good_clip" in tags else 2.2, "frontend_good_clip" if "good_clip" in tags else "frontend_good")
        elif rating == "boundary_bad":
            labels[candidate_id] = (0, 1.4, "frontend_boundary_bad")
        elif rating == "bad":
            if {"wrong_boundary", "wrong_duration", "bad_boundary", "too_long"} & tags:
                labels[candidate_id] = (0, 1.1, "frontend_boundary_or_duration")
            else:
                labels[candidate_id] = (0, 2.1 if "false_positive" in tags else 1.8, "frontend_false_positive" if "false_positive" in tags else "frontend_bad")
        elif bool(row.get("selected")):
            labels.setdefault(candidate_id, (1, 1.0, "pipeline_selected"))
        elif row.get("rejection_source") == "low_rank":
            labels.setdefault(candidate_id, (0, 1.0, "pipeline_low_rank"))

    for row in _read_jsonl(debug_dir / "serving_quality_gate.jsonl"):
        candidate_id = str(row.get("candidate_id") or "").strip()
        if not candidate_id or candidate_id in labels:
            continue
        if not _has_judge_result(row):
            continue
        if bool(row.get("accepted")):
            labels[candidate_id] = (1, _judge_sample_weight(row, accepted=True), "judge_quality_accepted")
        else:
            labels[candidate_id] = (0, _judge_sample_weight(row, accepted=False), "judge_quality_rejected")
    return labels


def _has_judge_result(row: dict) -> bool:
    if row.get("has_judge_result") is not None:
        return bool(row.get("has_judge_result"))
    return row.get("judge_score") not in (None, "")


def _judge_sample_weight(row: dict, *, accepted: bool) -> float:
    judge_score = _as_float(row.get("judge_score"))
    confidence = _as_float(row.get("judge_confidence"))
    base = 1.6 if accepted else 1.3
    if judge_score is not None:
        if accepted:
            base += max(0.0, min((judge_score - 6.0) / 4.0, 1.0)) * 0.45
        else:
            base += max(0.0, min((6.0 - judge_score) / 5.0, 1.0)) * 0.35
    if confidence is not None:
        base *= 0.9 + max(0.0, min(confidence, 1.0)) * 0.2
    return round(base, 3)


def _cross_platform_labels(
    debug_dir: Path,
    proposal_rows: list[dict],
    positive_index: dict[str, list[tuple[float, float]]],
) -> dict[str, tuple[int, float, str]]:
    vod_id = _job_vod_id(debug_dir.parent)
    positives = positive_index.get(vod_id, [])
    if not positives:
        return {}
    labels: dict[str, tuple[int, float, str]] = {}
    for row in proposal_rows:
        candidate_id = str(row.get("candidate_id") or "").strip()
        start = _as_float(row.get("start"))
        end = _as_float(row.get("end"))
        if not candidate_id or start is None or end is None:
            continue
        best_overlap = 0.0
        nearest_gap = None
        for pos_start, pos_end in positives:
            overlap = max(0.0, min(end, pos_end) - max(start, pos_start))
            shorter = max(min(end - start, pos_end - pos_start), 1e-6)
            best_overlap = max(best_overlap, overlap / shorter)
            if overlap > 0:
                nearest_gap = 0.0
            else:
                gap = min(abs(start - pos_end), abs(pos_start - end))
                nearest_gap = gap if nearest_gap is None else min(nearest_gap, gap)
        if best_overlap >= 0.45:
            accepted = bool(row.get("accepted") is True)
            if accepted:
                labels[candidate_id] = (1, _CROSS_PLATFORM_POSITIVE_WEIGHT, "cross_platform_positive")
            else:
                labels[candidate_id] = (
                    1,
                    _CROSS_PLATFORM_PROPOSAL_FALSE_NEGATIVE_WEIGHT,
                    "cross_platform_proposal_false_negative",
                )
        elif nearest_gap is not None and nearest_gap <= 180.0:
            labels[candidate_id] = (0, _CROSS_PLATFORM_NEAR_MISS_WEIGHT, "cross_platform_near_miss")
    return labels


def _is_manual_feedback_label(label_source: str) -> bool:
    return str(label_source or "").startswith("frontend_")


def _positive_window_index(path: Path | None) -> dict[str, list[tuple[float, float]]]:
    if path is None:
        return {}
    index: dict[str, list[tuple[float, float]]] = {}
    for row in _read_jsonl(path):
        vod_id = str(row.get("vod_id") or "").strip()
        start = _as_float(row.get("window_start") or row.get("start"))
        end = _as_float(row.get("window_end") or row.get("end"))
        label = row.get("label")
        try:
            is_positive = int(label) == 1 if label is not None else True
        except (TypeError, ValueError):
            is_positive = True
        if vod_id and start is not None and end is not None and end > start and is_positive:
            index.setdefault(vod_id, []).append((start, end))
    return index


def _job_vod_id(job_dir: Path) -> str:
    payload = {}
    request_path = job_dir / "request.json"
    if request_path.exists():
        try:
            payload = json.loads(request_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    source_url = str(payload.get("url") or "").strip()
    if "/videos/" in source_url:
        return source_url.rstrip("/").rsplit("/", 1)[-1]
    return str(payload.get("vod_id") or "").strip()


def _as_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _grouped_train_eval_split(rows: list[dict]) -> tuple[list[dict], list[dict], dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(_group_key(row), []).append(row)
    train_rows: list[dict] = []
    eval_rows: list[dict] = []
    for group, group_rows in sorted(groups.items()):
        bucket = int(hashlib.sha1(group.encode("utf-8")).hexdigest()[:8], 16) % 5
        if bucket == 0:
            eval_rows.extend(group_rows)
        else:
            train_rows.extend(group_rows)
    if not _has_two_classes(train_rows) or not _has_two_classes(eval_rows):
        return rows, [], {
            "mode": "train_fallback",
            "reason": "group_split_missing_both_classes",
            "groups": len(groups),
        }
    return train_rows, eval_rows, {
        "mode": "group_hash",
        "groups": len(groups),
        "train_groups": len({_group_key(row) for row in train_rows}),
        "eval_groups": len({_group_key(row) for row in eval_rows}),
    }


def _group_key(row: dict) -> str:
    for key in ("vod_id", "job_id", "source_url"):
        value = str(row.get(key) or "").strip()
        if value:
            return f"{key}:{value}"
    return f"candidate:{row.get('candidate_id') or ''}"


def _has_two_classes(rows: list[dict]) -> bool:
    labels = {int(row.get("label", 0)) for row in rows}
    return 0 in labels and 1 in labels


def _weighted_feature_stats(rows: list[dict], names: Iterable[str]) -> tuple[dict[str, float], dict[str, float]]:
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for name in names:
        total_weight = 0.0
        weighted_total = 0.0
        for row in rows:
            weight = _sample_weight(row)
            weighted_total += _feature_value(row, name) * weight
            total_weight += weight
        mean = weighted_total / max(total_weight, 1e-6)
        variance_total = 0.0
        for row in rows:
            weight = _sample_weight(row)
            delta = _feature_value(row, name) - mean
            variance_total += weight * delta * delta
        scale = math.sqrt(variance_total / max(total_weight, 1e-6))
        means[name] = mean
        scales[name] = max(scale, 1e-6)
    return means, scales


def _fit_logistic_regression(
    rows: list[dict],
    names: Iterable[str],
    means: dict[str, float],
    scales: dict[str, float],
    *,
    l2: float,
    learning_rate: float,
    epochs: int,
) -> tuple[dict[str, float], float]:
    names = tuple(names)
    weights = {name: 0.0 for name in names}
    positive_weight = sum(_sample_weight(row) for row in rows if int(row.get("label", 0)) == 1)
    total_weight = sum(_sample_weight(row) for row in rows)
    prior = min(max(positive_weight / max(total_weight, 1e-6), 1e-4), 1.0 - 1e-4)
    bias = math.log(prior / (1.0 - prior))
    for _epoch in range(epochs):
        gradients = {name: 0.0 for name in names}
        bias_gradient = 0.0
        for row in rows:
            y = 1.0 if int(row.get("label", 0)) == 1 else 0.0
            sample_weight = _sample_weight(row)
            raw = bias + sum(
                weights[name] * _standardized_feature(row, name, means, scales)
                for name in names
            )
            prediction = _sigmoid(raw)
            error = (prediction - y) * sample_weight
            bias_gradient += error
            for name in names:
                gradients[name] += error * _standardized_feature(row, name, means, scales)
        normalizer = max(total_weight, 1e-6)
        bias -= learning_rate * (bias_gradient / normalizer)
        for name in names:
            gradient = gradients[name] / normalizer + l2 * weights[name]
            weights[name] -= learning_rate * gradient
    return weights, bias


def _classification_metrics(
    rows: list[dict],
    weights: dict[str, float],
    bias: float,
    means: dict[str, float],
    scales: dict[str, float],
    names: Iterable[str],
) -> dict[str, float | int]:
    scored = [
        (
            _predict_probability(row, weights, bias, means, scales, names),
            1 if int(row.get("label", 0)) == 1 else 0,
        )
        for row in rows
    ]
    if not scored:
        return {"rows": 0}
    positives = sum(label for _score, label in scored)
    negatives = len(scored) - positives
    logloss = -sum(
        label * math.log(max(score, 1e-9)) + (1 - label) * math.log(max(1.0 - score, 1e-9))
        for score, label in scored
    ) / len(scored)
    accuracy = sum((score >= 0.5) == bool(label) for score, label in scored) / len(scored)
    top_k = max(1, min(positives or 1, len(scored)))
    top = sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]
    precision_at_k = sum(label for _score, label in top) / top_k
    recall_at_k = sum(label for _score, label in top) / max(positives, 1)
    return {
        "rows": len(scored),
        "positive_rows": positives,
        "negative_rows": negatives,
        "auc": round(_auc(scored), 6) if positives and negatives else None,
        "accuracy_at_0_5": round(accuracy, 6),
        "logloss": round(logloss, 6),
        "precision_at_k": round(precision_at_k, 6),
        "recall_at_k": round(recall_at_k, 6),
        "k": top_k,
    }


def _predict_probability(
    row: dict,
    weights: dict[str, float],
    bias: float,
    means: dict[str, float],
    scales: dict[str, float],
    names: Iterable[str],
) -> float:
    raw = bias + sum(weights[name] * _standardized_feature(row, name, means, scales) for name in names)
    return _sigmoid(raw)


def _standardized_feature(row: dict, name: str, means: dict[str, float], scales: dict[str, float]) -> float:
    return (_feature_value(row, name) - means.get(name, 0.0)) / max(scales.get(name, 1.0), 1e-6)


def _sample_weight(row: dict) -> float:
    try:
        return max(float(row.get("sample_weight", 1.0) or 1.0), 0.0)
    except (TypeError, ValueError):
        return 1.0


def _sigmoid(value: float) -> float:
    value = max(min(value, 40.0), -40.0)
    return 1.0 / (1.0 + math.exp(-value))


def _auc(scored: list[tuple[float, int]]) -> float:
    ordered = sorted(scored, key=lambda item: item[0])
    rank_sum = 0.0
    for rank, (_score, label) in enumerate(ordered, start=1):
        if label == 1:
            rank_sum += rank
    positives = sum(label for _score, label in scored)
    negatives = len(scored) - positives
    if positives == 0 or negatives == 0:
        return 0.0
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _is_gold_eval_row(row: dict) -> bool:
    haystack = " ".join(
        str(row.get(key) or "").lower()
        for key in (
            "dataset_source",
            "label_source",
            "supervision_tier",
            "label_class",
            "dataset_role",
            "original_gold_split",
        )
    )
    return any(token in haystack for token in ("gold_eval", "gold-eval", "frontend_gold"))


def _feature_stats(rows: list[dict], names: Iterable[str]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for name in names:
        values = [_feature_value(row, name) for row in rows]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        stats[name] = {"mean": mean, "std": math.sqrt(variance)}
    return stats


def _weighted_mean(rows: list[dict], feature_name: str) -> float:
    weighted_total = 0.0
    total_weight = 0.0
    for row in rows:
        weight = float(row.get("sample_weight", 1.0) or 1.0)
        weighted_total += _feature_value(row, feature_name) * weight
        total_weight += weight
    return weighted_total / max(total_weight, 1e-6)


def _feature_value(row: dict, feature_name: str) -> float:
    value = (row.get("features") or {}).get(feature_name, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
