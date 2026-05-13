from __future__ import annotations

import argparse
from collections import Counter
import json
import pathlib
import sys

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import build_eval_dataset, train_ranker
from services import selector
from services.ranker import Ranker
from services.timeline_types import Candidate, CandidateFeatures, RankedCandidate


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate heuristic pipeline ranking on an eval dataset.")
    parser.add_argument(
        "--dataset",
        default=str(BACKEND_ROOT / "ml" / "artifacts" / "eval_dataset.jsonl"),
    )
    parser.add_argument("--profile", default="general")
    parser.add_argument("--artifact-dir", default=str(BACKEND_ROOT / "ml" / "artifacts"))
    parser.add_argument("--top-groups", type=int, default=5)
    parser.add_argument("--selection-top-k", type=int, default=12)
    parser.add_argument(
        "--evaluation-mode",
        choices=("heuristic", "trained_product"),
        default="heuristic",
        help="Use trained_product to evaluate Ranker(enable_trained_model=True) plus product diversity selection.",
    )
    parser.add_argument(
        "--exclude-proxy-labels",
        action="store_true",
        help="Exclude noisy Twitch clip proxy rows from the heuristic pipeline eval scope.",
    )
    args = parser.parse_args()

    rows = train_ranker._read_jsonl(pathlib.Path(args.dataset))
    if not rows:
        print("No eval dataset rows found.")
        return 1

    result = evaluate_pipeline_rows(
        rows,
        profile=args.profile,
        artifact_dir=pathlib.Path(args.artifact_dir),
        top_groups=args.top_groups,
        selection_top_k=args.selection_top_k,
        evaluation_mode=args.evaluation_mode,
        exclude_proxy_labels=args.exclude_proxy_labels,
    )
    print(json.dumps(result, ensure_ascii=True))
    return 0


def evaluate_pipeline_rows(
    rows: list[dict],
    *,
    profile: str,
    artifact_dir: pathlib.Path | None = None,
    top_groups: int = 5,
    selection_top_k: int = 12,
    evaluation_mode: str = "heuristic",
    exclude_proxy_labels: bool = False,
) -> dict[str, object]:
    input_rows = len(rows)
    rows, excluded_counts = _pipeline_eval_rows(rows, exclude_proxy_labels=exclude_proxy_labels)
    excluded_sources = ["boundary_perturbation"]
    if exclude_proxy_labels:
        excluded_sources.append("twitch_clip_proxy")
    mode = str(evaluation_mode or "heuristic").strip().lower()
    trained_mode = mode == "trained_product"
    ranker = Ranker(artifact_dir=artifact_dir, enable_trained_model=trained_mode)
    artifact_inspection = ranker.inspect_artifact(profile) if trained_mode else {}
    features = [CandidateFeatures.from_dict(row["features"]) for row in rows]
    details = ranker.score_details(features, profile=profile)
    scores = {
        str(item["candidate_id"]): float(item["score"])
        for item in details
    }
    detail_by_id = {str(item["candidate_id"]): item for item in details}
    labels = train_ranker.np.array([int(row["label"]) for row in rows], dtype=int)
    score_array = train_ranker.np.array([float(scores.get(row["candidate_id"], 0.0)) for row in rows], dtype=float)
    groups = [str(row.get("eval_group") or build_eval_dataset._eval_group(row)) for row in rows]
    predictions = train_ranker.np.array(
        [
            bool(detail_by_id.get(str(row["candidate_id"]), {}).get("passes_serving_threshold", False))
            if trained_mode
            else float(scores.get(row["candidate_id"], 0.0)) >= train_ranker._default_classification_threshold("margin")
            for row in rows
        ],
        dtype=bool,
    )
    ranked_candidates = _ranked_candidates(rows, features, detail_by_id)
    selected_candidates = _select_eval_candidates(
        rows,
        ranked_candidates,
        details=detail_by_id,
        trained_mode=trained_mode,
        selection_top_k=int(selection_top_k),
    )
    selected_candidate_ids = {item.candidate.id for item in selected_candidates}

    return {
        "evaluation_kind": "trained_product_pipeline" if trained_mode else "heuristic_pipeline",
        "score_source": (
            "services.ranker.Ranker(enable_trained_model=True)+selector.select_diverse_ranked_candidates"
            if trained_mode
            else "services.ranker.Ranker(enable_trained_model=False)"
        ),
        "trained_model_used": bool(trained_mode and any(bool(item.get("trained_model_used")) for item in details)),
        "model_evaluation": bool(trained_mode),
        "interpretation_warning": _interpretation_warning(trained_mode),
        "eval_scope": "no_twitch_clip_proxy" if exclude_proxy_labels else "all_non_boundary_rows",
        "selection_top_k": int(selection_top_k),
        "input_rows": input_rows,
        "rows": len(rows),
        "excluded_rows": input_rows - len(rows),
        "excluded_rows_by_reason": dict(sorted(excluded_counts.items())),
        "excluded_sources": excluded_sources,
        "positive_rows": int(sum(labels)),
        "negative_rows": int(len(rows) - sum(labels)),
        "artifact": _artifact_summary(artifact_inspection) if trained_mode else {},
        "metrics": {
            "classification": (
                _classification_from_predictions(labels, predictions, groups=groups)
                if trained_mode
                else train_ranker._evaluate_scores(labels, score_array, score_kind="margin", groups=groups)
            ),
            "ranking": train_ranker._evaluate_scores(labels, score_array, score_kind="margin", groups=groups),
            "diverse_selection": _selection_metrics(rows, selected_candidates, detail_by_id),
            "top1_positive_rate": round(_top1_positive_rate(rows, scores), 3),
            "recall_at_10": round(_group_recall_at_k(rows, scores, 10), 3),
            "recall_at_25": round(_group_recall_at_k(rows, scores, 25), 3),
            "mrr": round(_mean_reciprocal_rank(rows, scores), 3),
        },
        "by_platform": _bucket_metrics(rows, scores, "platform_bucket"),
        "by_content_bucket": _bucket_metrics(rows, scores, "content_bucket"),
        "by_source": _bucket_metrics(rows, scores, "source_bucket"),
        "serving_reranker": _serving_reranker_summary(details),
        "top_groups": _top_groups(rows, scores, top_groups),
        "top_candidates": _top_candidates(
            rows,
            scores,
            detail_by_id,
            selected_candidate_ids=selected_candidate_ids,
            top_k=25,
        ),
    }


def _group_rows(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        group = str(row.get("eval_group") or build_eval_dataset._eval_group(row))
        grouped.setdefault(group, []).append(row)
    return grouped


def _pipeline_eval_rows(rows: list[dict], *, exclude_proxy_labels: bool = False) -> tuple[list[dict], Counter[str]]:
    filtered: list[dict] = []
    excluded_counts: Counter[str] = Counter()
    for row in rows:
        if _is_boundary_perturbation_row(row):
            excluded_counts["boundary_perturbation"] += 1
            continue
        if exclude_proxy_labels and _is_twitch_proxy_eval_row(row):
            excluded_counts["twitch_clip_proxy"] += 1
            continue
        filtered.append(row)
    return filtered, excluded_counts


def _row_source(row: dict) -> str:
    return str(
        row.get("source_bucket")
        or row.get("candidate_source")
        or (row.get("features") or {}).get("source")
        or ""
    ).strip().lower()


def _is_boundary_perturbation_row(row: dict) -> bool:
    return _row_source(row) == "boundary_perturbation"


def _is_twitch_proxy_eval_row(row: dict) -> bool:
    supervision_tier = str(row.get("supervision_tier") or "").strip().lower()
    dataset_source = str(row.get("dataset_source") or "").strip().lower()
    return (
        supervision_tier == "twitch_clip_proxy"
        or dataset_source == "twitch_vod_primary"
        or _row_source(row) == "twitch_label_proxy"
    )


def _top1_positive_rate(rows: list[dict], scores: dict[str, float]) -> float:
    grouped = _group_rows(rows)
    hits = 0
    considered = 0
    for subset in grouped.values():
        if not subset:
            continue
        ordered = sorted(subset, key=lambda row: float(scores.get(row["candidate_id"], 0.0)), reverse=True)
        hits += int(ordered[0].get("label", 0) == 1)
        considered += 1
    return hits / max(considered, 1)


def _group_recall_at_k(rows: list[dict], scores: dict[str, float], k: int) -> float:
    grouped = _group_rows(rows)
    values: list[float] = []
    for subset in grouped.values():
        positives = [row for row in subset if int(row.get("label", 0)) == 1]
        if not positives:
            continue
        ordered = sorted(subset, key=lambda row: float(scores.get(row["candidate_id"], 0.0)), reverse=True)
        top = ordered[: min(k, len(ordered))]
        hits = sum(1 for row in top if int(row.get("label", 0)) == 1)
        values.append(hits / max(len(positives), 1))
    return sum(values) / max(len(values), 1)


def _mean_reciprocal_rank(rows: list[dict], scores: dict[str, float]) -> float:
    grouped = _group_rows(rows)
    values: list[float] = []
    for subset in grouped.values():
        ordered = sorted(subset, key=lambda row: float(scores.get(row["candidate_id"], 0.0)), reverse=True)
        reciprocal = 0.0
        for index, row in enumerate(ordered, start=1):
            if int(row.get("label", 0)) == 1:
                reciprocal = 1.0 / index
                break
        values.append(reciprocal)
    return sum(values) / max(len(values), 1)


def _bucket_metrics(rows: list[dict], scores: dict[str, float], key: str) -> dict[str, dict[str, object]]:
    values = sorted({str(row.get(key) or "").strip() for row in rows if str(row.get(key) or "").strip()})
    bucketed: dict[str, dict[str, object]] = {}
    for value in values:
        subset = [row for row in rows if str(row.get(key) or "").strip() == value]
        labels = train_ranker.np.array([int(row["label"]) for row in subset], dtype=int)
        subset_scores = train_ranker.np.array([float(scores.get(row["candidate_id"], 0.0)) for row in subset], dtype=float)
        groups = [str(row.get("eval_group") or build_eval_dataset._eval_group(row)) for row in subset]
        bucketed[value] = {
            "rows": len(subset),
            "classification": train_ranker._evaluate_scores(labels, subset_scores, score_kind="margin", groups=groups),
            "top1_positive_rate": round(_top1_positive_rate(subset, scores), 3),
            "recall_at_10": round(_group_recall_at_k(subset, scores, 10), 3),
        }
    return bucketed


def _top_groups(rows: list[dict], scores: dict[str, float], top_k: int) -> list[dict[str, object]]:
    grouped = _group_rows(rows)
    summaries: list[dict[str, object]] = []
    for group, subset in grouped.items():
        labels = train_ranker.np.array([int(row["label"]) for row in subset], dtype=int)
        subset_scores = train_ranker.np.array([float(scores.get(row["candidate_id"], 0.0)) for row in subset], dtype=float)
        summaries.append(
            {
                "group": group,
                "rows": len(subset),
                "top1_positive_rate": round(_top1_positive_rate(subset, scores), 3),
                "mrr": round(_mean_reciprocal_rank(subset, scores), 3),
                "classification": train_ranker._evaluate_scores(
                    labels,
                    subset_scores,
                    score_kind="margin",
                    groups=[group for _ in subset],
                ),
            }
        )
    summaries.sort(key=lambda item: (item["rows"], item["group"]), reverse=True)
    return summaries[: max(int(top_k), 0)]


def _serving_reranker_summary(details: list[dict[str, object]]) -> dict[str, object]:
    rationale_counts: Counter[str] = Counter()
    for item in details:
        rationale_counts.update(str(reason) for reason in (item.get("rationale") or []))
    return {
        "rows": len(details),
        "hook_reranker_boost": int(rationale_counts.get("hook_reranker_boost", 0)),
        "hook_reranker_penalty": int(rationale_counts.get("hook_reranker_penalty", 0)),
        "boundary_variant_set_max": int(rationale_counts.get("boundary_variant_set_max", 0)),
        "rationale_counts": dict(sorted(rationale_counts.items())),
    }


def _interpretation_warning(trained_mode: bool) -> str:
    if trained_mode:
        return (
            "This report evaluates the trained ranker with serving thresholds "
            "and product-like per-group diversity selection."
        )
    return (
        "This report evaluates the heuristic pipeline ranker only. "
        "Use ml.evaluate_ranker outputs for trained-model metrics."
    )


def _artifact_summary(inspection: dict[str, object]) -> dict[str, object]:
    artifact = inspection.get("artifact") if isinstance(inspection.get("artifact"), dict) else {}
    return {
        "available": bool(artifact),
        "status": str(inspection.get("status") or ""),
        "reason": str(inspection.get("reason") or ""),
        "requested_profile": str(inspection.get("requested_profile") or ""),
        "resolved_profile": str(inspection.get("resolved_profile") or ""),
        "selected_model": str(artifact.get("selected_model") or ""),
        "model_type": str(artifact.get("model_type") or ""),
        "serving_threshold": artifact.get("serving_threshold"),
        "serving_threshold_calibration": artifact.get("serving_threshold_calibration") or {},
    }


def _classification_from_predictions(
    labels: train_ranker.np.ndarray,
    predictions: train_ranker.np.ndarray,
    *,
    groups: list[str],
) -> dict[str, float]:
    return train_ranker._evaluate_scores(
        labels,
        predictions.astype(float),
        score_kind="probability",
        groups=groups,
        threshold=0.5,
    )


def _ranked_candidates(
    rows: list[dict],
    features: list[CandidateFeatures],
    details: dict[str, dict[str, object]],
) -> list[RankedCandidate]:
    feature_by_id = {feature.candidate_id: feature for feature in features}
    ranked: list[RankedCandidate] = []
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "")
        detail = details.get(candidate_id, {})
        feature = feature_by_id.get(candidate_id)
        ranked.append(
            RankedCandidate(
                candidate=_candidate_from_row(row, feature),
                features=feature,
                rank_score=float(detail.get("score") or 0.0),
                rationale=list(detail.get("rationale") or []),
                heuristic_rank_score=(
                    float(detail["heuristic_score"])
                    if detail.get("heuristic_score") is not None
                    else None
                ),
                trained_rank_score=(
                    float(detail["trained_score"])
                    if detail.get("trained_score") is not None
                    else None
                ),
                model_blend=float(detail.get("blend") or 0.0),
                trained_model_used=bool(detail.get("trained_model_used", False)),
            )
        )
    ranked.sort(key=lambda item: item.rank_score, reverse=True)
    return ranked


def _candidate_from_row(row: dict, feature: CandidateFeatures | None) -> Candidate:
    start = _row_time(row, "window_start", "final_start", "proxy_window_start", default=0.0)
    end = _row_time(row, "window_end", "final_end", "proxy_window_end", default=0.0)
    if end <= start:
        duration = float(getattr(feature, "duration", 0.0) or 0.0)
        end = start + max(duration, 1.0)
    source = _row_source(row) or str(getattr(feature, "source", "") or "unknown")
    return Candidate(
        id=str(row.get("candidate_id") or ""),
        start=float(start),
        end=float(end),
        source=source or "unknown",
        anchor_time=(float(start) + float(end)) / 2.0,
        score_hint=float(row.get("viral_score") or row.get("label_score") or 0.0),
    )


def _row_time(row: dict, *keys: str, default: float) -> float:
    for key in keys:
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return float(default)


def _select_eval_candidates(
    rows: list[dict],
    ranked: list[RankedCandidate],
    *,
    details: dict[str, dict[str, object]],
    trained_mode: bool,
    selection_top_k: int,
) -> list[RankedCandidate]:
    if selection_top_k <= 0:
        return []
    ranked_by_id = {item.candidate.id: item for item in ranked}
    selected: list[RankedCandidate] = []
    for group, subset in _group_rows(rows).items():
        del group
        group_ranked = [ranked_by_id[str(row.get("candidate_id") or "")] for row in subset if str(row.get("candidate_id") or "") in ranked_by_id]
        if trained_mode:
            group_ranked = [
                item for item in group_ranked
                if _passes_serving_threshold(details.get(item.candidate.id, {}))
            ]
        group_ranked.sort(key=lambda item: item.rank_score, reverse=True)
        selected.extend(
            selector.select_diverse_ranked_candidates(
                group_ranked,
                top_k=selection_top_k,
            )
        )
    return selected


def _passes_serving_threshold(detail: dict[str, object]) -> bool:
    if detail.get("serving_threshold") is None:
        return False
    try:
        return float(detail.get("score") or 0.0) >= float(detail.get("serving_threshold") or 0.0)
    except (TypeError, ValueError):
        return False


def _selection_metrics(
    rows: list[dict],
    selected_candidates: list[RankedCandidate],
    details: dict[str, dict[str, object]],
) -> dict[str, object]:
    selected_ids = {item.candidate.id for item in selected_candidates}
    positives = [row for row in rows if int(row.get("label", 0)) == 1]
    selected_rows = [row for row in rows if str(row.get("candidate_id") or "") in selected_ids]
    selected_positive_rows = [row for row in selected_rows if int(row.get("label", 0)) == 1]
    serving_pass_rows = [
        row for row in rows
        if _passes_serving_threshold(details.get(str(row.get("candidate_id") or ""), {}))
    ]
    return {
        "selection_policy": "per_eval_group_serving_threshold_then_diversity",
        "selected_rows": len(selected_rows),
        "selected_positive_rows": len(selected_positive_rows),
        "serving_pass_rows": len(serving_pass_rows),
        "precision": round(len(selected_positive_rows) / max(len(selected_rows), 1), 3),
        "recall": round(len(selected_positive_rows) / max(len(positives), 1), 3),
        "selected_rate": round(len(selected_rows) / max(len(rows), 1), 3),
        "selection_groups": len(_group_rows(rows)),
    }


def _top_candidates(
    rows: list[dict],
    scores: dict[str, float],
    details: dict[str, dict[str, object]],
    *,
    selected_candidate_ids: set[str] | None = None,
    top_k: int,
) -> list[dict[str, object]]:
    ordered = sorted(rows, key=lambda row: float(scores.get(row["candidate_id"], 0.0)), reverse=True)
    output: list[dict[str, object]] = []
    for row in ordered[: max(int(top_k), 0)]:
        candidate_id = str(row.get("candidate_id") or "")
        detail = details.get(candidate_id, {})
        output.append(
            {
                "candidate_id": candidate_id,
                "eval_group": str(row.get("eval_group") or build_eval_dataset._eval_group(row)),
                "label": int(row.get("label", 0)),
                "score": round(float(scores.get(candidate_id, 0.0)), 3),
                "serving_threshold": detail.get("serving_threshold"),
                "passes_serving_threshold": _passes_serving_threshold(detail),
                "score_margin": (
                    round(float(scores.get(candidate_id, 0.0)) - float(detail.get("serving_threshold")), 3)
                    if detail.get("serving_threshold") is not None
                    else None
                ),
                "selected_after_diversity": candidate_id in (selected_candidate_ids or set()),
                "trained_model_used": bool(detail.get("trained_model_used", False)),
                "rationale": list(detail.get("rationale") or []),
                "hook_reranker_score": detail.get("hook_reranker_score"),
                "boundary_variant_set_score": detail.get("boundary_variant_set_score"),
            }
        )
    return output


if __name__ == "__main__":
    raise SystemExit(main())
