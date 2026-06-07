from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from config import settings
from services import clip_strategy, high_recall_protection
from services.timeline_types import RankedCandidate


@dataclass
class QualityGateResult:
    kept: list[RankedCandidate]
    audit_rows: list[dict[str, object]]
    summary: dict[str, object]


@dataclass
class FinalSelectionResult:
    selected: list[RankedCandidate]
    summary: dict[str, object]


def select_diverse_ranked_candidates(
    ranked: list[RankedCandidate],
    top_k: int,
    min_gap_sec: float = 20.0,
    strategy: dict | None = None,
) -> list[RankedCandidate]:
    if top_k <= 0 or not ranked:
        return []
    normalized_strategy = clip_strategy.normalize_strategy(strategy)
    if normalized_strategy["platform"] in {"tiktok", "instagram_reels"}:
        min_gap_sec = min(min_gap_sec, 12.0)
    elif normalized_strategy["platform"] == "linkedin":
        min_gap_sec = max(min_gap_sec, 24.0)

    kept: list[RankedCandidate] = []
    source_counts: Counter[str] = Counter()
    source_cap = max(1, min(4, top_k // 2 + top_k % 2))

    for item in ranked:
        if len(kept) >= top_k:
            break
        if source_counts[item.candidate.source] >= source_cap:
            continue
        if _too_close_to_any(item, kept, min_gap_sec=min_gap_sec):
            continue
        kept.append(item)
        source_counts[item.candidate.source] += 1

    if len(kept) < top_k:
        for item in ranked:
            if len(kept) >= top_k:
                break
            if item in kept:
                continue
            kept.append(item)

    return kept


def select_post_judge_final_candidates(
    ranked: list[RankedCandidate],
    top_k: int,
    min_gap_sec: float = 20.0,
    strategy: dict | None = None,
) -> FinalSelectionResult:
    """Select the final candidate pool after the LLM judge.

    The normal selector is intentionally source-balanced because it runs before
    the judge. After an LLM judge result exists, the judge should be the final
    decision maker: accepted judged candidates are ordered by judge quality and
    only dropped for temporal overlap/diversity. If too few judged candidates
    survive, return fewer clips instead of filling with unjudged candidates.
    """
    if top_k <= 0 or not ranked:
        return FinalSelectionResult(
            selected=[],
            summary=_post_judge_selection_summary([], [], [], [], 0, top_k, min_gap_sec, strategy),
        )

    normalized_strategy = clip_strategy.normalize_strategy(strategy)
    if normalized_strategy["platform"] in {"tiktok", "instagram_reels"}:
        min_gap_sec = min(min_gap_sec, 12.0)
    elif normalized_strategy["platform"] == "linkedin":
        min_gap_sec = max(min_gap_sec, 24.0)

    judged = [item for item in ranked if item.judge_result is not None]
    unjudged = [item for item in ranked if item.judge_result is None]
    judged_sorted = sorted(judged, key=_post_judge_sort_key, reverse=True)
    unjudged_sorted = sorted(unjudged, key=lambda item: item.rank_score, reverse=True)
    ranker_bypass_unjudged = [
        item
        for item in unjudged_sorted
        if _passes_unjudged_ranker_final_bypass(item)
    ]

    selected = _select_with_gap(judged_sorted, top_k=top_k, min_gap_sec=min_gap_sec)
    strict_selected_count = len(selected)
    relaxed_gap_sec = min(15.0, min_gap_sec)
    if len(selected) < min(top_k, len(judged_sorted)) and relaxed_gap_sec < min_gap_sec:
        selected = _select_with_gap(judged_sorted, top_k=top_k, min_gap_sec=relaxed_gap_sec)
    ranker_bypass_selected_count = 0
    if len(selected) < top_k and ranker_bypass_unjudged:
        for item in ranker_bypass_unjudged:
            if len(selected) >= top_k:
                break
            if _too_close_to_any(item, selected, min_gap_sec=relaxed_gap_sec):
                continue
            if "post_judge_final_ranker_bypass" not in item.rationale:
                item.rationale.append("post_judge_final_ranker_bypass")
            selected.append(item)
            ranker_bypass_selected_count += 1

    for item in selected:
        if item.judge_result is not None and "post_judge_final_primary" not in item.rationale:
            item.rationale.append("post_judge_final_primary")

    return FinalSelectionResult(
        selected=selected,
        summary=_post_judge_selection_summary(
            ranked,
            judged_sorted,
            unjudged_sorted,
            selected,
            0,
            top_k,
            min_gap_sec,
            normalized_strategy,
            strict_selected_count=strict_selected_count,
            relaxed_gap_sec=relaxed_gap_sec,
            ranker_bypass_unjudged_candidates=len(ranker_bypass_unjudged),
            ranker_bypass_unjudged_selected=ranker_bypass_selected_count,
        ),
    )


def apply_serving_quality_gate(ranked: list[RankedCandidate]) -> QualityGateResult:
    thresholds = _serving_quality_gate_thresholds()
    enabled = bool(getattr(settings, "serving_quality_gate_enabled", True))
    if not enabled:
        return QualityGateResult(
            kept=list(ranked),
            audit_rows=[_quality_gate_audit_row(item, accepted=True, reasons=[]) for item in ranked],
            summary={
                "enabled": False,
                "input_candidates": len(ranked),
                "kept_candidates": len(ranked),
                "rejected_candidates": 0,
                "thresholds": thresholds,
                "rejected_by_reason": {},
                "kept_without_judge_result": sum(1 for item in ranked if item.judge_result is None),
            },
        )

    kept: list[RankedCandidate] = []
    audit_rows: list[dict[str, object]] = []
    rejected_by_reason: Counter[str] = Counter()
    kept_without_judge_result = 0
    for item in ranked:
        reasons = _serving_quality_gate_rejection_reasons(item, thresholds)
        accepted = not reasons
        audit_rows.append(_quality_gate_audit_row(item, accepted=accepted, reasons=reasons))
        if accepted:
            kept.append(item)
            if item.judge_result is None:
                kept_without_judge_result += 1
            continue
        rejected_by_reason.update(reasons)

    return QualityGateResult(
        kept=kept,
        audit_rows=audit_rows,
        summary={
            "enabled": True,
            "input_candidates": len(ranked),
            "kept_candidates": len(kept),
            "rejected_candidates": len(ranked) - len(kept),
            "thresholds": thresholds,
            "rejected_by_reason": dict(sorted(rejected_by_reason.items())),
            "kept_without_judge_result": kept_without_judge_result,
        },
    )


def _serving_quality_gate_thresholds() -> dict[str, float]:
    return {
        "judge_score": float(getattr(settings, "serving_quality_min_judge_score", 4.4)),
        "standalone_clarity": float(
            getattr(settings, "serving_quality_min_standalone_clarity", 0.35)
        ),
        "payoff_strength": float(getattr(settings, "serving_quality_min_payoff_strength", 0.35)),
        "narrative_completeness": float(
            getattr(settings, "serving_quality_min_narrative_completeness", 0.45)
        ),
            "judge_confidence": float(getattr(settings, "serving_quality_min_judge_confidence", 0.45)),
        "hard_min_judge_score": float(
            getattr(settings, "serving_quality_hard_min_judge_score", 3.0)
        ),
    }


def _serving_quality_gate_rejection_reasons(
    item: RankedCandidate,
    thresholds: dict[str, float],
) -> list[str]:
    if item.judge_result is None:
        if "judge_recall_filtered" in item.rationale:
            return ["judge_recall_filtered"]
        return []

    judge = item.judge_result
    if _is_cross_platform_recall_target(item):
        min_judge = float(getattr(settings, "serving_cross_platform_quality_min_judge_score", 3.0) or 0.0)
        if float(judge.judge_score) >= min_judge and float(judge.judge_confidence) >= thresholds["judge_confidence"]:
            return []
    if _passes_ranker_quality_bypass(item):
        return []
    if _serving_quality_subscore_gate_mode() == "soft":
        return _soft_serving_quality_gate_rejection_reasons(judge, thresholds)
    if _passes_soft_judge_quality_gate(judge, thresholds):
        return []
    checks = {
        "judge_score": judge.judge_score,
        "standalone_clarity": judge.standalone_clarity,
        "payoff_strength": judge.payoff_strength,
        "narrative_completeness": judge.narrative_completeness,
        "judge_confidence": judge.judge_confidence,
    }
    return [
        f"{name}_below_min"
        for name, value in checks.items()
        if float(value) < thresholds[name]
    ]


def _quality_gate_audit_row(
    item: RankedCandidate,
    *,
    accepted: bool,
    reasons: list[str],
) -> dict[str, object]:
    judge = item.judge_result
    return {
        "candidate_id": item.candidate.id,
        "source": item.candidate.source,
        "start": item.candidate.start,
        "end": item.candidate.end,
        "rank_score": item.rank_score,
        "accepted": accepted,
        "rejection_reasons": list(reasons),
        "rationale": list(item.rationale),
        "has_judge_result": judge is not None,
        "judge_score": judge.judge_score if judge is not None else None,
        "standalone_clarity": judge.standalone_clarity if judge is not None else None,
        "payoff_strength": judge.payoff_strength if judge is not None else None,
        "narrative_completeness": judge.narrative_completeness if judge is not None else None,
        "judge_confidence": judge.judge_confidence if judge is not None else None,
        "judge_reason": judge.judge_reason if judge is not None else "",
        "judge_provider": judge.provider if judge is not None else "",
        "cross_platform_recall_target": _is_cross_platform_recall_target(item),
        "ranker_quality_bypass": _passes_ranker_quality_bypass(item),
        "ranker_quality_bypass_signal": _has_ranker_quality_bypass_signal(item),
        "high_recall_protection": high_recall_protection.oracle_free_recall_score(item),
        "soft_judge_quality_gate": (
            _passes_soft_judge_quality_gate(judge, _serving_quality_gate_thresholds())
            if judge is not None
            else False
        ),
        "serving_quality_subscore_gate_mode": _serving_quality_subscore_gate_mode(),
    }


def _is_cross_platform_recall_target(item: RankedCandidate) -> bool:
    if "cross_platform_recall_target" in item.rationale:
        return True
    feature = item.features
    return bool(feature is not None and getattr(feature, "cross_platform_recall_target", False))


def _passes_ranker_quality_bypass(item: RankedCandidate) -> bool:
    if item.judge_result is None:
        return False
    return _has_ranker_quality_bypass_signal(item)


def _passes_unjudged_ranker_final_bypass(item: RankedCandidate) -> bool:
    if item.judge_result is not None:
        return False
    return _has_ranker_quality_bypass_signal(item)


def _has_ranker_quality_bypass_signal(item: RankedCandidate) -> bool:
    return False


def _passes_soft_judge_quality_gate(
    judge,
    thresholds: dict[str, float],
) -> bool:
    if not bool(getattr(settings, "serving_quality_soft_single_subscore_miss_enabled", True)):
        return False
    if float(judge.judge_confidence) < thresholds["judge_confidence"]:
        return False
    min_judge = float(
        getattr(
            settings,
            "serving_quality_soft_single_subscore_min_judge_score",
            max(thresholds["judge_score"], 6.0),
        )
        or 0.0
    )
    if float(judge.judge_score) < max(thresholds["judge_score"], min_judge):
        return False
    axis_values = {
        "standalone_clarity": float(judge.standalone_clarity),
        "payoff_strength": float(judge.payoff_strength),
        "narrative_completeness": float(judge.narrative_completeness),
    }
    below = [
        name
        for name, value in axis_values.items()
        if value < thresholds[name]
    ]
    if len(below) != 1:
        return False
    axis_mean = sum(axis_values.values()) / len(axis_values)
    min_axis_mean = float(
        getattr(settings, "serving_quality_soft_single_subscore_min_axis_mean", 0.55)
        or 0.0
    )
    return axis_mean >= min_axis_mean


def _serving_quality_subscore_gate_mode() -> str:
    mode = str(getattr(settings, "serving_quality_subscore_gate_mode", "soft") or "soft").strip().lower()
    if mode not in {"soft", "hard"}:
        return "soft"
    return mode


def _soft_serving_quality_gate_rejection_reasons(
    judge,
    thresholds: dict[str, float],
) -> list[str]:
    reasons: list[str] = []
    if float(judge.judge_confidence) < thresholds["judge_confidence"]:
        reasons.append("judge_confidence_below_min")
    if float(judge.judge_score) < thresholds["hard_min_judge_score"]:
        reasons.append("judge_score_below_hard_min")
    return reasons


def _too_close_to_any(
    item: RankedCandidate,
    kept: list[RankedCandidate],
    *,
    min_gap_sec: float,
) -> bool:
    for other in kept:
        gap = (
            max(item.candidate.start, other.candidate.start)
            - min(item.candidate.end, other.candidate.end)
        )
        if gap < min_gap_sec:
            return True
    return False


def _select_with_gap(
    ranked: list[RankedCandidate],
    *,
    top_k: int,
    min_gap_sec: float,
) -> list[RankedCandidate]:
    kept: list[RankedCandidate] = []
    for item in ranked:
        if len(kept) >= top_k:
            break
        if _too_close_to_any(item, kept, min_gap_sec=min_gap_sec):
            continue
        kept.append(item)
    return kept


def _post_judge_sort_key(item: RankedCandidate) -> tuple[float, float, float, float, float, float, float]:
    judge = item.judge_result
    if judge is None:
        return (0.0, 0.0, 0.0, 0.0, 0.0, float(item.rank_score), -float(item.candidate.start))
    return (
        float(item.rank_score),
        float(judge.judge_score),
        float(judge.judge_confidence),
        float(judge.platform_fit_shortform),
        float(judge.payoff_strength),
        float(judge.scroll_stop_power),
        -float(item.candidate.start),
    )


def _post_judge_selection_summary(
    input_ranked: list[RankedCandidate],
    judged_sorted: list[RankedCandidate],
    unjudged_sorted: list[RankedCandidate],
    selected: list[RankedCandidate],
    fallback_used: int,
    top_k: int,
    min_gap_sec: float,
    strategy: dict | None,
    *,
    strict_selected_count: int | None = None,
    relaxed_gap_sec: float | None = None,
    ranker_bypass_unjudged_candidates: int = 0,
    ranker_bypass_unjudged_selected: int = 0,
) -> dict[str, object]:
    selected_ids = {item.candidate.id for item in selected}
    selected_judged = [item for item in selected if item.judge_result is not None]
    dropped_judged = [item for item in judged_sorted if item.candidate.id not in selected_ids]
    return {
        "policy": "blended_ranker_judge_temporal_diversity",
        "input_candidates": len(input_ranked),
        "judged_input_candidates": len(judged_sorted),
        "unjudged_input_candidates": len(unjudged_sorted),
        "selected_candidates": len(selected),
        "selected_judged_candidates": len(selected_judged),
        "selected_unjudged_candidates": len(selected) - len(selected_judged),
        "fallback_unjudged_candidates": int(fallback_used),
        "unjudged_fallback_enabled": False,
        "ranker_bypass_unjudged_candidates": int(ranker_bypass_unjudged_candidates),
        "ranker_bypass_unjudged_selected": int(ranker_bypass_unjudged_selected),
        "missing_judged_candidates": max(0, min(int(top_k), len(judged_sorted)) - len(selected)),
        "missing_final_slots": max(0, int(top_k) - len(selected)),
        "top_k": int(top_k),
        "min_gap_sec": float(min_gap_sec),
        "relaxed_gap_sec": float(relaxed_gap_sec if relaxed_gap_sec is not None else min_gap_sec),
        "strict_selected_candidates": strict_selected_count,
        "source_cap_enabled": False,
        "strategy_platform": (strategy or {}).get("platform") if isinstance(strategy, dict) else None,
        "selected": [_post_judge_candidate_summary(item) for item in selected],
        "dropped_judged_samples": [
            {
                **_post_judge_candidate_summary(item),
                "drop_reason": "temporal_overlap_or_below_selected_judge_rank",
            }
            for item in dropped_judged[:12]
        ],
    }


def _post_judge_candidate_summary(item: RankedCandidate) -> dict[str, object]:
    judge = item.judge_result
    return {
        "candidate_id": item.candidate.id,
        "source": item.candidate.source,
        "start": round(float(item.candidate.start), 3),
        "end": round(float(item.candidate.end), 3),
        "rank_score": round(float(item.rank_score), 3),
        "judge_score": round(float(judge.judge_score), 3) if judge is not None else None,
        "judge_confidence": round(float(judge.judge_confidence), 3) if judge is not None else None,
        "judge_provider": str(judge.provider) if judge is not None else "",
    }
