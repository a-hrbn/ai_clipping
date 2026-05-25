from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from config import settings
from services import clip_strategy
from services.timeline_types import Candidate, CandidateFeatures, RankedCandidate
from services.ranker import Ranker


@dataclass
class QualityGateResult:
    kept: list[RankedCandidate]
    audit_rows: list[dict[str, object]]
    summary: dict[str, object]


def rank_candidates(
    candidates: list[Candidate],
    features: list[CandidateFeatures],
    profile: str = "gaming",
    strategy: dict | None = None,
) -> list[RankedCandidate]:
    feature_map = {feature.candidate_id: feature for feature in features}
    scores = {
        str(item["candidate_id"]): item
        for item in Ranker().score_details(features, profile=profile, strategy=strategy)
    }
    ranked: list[RankedCandidate] = []
    for candidate in candidates:
        feature = feature_map.get(candidate.id)
        score_info = scores.get(candidate.id)
        if feature is None or score_info is None:
            continue
        ranked.append(
            RankedCandidate(
                candidate=candidate,
                features=feature,
                rank_score=float(score_info["score"]),
                rationale=list(score_info["rationale"]),
                heuristic_rank_score=float(score_info["heuristic_score"]),
                trained_rank_score=(
                    float(score_info["trained_score"])
                    if score_info.get("trained_score") is not None
                    else None
                ),
                model_blend=float(score_info.get("blend", 0.0)),
                trained_model_used=bool(score_info.get("trained_model_used", False)),
            )
        )

    ranked.sort(key=lambda item: item.rank_score, reverse=True)
    return ranked


def select_top_candidates(
    candidates: list[Candidate],
    features: list[CandidateFeatures],
    top_k: int,
    profile: str = "gaming",
    strategy: dict | None = None,
) -> list[RankedCandidate]:
    ranked = rank_candidates(candidates, features, profile=profile, strategy=strategy)
    return select_diverse_ranked_candidates(ranked, top_k=top_k, strategy=strategy)


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
        "judge_score": float(getattr(settings, "serving_quality_min_judge_score", 6.0)),
        "standalone_clarity": float(
            getattr(settings, "serving_quality_min_standalone_clarity", 0.55)
        ),
        "payoff_strength": float(getattr(settings, "serving_quality_min_payoff_strength", 0.55)),
        "narrative_completeness": float(
            getattr(settings, "serving_quality_min_narrative_completeness", 0.45)
        ),
        "judge_confidence": float(getattr(settings, "serving_quality_min_judge_confidence", 0.45)),
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
    }


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
