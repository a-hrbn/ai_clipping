from __future__ import annotations

from collections import Counter

from services import clip_strategy
from services.timeline_types import Candidate, CandidateFeatures, RankedCandidate
from services.ranker import Ranker


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
