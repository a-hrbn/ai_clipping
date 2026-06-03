from __future__ import annotations

from services.timeline_types import RankedCandidate


def oracle_free_recall_score(item: RankedCandidate, proposal_row: dict | None = None) -> dict[str, object]:
    """Score label-free recall signals shared by Judge reserve and gate audit.

    This deliberately excludes cross-platform recall targets. Cross-platform
    positives are training/eval supervision; normal serving protection should
    come from organic candidate shape, ranker, chat, context, and multimodal
    signals.
    """

    proposal_row = proposal_row or {}
    features = item.features
    reasons: list[str] = []
    score = 0.0

    proposal_score = _clamp01(_num(proposal_row.get("score") or proposal_row.get("model_score")))
    proposal_heuristic = _clamp01(_num(proposal_row.get("heuristic_score")))
    if proposal_score >= 0.30:
        reasons.append("proposal_score")
    if proposal_heuristic >= 0.45:
        reasons.append("proposal_heuristic")
    score += proposal_score * 0.18
    score += proposal_heuristic * 0.08

    rank_score = _clamp01(_num(item.rank_score) / 12.0)
    heuristic_rank = _clamp01(_num(item.heuristic_rank_score) / 12.0)
    if rank_score >= 0.60:
        reasons.append("ranker_score")
    if heuristic_rank >= 0.55:
        reasons.append("heuristic_ranker_score")
    score += rank_score * 0.16
    score += heuristic_rank * 0.08

    score_hint = _clamp01(_num(item.candidate.score_hint) / 10.0)
    if score_hint >= 0.50:
        reasons.append("candidate_signal")
    score += score_hint * 0.06

    if features is not None:
        context = max(
            _clamp01(getattr(features, "context_completeness_score", 0.0)),
            _clamp01(getattr(features, "context_strength_score", 0.0)),
            _clamp01(getattr(features, "context_pattern_score", 0.0)),
        )
        hook = _clamp01(_num(getattr(features, "hook_strength", 0.0)) / 10.0)
        payoff = _clamp01(_num(getattr(features, "payoff_score", 0.0)) / 10.0)
        chat = max(
            _clamp01(getattr(features, "chat_velocity_score", 0.0)),
            _clamp01(getattr(features, "chat_rate_derivative_score", 0.0)),
            _clamp01(getattr(features, "chat_ema_crossing_score", 0.0)),
            _clamp01(getattr(features, "chat_shock_score", 0.0)),
        )
        multimodal = max(
            _clamp01(getattr(features, "face_reaction_peak", 0.0)),
            _clamp01(getattr(features, "audio_pop_score", 0.0)),
            _clamp01(getattr(features, "funny_reaction_score", 0.0)),
            _clamp01(getattr(features, "multimodal_proxy_score", 0.0)),
        )
        outcome_prior = max(
            _clamp01(getattr(features, "projected_retention_score", 0.0)),
            _clamp01(getattr(features, "outcome_prior_score", 0.0)),
        )
        if context >= 0.55:
            reasons.append("context")
        if hook >= 0.55:
            reasons.append("hook")
        if payoff >= 0.55:
            reasons.append("payoff")
        if chat >= 0.45:
            reasons.append("chat_velocity")
        if multimodal >= 0.45:
            reasons.append("reaction_signal")
        if outcome_prior >= 0.45:
            reasons.append("outcome_prior")
        score += context * 0.14
        score += hook * 0.08
        score += payoff * 0.10
        score += chat * 0.10
        score += multimodal * 0.10
        score += outcome_prior * 0.06

    return {"score": round(_clamp01(score), 6), "reasons": sorted(set(reasons))}


def _num(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clamp01(value) -> float:
    return min(max(_num(value), 0.0), 1.0)
