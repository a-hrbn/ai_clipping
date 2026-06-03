from __future__ import annotations


def clamp01(value: float) -> float:
    return max(0.0, min(float(value or 0.0), 1.0))


def token_coverage_score(token_count: int | float, duration: float) -> float:
    tokens = max(float(token_count or 0.0), 0.0)
    seconds = max(float(duration or 0.0), 1.0)
    # Good short-form context usually has enough speech density to explain the
    # moment, but avoid rewarding rambling transcripts without a payoff.
    density = tokens / seconds
    return clamp01(min(tokens / 24.0, 1.0) * 0.55 + min(density / 2.2, 1.0) * 0.45)


def completeness_score(
    *,
    setup_present: float,
    payoff_present: float,
    standalone: float,
    context_efficiency: float,
    pattern_confidence: float,
    token_coverage: float,
    context_dependency: float = 0.0,
    weak_resolution: float = 0.0,
) -> float:
    raw = (
        clamp01(setup_present) * 0.20
        + clamp01(payoff_present) * 0.28
        + clamp01(standalone) * 0.22
        + clamp01(context_efficiency) * 0.10
        + clamp01(pattern_confidence) * 0.12
        + clamp01(token_coverage) * 0.08
        - clamp01(context_dependency) * 0.16
        - clamp01(weak_resolution) * 0.14
    )
    return clamp01(raw)


def strength_score(
    *,
    pattern_confidence: float,
    pattern_score: float,
    pattern_views_prior: float,
    source_prior: float,
    clip_type_signal: float,
    payoff_present: float,
    setup_present: float,
) -> float:
    raw = (
        clamp01(pattern_confidence) * 0.26
        + clamp01(pattern_score) * 0.14
        + clamp01(pattern_views_prior) * 0.12
        + clamp01(source_prior) * 0.12
        + clamp01(clip_type_signal) * 0.16
        + clamp01(payoff_present) * 0.12
        + clamp01(setup_present) * 0.08
    )
    return clamp01(raw)


def primary_score(*, completeness: float, strength: float) -> float:
    return clamp01(clamp01(completeness) * 0.62 + clamp01(strength) * 0.38)


def source_prior(source: str) -> float:
    priors = {
        "peak": 0.95,
        "hype_peak": 0.85,
        "story_payoff": 0.85,
        "tension_build": 0.78,
        "quote": 0.76,
        "reaction": 0.72,
        "chat_sync": 0.72,
        "chat_velocity": 0.78,
        "goal_prompt": 0.70,
        "clean_reaction": 0.68,
        "controversy": 0.67,
        "narrative_reveal": 0.65,
        "dense_window": 0.48,
        "dense_inventory": 0.38,
    }
    return priors.get(source, 0.45)
