from __future__ import annotations

from services import clip_similarity, profile_config, semantic_service
from services.timeline_types import Candidate, CandidateFeatures
from services.transcriber import RichSegment


def build_features(
    candidates: list[Candidate],
    segments: list[RichSegment],
    profile: str = "gaming",
    use_llm_semantics: bool = True,
    llm_candidate_ids: set[str] | None = None,
    llm_provider: str | None = None,
) -> list[CandidateFeatures]:
    return [
        _build_candidate_features(
            candidate,
            segments,
            profile,
            use_llm_semantics=use_llm_semantics,
            llm_candidate_ids=llm_candidate_ids,
            llm_provider=llm_provider,
        )
        for candidate in candidates
    ]


def _build_candidate_features(
    candidate: Candidate,
    segments: list[RichSegment],
    profile: str,
    *,
    use_llm_semantics: bool = True,
    llm_candidate_ids: set[str] | None = None,
    llm_provider: str | None = None,
) -> CandidateFeatures:
    window = [seg for seg in segments if seg.end > candidate.start and seg.start < candidate.end]
    if not window:
        return CandidateFeatures(
            candidate_id=candidate.id,
            duration=max(0.0, candidate.end - candidate.start),
            hook_delay_sec=0.0,
            hook_strength=0.0,
            max_hype=0.0,
            mean_hype=0.0,
            arc_score=0.0,
            chat_peak=0.0,
            motion_peak=0.0,
            face_reaction_peak=0.0,
            speaker_turns=0,
            exclamation_count=0,
            laughter_count=0,
            question_open=False,
            context_cost=1.0,
            payoff_score=0.0,
            context_efficiency=0.0,
            quoteability_score=0.0,
            surprise_score=0.0,
            contradiction_score=0.0,
            standalone_score=0.0,
            semantic_density=0.0,
            profile_fit_score=0.0,
            pre_hook_hype=0.0,
            post_hook_hype=0.0,
            hype_delta=0.0,
            local_peak_density=0.0,
            winner_similarity_max=0.0,
            winner_similarity_mean=0.0,
            loser_similarity_max=0.0,
            similarity_margin=0.0,
            neighbor_quality_score=0.0,
            vod_progress=0.0,
            boundary_left_context_sec=0.0,
            boundary_right_context_sec=0.0,
            neighbor_positive_density=0.0,
            clip_overlap_count=0.0,
            has_llm_semantics=False,
            has_chat_features=False,
            has_vector_similarity=False,
            feature_build_version=2,
            source=candidate.source,
        )

    profile_weights = profile_config.get_profile(profile)
    duration = candidate.end - candidate.start
    first_segment = window[0]
    hook_delay = max(0.0, candidate.anchor_time - candidate.start)
    hook_heuristic = _hook_strength(first_segment, hook_delay)
    max_hype = max(seg.hype_score for seg in window)
    mean_hype = sum(seg.hype_score for seg in window) / len(window)
    arc_score = max(seg.arc_score for seg in window)
    chat_peak = max(seg.chat_spike_score for seg in window)
    motion_peak = max(seg.max_motion_score for seg in window)
    face_reaction_peak = max(seg.max_face_reaction for seg in window)
    speaker_turns = sum(
        1 for left, right in zip(window, window[1:])
        if left.speaker != "UNKNOWN" and right.speaker != "UNKNOWN" and left.speaker != right.speaker
    )
    exclamation_count = sum(1 for seg in window if "exclamation" in seg.emotions)
    laughter_count = sum(1 for seg in window if "laughter" in seg.emotions or "audio_laughter" in seg.emotions)
    question_open = first_segment.text.strip().endswith("?")
    context_cost = min(max(hook_delay - 1.5, 0.0) / 6.0, 1.0)
    payoff_score = _payoff_score(window, candidate.anchor_time)
    context_efficiency = max(0.0, 1.0 - context_cost) * min(1.0, 18.0 / max(duration, 8.0))
    should_use_llm = use_llm_semantics and (
        llm_candidate_ids is None or candidate.id in llm_candidate_ids
    )
    sem = semantic_service.score_window(
        window,
        hook_delay,
        use_llm=should_use_llm,
        llm_provider=llm_provider,
    )
    quoteability_score = sem["quoteability"]
    surprise_score = sem["surprise"]
    contradiction_score = sem["contradiction"]
    standalone_score = sem["standalone"]
    semantic_density = sem["density"]
    pre_window = [seg for seg in window if seg.end <= candidate.anchor_time]
    post_window = [seg for seg in window if seg.start >= candidate.anchor_time]
    pre_hook_hype = sum(seg.hype_score for seg in pre_window) / len(pre_window) if pre_window else 0.0
    post_hook_hype = sum(seg.hype_score for seg in post_window) / len(post_window) if post_window else 0.0
    hype_delta = max_hype - pre_hook_hype
    local_peak_density = (
        sum(1 for seg in window if seg.hype_score >= max(max_hype - 1.0, 2.5)) / len(window)
        if window else 0.0
    )
    # Blend heuristic (timing/punctuation) with LLM semantic hook score
    hook_llm = sem.get("hook_strength", hook_heuristic)
    hook_strength = round(0.4 * hook_heuristic + 0.6 * hook_llm, 3)
    profile_fit_score = _profile_fit_score(
        candidate.source,
        chat_peak,
        motion_peak,
        speaker_turns,
        quoteability_score,
        surprise_score,
        hook_strength,
        profile_weights,
    )
    similarity = clip_similarity.score_candidate_window(window, candidate.anchor_time, profile)
    has_chat_features = any(
        abs(float(getattr(seg, "chat_spike_score", 0.0) or 0.0)) > 1e-9
        for seg in window
    )
    has_vector_similarity = bool(similarity.get("vector_reference_available", False)) or any(
        abs(float(similarity.get(name, 0.0) or 0.0)) > 1e-9
        for name in (
            "winner_similarity_max",
            "winner_similarity_mean",
            "loser_similarity_max",
            "similarity_margin",
            "neighbor_quality_score",
        )
    )

    return CandidateFeatures(
        candidate_id=candidate.id,
        duration=round(duration, 2),
        hook_delay_sec=round(hook_delay, 2),
        hook_strength=round(hook_strength, 3),
        max_hype=round(max_hype, 3),
        mean_hype=round(mean_hype, 3),
        arc_score=round(arc_score, 3),
        chat_peak=round(chat_peak, 3),
        motion_peak=round(motion_peak, 3),
        face_reaction_peak=round(face_reaction_peak, 3),
        speaker_turns=speaker_turns,
        exclamation_count=exclamation_count,
        laughter_count=laughter_count,
        question_open=question_open,
        context_cost=round(context_cost, 3),
        payoff_score=round(payoff_score, 3),
        context_efficiency=round(context_efficiency, 3),
        quoteability_score=quoteability_score,
        surprise_score=surprise_score,
        contradiction_score=contradiction_score,
        standalone_score=standalone_score,
        semantic_density=semantic_density,
        profile_fit_score=round(profile_fit_score, 3),
        pre_hook_hype=round(pre_hook_hype, 3),
        post_hook_hype=round(post_hook_hype, 3),
        hype_delta=round(hype_delta, 3),
        local_peak_density=round(local_peak_density, 3),
        winner_similarity_max=similarity["winner_similarity_max"],
        winner_similarity_mean=similarity["winner_similarity_mean"],
        loser_similarity_max=similarity["loser_similarity_max"],
        similarity_margin=similarity["similarity_margin"],
        neighbor_quality_score=similarity["neighbor_quality_score"],
        vod_progress=0.0,
        boundary_left_context_sec=0.0,
        boundary_right_context_sec=0.0,
        neighbor_positive_density=0.0,
        clip_overlap_count=0.0,
        has_llm_semantics=bool(sem.get("llm_used", False)),
        has_chat_features=has_chat_features,
        has_vector_similarity=has_vector_similarity,
        feature_build_version=2,
        source=candidate.source,
    )


def _payoff_score(window: list[RichSegment], anchor_time: float) -> float:
    payoff_segments = [seg for seg in window if seg.end >= anchor_time]
    if not payoff_segments:
        return 0.0
    score = 0.0
    for seg in payoff_segments:
        score += seg.hype_score * 0.2
        score += seg.chat_spike_score * 1.0
        if "audio_laughter" in seg.emotions or "laughter" in seg.emotions:
            score += 1.0
        if "face_reaction" in seg.emotions:
            score += 0.8
        if "high_motion" in seg.emotions:
            score += 0.5
    return min(score, 10.0)


def _hook_strength(first_segment: RichSegment, hook_delay: float) -> float:
    score = max(0.0, 1.0 - min(hook_delay, 3.0) / 3.0) * 0.45
    if "exclamation" in first_segment.emotions:
        score += 0.15
    if "question" in first_segment.emotions or first_segment.text.strip().endswith("?"):
        score += 0.15
    if "laughter" in first_segment.emotions or "audio_laughter" in first_segment.emotions:
        score += 0.1
    if first_segment.hype_score >= 3.0:
        score += 0.15
    return min(score, 1.0)


def _profile_fit_score(
    source: str,
    chat_peak: float,
    motion_peak: float,
    speaker_turns: int,
    quoteability_score: float,
    surprise_score: float,
    hook_strength: float,
    profile_weights: dict[str, float],
) -> float:
    base = (
        chat_peak * profile_weights["chat_weight"]
        + motion_peak * profile_weights["visual_weight"]
        + min(speaker_turns / 4.0, 1.0) * profile_weights["dialogue_weight"]
        + quoteability_score * profile_weights["quote_weight"]
        + surprise_score * profile_weights["surprise_weight"]
        + hook_strength * profile_weights["hook_weight"]
    ) / 6.0

    source_bonus = {
        "chat_sync": 0.08 * profile_weights["chat_weight"],
        "dialogue": 0.08 * profile_weights["dialogue_weight"],
        "quote": 0.08 * profile_weights["quote_weight"],
    }.get(source, 0.0)

    return min(base + source_bonus, 1.0)
