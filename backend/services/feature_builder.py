from __future__ import annotations

import concurrent.futures
import pathlib
import statistics
import re

from config import settings
from services import boundary_refiner, chapter_segmenter, clip_similarity, clip_strategy, clip_taxonomy, coherence_scorer, context_patterns, context_quality, game_event_features, hook_rewrite, profile_config, semantic_service, trend_prior, vision_language_features, vod_memory
from services.game_events import GameEvent
from services.timeline_types import Candidate, CandidateFeatures, Chapter
from services.transcriber import RichSegment

_TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")
FEATURE_BUILD_VERSION = 13
_LLM_SEMANTIC_MAX_WORKERS = 4


def build_features(
    candidates: list[Candidate],
    segments: list[RichSegment],
    profile: str = "gaming",
    use_llm_semantics: bool = True,
    llm_candidate_ids: set[str] | None = None,
    llm_provider: str | None = None,
    chapters: list[Chapter] | None = None,
    strategy: dict | None = None,
    game_events: list[GameEvent] | None = None,
    video_path: pathlib.Path | None = None,
) -> list[CandidateFeatures]:
    resolved_chapters = chapters if chapters is not None else chapter_segmenter.segment_chapters(segments)
    chapter_contexts = _prepare_chapter_contexts(candidates, segments, resolved_chapters)
    memory_spans = vod_memory.build_memory_spans(segments, resolved_chapters)
    candidate_windows = {
        candidate.id: _candidate_window(candidate, segments)
        for candidate in candidates
    }
    memory_contexts = {
        candidate.id: vod_memory.candidate_memory_context(candidate, segments, memory_spans)
        for candidate in candidates
    }
    semantic_scores = _semantic_scores_for_candidates(
        candidates,
        candidate_windows,
        use_llm_semantics=use_llm_semantics,
        llm_candidate_ids=llm_candidate_ids,
        llm_provider=llm_provider,
    )
    normalized_strategy = clip_strategy.normalize_strategy(strategy)
    vision_scores = vision_language_features.score_candidates(video_path, candidates)
    boundary_variant_scores = _boundary_variant_scores(candidates, segments, profile)
    return [
        _build_candidate_features(
            candidate,
            segments,
            profile,
            use_llm_semantics=use_llm_semantics,
            llm_candidate_ids=llm_candidate_ids,
            llm_provider=llm_provider,
            chapter_context=chapter_contexts.get(candidate.id, _empty_chapter_context()),
            memory_context=memory_contexts.get(candidate.id, _empty_memory_context()),
            strategy=normalized_strategy,
            window=candidate_windows.get(candidate.id),
            semantic_scores=semantic_scores.get(candidate.id),
            game_events=game_events or [],
            vision_scores=vision_scores.get(candidate.id),
            boundary_scores=boundary_variant_scores.get(candidate.id),
        )
        for candidate in candidates
    ]


def _candidate_window(candidate: Candidate, segments: list[RichSegment]) -> list[RichSegment]:
    return [seg for seg in segments if seg.end > candidate.start and seg.start < candidate.end]


def _boundary_variant_scores(
    candidates: list[Candidate],
    segments: list[RichSegment],
    profile: str,
) -> dict[str, dict[str, float]]:
    if not bool(getattr(settings, "enable_boundary_refinement_v2", False)):
        return {}
    if not candidates or not segments:
        return {}
    scores: dict[str, dict[str, float]] = {}
    for candidate in candidates:
        try:
            pairs = boundary_refiner.build_boundary_pairs(candidate, segments, profile=profile)
        except Exception:
            continue
        if not pairs:
            continue
        best = max(float(pair.pair_score) for pair in pairs)
        baseline = max(0.0, min(1.0, float(candidate.score_hint) / 10.0))
        scores[candidate.id] = {
            "boundary_variant_score": max(0.0, min(best, 1.0)),
            "boundary_variant_margin_score": max(0.0, min(best - baseline, 1.0)),
            "boundary_variant_count": float(len(pairs)),
        }
    return scores


def _semantic_scores_for_candidates(
    candidates: list[Candidate],
    candidate_windows: dict[str, list[RichSegment]],
    *,
    use_llm_semantics: bool,
    llm_candidate_ids: set[str] | None,
    llm_provider: str | None,
) -> dict[str, dict[str, float]]:
    llm_requests: list[tuple[str, list[RichSegment], float]] = []
    for candidate in candidates:
        should_use_llm = use_llm_semantics and (
            llm_candidate_ids is None or candidate.id in llm_candidate_ids
        )
        window = candidate_windows.get(candidate.id, [])
        if not should_use_llm or not window:
            continue
        llm_requests.append((
            candidate.id,
            window,
            max(0.0, candidate.anchor_time - candidate.start),
        ))

    if not llm_requests:
        return {}

    if len(llm_requests) == 1:
        candidate_id, window, hook_delay = llm_requests[0]
        return {
            candidate_id: semantic_service.score_window(
                window,
                hook_delay,
                use_llm=True,
                llm_provider=llm_provider,
            ),
        }

    resolved_workers = min(len(llm_requests), _LLM_SEMANTIC_MAX_WORKERS)
    semantic_scores: dict[str, dict[str, float]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=resolved_workers) as executor:
        futures = {
            candidate_id: executor.submit(
                semantic_service.score_window,
                window,
                hook_delay,
                use_llm=True,
                llm_provider=llm_provider,
            )
            for candidate_id, window, hook_delay in llm_requests
        }
        for candidate in candidates:
            future = futures.get(candidate.id)
            if future is not None:
                semantic_scores[candidate.id] = future.result()
    return semantic_scores


def _build_candidate_features(
    candidate: Candidate,
    segments: list[RichSegment],
    profile: str,
    *,
    use_llm_semantics: bool = True,
    llm_candidate_ids: set[str] | None = None,
    llm_provider: str | None = None,
    chapter_context: dict | None = None,
    memory_context: dict | None = None,
    strategy: dict | None = None,
    window: list[RichSegment] | None = None,
    semantic_scores: dict[str, float] | None = None,
    game_events: list[GameEvent] | None = None,
    vision_scores: dict[str, float] | None = None,
    boundary_scores: dict[str, float] | None = None,
) -> CandidateFeatures:
    chapter_context = chapter_context or _empty_chapter_context()
    memory_context = memory_context or _empty_memory_context()
    strategy = clip_strategy.normalize_strategy(strategy)
    window = list(window) if window is not None else _candidate_window(candidate, segments)
    if not window:
        return CandidateFeatures.create(
            candidate_id=candidate.id,
            duration=max(0.0, candidate.end - candidate.start),
            hook_delay_sec=0.0,
            hook_strength=0.0,
            max_hype=0.0,
            mean_hype=0.0,
            arc_score=0.0,
            chat_peak=0.0,
            chat_mean=0.0,
            chat_early_peak=0.0,
            chat_post_hook_peak=0.0,
            chat_delta=0.0,
            chat_spike_coverage=0.0,
            chat_activity_coverage=0.0,
            chat_message_rate_peak=0.0,
            chat_message_rate_mean=0.0,
            chat_density_peak=0.0,
            chat_density_mean=0.0,
            chat_hype_score=0.0,
            chat_laugh_score=0.0,
            chat_shock_score=0.0,
            chat_sadge_score=0.0,
            chat_velocity_score=0.0,
            chat_rate_derivative_score=0.0,
            chat_ema_crossing_score=0.0,
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
            payoff_clarity_score=0.0,
            slow_open_penalty=1.0,
            context_dependency_score=0.0,
            weak_resolution_penalty=1.0,
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
            vod_progress=chapter_context["vod_progress"],
            boundary_left_context_sec=chapter_context["boundary_left_context_sec"],
            boundary_right_context_sec=chapter_context["boundary_right_context_sec"],
            # Offline-only dataset context; serving does not try to approximate it.
            neighbor_positive_density=0.0,
            clip_overlap_count=chapter_context["clip_overlap_count"],
            chapter_position=chapter_context["chapter_position"],
            topic_shift_strength=chapter_context["topic_shift_strength"],
            chapter_local_density=chapter_context["chapter_local_density"],
            chapter_boundary_distance=chapter_context["chapter_boundary_distance"],
            chapter_coherence_score=chapter_context["chapter_coherence_score"],
            setup_depth_score=memory_context["setup_depth_score"],
            callback_strength=memory_context["callback_strength"],
            arc_payoff_match=memory_context["arc_payoff_match"],
            topic_continuity_score=memory_context["topic_continuity_score"],
            speaker_context_clarity=memory_context["speaker_context_clarity"],
            memory_dependency_score=memory_context["memory_dependency_score"],
            story_stage_score=memory_context["story_stage_score"],
            required_setup_lead_sec=memory_context["required_setup_lead_sec"],
            retrieved_memory_span_count=memory_context["retrieved_memory_span_count"],
            trend_hook_score=0.0,
            trend_topic_score=0.0,
            trend_prior_score=0.0,
            dynamic_trend_score=0.0,
            outcome_prior_score=0.0,
            visual_hook_score=0.0,
            audio_pop_score=0.0,
            streamer_shout_score=0.0,
            voice_energy_burst_score=0.0,
            pitch_variation_score=0.0,
            f0_sigma_score=0.0,
            vad_voice_ratio=0.0,
            shout_onset_score=0.0,
            event_surprise_score=0.0,
            nonverbal_payoff_score=0.0,
            goal_relevance_score=0.0,
            platform_relevance_score=0.0,
            projected_hook_score=0.0,
            projected_retention_score=0.0,
            vl_prompt_max_score=0.0,
            vl_clutch_score=0.0,
            vl_comedy_score=0.0,
            vl_reaction_face_score=0.0,
            vl_scoreboard_score=0.0,
            judge_score=0.0,
            judge_confidence=0.0,
            judge_scroll_stop_power=0.0,
            judge_standalone_clarity=0.0,
            judge_payoff_strength=0.0,
            judge_narrative_completeness=0.0,
            judge_platform_fit_shortform=0.0,
            has_judge_score=False,
            judge_llm_used=False,
            has_llm_semantics=False,
            has_chat_features=False,
            has_vector_similarity=False,
            feature_build_version=FEATURE_BUILD_VERSION,
            source=candidate.source,
            clip_type="general",
            multimodal_proxy_score=0.0,
            rage_score=0.0,
            clutch_score=0.0,
            funny_reaction_score=0.0,
            trash_talk_payoff_score=0.0,
            jynxzi_chat_shock_score=0.0,
            ranked_tension_score=0.0,
            failed_setup_score=0.0,
            streamer_persona_beat_score=0.0,
            context_pattern_score=0.0,
            context_pattern_support=0.0,
            context_pattern_views_prior=0.0,
            context_pattern_confidence=0.0,
            context_pattern_similarity_margin=0.0,
            context_pattern_id="",
            context_pattern_label="",
        )

    profile_weights = profile_config.get_profile(profile)
    duration = candidate.end - candidate.start
    first_segment = window[0]
    hook_delay = max(0.0, candidate.anchor_time - candidate.start)
    hook_heuristic = _hook_strength(first_segment, hook_delay)
    max_hype = max(seg.hype_score for seg in window)
    mean_hype = sum(seg.hype_score for seg in window) / len(window)
    arc_score = max(seg.arc_score for seg in window)
    chat_peak = max(_chat_activity_score(seg) for seg in window)
    chat_stats = _chat_window_stats(window, candidate)
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
    sem = semantic_scores or semantic_service.score_window(
        window,
        hook_delay,
        use_llm=should_use_llm,
        llm_provider=llm_provider,
    )
    quoteability_score = sem["quoteability"]
    surprise_score = sem["surprise"]
    contradiction_score = sem["contradiction"]
    coherence = coherence_scorer.score_window(window)
    standalone_score = max(float(sem["standalone"]), float(coherence["coherence_score"]))
    semantic_density = sem["density"]
    payoff_clarity_score = sem.get("payoff_clarity", 0.0)
    semantic_context_dependency = max(float(sem.get("context_dependency", 0.0)), float(coherence["context_dependency_score"]))
    pre_window = [seg for seg in window if seg.end <= candidate.anchor_time]
    post_window = [seg for seg in window if seg.start >= candidate.anchor_time]
    pre_hook_hype = sum(seg.hype_score for seg in pre_window) / len(pre_window) if pre_window else 0.0
    post_hook_hype = sum(seg.hype_score for seg in post_window) / len(post_window) if post_window else 0.0
    hype_delta = max_hype - pre_hook_hype
    visual_hook_score = _visual_hook_score(window, candidate.anchor_time)
    audio_pop_score = _audio_pop_score(window, candidate.anchor_time)
    shout_scores = _streamer_shout_scores(window, candidate.anchor_time)
    game_scores = game_event_features.features_for_window(game_events or [], start=candidate.start, end=candidate.end)
    taxonomy = clip_taxonomy.classify_segments(window, source=candidate.source)
    context_pattern = context_patterns.score_segments(window, profile=profile)
    multimodal_proxy_score = _multimodal_proxy_score(
        chat_peak=chat_peak,
        motion_peak=motion_peak,
        face_peak=face_reaction_peak,
        visual_hook_score=visual_hook_score,
        audio_pop_score=audio_pop_score,
        game_event_score=float(game_scores.get("game_event_score", 0.0) or 0.0),
        laughter_count=laughter_count,
        exclamation_count=exclamation_count,
        max_hype=max_hype,
    )
    hook_scores = _hook_reranker_scores(window)
    vl_scores = {**vision_language_features.empty_scores(), **(vision_scores or {})}
    boundary_set_scores = {
        "boundary_variant_score": 0.0,
        "boundary_variant_margin_score": 0.0,
        "boundary_variant_count": 0.0,
        **(boundary_scores or {}),
    }
    event_surprise_score = _event_surprise_score(window)
    nonverbal_payoff_score = _nonverbal_payoff_score(window, candidate.anchor_time)
    local_peak_density = (
        sum(1 for seg in window if seg.hype_score >= max(max_hype - 1.0, 2.5)) / len(window)
        if window else 0.0
    )
    # Blend heuristic (timing/punctuation) with LLM semantic hook score
    hook_llm = sem.get("hook_strength", hook_heuristic)
    hook_strength = round(0.4 * hook_heuristic + 0.6 * hook_llm, 3)
    slow_open_penalty = _slow_open_penalty(first_segment, hook_delay, hook_strength, window)
    context_dependency_score = max(float(coherence["context_dependency_score"]), _context_dependency_score(
        window,
        hook_delay,
        standalone_score,
        semantic_context_dependency,
    ))
    weak_resolution_penalty = max(float(coherence["weak_resolution_penalty"]), _weak_resolution_penalty(
        window,
        payoff_score,
        payoff_clarity_score,
        post_hook_hype,
    ))
    token_count = len(_TOKEN_RE.findall(" ".join(seg.text.lower() for seg in window)))
    clip_type_signal = max(taxonomy.scores.values(), default=0.0)
    setup_present_score = max(
        hook_strength,
        memory_context["setup_depth_score"],
        memory_context["topic_continuity_score"] * 0.8,
        min(len(window) / 3.0, 1.0) * 0.35 + min(token_count / 18.0, 1.0) * 0.45 + context_pattern.confidence * 0.20,
    )
    payoff_present_score = max(
        payoff_score / 5.0,
        payoff_clarity_score,
        memory_context["callback_strength"],
        nonverbal_payoff_score,
        clip_type_signal * 0.75,
        context_pattern.confidence * 0.45,
    )
    context_strength_score = context_quality.strength_score(
        pattern_confidence=context_pattern.confidence,
        pattern_score=context_pattern.score,
        pattern_views_prior=context_pattern.views_prior,
        source_prior=context_quality.source_prior(candidate.source),
        clip_type_signal=clip_type_signal,
        payoff_present=payoff_present_score,
        setup_present=setup_present_score,
    )
    context_completeness_score = context_quality.completeness_score(
        setup_present=setup_present_score,
        payoff_present=payoff_present_score,
        standalone=standalone_score,
        context_efficiency=context_efficiency,
        pattern_confidence=context_pattern.confidence,
        token_coverage=context_quality.token_coverage_score(token_count, duration),
        context_dependency=context_dependency_score,
        weak_resolution=weak_resolution_penalty,
    )
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
    trend_scores = trend_prior.score_candidate_window(window, profile, strategy=strategy)
    goal_relevance_score = clip_strategy.text_relevance_score(
        " ".join(seg.text.strip() for seg in window if seg.text.strip()),
        strategy,
    )
    platform_relevance_score = _platform_relevance_score(
        duration,
        hook_strength,
        context_efficiency,
        payoff_clarity_score,
        visual_hook_score,
        strategy,
    )
    projected_hook_score = _projected_hook_score(
        hook_strength,
        visual_hook_score,
        event_surprise_score,
        strategy,
    )
    projected_retention_score = _projected_retention_score(
        context_efficiency,
        payoff_clarity_score,
        nonverbal_payoff_score,
        goal_relevance_score,
        platform_relevance_score,
        float(trend_scores.get("outcome_prior_score", 0.0)),
        strategy,
    )
    has_chat_features = any(
        _chat_activity_score(seg) > 1e-9
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
    ) or any(
        abs(float(vl_scores.get(name, 0.0) or 0.0)) > 1e-9
        for name in (
            "vl_prompt_max_score",
            "vl_clutch_score",
            "vl_comedy_score",
            "vl_reaction_face_score",
            "vl_scoreboard_score",
        )
    )

    return CandidateFeatures.create(
        candidate_id=candidate.id,
        duration=round(duration, 2),
        hook_delay_sec=round(hook_delay, 2),
        hook_strength=round(hook_strength, 3),
        max_hype=round(max_hype, 3),
        mean_hype=round(mean_hype, 3),
        arc_score=round(arc_score, 3),
        chat_peak=round(chat_peak, 3),
        chat_mean=round(chat_stats["mean"], 3),
        chat_early_peak=round(chat_stats["early_peak"], 3),
        chat_post_hook_peak=round(chat_stats["post_hook_peak"], 3),
        chat_delta=round(chat_stats["delta"], 3),
        chat_spike_coverage=round(chat_stats["coverage"], 3),
        chat_activity_coverage=round(chat_stats["activity_coverage"], 3),
        chat_message_rate_peak=round(chat_stats["message_rate_peak"], 3),
        chat_message_rate_mean=round(chat_stats["message_rate_mean"], 3),
        chat_density_peak=round(chat_stats["density_peak"], 3),
        chat_density_mean=round(chat_stats["density_mean"], 3),
        chat_hype_score=round(chat_stats["hype"], 3),
        chat_laugh_score=round(chat_stats["laugh"], 3),
        chat_shock_score=round(chat_stats["shock"], 3),
        chat_sadge_score=round(chat_stats["sadge"], 3),
        chat_velocity_score=round(chat_stats["velocity"], 3),
        chat_rate_derivative_score=round(chat_stats["rate_derivative"], 3),
        chat_ema_crossing_score=round(chat_stats["ema_crossing"], 3),
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
        payoff_clarity_score=round(payoff_clarity_score, 3),
        slow_open_penalty=round(slow_open_penalty, 3),
        context_dependency_score=round(context_dependency_score, 3),
        weak_resolution_penalty=round(weak_resolution_penalty, 3),
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
        vod_progress=chapter_context["vod_progress"],
        boundary_left_context_sec=chapter_context["boundary_left_context_sec"],
        boundary_right_context_sec=chapter_context["boundary_right_context_sec"],
        # Offline-only dataset context; serving does not try to approximate it.
        neighbor_positive_density=0.0,
        clip_overlap_count=chapter_context["clip_overlap_count"],
        chapter_position=chapter_context["chapter_position"],
        topic_shift_strength=chapter_context["topic_shift_strength"],
        chapter_local_density=chapter_context["chapter_local_density"],
        chapter_boundary_distance=chapter_context["chapter_boundary_distance"],
        chapter_coherence_score=chapter_context["chapter_coherence_score"],
        setup_depth_score=memory_context["setup_depth_score"],
        callback_strength=memory_context["callback_strength"],
        arc_payoff_match=memory_context["arc_payoff_match"],
        topic_continuity_score=memory_context["topic_continuity_score"],
        speaker_context_clarity=memory_context["speaker_context_clarity"],
        memory_dependency_score=memory_context["memory_dependency_score"],
        story_stage_score=memory_context["story_stage_score"],
        required_setup_lead_sec=memory_context["required_setup_lead_sec"],
        retrieved_memory_span_count=memory_context["retrieved_memory_span_count"],
        trend_hook_score=float(trend_scores["trend_hook_score"]),
        trend_topic_score=float(trend_scores["trend_topic_score"]),
        trend_prior_score=float(trend_scores["trend_prior_score"]),
        dynamic_trend_score=float(trend_scores.get("dynamic_trend_score", 0.0)),
        outcome_prior_score=float(trend_scores.get("outcome_prior_score", 0.0)),
        visual_hook_score=round(visual_hook_score, 3),
        audio_pop_score=round(audio_pop_score, 3),
        streamer_shout_score=round(shout_scores["shout"], 3),
        voice_energy_burst_score=round(shout_scores["voice_energy"], 3),
        pitch_variation_score=round(shout_scores["pitch_variation"], 3),
        f0_sigma_score=round(shout_scores["f0_sigma"], 3),
        vad_voice_ratio=round(shout_scores["vad_voice_ratio"], 3),
        shout_onset_score=round(shout_scores["onset"], 3),
        event_surprise_score=round(event_surprise_score, 3),
        nonverbal_payoff_score=round(nonverbal_payoff_score, 3),
        goal_relevance_score=round(goal_relevance_score, 3),
        platform_relevance_score=round(platform_relevance_score, 3),
        projected_hook_score=round(projected_hook_score, 3),
        projected_retention_score=round(projected_retention_score, 3),
        vl_prompt_max_score=round(float(vl_scores.get("vl_prompt_max_score", 0.0) or 0.0), 3),
        vl_clutch_score=round(float(vl_scores.get("vl_clutch_score", 0.0) or 0.0), 3),
        vl_comedy_score=round(float(vl_scores.get("vl_comedy_score", 0.0) or 0.0), 3),
        vl_reaction_face_score=round(float(vl_scores.get("vl_reaction_face_score", 0.0) or 0.0), 3),
        vl_scoreboard_score=round(float(vl_scores.get("vl_scoreboard_score", 0.0) or 0.0), 3),
        game_event_score=round(float(game_scores.get("game_event_score", 0.0) or 0.0), 3),
        kill_density_in_window=round(float(game_scores.get("kill_density_in_window", 0.0) or 0.0), 4),
        is_clutch_window=bool(game_scores.get("is_clutch_window", False)),
        is_match_endpoint=bool(game_scores.get("is_match_endpoint", False)),
        boundary_variant_score=round(float(boundary_set_scores.get("boundary_variant_score", 0.0) or 0.0), 3),
        boundary_variant_margin_score=round(float(boundary_set_scores.get("boundary_variant_margin_score", 0.0) or 0.0), 3),
        boundary_variant_count=round(float(boundary_set_scores.get("boundary_variant_count", 0.0) or 0.0), 3),
        hook_rewrite_score=round(hook_scores["hook_rewrite_score"], 3),
        hook_curiosity_gap_score=round(hook_scores["hook_curiosity_gap_score"], 3),
        has_hook_rewrite=bool(hook_scores["has_hook_rewrite"]),
        judge_score=0.0,
        judge_confidence=0.0,
        judge_scroll_stop_power=0.0,
        judge_standalone_clarity=0.0,
        judge_payoff_strength=0.0,
        judge_narrative_completeness=0.0,
        judge_platform_fit_shortform=0.0,
        has_judge_score=False,
        judge_llm_used=False,
        has_llm_semantics=bool(sem.get("llm_used", False)),
        has_chat_features=has_chat_features,
        has_vector_similarity=has_vector_similarity,
        feature_build_version=FEATURE_BUILD_VERSION,
        source=candidate.source,
        clip_type=taxonomy.clip_type,
        multimodal_proxy_score=round(multimodal_proxy_score, 3),
        rage_score=taxonomy.scores.get("rage", 0.0),
        clutch_score=taxonomy.scores.get("clutch", 0.0),
        funny_reaction_score=taxonomy.scores.get("funny_reaction", 0.0),
        trash_talk_payoff_score=taxonomy.scores.get("trash_talk_payoff", 0.0),
        jynxzi_chat_shock_score=taxonomy.scores.get("chat_shock", 0.0),
        ranked_tension_score=taxonomy.scores.get("ranked_tension", 0.0),
        failed_setup_score=taxonomy.scores.get("failed_setup", 0.0),
        streamer_persona_beat_score=taxonomy.scores.get("streamer_persona_beat", 0.0),
        context_pattern_score=context_pattern.score,
        context_pattern_support=context_pattern.support,
        context_pattern_views_prior=context_pattern.views_prior,
        context_pattern_confidence=context_pattern.confidence,
        context_pattern_similarity_margin=context_pattern.similarity_margin,
        context_pattern_id=context_pattern.pattern_id,
        context_pattern_label=context_pattern.label,
        setup_present_score=round(context_quality.clamp01(setup_present_score), 3),
        payoff_present_score=round(context_quality.clamp01(payoff_present_score), 3),
        context_strength_score=round(context_strength_score, 3),
        context_completeness_score=round(context_completeness_score, 3),
    )


def _chat_window_stats(window: list[RichSegment], candidate: Candidate) -> dict[str, float]:
    scores = [_chat_activity_score(seg) for seg in window]
    spike_scores = [max(0.0, float(getattr(seg, "chat_spike_score", 0.0) or 0.0)) for seg in window]
    density_scores = [max(0.0, float(getattr(seg, "chat_density_score", 0.0) or 0.0)) for seg in window]
    message_rates = [max(0.0, float(getattr(seg, "chat_message_rate", 0.0) or 0.0)) for seg in window]
    if not scores:
        return {
            "mean": 0.0,
            "early_peak": 0.0,
            "post_hook_peak": 0.0,
            "delta": 0.0,
            "coverage": 0.0,
            "activity_coverage": 0.0,
            "message_rate_peak": 0.0,
            "message_rate_mean": 0.0,
            "density_peak": 0.0,
            "density_mean": 0.0,
            "hype": 0.0,
            "laugh": 0.0,
            "shock": 0.0,
            "sadge": 0.0,
            "velocity": 0.0,
            "rate_derivative": 0.0,
            "ema_crossing": 0.0,
        }

    early_end = min(candidate.end, candidate.start + 5.0)
    early_scores = [
        score
        for seg, score in zip(window, scores, strict=False)
        if seg.end > candidate.start and seg.start < early_end
    ]
    pre_hook_scores = [
        score
        for seg, score in zip(window, scores, strict=False)
        if seg.start < candidate.anchor_time
    ]
    post_hook_scores = [
        score
        for seg, score in zip(window, scores, strict=False)
        if seg.end > candidate.anchor_time
    ]
    spiked_duration = 0.0
    activity_duration = 0.0
    for seg, score, spike_score in zip(window, scores, spike_scores, strict=False):
        if score <= 0.0:
            continue
        overlap = min(seg.end, candidate.end) - max(seg.start, candidate.start)
        overlap = max(0.0, overlap)
        activity_duration += overlap
        if spike_score > 0.0:
            spiked_duration += overlap

    return {
        "mean": sum(scores) / len(scores),
        "early_peak": max(early_scores, default=0.0),
        "post_hook_peak": max(post_hook_scores, default=0.0),
        "delta": max(post_hook_scores, default=0.0) - max(pre_hook_scores, default=0.0),
        "coverage": min(spiked_duration / max(candidate.end - candidate.start, 1e-6), 1.0),
        "activity_coverage": min(activity_duration / max(candidate.end - candidate.start, 1e-6), 1.0),
        "message_rate_peak": max(message_rates, default=0.0),
        "message_rate_mean": sum(message_rates) / len(message_rates),
        "density_peak": max(density_scores, default=0.0),
        "density_mean": sum(density_scores) / len(density_scores),
        "hype": _chat_category_peak(window, "hype"),
        "laugh": _chat_category_peak(window, "laugh"),
        "shock": _chat_category_peak(window, "shock"),
        "sadge": _chat_category_peak(window, "sadge"),
        "velocity": _chat_velocity_score(message_rates, scores),
        "rate_derivative": _chat_derivative_score(window, message_rates),
        "ema_crossing": _chat_ema_crossing_score(window, scores),
    }


def _chat_activity_score(segment: RichSegment) -> float:
    spike = max(0.0, float(getattr(segment, "chat_spike_score", 0.0) or 0.0))
    density = max(0.0, float(getattr(segment, "chat_density_score", 0.0) or 0.0))
    return min(max(spike, density), 1.0)


def _chat_category_peak(window: list[RichSegment], category: str) -> float:
    attr_name = f"chat_{category}_score"
    numeric_peak = max(
        (
            max(0.0, float(getattr(seg, attr_name, 0.0) or 0.0))
            for seg in window
        ),
        default=0.0,
    )
    if numeric_peak > 0.0:
        return min(numeric_peak, 1.0)
    tag = f"chat_{category}"
    return max(
        (
            _chat_activity_score(seg)
            for seg in window
            if tag in (getattr(seg, "emotions", []) or [])
        ),
        default=0.0,
    )


def _chat_velocity_score(message_rates: list[float], activity_scores: list[float]) -> float:
    if len(message_rates) < 2:
        return 0.0
    rate_delta = max(
        (
            max(0.0, right - left)
            for left, right in zip(message_rates, message_rates[1:], strict=False)
        ),
        default=0.0,
    )
    activity_delta = max(
        (
            max(0.0, right - left)
            for left, right in zip(activity_scores, activity_scores[1:], strict=False)
        ),
        default=0.0,
    )
    return _clamp01(max(rate_delta / 8.0, activity_delta))


def _chat_derivative_score(window: list[RichSegment], message_rates: list[float]) -> float:
    segment_derivative = max(
        (
            max(0.0, float(getattr(seg, "chat_rate_derivative", 0.0) or 0.0))
            for seg in window
        ),
        default=0.0,
    )
    raw_rate_delta = max(
        (
            max(0.0, right - left)
            for left, right in zip(message_rates, message_rates[1:], strict=False)
        ),
        default=0.0,
    )
    return _clamp01(max(segment_derivative / 4.0, raw_rate_delta / 8.0))


def _chat_ema_crossing_score(window: list[RichSegment], activity_scores: list[float]) -> float:
    crossing = max(
        (
            max(0.0, float(getattr(seg, "chat_ema_crossing_score", 0.0) or 0.0))
            for seg in window
        ),
        default=0.0,
    )
    if crossing > 0.0:
        return _clamp01(crossing)
    return _clamp01(
        max(
            (
                max(0.0, right - left)
                for left, right in zip(activity_scores, activity_scores[1:], strict=False)
            ),
            default=0.0,
        )
    )


def _slow_open_penalty(
    first_segment: RichSegment,
    hook_delay: float,
    hook_strength: float,
    window: list[RichSegment],
) -> float:
    early_peak = max((seg.hype_score for seg in window[:2]), default=0.0)
    score = 0.0
    if hook_delay > 2.0:
        score += min((hook_delay - 2.0) / 6.0, 0.45)
    if hook_strength < 0.45:
        score += 0.25
    if early_peak < 2.0:
        score += 0.15
    if not first_segment.text.strip().endswith(("?", "!")) and "exclamation" not in first_segment.emotions:
        score += 0.1
    return _clamp01(score)


def _visual_hook_score(window: list[RichSegment], anchor_time: float) -> float:
    opening = [seg for seg in window if seg.start < min(anchor_time, window[0].start + 3.0)] or window[:2]
    if not opening:
        return 0.0
    score = 0.0
    for seg in opening:
        if seg.has_scene_change:
            score += 0.3
        score += min(seg.max_motion_score, 1.0) * 0.25
        score += min(seg.max_face_reaction, 1.0) * 0.25
        score += min(seg.visual_activity / 6.0, 1.0) * 0.2
    return _clamp01(score)


def _multimodal_proxy_score(
    *,
    chat_peak: float,
    motion_peak: float,
    face_peak: float,
    visual_hook_score: float,
    audio_pop_score: float,
    game_event_score: float,
    laughter_count: int,
    exclamation_count: int,
    max_hype: float,
) -> float:
    return _clamp01(
        min(chat_peak, 1.0) * 0.16
        + max(min(motion_peak, 1.0), min(face_peak, 1.0)) * 0.20
        + min(visual_hook_score, 1.0) * 0.16
        + min(audio_pop_score, 1.0) * 0.14
        + min(game_event_score, 1.0) * 0.12
        + min((laughter_count + exclamation_count) / 4.0, 1.0) * 0.10
        + min(max_hype / 6.0, 1.0) * 0.12
    )


def _audio_pop_score(window: list[RichSegment], anchor_time: float) -> float:
    early = [seg for seg in window if seg.end <= anchor_time + 2.0] or window[:2]
    if not early:
        return 0.0
    score = 0.0
    for seg in early:
        score += min(seg.laughter_prob, 1.0) * 0.25
        score += min(max(seg.max_onset_rate, 0.0) / 8.0, 1.0) * 0.2
        score += min(max(seg.max_pitch_hz, 0.0) / 550.0, 1.0) * 0.15
        if "audio_laughter" in seg.emotions or "laughter" in seg.emotions:
            score += 0.18
        if seg.hype_score >= 4.0:
            score += 0.12
    return _clamp01(score)


def _streamer_shout_scores(window: list[RichSegment], anchor_time: float) -> dict[str, float]:
    early = [seg for seg in window if seg.end <= anchor_time + 2.0] or window[:2]
    empty = {
        "shout": 0.0,
        "voice_energy": 0.0,
        "pitch_variation": 0.0,
        "f0_sigma": 0.0,
        "vad_voice_ratio": 0.0,
        "onset": 0.0,
    }
    if not early:
        return empty

    known_speaker_segments = [
        seg for seg in early
        if str(getattr(seg, "speaker", "")).strip().upper() not in {"", "UNKNOWN"}
    ]
    streamer_segments = [
        seg for seg in known_speaker_segments
        if str(getattr(seg, "speaker", "")).strip().upper() == "STREAMER"
    ]
    if known_speaker_segments and not streamer_segments:
        return empty
    speaker_segments = streamer_segments or early
    voiced = [
        seg for seg in speaker_segments
        if str(getattr(seg, "text", "") or "").strip()
        and float(getattr(seg, "max_rms_db", 0.0) or 0.0) < 0.0
        and (
            float(getattr(seg, "max_pitch_hz", 0.0) or 0.0) > 0.0
            or float(getattr(seg, "max_onset_rate", 0.0) or 0.0) > 0.0
        )
    ]
    if not voiced:
        return empty

    rms_values = [float(getattr(seg, "max_rms_db", 0.0) or 0.0) for seg in voiced]
    rms_burst_values = [float(getattr(seg, "rms_burst_score", 0.0) or 0.0) for seg in voiced]
    pitch_values = [
        float(getattr(seg, "max_pitch_hz", 0.0) or 0.0)
        for seg in voiced
        if float(getattr(seg, "max_pitch_hz", 0.0) or 0.0) > 0.0
    ]
    pitch_std_values = [
        float(getattr(seg, "pitch_std_hz", 0.0) or 0.0)
        for seg in voiced
        if float(getattr(seg, "pitch_std_hz", 0.0) or 0.0) > 0.0
    ]
    onset_values = [float(getattr(seg, "max_onset_rate", 0.0) or 0.0) for seg in voiced]
    voice_ratios = [float(getattr(seg, "voice_activity_score", 0.0) or 0.0) for seg in speaker_segments]

    peak_rms = max(rms_values, default=-60.0)
    local_rms_baseline = statistics.median(rms_values) if rms_values else -60.0
    relative_rms_burst = _clamp01((peak_rms - local_rms_baseline) / 8.0)
    absolute_voice_energy = _clamp01((peak_rms + 34.0) / 22.0)
    voice_energy = max(absolute_voice_energy, relative_rms_burst, max(rms_burst_values, default=0.0))
    onset_score = _clamp01(max(onset_values, default=0.0) / 8.0)
    f0_sigma_score = _clamp01(max(pitch_std_values, default=0.0) / 180.0)
    vad_voice_ratio = _clamp01(statistics.fmean(voice_ratios) if voice_ratios else len(voiced) / max(len(speaker_segments), 1))
    if len(pitch_values) >= 2:
        pitch_std = statistics.pstdev(pitch_values)
        pitch_mean = max(statistics.fmean(pitch_values), 1.0)
        pitch_variation = max(_clamp01((pitch_std / pitch_mean) * 4.0), f0_sigma_score)
    elif pitch_values:
        pitch_variation = max(_clamp01(max(pitch_values) / 650.0), f0_sigma_score)
    else:
        pitch_variation = f0_sigma_score

    tag_bonus = 0.0
    if any("energy_burst" in (getattr(seg, "emotions", []) or []) for seg in voiced):
        tag_bonus += 0.15
    if any("pitch_spike" in (getattr(seg, "emotions", []) or []) for seg in voiced):
        tag_bonus += 0.15
    shout = _clamp01(
        0.38 * voice_energy
        + 0.28 * pitch_variation
        + 0.16 * f0_sigma_score
        + 0.12 * onset_score
        + 0.06 * vad_voice_ratio
        + tag_bonus
    )
    return {
        "shout": shout,
        "voice_energy": voice_energy,
        "pitch_variation": pitch_variation,
        "f0_sigma": f0_sigma_score,
        "vad_voice_ratio": vad_voice_ratio,
        "onset": onset_score,
    }


def _hook_reranker_scores(window: list[RichSegment]) -> dict[str, float | bool]:
    if not bool(getattr(settings, "enable_hook_subclassifier", False)) or not window:
        return {
            "hook_rewrite_score": 0.0,
            "hook_curiosity_gap_score": 0.0,
            "has_hook_rewrite": False,
        }
    opening_text = " ".join(seg.text.strip() for seg in window[:2] if seg.text.strip())
    if not opening_text:
        return {
            "hook_rewrite_score": 0.0,
            "hook_curiosity_gap_score": 0.0,
            "has_hook_rewrite": False,
        }
    try:
        payload = hook_rewrite.score_hook(opening_text)
    except Exception:
        return {
            "hook_rewrite_score": 0.0,
            "hook_curiosity_gap_score": 0.0,
            "has_hook_rewrite": False,
        }
    return {
        "hook_rewrite_score": _clamp01(float(payload.get("score", 0.0) or 0.0) / 10.0),
        "hook_curiosity_gap_score": _clamp01(float(payload.get("gap", 0.0) or 0.0) / 10.0),
        "has_hook_rewrite": True,
    }


def _event_surprise_score(window: list[RichSegment]) -> float:
    if not window:
        return 0.0
    score = 0.0
    for prev, seg in zip(window, window[1:]):
        if seg.has_scene_change:
            score += 0.18
        if seg.hype_score - prev.hype_score >= 2.0:
            score += 0.16
        if max(seg.max_face_reaction, seg.max_motion_score) - max(prev.max_face_reaction, prev.max_motion_score) >= 0.3:
            score += 0.12
        if seg.chat_spike_score - prev.chat_spike_score >= 0.25:
            score += 0.08
    return _clamp01(score)


def _nonverbal_payoff_score(window: list[RichSegment], anchor_time: float) -> float:
    tail = [seg for seg in window if seg.start >= anchor_time] or window[-2:]
    if not tail:
        return 0.0
    score = 0.0
    for seg in tail:
        score += min(seg.max_face_reaction, 1.0) * 0.3
        score += min(seg.max_motion_score, 1.0) * 0.2
        score += min(seg.chat_spike_score, 1.0) * 0.15
        if "audio_laughter" in seg.emotions or "laughter" in seg.emotions:
            score += 0.2
        if seg.has_scene_change:
            score += 0.1
    return _clamp01(score)


def _platform_relevance_score(
    duration: float,
    hook_strength: float,
    context_efficiency: float,
    payoff_clarity_score: float,
    visual_hook_score: float,
    strategy: dict,
) -> float:
    low, high, ideal = clip_strategy.platform_duration_target(strategy)
    duration_penalty = min(abs(duration - ideal) / max(high - low, 1.0), 1.0)
    platform_bias = clip_strategy.platform_biases(strategy)
    score = (
        hook_strength * platform_bias.get("hook", 1.0) * 0.35
        + context_efficiency * platform_bias.get("context", 1.0) * 0.2
        + payoff_clarity_score * platform_bias.get("payoff", 1.0) * 0.25
        + visual_hook_score * platform_bias.get("visual", 1.0) * 0.2
    )
    return max(0.0, min(score - duration_penalty * 0.25, 1.0))


def _projected_hook_score(
    hook_strength: float,
    visual_hook_score: float,
    event_surprise_score: float,
    strategy: dict,
) -> float:
    bias = clip_strategy.platform_biases(strategy)
    return max(
        0.0,
        min(
            hook_strength * bias.get("hook", 1.0) * 0.55
            + visual_hook_score * bias.get("visual", 1.0) * 0.25
            + event_surprise_score * 0.2,
            1.0,
        ),
    )


def _projected_retention_score(
    context_efficiency: float,
    payoff_clarity_score: float,
    nonverbal_payoff_score: float,
    goal_relevance_score: float,
    platform_relevance_score: float,
    outcome_prior_score: float,
    strategy: dict,
) -> float:
    bias = clip_strategy.platform_biases(strategy)
    score = (
        context_efficiency * bias.get("context", 1.0) * 0.22
        + payoff_clarity_score * bias.get("payoff", 1.0) * 0.24
        + nonverbal_payoff_score * 0.18
        + goal_relevance_score * 0.14
        + platform_relevance_score * 0.12
        + outcome_prior_score * 0.1
    )
    return max(0.0, min(score, 1.0))


def _context_dependency_score(
    window: list[RichSegment],
    hook_delay: float,
    standalone_score: float,
    semantic_context_dependency: float,
) -> float:
    if not window:
        return 0.0
    first = window[0].text.strip().lower()
    score = semantic_context_dependency * 0.65
    if standalone_score < 0.45:
        score += 0.2
    if hook_delay > 2.5:
        score += 0.08
    if first.startswith(("and ", "but ", "so ", "weil ", "aber ", "und ")):
        score += 0.15
    if any(token in first for token in ("he ", "she ", "they ", "this ", "that ", "dies", "das ")):
        score += 0.08
    return _clamp01(score)


def _weak_resolution_penalty(
    window: list[RichSegment],
    payoff_score: float,
    payoff_clarity_score: float,
    post_hook_hype: float,
) -> float:
    if not window:
        return 1.0
    score = 0.0
    if payoff_score < 2.5:
        score += 0.35
    if payoff_clarity_score < 0.45:
        score += 0.35
    if post_hook_hype < 1.5:
        score += 0.15
    tail = window[max(len(window) // 2, 0):] or window
    if not any(seg.text.strip().endswith(("!", ".", "?")) for seg in tail):
        score += 0.1
    return _clamp01(score)


def _prepare_chapter_contexts(
    candidates: list[Candidate],
    segments: list[RichSegment],
    chapters: list[Chapter],
) -> dict[str, dict]:
    if not candidates or not segments:
        return {}

    if not chapters:
        chapters = chapter_segmenter.segment_chapters(segments)
    if not chapters:
        return {candidate.id: _empty_chapter_context() for candidate in candidates}

    total_duration = max(segments[-1].end - segments[0].start, 1.0)
    chapter_tokens = {
        chapter.id: _tokens_for_text(
            " ".join(seg.text.strip() for seg in chapter_segmenter.segments_for_chapter(segments, chapter))
        )
        for chapter in chapters
    }
    chapter_assignments: dict[str, Chapter] = {}
    chapter_counts: dict[str, int] = {chapter.id: 0 for chapter in chapters}
    for candidate in candidates:
        center = (candidate.start + candidate.end) / 2.0
        chapter = chapter_segmenter.chapter_for_time(chapters, center)
        if chapter is None:
            continue
        chapter_assignments[candidate.id] = chapter
        chapter_counts[chapter.id] = chapter_counts.get(chapter.id, 0) + 1

    overlaps_by_candidate: dict[str, float] = {}
    for candidate in candidates:
        chapter = chapter_assignments.get(candidate.id)
        same_chapter = [
            other for other in candidates
            if other.id != candidate.id and chapter_assignments.get(other.id) == chapter
        ]
        overlap_count = 0.0
        for other in same_chapter:
            overlap = min(candidate.end, other.end) - max(candidate.start, other.start)
            if overlap > 0:
                overlap_count += 1.0
        overlaps_by_candidate[candidate.id] = overlap_count

    contexts: dict[str, dict] = {}
    for candidate in candidates:
        chapter = chapter_assignments.get(candidate.id)
        if chapter is None:
            contexts[candidate.id] = _empty_chapter_context()
            continue
        chapter_duration = max(chapter.end - chapter.start, 0.1)
        center = (candidate.start + candidate.end) / 2.0
        left_context = max(candidate.start - chapter.start, 0.0)
        right_context = max(chapter.end - candidate.end, 0.0)
        chapter_position = _clamp01((center - chapter.start) / chapter_duration)
        chapter_boundary_distance = min(left_context, right_context)
        boundary_proximity = 1.0 - min(chapter_boundary_distance / 30.0, 1.0)
        has_neighboring_chapters = 1.0 if len(chapters) > 1 else 0.0
        topic_shift_strength = round(chapter.confidence * boundary_proximity * has_neighboring_chapters, 3)
        candidates_per_minute = chapter_counts.get(chapter.id, 0) / max(chapter_duration / 60.0, 0.5)
        chapter_local_density = round(min(candidates_per_minute / 6.0, 1.0), 3)
        candidate_tokens = _tokens_for_text("")
        candidate_window = [seg for seg in segments if seg.end > candidate.start and seg.start < candidate.end]
        if candidate_window:
            candidate_tokens = _tokens_for_text(" ".join(seg.text.strip() for seg in candidate_window))
        chapter_coherence = round(_jaccard_similarity(candidate_tokens, chapter_tokens.get(chapter.id, set())), 3)
        vod_progress = round((center - segments[0].start) / total_duration, 3)

        contexts[candidate.id] = {
            "vod_progress": vod_progress,
            "boundary_left_context_sec": round(left_context, 3),
            "boundary_right_context_sec": round(right_context, 3),
            "clip_overlap_count": overlaps_by_candidate.get(candidate.id, 0.0),
            "chapter_position": round(chapter_position, 3),
            "topic_shift_strength": topic_shift_strength,
            "chapter_local_density": chapter_local_density,
            "chapter_boundary_distance": round(chapter_boundary_distance, 3),
            "chapter_coherence_score": chapter_coherence,
        }
    return contexts


def _empty_chapter_context() -> dict:
    return {
        "vod_progress": 0.0,
        "boundary_left_context_sec": 0.0,
        "boundary_right_context_sec": 0.0,
        "clip_overlap_count": 0.0,
        "chapter_position": 0.0,
        "topic_shift_strength": 0.0,
        "chapter_local_density": 0.0,
        "chapter_boundary_distance": 0.0,
        "chapter_coherence_score": 0.0,
    }


def _empty_memory_context() -> dict:
    return {
        "setup_depth_score": 0.0,
        "callback_strength": 0.0,
        "arc_payoff_match": 0.0,
        "topic_continuity_score": 0.0,
        "speaker_context_clarity": 0.0,
        "memory_dependency_score": 0.0,
        "story_stage_score": 0.0,
        "required_setup_lead_sec": 0.0,
        "retrieved_memory_span_count": 0.0,
    }


def _tokens_for_text(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(len(left | right), 1)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


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
