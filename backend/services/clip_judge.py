from __future__ import annotations

import base64
from collections import Counter
import hashlib
import json
import logging
import pathlib
import time
from typing import Callable
import urllib.error
import urllib.request

import ollama

from config import settings
from services.keyframe_sampler import SampledKeyframe
from services.timeline_types import CandidateFeatures, JudgeResult, RankedCandidate
from services.transcriber import RichSegment, format_transcript

logger = logging.getLogger(__name__)
JudgeProviderCall = Callable[[str, list[SampledKeyframe] | None], JudgeResult | None]


class ClipJudgeError(RuntimeError):
    """Raised when the configured clip judge cannot produce a required LLM result."""

_AXIS_NAMES = (
    "scroll_stop_power",
    "standalone_clarity",
    "payoff_strength",
    "narrative_completeness",
    "platform_fit_shortform",
)
_JUDGE_WEIGHTS = {
    "scroll_stop_power": 0.30,
    "standalone_clarity": 0.18,
    "payoff_strength": 0.28,
    "narrative_completeness": 0.12,
    "platform_fit_shortform": 0.12,
}
_JUDGE_PROMPT_STATIC = """You are judging whether a candidate clip has real viral short-form potential, not just whether it is a complete stream moment.

Score each axis from 0.0 to 1.0:
- scroll_stop_power: does the first 1-3 seconds create immediate curiosity, shock, conflict, comedy, or reaction?
- standalone_clarity: can a new viewer quickly understand the reaction spike, clutch, argument, unboxing reveal, or punchline? Do not require a complete stream story when the moment is a clear reaction beat.
- payoff_strength: is there a strong reaction, reveal, punchline, clutch, argument, or quote worth replaying?
- narrative_completeness: does it feel like a coherent mini-story rather than a fragment?
- platform_fit_shortform: is the pacing, density, and replay value right for Shorts/Reels/TikTok?

For Jynxzi-style content, value scroll-stopping reaction/clutch/rage/unboxing spikes even when the context is compact or
the full narrative is not self-contained. Be strict about generic, slow, or payoff-free moments, but do not over-penalize
short reaction clips just because they are less narratively standalone than an essay-style clip.

Calibration anchors for this profile:
- Strong: immediate streamer reaction to a clutch, rage moment, fanmail/unboxing reveal, absurd chat prompt, or quotable hot take.
- Medium: understandable setup with a clear payoff but weaker first-second hook or replay value.
- Weak: generic talking, slow setup with no sharp reaction, or a clip that only makes sense with long stream context.
Return raw JSON only:
{{
  "scroll_stop_power": <float>,
  "standalone_clarity": <float>,
  "payoff_strength": <float>,
  "narrative_completeness": <float>,
  "platform_fit_shortform": <float>,
  "judge_reason": "<short sentence>",
  "judge_confidence": <float>
}}

"""
_JUDGE_PROMPT_DYNAMIC = """\
Profile: {profile}
Current rank score: {rank_score}
Source: {source}
Candidate duration seconds: {duration}

Feature summary:
{feature_summary}

Prior context:
{context_summary}

Transcript excerpt:
{transcript}
"""
_JUDGE_PROMPT = _JUDGE_PROMPT_STATIC + _JUDGE_PROMPT_DYNAMIC

def rerank_candidates(
    ranked_candidates: list[RankedCandidate],
    segments: list[RichSegment],
    profile: str = "general",
    shortlist_top_k: int | None = None,
    provider: str | None = None,
    keyframes_by_candidate: dict[str, list[SampledKeyframe]] | None = None,
    progress_cb: Callable[[dict[str, object]], None] | None = None,
) -> list[RankedCandidate]:
    if not ranked_candidates:
        return []

    limit = len(ranked_candidates) if shortlist_top_k is None else min(max(shortlist_top_k, 0), len(ranked_candidates))
    if limit <= 0:
        return ranked_candidates

    reranked: list[RankedCandidate] = list(ranked_candidates)
    started_at = time.perf_counter()
    counters: Counter[str] = Counter()
    provider_counts: Counter[str] = Counter()
    keyframe_candidate_count = sum(1 for frames in (keyframes_by_candidate or {}).values() if frames)
    batch_size = _effective_ollama_batch_size(provider, keyframes_by_candidate)
    snapshot_path = pathlib.Path(settings.storage_dir) / "feedback" / "mm_judge_snapshots.jsonl"
    _log_provider_event(
        "clip_judge_rerank_start",
        profile=profile,
        candidates_total=len(ranked_candidates),
        candidates_limit=limit,
        requested_provider=(provider or "auto"),
        keyframe_candidate_count=keyframe_candidate_count,
        fail_fast=bool(getattr(settings, "clip_judge_fail_fast", False)),
        provider_fallback_enabled=bool(getattr(settings, "judge_provider_fallback_enabled", True)),
        batch_size=batch_size,
        sample_count=_clip_judge_sample_count(),
        uncertain_sample_count=_clip_judge_uncertain_sample_count(),
        cache_enabled=bool(getattr(settings, "clip_judge_cache_enabled", True)),
    )

    def emit_progress(index: int, ranked: RankedCandidate | None = None) -> None:
        if progress_cb is None:
            return
        payload = {
            "processed_candidates": index,
            "total_candidates": limit,
            "attempted_candidates": int(counters["judge_attempted"]),
            "judged_candidates": int(counters["llm_used"] + counters["heuristic_used"]),
            "recall_filtered_candidates": int(counters["recall_filtered"]),
            "elapsed_sec": round(time.perf_counter() - started_at, 3),
        }
        if ranked is not None:
            payload["candidate_id"] = ranked.candidate.id
            payload["candidate_start"] = round(float(ranked.candidate.start), 3)
            payload["candidate_end"] = round(float(ranked.candidate.end), 3)
        progress_cb(payload)

    def apply_judge_result(idx: int, ranked: RankedCandidate, judge_result: JudgeResult) -> None:
        provider_counts[str(judge_result.provider or "heuristic")] += 1
        if judge_result.llm_used:
            counters["llm_used"] += 1
        else:
            counters["heuristic_used"] += 1
        _attach_judge_features(ranked.features, judge_result)
        _append_judge_snapshot(ranked, judge_result, profile=profile)
        counters["snapshots_written"] += 1
        final_score, score_reason = _judge_adjusted_score(ranked.rank_score, judge_result)
        counters[score_reason] += 1
        rationale = list(ranked.rationale)
        rationale.append("judge_reranked")
        rationale.append(score_reason)
        if _is_cross_platform_recall_target(ranked):
            score_floor = float(getattr(settings, "serving_cross_platform_judge_score_floor", 7.2) or 0.0)
            if final_score < score_floor:
                final_score = round(score_floor + min(max(float(ranked.rank_score) / 12.0, -0.05), 0.05), 3)
                rationale.append("cross_platform_judge_calibrated")
                counters["cross_platform_judge_calibrated"] += 1
        if judge_result.llm_used:
            rationale.append("judge_llm")
        if judge_result.provider == "ollama_batch":
            rationale.append("judge_batch")
        reranked[idx] = RankedCandidate(
            candidate=ranked.candidate,
            features=ranked.features,
            rank_score=final_score,
            rationale=rationale,
            base_rank_score=ranked.base_rank_score if ranked.base_rank_score is not None else ranked.rank_score,
            heuristic_rank_score=ranked.heuristic_rank_score,
            trained_rank_score=ranked.trained_rank_score,
            model_blend=ranked.model_blend,
            trained_model_used=ranked.trained_model_used,
            judge_result=judge_result,
        )

    def recall_filter_candidate(idx: int, ranked: RankedCandidate) -> bool:
        if _passes_judge_recall_filter(ranked):
            return False
        counters["recall_filtered"] += 1
        rationale = list(ranked.rationale)
        rationale.append("judge_recall_filtered")
        reranked[idx] = RankedCandidate(
            candidate=ranked.candidate,
            features=ranked.features,
            rank_score=ranked.rank_score,
            rationale=rationale,
            base_rank_score=ranked.base_rank_score,
            heuristic_rank_score=ranked.heuristic_rank_score,
            trained_rank_score=ranked.trained_rank_score,
            model_blend=ranked.model_blend,
            trained_model_used=ranked.trained_model_used,
            judge_result=ranked.judge_result,
        )
        emit_progress(idx + 1, ranked)
        return True

    try:
        emit_progress(0)
        idx = 0
        while idx < limit:
            ranked = reranked[idx]
            if not _passes_judge_recall_filter(ranked):
                recall_filter_candidate(idx, ranked)
                idx += 1
                continue

            if batch_size > 1:
                batch: list[tuple[int, RankedCandidate]] = []
                cursor = idx
                while cursor < limit and len(batch) < batch_size:
                    candidate = reranked[cursor]
                    if not _passes_judge_recall_filter(candidate):
                        break
                    batch.append((cursor, candidate))
                    cursor += 1
                if batch:
                    counters["judge_attempted"] += len(batch)
                    batch_results = _judge_ollama_batch(
                        [candidate for _, candidate in batch],
                        segments,
                        profile=profile,
                    )
                    if batch_results:
                        counters["batch_calls"] += 1
                    for batch_idx, batch_ranked in batch:
                        judge_result = batch_results.get(batch_ranked.candidate.id) if batch_results else None
                        if judge_result is None:
                            counters["batch_missing_result"] += 1
                            judge_result = judge_candidate(
                                batch_ranked,
                                segments,
                                profile=profile,
                                provider=provider,
                                keyframes=(keyframes_by_candidate or {}).get(batch_ranked.candidate.id),
                            )
                        apply_judge_result(batch_idx, batch_ranked, judge_result)
                        emit_progress(batch_idx + 1, reranked[batch_idx])
                idx = max(cursor, idx + 1)
                continue

            counters["judge_attempted"] += 1
            judge_result = judge_candidate(
                ranked,
                segments,
                profile=profile,
                provider=provider,
                keyframes=(keyframes_by_candidate or {}).get(ranked.candidate.id),
            )
            apply_judge_result(idx, ranked, judge_result)
            emit_progress(idx + 1, reranked[idx])
            idx += 1
    except Exception as exc:
        _log_provider_event(
            "clip_judge_rerank_failed",
            profile=profile,
            candidates_total=len(ranked_candidates),
            candidates_limit=limit,
            elapsed_sec=round(time.perf_counter() - started_at, 3),
            counters=dict(counters),
            provider_counts=dict(provider_counts),
            error_type=type(exc).__name__,
            error=_safe_error_message(exc),
            fail_fast=bool(getattr(settings, "clip_judge_fail_fast", False)),
        )
        raise

    reranked.sort(key=lambda item: item.rank_score, reverse=True)
    _log_provider_event(
        "clip_judge_rerank_done",
        profile=profile,
        candidates_total=len(ranked_candidates),
        candidates_limit=limit,
        elapsed_sec=round(time.perf_counter() - started_at, 3),
        counters=dict(counters),
        provider_counts=dict(provider_counts),
        snapshot_path=str(snapshot_path),
    )
    return reranked


def _passes_judge_recall_filter(ranked: RankedCandidate) -> bool:
    if not bool(getattr(settings, "judge_recall_filter_enabled", True)):
        return True
    feature = ranked.features
    min_score = float(getattr(settings, "judge_recall_filter_min_score", 0.0) or 0.0)
    if ranked.rank_score >= min_score:
        return True
    if feature is None:
        return ranked.candidate.score_hint > 0.0
    if bool(getattr(feature, "cross_platform_recall_target", False)):
        return True
    recall_signals = (
        float(feature.hook_strength) > 0.0,
        float(feature.chat_peak) > 0.0,
        float(feature.chat_hype_score) > 0.0,
        float(feature.max_hype) > 0.0,
        float(feature.payoff_score) > 0.0,
        float(feature.face_reaction_peak) > 0.0,
        float(feature.motion_peak) > 0.0,
        float(feature.visual_hook_score) > 0.0,
        float(feature.audio_pop_score) > 0.0,
        float(feature.streamer_shout_score) > 0.0,
        float(feature.event_surprise_score) > 0.0,
        float(feature.game_event_score) > 0.0,
        float(feature.vl_prompt_max_score) > 0.0,
        bool(feature.is_clutch_window),
        ranked.candidate.score_hint > 0.0,
    )
    return any(recall_signals)


def _attach_judge_features(features: CandidateFeatures | None, judge_result: JudgeResult) -> None:
    if features is None:
        return
    features.judge_score = round(float(judge_result.judge_score), 3)
    features.judge_confidence = round(float(judge_result.judge_confidence), 3)
    features.judge_scroll_stop_power = round(float(judge_result.scroll_stop_power), 3)
    features.judge_standalone_clarity = round(float(judge_result.standalone_clarity), 3)
    features.judge_payoff_strength = round(float(judge_result.payoff_strength), 3)
    features.judge_narrative_completeness = round(float(judge_result.narrative_completeness), 3)
    features.judge_platform_fit_shortform = round(float(judge_result.platform_fit_shortform), 3)
    features.has_judge_score = True
    features.judge_llm_used = bool(judge_result.llm_used)


def _is_cross_platform_recall_target(ranked: RankedCandidate) -> bool:
    if "cross_platform_recall_target" in ranked.rationale:
        return True
    feature = ranked.features
    return bool(feature is not None and getattr(feature, "cross_platform_recall_target", False))


def _judge_adjusted_score(existing_score: float, judge_result: JudgeResult) -> tuple[float, str]:
    if not bool(getattr(settings, "judge_llm_first", True)):
        return (
            _blend_scores(existing_score, judge_result.judge_score, judge_result.judge_confidence),
            "judge_blended",
        )
    min_confidence = float(getattr(settings, "judge_llm_first_min_confidence", 0.35) or 0.0)
    if not judge_result.llm_used:
        return (
            _blend_scores(existing_score, judge_result.judge_score, judge_result.judge_confidence),
            "judge_fallback_blended",
        )
    if float(judge_result.judge_confidence) < min_confidence:
        return (
            _blend_scores(existing_score, judge_result.judge_score, judge_result.judge_confidence),
            "judge_low_confidence_blended",
        )
    tiebreaker_weight = max(0.0, float(getattr(settings, "judge_llm_first_tiebreaker_weight", 0.01) or 0.0))
    normalized_tiebreaker = max(-1.0, min(float(existing_score) / 12.0, 1.0))
    score = float(judge_result.judge_score) + normalized_tiebreaker * tiebreaker_weight
    return round(score, 3), "judge_primary_ranked"


def _append_judge_snapshot(ranked: RankedCandidate, judge_result: JudgeResult, *, profile: str) -> None:
    path = pathlib.Path(settings.storage_dir) / "feedback" / "mm_judge_snapshots.jsonl"
    payload = {
        "profile": profile,
        "candidate_id": ranked.candidate.id,
        "candidate_source": ranked.candidate.source,
        "window_start": float(ranked.candidate.start),
        "window_end": float(ranked.candidate.end),
        "rank_score": float(ranked.rank_score),
        "judge_score": float(judge_result.judge_score),
        "judge_confidence": float(judge_result.judge_confidence),
        "judge_provider": str(judge_result.provider),
        "judge_model": str(judge_result.model),
        "judge_llm_used": bool(judge_result.llm_used),
        "judge_reason": str(judge_result.judge_reason),
        "features": ranked.features.to_dict() if ranked.features is not None else {},
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except OSError as exc:
        logger.debug("clip judge snapshot persist failed: %s", exc)


def judge_candidate(
    ranked: RankedCandidate,
    segments: list[RichSegment],
    *,
    profile: str = "general",
    provider: str | None = None,
    keyframes: list[SampledKeyframe] | None = None,
) -> JudgeResult:
    window = [
        seg for seg in segments
        if seg.end > ranked.candidate.start and seg.start < ranked.candidate.end
    ]
    heuristic = _heuristic_judge(ranked, window)
    if not window:
        if _clip_judge_fail_fast_enabled():
            _log_provider_event(
                "clip_judge_candidate_failed",
                candidate_id=ranked.candidate.id,
                reason="empty_transcript_window",
                fail_fast=True,
            )
            raise ClipJudgeError(f"clip judge has no transcript window for candidate {ranked.candidate.id}")
        return heuristic

    llm_result = _call_llm_judge(
        ranked,
        segments,
        window,
        profile=profile,
        provider=provider,
        keyframes=keyframes,
    )
    if llm_result is not None:
        return llm_result
    if _clip_judge_fail_fast_enabled():
        _log_provider_event(
            "clip_judge_candidate_failed",
            candidate_id=ranked.candidate.id,
            reason="llm_result_missing",
            fail_fast=True,
        )
        raise ClipJudgeError(f"clip judge LLM failed for candidate {ranked.candidate.id}")
    return heuristic


def _effective_ollama_batch_size(
    provider: str | None,
    keyframes_by_candidate: dict[str, list[SampledKeyframe]] | None,
) -> int:
    if keyframes_by_candidate and any(keyframes_by_candidate.values()):
        return 1
    preferred = (provider or "auto").strip().lower()
    if preferred != "ollama":
        return 1
    return max(int(getattr(settings, "clip_judge_batch_size", 1) or 1), 1)


def _judge_ollama_batch(
    ranked_candidates: list[RankedCandidate],
    segments: list[RichSegment],
    *,
    profile: str,
) -> dict[str, JudgeResult]:
    if not ranked_candidates:
        return {}
    items: list[dict[str, object]] = []
    for ranked in ranked_candidates:
        window = [
            seg for seg in segments
            if seg.end > ranked.candidate.start and seg.start < ranked.candidate.end
        ]
        if not window:
            continue
        items.append(
            {
                "candidate_id": ranked.candidate.id,
                "rank_score": round(float(ranked.rank_score), 3),
                "source": ranked.candidate.source,
                "duration": round(max(ranked.candidate.end - ranked.candidate.start, 0.0), 2),
                "feature_summary": _feature_summary(ranked),
                "prior_context": _build_context_summary(segments, ranked.candidate.start),
                "transcript": format_transcript(window, use_seconds=True),
            }
        )
    if not items:
        return {}

    prompt = _batch_judge_prompt(profile=profile, items=items)
    try:
        client = ollama.Client(host=settings.ollama_host)
        response = client.chat(
            model=settings.ollama_model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0.0},
        )
        content = response["message"]["content"].strip()
        results = _parse_batch_judge_results(content)
        parsed: dict[str, JudgeResult] = {}
        for candidate_id, payload in results.items():
            result = _parse_judge_result(json.dumps(payload), provider="ollama_batch", llm_used=True)
            if result is None:
                continue
            result.model = settings.ollama_model
            parsed[candidate_id] = result
        _log_provider_event(
            "clip_judge_batch_done",
            provider="ollama_batch",
            model=settings.ollama_model,
            requested_candidates=len(items),
            parsed_candidates=len(parsed),
        )
        return parsed
    except Exception as exc:
        _log_provider_event(
            "clip_judge_batch_failed",
            provider="ollama_batch",
            model=settings.ollama_model,
            requested_candidates=len(items),
            error_type=type(exc).__name__,
            error=_safe_error_message(exc),
        )
        return {}


def _batch_judge_prompt(*, profile: str, items: list[dict[str, object]]) -> str:
    return (
        "You are judging whether candidate clips have real viral short-form potential, not just whether they are complete stream moments.\n"
        "Score each axis from 0.0 to 1.0:\n"
        "- scroll_stop_power: does the first 1-3 seconds create immediate curiosity, shock, conflict, comedy, or reaction?\n"
        "- standalone_clarity: can a new viewer quickly understand the reaction spike, clutch, argument, unboxing reveal, or punchline? Do not require a complete stream story when the moment is a clear reaction beat.\n"
        "- payoff_strength: is there a strong reaction, reveal, punchline, clutch, argument, or quote worth replaying?\n"
        "- narrative_completeness: does it feel like a coherent mini-story rather than a fragment?\n"
        "- platform_fit_shortform: is the pacing, density, and replay value right for Shorts/Reels/TikTok?\n\n"
        "For Jynxzi-style content, value scroll-stopping reaction/clutch/rage/unboxing spikes even when the context is compact "
        "or the full narrative is not self-contained. Be strict about generic, slow, or payoff-free moments, but do not "
        "over-penalize short reaction clips just because they are less narratively standalone than an essay-style clip.\n"
        "Calibration anchors for this profile: Strong = immediate streamer reaction to a clutch, rage moment, "
        "fanmail/unboxing reveal, absurd chat prompt, or quotable hot take. Medium = understandable setup with a clear "
        "payoff but weaker first-second hook or replay value. Weak = generic talking, slow setup with no sharp reaction, "
        "or a clip that only makes sense with long stream context.\n"
        "Return raw JSON only with exactly this shape:\n"
        '{"results":[{"candidate_id":"<id>","scroll_stop_power":0.0,'
        '"standalone_clarity":0.0,"payoff_strength":0.0,'
        '"narrative_completeness":0.0,"platform_fit_shortform":0.0,'
        '"judge_reason":"<short sentence>","judge_confidence":0.0}]}\n\n'
        f"Profile: {profile}\n"
        "Candidates:\n"
        f"{json.dumps(items, ensure_ascii=True)}"
    )


def _parse_batch_judge_results(content: str) -> dict[str, dict]:
    start = content.find("{")
    end = content.rfind("}")
    if start < 0 or end < start:
        return {}
    try:
        data = json.loads(content[start:end + 1])
    except json.JSONDecodeError:
        return {}
    raw_results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(raw_results, list):
        return {}
    parsed: dict[str, dict] = {}
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        candidate_id = str(item.get("candidate_id") or "").strip()
        if not candidate_id:
            continue
        parsed[candidate_id] = item
    return parsed


def _call_llm_judge(
    ranked: RankedCandidate,
    segments: list[RichSegment],
    window: list[RichSegment],
    *,
    profile: str,
    provider: str | None,
    keyframes: list[SampledKeyframe] | None,
) -> JudgeResult | None:
    prompt = _JUDGE_PROMPT.format(
        profile=profile,
        rank_score=f"{ranked.rank_score:.3f}",
        source=ranked.candidate.source,
        duration=f"{max(ranked.candidate.end - ranked.candidate.start, 0.0):.2f}",
        feature_summary=_feature_summary(ranked),
        context_summary=_build_context_summary(segments, ranked.candidate.start),
        transcript=format_transcript(window, use_seconds=True),
    )

    cache_key = _judge_cache_key(prompt, provider=provider, keyframes=keyframes)
    cached = _load_cached_judge_result(cache_key)
    if cached is not None:
        _log_provider_event(
            "clip_judge_cache_hit",
            candidate_id=ranked.candidate.id,
            provider=provider or "auto",
            keyframes=bool(keyframes),
        )
        return cached

    sample_target = _clip_judge_sample_count()
    uncertain_target = _clip_judge_uncertain_sample_count()
    results: list[JudgeResult] = []
    attempts = sample_target
    while len(results) < attempts:
        result = _call_text_judge_prompt(prompt, provider=provider, keyframes=keyframes)
        if result is not None:
            results.append(result)
            if (
                len(results) == sample_target
                and uncertain_target > sample_target
                and _judge_result_needs_consensus(result)
            ):
                attempts = uncertain_target
        elif not results:
            attempts = max(attempts, 1)
            break
        else:
            break

    if not results:
        return None

    final_result = _median_judge_results(results)
    _store_cached_judge_result(cache_key, final_result, samples=len(results))
    if len(results) > 1:
        _log_provider_event(
            "clip_judge_consensus_done",
            candidate_id=ranked.candidate.id,
            samples=len(results),
            provider=final_result.provider,
            judge_score=final_result.judge_score,
            judge_confidence=final_result.judge_confidence,
        )
    return final_result


def _clip_judge_sample_count() -> int:
    return max(int(getattr(settings, "clip_judge_sample_count", 1) or 1), 1)


def _clip_judge_uncertain_sample_count() -> int:
    return max(
        int(getattr(settings, "clip_judge_uncertain_sample_count", 1) or 1),
        _clip_judge_sample_count(),
    )


def _judge_result_needs_consensus(result: JudgeResult) -> bool:
    if _clip_judge_uncertain_sample_count() <= _clip_judge_sample_count():
        return False
    margin = max(float(getattr(settings, "clip_judge_uncertain_margin", 0.08) or 0.0), 0.0)
    if margin <= 0.0:
        return False
    if float(result.judge_confidence) < 0.7:
        return True
    axis_thresholds = {
        "standalone_clarity": float(getattr(settings, "serving_quality_min_standalone_clarity", 0.35)),
        "payoff_strength": float(getattr(settings, "serving_quality_min_payoff_strength", 0.35)),
        "narrative_completeness": float(getattr(settings, "serving_quality_min_narrative_completeness", 0.45)),
    }
    axis_values = {
        "standalone_clarity": float(result.standalone_clarity),
        "payoff_strength": float(result.payoff_strength),
        "narrative_completeness": float(result.narrative_completeness),
    }
    if any(abs(axis_values[name] - threshold) <= margin for name, threshold in axis_thresholds.items()):
        return True
    judge_threshold = float(getattr(settings, "serving_quality_min_judge_score", 4.4))
    return abs(float(result.judge_score) - judge_threshold) <= margin * 10.0


def _median_judge_results(results: list[JudgeResult]) -> JudgeResult:
    if len(results) == 1:
        return results[0]

    def median(values: list[float]) -> float:
        ordered = sorted(values)
        midpoint = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[midpoint]
        return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0

    axes = {
        "scroll_stop_power": _clamp01(median([float(result.scroll_stop_power) for result in results])),
        "standalone_clarity": _clamp01(median([float(result.standalone_clarity) for result in results])),
        "payoff_strength": _clamp01(median([float(result.payoff_strength) for result in results])),
        "narrative_completeness": _clamp01(median([float(result.narrative_completeness) for result in results])),
        "platform_fit_shortform": _clamp01(median([float(result.platform_fit_shortform) for result in results])),
    }
    confidence = _clamp01(median([float(result.judge_confidence) for result in results]))
    reason_source = max(results, key=lambda result: float(result.judge_confidence))
    providers = {str(result.provider) for result in results}
    models = {str(result.model) for result in results if result.model}
    return JudgeResult(
        judge_score=_axes_to_score(axes),
        scroll_stop_power=axes["scroll_stop_power"],
        standalone_clarity=axes["standalone_clarity"],
        payoff_strength=axes["payoff_strength"],
        narrative_completeness=axes["narrative_completeness"],
        platform_fit_shortform=axes["platform_fit_shortform"],
        judge_reason=str(reason_source.judge_reason)[:220],
        judge_confidence=confidence,
        provider=providers.pop() if len(providers) == 1 else "judge_consensus",
        llm_used=any(result.llm_used for result in results),
        model=models.pop() if len(models) == 1 else "",
    )


def _judge_cache_key(
    prompt: str,
    *,
    provider: str | None,
    keyframes: list[SampledKeyframe] | None,
) -> str:
    keyframe_payload: list[dict[str, object]] = []
    for frame in keyframes or []:
        path = pathlib.Path(frame.path)
        try:
            stat = path.stat()
            path_payload = {
                "path": str(path),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        except OSError:
            path_payload = {"path": str(path), "size": None, "mtime_ns": None}
        keyframe_payload.append(
            {
                "label": frame.label,
                "time_sec": round(float(frame.time_sec), 3),
                "mime_type": frame.mime_type,
                **path_payload,
            }
        )
    payload = {
        "version": 3,
        "provider": provider or "auto",
        "prompt": prompt,
        "keyframes": keyframe_payload,
        "sampling": {
            "sample_count": _clip_judge_sample_count(),
            "uncertain_sample_count": _clip_judge_uncertain_sample_count(),
            "uncertain_margin": float(getattr(settings, "clip_judge_uncertain_margin", 0.08) or 0.0),
        },
        "models": {
            "gemini": settings.gemini_model,
            "ollama": settings.ollama_model,
            "groq": settings.groq_model,
            "anthropic_bulk": settings.anthropic_model_bulk,
            "anthropic_topk": settings.anthropic_model_topk,
        },
    }
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _judge_cache_path(cache_key: str) -> pathlib.Path:
    storage_root = pathlib.Path(settings.storage_dir)
    cache_root = storage_root.parent / "cache" / "clip_judge"
    return cache_root / f"{cache_key}.json"


def _load_cached_judge_result(cache_key: str) -> JudgeResult | None:
    if not bool(getattr(settings, "clip_judge_cache_enabled", True)):
        return None
    path = _judge_cache_path(cache_key)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    result_payload = payload.get("judge_result") if isinstance(payload, dict) else None
    if not isinstance(result_payload, dict):
        return None
    try:
        return JudgeResult(**result_payload)
    except TypeError:
        return None


def _store_cached_judge_result(cache_key: str, result: JudgeResult, *, samples: int) -> None:
    if not bool(getattr(settings, "clip_judge_cache_enabled", True)):
        return
    path = _judge_cache_path(cache_key)
    payload = {
        "cache_key": cache_key,
        "created_at": time.time(),
        "samples": int(samples),
        "judge_result": result.to_dict(),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True), encoding="utf-8")
        tmp_path.replace(path)
    except OSError as exc:
        logger.debug("clip judge cache persist failed: %s", exc)


def _call_text_judge_prompt(
    prompt: str,
    *,
    provider: str | None,
    keyframes: list[SampledKeyframe] | None = None,
) -> JudgeResult | None:
    preferred = (provider or "auto").strip().lower()
    attempts = _judge_provider_attempts(preferred, bool(keyframes))
    if not attempts:
        _log_provider_event(
            "clip_judge_no_provider_configured",
            preferred=preferred,
            keyframes=bool(keyframes),
            fail_fast=_clip_judge_fail_fast_enabled(),
        )
        if _clip_judge_fail_fast_enabled():
            raise ClipJudgeError(f"clip judge has no configured LLM provider for preferred={preferred}")
        return None
    failed_providers: list[str] = []
    for provider_name, call in attempts:
        result = call(prompt, keyframes)
        if result is not None:
            _log_provider_event(
                "clip_judge_provider_selected",
                provider=provider_name,
                preferred=preferred,
                fallback_from=failed_providers,
                keyframes=bool(keyframes),
                model=getattr(result, "model", ""),
            )
            return result
        _log_provider_event(
            "clip_judge_provider_failed",
            provider=provider_name,
            preferred=preferred,
            keyframes=bool(keyframes),
            failure_reason="empty_result_or_parse_failure",
        )
        failed_providers.append(provider_name)
        if len(failed_providers) < len(attempts):
            _log_provider_event(
                "clip_judge_provider_fallback",
                failed_provider=provider_name,
                next_provider=attempts[len(failed_providers)][0],
                preferred=preferred,
                keyframes=bool(keyframes),
            )
    _log_provider_event(
        "clip_judge_all_providers_failed",
        preferred=preferred,
        attempted_providers=failed_providers,
        keyframes=bool(keyframes),
        fail_fast=_clip_judge_fail_fast_enabled(),
    )
    if _clip_judge_fail_fast_enabled():
        raise ClipJudgeError(
            "clip judge LLM provider failed"
            f" preferred={preferred} attempted={','.join(failed_providers)}"
        )
    return None


def _judge_provider_attempts(preferred: str, has_keyframes: bool) -> list[tuple[str, JudgeProviderCall]]:
    if preferred == "anthropic" and has_keyframes and settings.enable_multimodal_clip_judge:
        return [("anthropic_multimodal", _call_anthropic_multimodal_from_optional_keyframes)]
    if preferred == "anthropic":
        return [("anthropic", _call_anthropic_text)]
    if preferred == "anthropic_multimodal":
        return [("anthropic_multimodal", _call_anthropic_multimodal_from_optional_keyframes)]
    if preferred == "anthropic_text":
        return [("anthropic", _call_anthropic_text)]
    if preferred == "gemini" and has_keyframes and settings.enable_multimodal_clip_judge:
        return [("gemini_multimodal", _call_gemini_multimodal_from_optional_keyframes)]
    if preferred == "gemini":
        return [("gemini", _call_gemini_text)]
    if preferred == "groq":
        return [("groq", _call_groq_text)]
    if preferred == "ollama":
        return [("ollama", _call_ollama_text)]

    attempts: list[tuple[str, JudgeProviderCall]] = []
    if settings.anthropic_api_key and has_keyframes and settings.enable_multimodal_clip_judge:
        attempts.append(("anthropic_multimodal", _call_anthropic_multimodal_from_optional_keyframes))
    if settings.anthropic_api_key:
        attempts.append(("anthropic", _call_anthropic_text))
    if settings.gemini_api_key and has_keyframes and settings.enable_multimodal_clip_judge:
        attempts.append(("gemini_multimodal", _call_gemini_multimodal_from_optional_keyframes))
    if settings.gemini_api_key:
        attempts.append(("gemini", _call_gemini_text))
    if settings.groq_api_key:
        attempts.append(("groq", _call_groq_text))
    fallback_enabled = bool(getattr(settings, "judge_provider_fallback_enabled", True))
    if fallback_enabled:
        attempts.append(("ollama", _call_ollama_text))
    elif attempts:
        return attempts[:1]
    return attempts


def _log_provider_event(event: str, **fields) -> None:
    logger.info(json.dumps({"event": event, **fields}, ensure_ascii=True, sort_keys=True))


def _clip_judge_fail_fast_enabled() -> bool:
    return bool(getattr(settings, "clip_judge_fail_fast", False))


def _safe_error_message(exc: BaseException, *, limit: int = 240) -> str:
    text = str(exc).replace("\n", " ").strip()
    if len(text) > limit:
        return text[:limit].rstrip() + "..."
    return text


def _log_anthropic_usage(response_payload: dict, *, provider: str, model: str, keyframes: bool) -> None:
    usage = response_payload.get("usage")
    if not isinstance(usage, dict):
        return
    _log_provider_event(
        "clip_judge_provider_usage",
        provider=provider,
        model=model,
        keyframes=keyframes,
        input_tokens=_int_or_zero(usage.get("input_tokens")),
        output_tokens=_int_or_zero(usage.get("output_tokens")),
        cache_creation_input_tokens=_int_or_zero(usage.get("cache_creation_input_tokens")),
        cache_read_input_tokens=_int_or_zero(usage.get("cache_read_input_tokens")),
    )


def _log_openai_usage(response, *, provider: str, model: str, keyframes: bool) -> None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    _log_provider_event(
        "clip_judge_provider_usage",
        provider=provider,
        model=model,
        keyframes=keyframes,
        input_tokens=_int_or_zero(getattr(usage, "prompt_tokens", 0)),
        output_tokens=_int_or_zero(getattr(usage, "completion_tokens", 0)),
        total_tokens=_int_or_zero(getattr(usage, "total_tokens", 0)),
    )


def _int_or_zero(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _call_gemini_text(prompt: str, _keyframes: list[SampledKeyframe] | None = None) -> JudgeResult | None:
    return _call_gemini(prompt)


def _call_anthropic_text(prompt: str, _keyframes: list[SampledKeyframe] | None = None) -> JudgeResult | None:
    return _call_anthropic(
        prompt,
        keyframes=None,
        model=settings.anthropic_model_bulk,
        provider="anthropic",
    )


def _call_anthropic_multimodal_from_optional_keyframes(
    prompt: str,
    keyframes: list[SampledKeyframe] | None = None,
) -> JudgeResult | None:
    if not keyframes:
        return None
    return _call_anthropic(
        prompt,
        keyframes=keyframes,
        model=settings.anthropic_model_topk,
        provider="anthropic_multimodal",
    )


def _call_gemini_multimodal_from_optional_keyframes(
    prompt: str,
    keyframes: list[SampledKeyframe] | None = None,
) -> JudgeResult | None:
    if not keyframes:
        return None
    return _call_gemini_multimodal(prompt, keyframes)


def _call_groq_text(prompt: str, _keyframes: list[SampledKeyframe] | None = None) -> JudgeResult | None:
    return _call_groq(prompt)


def _call_ollama_text(prompt: str, _keyframes: list[SampledKeyframe] | None = None) -> JudgeResult | None:
    return _call_ollama(prompt)


def _call_anthropic(
    prompt: str,
    *,
    keyframes: list[SampledKeyframe] | None,
    model: str,
    provider: str,
) -> JudgeResult | None:
    if not settings.anthropic_api_key:
        return None
    try:
        system_blocks, user_blocks = _anthropic_prompt_blocks(prompt, keyframes)
        payload = {
            "model": model,
            "max_tokens": 220,
            "temperature": 0.0,
            "system": system_blocks,
            "messages": [{"role": "user", "content": user_blocks}],
        }
        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "x-api-key": str(settings.anthropic_api_key),
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        timeout_sec = float(settings.clip_judge_timeout_sec or 0.0)
        with urllib.request.urlopen(request, timeout=timeout_sec if timeout_sec > 0 else None) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
        _log_anthropic_usage(response_payload, provider=provider, model=model, keyframes=bool(keyframes))
        content = _anthropic_text_content(response_payload)
        result = _parse_judge_result(content, provider=provider, llm_used=True)
        if result is not None:
            result.model = model
        else:
            _log_provider_event(
                "clip_judge_provider_parse_failed",
                provider=provider,
                model=model,
                keyframes=bool(keyframes),
                response_chars=len(content),
            )
        return result
    except urllib.error.HTTPError as exc:
        _log_provider_event(
            "clip_judge_provider_error",
            provider=provider,
            model=model,
            keyframes=bool(keyframes),
            error_type=type(exc).__name__,
            http_status=int(exc.code),
        )
        logger.debug("Anthropic clip judge failed: status=%s", exc.code)
        return None
    except Exception as exc:
        _log_provider_event(
            "clip_judge_provider_error",
            provider=provider,
            model=model,
            keyframes=bool(keyframes),
            error_type=type(exc).__name__,
            error=_safe_error_message(exc),
        )
        logger.debug("Anthropic clip judge failed: %s", exc)
        return None


def _anthropic_prompt_blocks(
    prompt: str,
    keyframes: list[SampledKeyframe] | None,
) -> tuple[list[dict], list[dict]]:
    static_prompt, dynamic_prompt = _split_judge_prompt(prompt)
    system_block = {"type": "text", "text": static_prompt}
    if bool(getattr(settings, "anthropic_prompt_caching", True)):
        system_block["cache_control"] = {"type": "ephemeral"}
    user_blocks: list[dict] = []
    if keyframes:
        for frame in keyframes:
            user_blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": frame.mime_type,
                        "data": _base64_for_frame(frame.path),
                    },
                }
            )
        dynamic_prompt = _multimodal_prompt(dynamic_prompt, keyframes)
    user_blocks.append({"type": "text", "text": dynamic_prompt})
    return [system_block], user_blocks


def _split_judge_prompt(prompt: str) -> tuple[str, str]:
    if prompt.startswith(_JUDGE_PROMPT_STATIC):
        return _JUDGE_PROMPT_STATIC, prompt[len(_JUDGE_PROMPT_STATIC):]
    marker = "\nProfile: "
    if marker in prompt:
        static_prompt, dynamic_prompt = prompt.split(marker, 1)
        return static_prompt.rstrip() + "\n\n", "Profile: " + dynamic_prompt
    return _JUDGE_PROMPT_STATIC, prompt


def _anthropic_text_content(response_payload: dict) -> str:
    chunks: list[str] = []
    for block in response_payload.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            chunks.append(str(block.get("text") or ""))
    return "\n".join(chunks).strip()


def _call_gemini(prompt: str) -> JudgeResult | None:
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=settings.gemini_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        response = client.chat.completions.create(
            model=settings.gemini_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=220,
        )
        _log_openai_usage(response, provider="gemini", model=settings.gemini_model, keyframes=False)
        content = (response.choices[0].message.content or "").strip()
        result = _parse_judge_result(content, provider="gemini", llm_used=True)
        if result is not None:
            result.model = settings.gemini_model
        else:
            _log_provider_event(
                "clip_judge_provider_parse_failed",
                provider="gemini",
                model=settings.gemini_model,
                keyframes=False,
                response_chars=len(content),
            )
        return result
    except Exception as exc:
        _log_provider_event(
            "clip_judge_provider_error",
            provider="gemini",
            model=settings.gemini_model,
            keyframes=False,
            error_type=type(exc).__name__,
            error=_safe_error_message(exc),
        )
        logger.debug("Gemini clip judge failed: %s", exc)
        return None


def _call_gemini_multimodal(prompt: str, keyframes: list[SampledKeyframe]) -> JudgeResult | None:
    try:
        from openai import OpenAI

        parts: list[dict] = [{"type": "text", "text": _multimodal_prompt(prompt, keyframes)}]
        for frame in keyframes:
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _data_url_for_frame(frame.path, frame.mime_type),
                    },
                }
            )

        client = OpenAI(
            api_key=settings.gemini_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        response = client.chat.completions.create(
            model=settings.gemini_model,
            messages=[{"role": "user", "content": parts}],
            temperature=0.0,
            max_tokens=220,
        )
        _log_openai_usage(response, provider="gemini_multimodal", model=settings.gemini_model, keyframes=True)
        content = (response.choices[0].message.content or "").strip()
        result = _parse_judge_result(content, provider="gemini_multimodal", llm_used=True)
        if result is not None:
            result.model = settings.gemini_model
        else:
            _log_provider_event(
                "clip_judge_provider_parse_failed",
                provider="gemini_multimodal",
                model=settings.gemini_model,
                keyframes=True,
                response_chars=len(content),
            )
        return result
    except Exception as exc:
        _log_provider_event(
            "clip_judge_provider_error",
            provider="gemini_multimodal",
            model=settings.gemini_model,
            keyframes=True,
            error_type=type(exc).__name__,
            error=_safe_error_message(exc),
        )
        logger.debug("Gemini multimodal clip judge failed: %s", exc)
        return None


def _call_groq(prompt: str) -> JudgeResult | None:
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=220,
        )
        _log_openai_usage(response, provider="groq", model=settings.groq_model, keyframes=False)
        content = (response.choices[0].message.content or "").strip()
        result = _parse_judge_result(content, provider="groq", llm_used=True)
        if result is not None:
            result.model = settings.groq_model
        else:
            _log_provider_event(
                "clip_judge_provider_parse_failed",
                provider="groq",
                model=settings.groq_model,
                keyframes=False,
                response_chars=len(content),
            )
        return result
    except Exception as exc:
        _log_provider_event(
            "clip_judge_provider_error",
            provider="groq",
            model=settings.groq_model,
            keyframes=False,
            error_type=type(exc).__name__,
            error=_safe_error_message(exc),
        )
        logger.debug("Groq clip judge failed: %s", exc)
        return None


def _call_ollama(prompt: str) -> JudgeResult | None:
    try:
        client = ollama.Client(host=settings.ollama_host)
        response = client.chat(
            model=settings.ollama_model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0.0},
        )
        content = response["message"]["content"].strip()
        result = _parse_judge_result(content, provider="ollama", llm_used=True)
        if result is not None:
            result.model = settings.ollama_model
        else:
            _log_provider_event(
                "clip_judge_provider_parse_failed",
                provider="ollama",
                model=settings.ollama_model,
                keyframes=False,
                response_chars=len(content),
            )
        return result
    except Exception as exc:
        _log_provider_event(
            "clip_judge_provider_error",
            provider="ollama",
            model=settings.ollama_model,
            keyframes=False,
            error_type=type(exc).__name__,
            error=_safe_error_message(exc),
        )
        logger.debug("Ollama clip judge failed: %s", exc)
        return None


def _parse_judge_result(content: str, *, provider: str, llm_used: bool) -> JudgeResult | None:
    try:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end < start:
            return None
        data = json.loads(content[start:end + 1])
    except Exception:
        return None

    try:
        axes = {name: _clamp01(float(data[name])) for name in _AXIS_NAMES}
        reason = str(data.get("judge_reason") or "Strongest short-form candidate in this batch.").strip()
        confidence = _clamp01(float(data.get("judge_confidence", 0.65)))
    except Exception:
        return None

    judge_score = _axes_to_score(axes)
    return JudgeResult(
        judge_score=judge_score,
        scroll_stop_power=axes["scroll_stop_power"],
        standalone_clarity=axes["standalone_clarity"],
        payoff_strength=axes["payoff_strength"],
        narrative_completeness=axes["narrative_completeness"],
        platform_fit_shortform=axes["platform_fit_shortform"],
        judge_reason=reason[:220],
        judge_confidence=confidence,
        provider=provider,
        llm_used=llm_used,
    )


def _heuristic_judge(ranked: RankedCandidate, window: list[RichSegment]) -> JudgeResult:
    features = ranked.features
    if features is None:
        return JudgeResult(
            judge_score=0.0,
            scroll_stop_power=0.0,
            standalone_clarity=0.0,
            payoff_strength=0.0,
            narrative_completeness=0.0,
            platform_fit_shortform=0.0,
            judge_reason="No candidate features available for clip judging.",
            judge_confidence=0.2,
        )

    duration = max(ranked.candidate.end - ranked.candidate.start, 0.0)
    duration_fit = 1.0 - min(abs(duration - 24.0) / 24.0, 1.0)
    chapter_coherence = _clamp01(features.chapter_coherence_score)
    topic_shift = _clamp01(features.topic_shift_strength)
    chapter_density = _clamp01(features.chapter_local_density)
    trend_hook = _clamp01(features.trend_hook_score)
    trend_prior = _clamp01(features.trend_prior_score)
    overlap_risk = _clamp01(features.clip_overlap_count / 3.0)
    boundary_closeness = _clamp01(1.0 - min(features.chapter_boundary_distance / 18.0, 1.0))
    scroll_stop = _clamp01(
        0.40 * features.hook_strength
        + 0.20 * min(features.max_hype / 8.0, 1.0)
        + 0.15 * features.chat_peak
        + 0.10 * max(features.motion_peak, features.face_reaction_peak)
        + 0.10 * (1.0 if features.question_open else 0.0)
        + 0.05 * (topic_shift * boundary_closeness)
        + 0.05 * trend_hook
        - 0.18 * features.slow_open_penalty
    )
    standalone = _clamp01(
        0.38 * features.standalone_score
        + 0.24 * features.context_efficiency
        + 0.15 * (1.0 - min(features.context_cost, 1.0))
        + 0.10 * min(features.quoteability_score + features.contradiction_score, 1.0)
        + 0.13 * chapter_coherence
        - 0.22 * features.context_dependency_score
    )
    payoff = _clamp01(
        0.55 * min(features.payoff_score / 8.0, 1.0)
        + 0.15 * features.payoff_clarity_score
        + 0.15 * min(features.post_hook_hype / 6.0, 1.0)
        + 0.15 * min(features.laughter_count / 2.0, 1.0)
        + 0.15 * max(features.chat_peak, features.face_reaction_peak)
        - 0.20 * features.weak_resolution_penalty
    )
    narrative = _clamp01(
        0.25 * features.context_efficiency
        + 0.25 * min(features.arc_score / 5.0, 1.0)
        + 0.18 * features.standalone_score
        + 0.15 * duration_fit
        + 0.07 * min(features.speaker_turns / 3.0, 1.0)
        + 0.10 * chapter_coherence
        + 0.07 * (1.0 - overlap_risk)
        + 0.03 * topic_shift
        - 0.10 * features.context_dependency_score
        - 0.08 * features.weak_resolution_penalty
    )
    platform_fit = _clamp01(
        0.23 * scroll_stop
        + 0.20 * standalone
        + 0.19 * payoff
        + 0.15 * duration_fit
        + 0.15 * features.profile_fit_score
        + 0.05 * chapter_density
        + 0.03 * chapter_coherence
        + 0.05 * trend_prior
    )

    axes = {
        "scroll_stop_power": scroll_stop,
        "standalone_clarity": standalone,
        "payoff_strength": payoff,
        "narrative_completeness": narrative,
        "platform_fit_shortform": platform_fit,
    }
    judge_score = _axes_to_score(axes)
    strongest = max(axes, key=axes.get)
    weakest = min(axes, key=axes.get)
    return JudgeResult(
        judge_score=judge_score,
        scroll_stop_power=scroll_stop,
        standalone_clarity=standalone,
        payoff_strength=payoff,
        narrative_completeness=narrative,
        platform_fit_shortform=platform_fit,
        judge_reason=_heuristic_reason(strongest, weakest, features),
        judge_confidence=0.55 if window else 0.2,
        provider="heuristic",
        llm_used=False,
    )


def _feature_summary(ranked: RankedCandidate) -> str:
    features = ranked.features
    if features is None:
        return "No feature summary available."
    return (
        f"hook_strength={features.hook_strength:.3f}, "
        f"payoff_score={features.payoff_score:.3f}, "
        f"context_efficiency={features.context_efficiency:.3f}, "
        f"standalone_score={features.standalone_score:.3f}, "
        f"payoff_clarity_score={features.payoff_clarity_score:.3f}, "
        f"slow_open_penalty={features.slow_open_penalty:.3f}, "
        f"context_dependency_score={features.context_dependency_score:.3f}, "
        f"weak_resolution_penalty={features.weak_resolution_penalty:.3f}, "
        f"chat_peak={features.chat_peak:.3f}, "
        f"face_reaction_peak={features.face_reaction_peak:.3f}, "
        f"motion_peak={features.motion_peak:.3f}, "
        f"arc_score={features.arc_score:.3f}, "
        f"profile_fit_score={features.profile_fit_score:.3f}, "
        f"chapter_position={features.chapter_position:.3f}, "
        f"topic_shift_strength={features.topic_shift_strength:.3f}, "
        f"chapter_local_density={features.chapter_local_density:.3f}, "
        f"chapter_boundary_distance={features.chapter_boundary_distance:.3f}, "
        f"chapter_coherence_score={features.chapter_coherence_score:.3f}, "
        f"clip_overlap_count={features.clip_overlap_count:.3f}, "
        f"trend_hook_score={features.trend_hook_score:.3f}, "
        f"trend_topic_score={features.trend_topic_score:.3f}, "
        f"trend_prior_score={features.trend_prior_score:.3f}"
    )


def _multimodal_prompt(prompt: str, keyframes: list[SampledKeyframe]) -> str:
    frame_lines = [
        f"- {frame.label} at {frame.time_sec:.2f}s"
        for frame in keyframes
    ]
    return (
        f"{prompt}\n\n"
        "You are also given keyframes from the candidate clip window.\n"
        "Use them to judge visible facial reactions, scene energy, framing, and whether the clip looks visually compelling.\n"
        "Keyframes:\n"
        f"{chr(10).join(frame_lines)}"
    )


def _build_context_summary(segments: list[RichSegment], candidate_start: float) -> str:
    if not segments:
        return "No prior context available."
    lookback_start = max(0.0, candidate_start - float(settings.judge_context_lookback_sec))
    prior = [
        seg.text.strip()
        for seg in segments
        if seg.end <= candidate_start and seg.start >= lookback_start and seg.text.strip()
    ]
    if not prior:
        return "Limited prior context available."
    return " | ".join(prior[-4:])


def _data_url_for_frame(path: str, mime_type: str) -> str:
    encoded = _base64_for_frame(path)
    return f"data:{mime_type};base64,{encoded}"


def _base64_for_frame(path: str) -> str:
    data = pathlib.Path(path).read_bytes()
    return base64.b64encode(data).decode("ascii")


def _axes_to_score(axes: dict[str, float]) -> float:
    weighted = sum(_JUDGE_WEIGHTS[name] * axes[name] for name in _AXIS_NAMES)
    return round(weighted * 10.0, 3)


def _blend_scores(existing_score: float, judge_score: float, judge_confidence: float) -> float:
    blend = min(max(float(settings.judge_blend_weight), 0.0), 1.0)
    confidence_scaled = blend * (0.55 + 0.45 * _clamp01(judge_confidence))
    return round(existing_score * (1.0 - confidence_scaled) + judge_score * confidence_scaled, 3)


def _heuristic_reason(strongest: str, weakest: str, features) -> str:
    notes: list[str] = [f"Strongest axis: {strongest}", f"main risk: {weakest}"]
    if features.chapter_coherence_score >= 0.65:
        notes.append("clean chapter coherence")
    elif features.chapter_coherence_score <= 0.35:
        notes.append("weak topic coherence")
    if features.slow_open_penalty >= 0.45:
        notes.append("slow opening risk")
    if features.context_dependency_score >= 0.5:
        notes.append("needs prior stream context")
    if features.payoff_clarity_score >= 0.65:
        notes.append("clear ending payoff")
    if features.weak_resolution_penalty >= 0.5:
        notes.append("resolution feels weak")
    if features.topic_shift_strength >= 0.5:
        notes.append("benefits from a topic pivot")
    if features.clip_overlap_count >= 2.0:
        notes.append("overlaps nearby candidates")
    if features.trend_prior_score >= 0.55:
        notes.append("matches proven short-form patterns")
    return "; ".join(notes) + "."


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
