from __future__ import annotations

import base64
import json
import logging
import pathlib

import ollama

from config import settings
from services.keyframe_sampler import SampledKeyframe
from services.timeline_types import JudgeResult, RankedCandidate
from services.transcriber import RichSegment, format_transcript

logger = logging.getLogger(__name__)

_AXIS_NAMES = (
    "scroll_stop_power",
    "standalone_clarity",
    "payoff_strength",
    "narrative_completeness",
    "platform_fit_shortform",
)
_JUDGE_WEIGHTS = {
    "scroll_stop_power": 0.28,
    "standalone_clarity": 0.22,
    "payoff_strength": 0.24,
    "narrative_completeness": 0.16,
    "platform_fit_shortform": 0.10,
}
_JUDGE_PROMPT = """You are judging whether a candidate clip would work as a high-performing short-form video.

Score each axis from 0.0 to 1.0:
- scroll_stop_power: does the opening stop the scroll quickly?
- standalone_clarity: can a new viewer understand it without prior stream context?
- payoff_strength: is there a satisfying reaction, reveal, punchline, or resolution?
- narrative_completeness: does it feel like a coherent mini-story rather than a fragment?
- platform_fit_shortform: is the pacing and shape right for Shorts/Reels/TikTok?

Be strict. Penalize clips with slow setup, missing payoff, weak standalone clarity, clips that start mid-thought,
clips that depend on prior stream context, and clips that fizzle without a clean ending.
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


def rerank_candidates(
    ranked_candidates: list[RankedCandidate],
    segments: list[RichSegment],
    profile: str = "general",
    shortlist_top_k: int | None = None,
    provider: str | None = None,
    keyframes_by_candidate: dict[str, list[SampledKeyframe]] | None = None,
) -> list[RankedCandidate]:
    if not ranked_candidates:
        return []

    limit = len(ranked_candidates) if shortlist_top_k is None else min(max(shortlist_top_k, 0), len(ranked_candidates))
    if limit <= 0:
        return ranked_candidates

    reranked: list[RankedCandidate] = list(ranked_candidates)
    for idx, ranked in enumerate(reranked[:limit]):
        judge_result = judge_candidate(
            ranked,
            segments,
            profile=profile,
            provider=provider,
            keyframes=(keyframes_by_candidate or {}).get(ranked.candidate.id),
        )
        blended_score = _blend_scores(ranked.rank_score, judge_result.judge_score, judge_result.judge_confidence)
        rationale = list(ranked.rationale)
        rationale.append("judge_reranked")
        if judge_result.llm_used:
            rationale.append("judge_llm")
        reranked[idx] = RankedCandidate(
            candidate=ranked.candidate,
            features=ranked.features,
            rank_score=blended_score,
            rationale=rationale,
            base_rank_score=ranked.base_rank_score if ranked.base_rank_score is not None else ranked.rank_score,
            judge_result=judge_result,
        )

    reranked.sort(key=lambda item: item.rank_score, reverse=True)
    return reranked


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
    return heuristic


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

    preferred = (provider or "auto").strip().lower()
    if preferred == "gemini" and keyframes and settings.enable_multimodal_clip_judge:
        return _call_gemini_multimodal(prompt, keyframes)
    if preferred == "gemini":
        return _call_gemini(prompt)
    if preferred == "groq":
        return _call_groq(prompt)
    if preferred == "ollama":
        return _call_ollama(prompt)

    if settings.gemini_api_key and keyframes and settings.enable_multimodal_clip_judge:
        result = _call_gemini_multimodal(prompt, keyframes)
        if result is not None:
            return result
    if settings.gemini_api_key:
        result = _call_gemini(prompt)
        if result is not None:
            return result
    if settings.groq_api_key:
        result = _call_groq(prompt)
        if result is not None:
            return result
    return _call_ollama(prompt)


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
        content = (response.choices[0].message.content or "").strip()
        return _parse_judge_result(content, provider="gemini", llm_used=True)
    except Exception as exc:
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
        content = (response.choices[0].message.content or "").strip()
        return _parse_judge_result(content, provider="gemini_multimodal", llm_used=True)
    except Exception as exc:
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
        content = (response.choices[0].message.content or "").strip()
        return _parse_judge_result(content, provider="groq", llm_used=True)
    except Exception as exc:
        logger.debug("Groq clip judge failed: %s", exc)
        return None


def _call_ollama(prompt: str) -> JudgeResult | None:
    try:
        client = ollama.Client(host=settings.ollama_host)
        response = client.chat(
            model=settings.ollama_model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response["message"]["content"].strip()
        return _parse_judge_result(content, provider="ollama", llm_used=True)
    except Exception as exc:
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
    data = pathlib.Path(path).read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


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
