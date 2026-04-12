import json
import logging
import re

import ollama

from config import settings

logger = logging.getLogger(__name__)
from services.timeline_types import Candidate, RankedCandidate
from services.transcriber import RichSegment, Segment, format_transcript

SYSTEM_PROMPT = """You refine short-form clip boundaries for stream highlights.
The content may be gaming, Just Chatting, IRL, podcast, interview, or any other format.
You are given:
- a narrative context summary of what happened before the clip window
- a candidate clip window already selected by upstream ranking
- the anchor time of the likely peak moment
- allowed boundary timestamps taken from the transcript

Your task:
- use the prior context to understand WHY this moment is significant (e.g. a contradiction, a payoff, a reveal, an insight, an emotional beat)
- choose the cleanest start and end boundary from the allowed timestamps
- preserve a strong hook in the first 3 seconds
- keep enough context for the moment to make sense standalone
- keep the payoff/reaction/punchline/insight inside the clip
- prefer concise clips over long clips when both are valid

Return raw JSON only."""

USER_TEMPLATE = """Prior context (what happened before this clip window):
{context_summary}

Candidate metadata:
- source: {source}
- profile: {profile}
- anchor_time: {anchor_time}
- heuristic_rank_score: {rank_score}
- rationale: {rationale}

Transcript excerpt:
{transcript}

Allowed boundaries:
{boundaries}

Return JSON matching exactly:
{{
  "clip": {{
    "start_time": <float>,
    "end_time": <float>,
    "viral_score": <float 1.0-10.0>,
    "reason": "<one sentence>"
  }}
}}"""

_CONTEXT_SUMMARY_PROMPT = """\
Summarize what happened in this stream transcript in 2-3 sentences.
Focus on: topic, mood, any setup or build-up relevant to what comes next.
Be factual and brief. Do not speculate about what happens after.

Transcript:
{text}

Summary:"""

# How many seconds before the candidate window to include as prior context
_CONTEXT_LOOKBACK_SEC = 120.0

Moment = dict
_MIN_STORY_DURATION = 8.0
_MIN_CONTEXT_BEFORE_ANCHOR = 1.5
_MIN_PAYOFF_AFTER_ANCHOR = 2.5


def analyze(segments: list[RichSegment] | list[Segment]) -> list[Moment]:
    if not segments:
        return []
    candidate = Candidate(
        id="fallback",
        start=_seg_start(segments[0]),
        end=_seg_end(segments[-1]),
        source="fallback",
        anchor_time=(_seg_start(segments[0]) + _seg_end(segments[-1])) / 2.0,
        score_hint=0.0,
        tags=[],
    )
    ranked = RankedCandidate(candidate=candidate, features=None, rank_score=0.0, rationale=["fallback"])
    return refine_candidates([ranked], segments, limit=1)


def refine_candidates(
    ranked_candidates: list[RankedCandidate],
    segments: list[RichSegment] | list[Segment],
    limit: int = 5,
    profile: str = "general",
) -> list[Moment]:
    moments: list[Moment] = []
    for index, ranked in enumerate(ranked_candidates[:limit], start=1):
        candidate_segments = _segments_in_candidate_window(segments, ranked.candidate)
        if not candidate_segments:
            continue

        context_summary = _build_context_summary(segments, ranked.candidate)
        transcript_text = format_transcript(candidate_segments, use_seconds=True)
        boundaries = _allowed_boundaries(candidate_segments)
        moment = _call_ollama(ranked, transcript_text, boundaries, candidate_segments, profile, context_summary)
        if moment is None:
            moment = _fallback_moment(ranked, boundaries, profile)
        moment = _enforce_story_shape(moment, ranked.candidate, candidate_segments)
        moment["rank"] = index
        moment["ranking_mode"] = "ml_blended" if "ml_ranked" in ranked.rationale else "heuristic"
        moment["candidate_id"] = ranked.candidate.id
        moments.append(moment)

    return _dedup_moments(moments)


def _build_context_summary(
    all_segments: list[RichSegment] | list[Segment],
    candidate: Candidate,
) -> str:
    """
    Build a 2-3 sentence narrative summary of what happened before the clip window.

    Strategy:
    - Gather up to _CONTEXT_LOOKBACK_SEC of transcript before candidate.start.
    - If Groq is available: ask it to summarize (fast, uses existing key).
    - Fallback: pick the top-3 highest-hype segments by text and join them as
      a bullet list — gives the LLM real signal without an extra API call.
    """
    lookback_start = candidate.start - _CONTEXT_LOOKBACK_SEC
    prior = [
        seg for seg in all_segments
        if _seg_end(seg) <= candidate.start and _seg_start(seg) >= lookback_start
    ]
    if not prior:
        return "No prior context available."

    prior_text = format_transcript(prior, use_seconds=True)

    # --- Groq summary (preferred) ---
    if settings.groq_api_key:
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=settings.groq_api_key,
                base_url="https://api.groq.com/openai/v1",
            )
            response = client.chat.completions.create(
                model=settings.groq_model,
                messages=[{
                    "role": "user",
                    "content": _CONTEXT_SUMMARY_PROMPT.format(text=prior_text),
                }],
                temperature=0.0,
                max_tokens=120,
            )
            summary = response.choices[0].message.content.strip()
            if summary:
                return summary
        except Exception as exc:
            logger.debug("Context summary via Groq failed: %s", exc)

    # --- Heuristic fallback: top-3 segments by hype_score ---
    rich = [s for s in prior if isinstance(s, RichSegment)]
    if rich:
        top = sorted(rich, key=lambda s: s.hype_score, reverse=True)[:3]
        top_sorted = sorted(top, key=lambda s: s.start)
        return " | ".join(s.text.strip() for s in top_sorted if s.text.strip())

    # Plain text fallback: last 3 segments
    last = prior[-3:]
    return " ".join(
        (seg.text if isinstance(seg, RichSegment) else seg[2]).strip()
        for seg in last
    )


def _segments_in_candidate_window(
    segments: list[RichSegment] | list[Segment],
    candidate: Candidate,
) -> list[RichSegment] | list[Segment]:
    return [seg for seg in segments if _seg_end(seg) > candidate.start and _seg_start(seg) < candidate.end]


def _call_ollama(
    ranked: RankedCandidate,
    transcript_text: str,
    boundaries: list[float],
    segments: list[RichSegment] | list[Segment],
    profile: str,
    context_summary: str = "No prior context available.",
) -> Moment | None:
    """Refine clip boundaries. Uses Gemini if configured, else falls back to Ollama."""
    if settings.gemini_api_key:
        return _call_gemini(ranked, transcript_text, boundaries, segments, profile, context_summary)
    return _call_ollama_refine(ranked, transcript_text, boundaries, segments, profile, context_summary)


def _call_gemini(
    ranked: RankedCandidate,
    transcript_text: str,
    boundaries: list[float],
    segments: list[RichSegment] | list[Segment],
    profile: str,
    context_summary: str = "No prior context available.",
) -> Moment | None:
    """Ask Gemini Flash to refine clip boundaries via its OpenAI-compatible endpoint."""
    user_msg = USER_TEMPLATE.format(
        context_summary=context_summary,
        source=ranked.candidate.source,
        profile=profile,
        anchor_time=f"{ranked.candidate.anchor_time:.2f}",
        rank_score=f"{ranked.rank_score:.2f}",
        rationale=", ".join(ranked.rationale) or "none",
        transcript=transcript_text,
        boundaries=json.dumps(boundaries),
    )
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=settings.gemini_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        response = client.chat.completions.create(
            model=settings.gemini_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        content = response.choices[0].message.content.strip()
        parsed = _parse_json(content, segments)
        return parsed[0] if parsed else None
    except Exception as exc:
        logger.warning(
            "Gemini call failed (model=%s): %s — falling back to Ollama",
            settings.gemini_model,
            exc,
        )
        return _call_ollama_refine(ranked, transcript_text, boundaries, segments, profile, context_summary)


def _call_ollama_refine(
    ranked: RankedCandidate,
    transcript_text: str,
    boundaries: list[float],
    segments: list[RichSegment] | list[Segment],
    profile: str,
    context_summary: str = "No prior context available.",
) -> Moment | None:
    """Ask Ollama to refine clip boundaries."""
    user_msg = USER_TEMPLATE.format(
        context_summary=context_summary,
        source=ranked.candidate.source,
        profile=profile,
        anchor_time=f"{ranked.candidate.anchor_time:.2f}",
        rank_score=f"{ranked.rank_score:.2f}",
        rationale=", ".join(ranked.rationale) or "none",
        transcript=transcript_text,
        boundaries=json.dumps(boundaries),
    )

    try:
        client = ollama.Client(host=settings.ollama_host)
        response = client.chat(
            model=settings.ollama_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
    except Exception as exc:
        logger.warning(
            "Ollama call failed (host=%s model=%s): %s — falling back to heuristic",
            settings.ollama_host,
            settings.ollama_model,
            exc,
        )
        return None

    content = response["message"]["content"].strip()
    parsed = _parse_json(content, segments)
    return parsed[0] if parsed else None


def _parse_json(
    content: str,
    segments: list[RichSegment] | list[Segment],
) -> list[Moment] | None:
    try:
        data = json.loads(content)
        return _validate(data, segments)
    except (json.JSONDecodeError, KeyError):
        pass

    match = re.search(r"\{[\s\S]*\}", content)
    if match:
        try:
            data = json.loads(match.group())
            return _validate(data, segments)
        except (json.JSONDecodeError, KeyError):
            pass

    return None


def _validate(
    data: dict,
    segments: list[RichSegment] | list[Segment],
) -> list[Moment] | None:
    if "clip" in data:
        clips = [data["clip"]]
    else:
        clips = data["clips"]

    boundaries = _allowed_boundaries(segments)
    min_time = boundaries[0]
    max_time = boundaries[-1]
    validated = []
    for clip in clips:
        try:
            start_time = _snap_time(float(clip["start_time"]), boundaries)
            end_time = _snap_time(float(clip["end_time"]), boundaries)
            viral_score = max(1.0, min(10.0, float(clip["viral_score"])))
        except (KeyError, TypeError, ValueError):
            continue

        if start_time < min_time or end_time > max_time or end_time <= start_time:
            continue

        validated.append(
            {
                "rank": int(clip.get("rank", len(validated) + 1)),
                "candidate_id": str(clip.get("candidate_id", "")),
                "start_time": start_time,
                "end_time": end_time,
                "viral_score": viral_score,
                "reason": str(clip["reason"]),
            }
        )
    if not validated:
        return None
    return sorted(validated, key=lambda c: c["viral_score"], reverse=True)


def _allowed_boundaries(segments: list[RichSegment] | list[Segment]) -> list[float]:
    boundaries: set[float] = set()
    for seg in segments:
        boundaries.add(round(_seg_start(seg), 2))
        boundaries.add(round(_seg_end(seg), 2))
    return sorted(boundaries)


def _snap_time(value: float, boundaries: list[float], tolerance: float = 3.0) -> float:
    nearest = min(boundaries, key=lambda boundary: abs(boundary - value))
    if abs(nearest - value) > tolerance:
        raise KeyError("timestamp outside allowed tolerance")
    return nearest


def _fallback_moment(ranked: RankedCandidate, boundaries: list[float], profile: str) -> Moment:
    start_time = _snap_time(ranked.candidate.start, boundaries, tolerance=10.0)
    end_time = _snap_time(ranked.candidate.end, boundaries, tolerance=10.0)
    if end_time <= start_time and len(boundaries) > 1:
        later = [boundary for boundary in boundaries if boundary > start_time]
        end_time = later[0] if later else boundaries[-1]
    return {
        "rank": 1,
        "candidate_id": ranked.candidate.id,
        "start_time": start_time,
        "end_time": end_time,
        "viral_score": max(1.0, min(10.0, ranked.rank_score)),
        "reason": f"Selected from {profile} {ranked.candidate.source} candidate with {', '.join(ranked.rationale) or 'strong signal'}.",
    }


def _enforce_story_shape(
    moment: Moment,
    candidate: Candidate,
    segments: list[RichSegment] | list[Segment],
) -> Moment:
    adjusted = dict(moment)
    boundaries = _allowed_boundaries(segments)
    if not boundaries:
        return adjusted

    earliest = _boundary_at_or_after(boundaries, candidate.start)
    latest = _boundary_at_or_before(boundaries, candidate.end)

    start_time = max(adjusted["start_time"], earliest)
    end_time = min(adjusted["end_time"], latest)

    target_start = candidate.anchor_time - _MIN_CONTEXT_BEFORE_ANCHOR
    target_end = candidate.anchor_time + _MIN_PAYOFF_AFTER_ANCHOR

    if start_time > target_start:
        start_time = _boundary_at_or_before(boundaries, target_start, floor=earliest)
    if end_time < target_end:
        end_time = _boundary_at_or_after(boundaries, target_end, ceiling=latest)

    duration = end_time - start_time
    if duration < _MIN_STORY_DURATION:
        desired_start = candidate.anchor_time - (_MIN_STORY_DURATION * 0.4)
        desired_end = candidate.anchor_time + (_MIN_STORY_DURATION * 0.6)
        start_time = _boundary_at_or_before(boundaries, min(start_time, desired_start), floor=earliest)
        end_time = _boundary_at_or_after(boundaries, max(end_time, desired_end), ceiling=latest)

    duration = end_time - start_time
    if duration < _MIN_STORY_DURATION:
        start_time = earliest
        end_time = latest

    if end_time <= start_time:
        end_time = latest
        start_time = earliest

    adjusted["start_time"] = start_time
    adjusted["end_time"] = end_time
    return adjusted


def _dedup_moments(moments: list[Moment], overlap_threshold: float = 0.6) -> list[Moment]:
    sorted_moments = sorted(moments, key=lambda m: m["viral_score"], reverse=True)
    kept: list[Moment] = []
    for moment in sorted_moments:
        moment_duration = moment["end_time"] - moment["start_time"]
        if moment_duration <= 0:
            continue
        duplicate = False
        for kept_moment in kept:
            overlap = min(moment["end_time"], kept_moment["end_time"]) - max(moment["start_time"], kept_moment["start_time"])
            if overlap > 0 and overlap / moment_duration > overlap_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(moment)

    for index, moment in enumerate(kept, 1):
        moment["rank"] = index
    return kept


def _seg_start(seg: RichSegment | Segment) -> float:
    return seg.start if isinstance(seg, RichSegment) else seg[0]


def _seg_end(seg: RichSegment | Segment) -> float:
    return seg.end if isinstance(seg, RichSegment) else seg[1]


def _boundary_at_or_before(
    boundaries: list[float],
    target: float,
    floor: float | None = None,
) -> float:
    candidates = [
        boundary for boundary in boundaries
        if boundary <= target and (floor is None or boundary >= floor)
    ]
    if candidates:
        return candidates[-1]
    if floor is not None:
        fallback = [boundary for boundary in boundaries if boundary >= floor]
        if fallback:
            return fallback[0]
    return boundaries[0]


def _boundary_at_or_after(
    boundaries: list[float],
    target: float,
    ceiling: float | None = None,
) -> float:
    candidates = [
        boundary for boundary in boundaries
        if boundary >= target and (ceiling is None or boundary <= ceiling)
    ]
    if candidates:
        return candidates[0]
    if ceiling is not None:
        fallback = [boundary for boundary in boundaries if boundary <= ceiling]
        if fallback:
            return fallback[-1]
    return boundaries[-1]
