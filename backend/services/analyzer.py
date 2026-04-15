import json
import logging
import re

import ollama

from config import settings
from services import boundary_refiner, clip_strategy

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

PAIR_SYSTEM_PROMPT = """You are selecting the best precomputed clip boundary pair for a short-form highlight.
Choose the pair that:
- starts cleanly and does not feel mid-thought
- keeps enough setup for the moment to make sense
- includes the payoff, reaction, punchline, or reveal
- ends at a natural stopping point
- feels most standalone for TikTok, Shorts, or Reels

You MUST choose one of the provided pair ids.
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

PAIR_USER_TEMPLATE = """Prior context (what happened before this clip window):
{context_summary}

Candidate metadata:
- source: {source}
- profile: {profile}
- anchor_time: {anchor_time}
- heuristic_rank_score: {rank_score}
- rationale: {rationale}

Transcript excerpt:
{transcript}

Precomputed boundary pairs:
{pair_summaries}

Return JSON matching exactly:
{{
  "selected_pair_id": "<pair id>",
  "viral_score": <float 1.0-10.0>,
  "reason": "<one sentence>"
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
_MAX_SELECTION_REASONS = 4

_RATIONALE_REASON_TEXT = {
    "strong_hook": "Der Einstieg hat einen starken Hook und zieht sofort Aufmerksamkeit.",
    "clear_payoff": "Der Clip enthält einen klaren Payoff statt nur Setup.",
    "chat_confirmed": "Live-Signale aus dem Chat bestätigen, dass der Moment stark angekommen ist.",
    "visual_reaction": "Sichtbare Reaktionen im Bild verstärken den Moment.",
    "question_open": "Der Einstieg öffnet mit einer Frage und baut direkt Spannung auf.",
    "quoteable": "Der Ausschnitt enthält eine zitierbare Aussage, die auch alleine funktioniert.",
    "surprising": "Der Moment wirkt überraschend und hat dadurch Stop-Scroll-Potenzial.",
    "standalone": "Der Ausschnitt funktioniert weitgehend ohne vorherigen Stream-Kontext.",
    "clear_resolution": "Der Moment hat eine klare Auflösung statt offen auszulaufen.",
    "chapter_coherent": "Der Clip bleibt in einem thematisch zusammenhängenden Beat.",
    "topic_shift_hook": "Ein Themenwechsel liefert einen natürlichen Einstiegspunkt.",
    "chapter_dense": "In diesem Abschnitt liegen mehrere starke Signale dicht beieinander.",
    "trend_aligned": "Der Moment passt gut zum erwarteten Short-Form-Format.",
    "trend_feedback_aligned": "Ähnliche Momente haben im jüngeren Feedback bereits gut funktioniert.",
    "goal_aligned": "Der Clip passt besonders gut zur angefragten Suchabsicht oder zum gewünschten Thema.",
    "projected_strong_open": "Die Variante ist auf einen starken Einstieg in den ersten Sekunden optimiert.",
    "projected_retention": "Die Variante ist auf bessere Retention und klareren Payoff optimiert.",
    "semantic": "Der Inhalt ist semantisch markant und wurde deshalb als Highlight erkannt.",
    "peak_first": "Der Moment lag auf einem starken Peak der Hype-Timeline.",
    "fallback": "Der Clip blieb als beste verfügbare Option im Fallback-Pfad übrig.",
}

_BOUNDARY_START_REASON_TEXT = {
    "clean sentence start": "Der Clip startet auf einem sauberen Satzanfang.",
    "scene-cut opening": "Der Clip startet direkt auf einem visuellen Wechsel.",
    "speaker-turn opening": "Der Clip startet auf einem klaren Sprecherwechsel.",
    "pause-reset opening": "Der Clip startet nach einer klaren Pause statt mitten im Gedanken.",
}

_BOUNDARY_END_REASON_TEXT = {
    "complete payoff": "Der vollständige Payoff oder Reaktionsmoment bleibt im Clip.",
    "sentence closure": "Der Clip endet auf einem sauberen Satzabschluss.",
    "natural pause ending": "Der Clip endet auf einer natürlichen Pause.",
}

_JUDGE_AXIS_LABELS = {
    "scroll_stop_power": "starken Einstieg",
    "standalone_clarity": "gute Verständlichkeit ohne Vorwissen",
    "payoff_strength": "klaren Payoff",
    "narrative_completeness": "runde Mini-Story",
    "platform_fit_shortform": "gutes Short-Form-Pacing",
}


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
    strategy: dict | None = None,
) -> list[Moment]:
    moments: list[Moment] = []
    normalized_strategy = clip_strategy.normalize_strategy(strategy)
    for index, ranked in enumerate(ranked_candidates[:limit], start=1):
        candidate_segments = _segments_in_candidate_window(segments, ranked.candidate)
        if not candidate_segments:
            continue

        if _should_use_boundary_v2(segments):
            moment = _refine_candidate_v2(ranked, segments, candidate_segments, profile)
        elif settings.enable_legacy_boundary_refinement_llm:
            moment = _refine_candidate_v1(ranked, segments, candidate_segments, profile)
        else:
            moment = None

        if moment is None:
            boundaries = _allowed_boundaries(candidate_segments)
            moment = _fallback_moment(ranked, boundaries, profile)
        moment = _enforce_story_shape(moment, ranked.candidate, candidate_segments)
        moment["selection_reasons"] = _selection_reasons(ranked, moment)
        moment["rank"] = index
        moment["ranking_mode"] = "ml_blended" if "ml_ranked" in ranked.rationale else "heuristic"
        moment["candidate_id"] = ranked.candidate.id
        moment["platform"] = normalized_strategy["platform"]
        moment["variant_key"] = normalized_strategy["clip_style"]
        moment["variant_label"] = clip_strategy.variant_label(normalized_strategy, normalized_strategy["clip_style"])
        if ranked.features is not None:
            moment["projected_hook_score"] = round(float(ranked.features.projected_hook_score), 3)
            moment["projected_retention_score"] = round(float(ranked.features.projected_retention_score), 3)
        moments.append(moment)

    return _dedup_moments(moments)


def expand_variants(
    moments: list[Moment],
    ranked_candidates: list[RankedCandidate],
    segments: list[RichSegment] | list[Segment],
    strategy: dict | None = None,
) -> list[Moment]:
    normalized_strategy = clip_strategy.normalize_strategy(strategy)
    specs = clip_strategy.build_variant_specs(normalized_strategy)
    if len(specs) <= 1 or not moments:
        return moments

    ranked_by_id = {ranked.candidate.id: ranked for ranked in ranked_candidates}
    expanded: list[Moment] = []
    for moment in moments:
        ranked = ranked_by_id.get(str(moment.get("candidate_id", "")).split("::", 1)[0])
        if ranked is None:
            expanded.append(moment)
            continue
        candidate_segments = _segments_in_candidate_window(segments, ranked.candidate)
        boundaries = _allowed_boundaries(candidate_segments) if candidate_segments else []
        for spec in specs:
            variant = dict(moment)
            variant["variant_key"] = spec["key"]
            variant["variant_label"] = clip_strategy.variant_label(normalized_strategy, spec["key"])
            variant["candidate_id"] = f"{ranked.candidate.id}::{spec['key']}"
            if boundaries:
                variant = _apply_variant_spec(variant, ranked.candidate, boundaries, spec)
            if ranked.features is not None:
                hook_base = float(ranked.features.projected_hook_score or ranked.features.hook_strength)
                retention_base = float(ranked.features.projected_retention_score or ranked.features.context_efficiency)
                variant["projected_hook_score"] = round(
                    max(0.0, min(hook_base + _variant_hook_delta(spec["key"]), 1.0)),
                    3,
                )
                variant["projected_retention_score"] = round(
                    max(0.0, min(retention_base + _variant_retention_delta(spec["key"]), 1.0)),
                    3,
                )
            expanded.append(variant)

    expanded.sort(
        key=lambda item: (
            float(item.get("projected_retention_score", item.get("viral_score", 0.0))),
            float(item.get("projected_hook_score", item.get("viral_score", 0.0))),
            float(item.get("viral_score", 0.0)),
        ),
        reverse=True,
    )
    for index, moment in enumerate(expanded, start=1):
        moment["rank"] = index
    return expanded


def _should_use_boundary_v2(segments: list[RichSegment] | list[Segment]) -> bool:
    return bool(settings.enable_boundary_refinement_v2 and all(isinstance(seg, RichSegment) for seg in segments))


def _refine_candidate_v1(
    ranked: RankedCandidate,
    segments: list[RichSegment] | list[Segment],
    candidate_segments: list[RichSegment] | list[Segment],
    profile: str,
) -> Moment | None:
    logger.debug("Using legacy raw-boundary LLM refinement path for candidate=%s", ranked.candidate.id)
    context_summary = _build_context_summary(segments, ranked.candidate)
    transcript_text = format_transcript(candidate_segments, use_seconds=True)
    boundaries = _allowed_boundaries(candidate_segments)
    return _call_ollama(ranked, transcript_text, boundaries, candidate_segments, profile, context_summary)


def _refine_candidate_v2(
    ranked: RankedCandidate,
    segments: list[RichSegment] | list[Segment],
    candidate_segments: list[RichSegment] | list[Segment],
    profile: str,
) -> Moment | None:
    rich_segments = [seg for seg in segments if isinstance(seg, RichSegment)]
    if not rich_segments:
        return None

    selected_pair, pairs = boundary_refiner.refine_candidate_boundaries(
        ranked,
        rich_segments,
        profile=profile,
    )
    if selected_pair is None:
        return None

    context_summary = _build_context_summary(segments, ranked.candidate)
    transcript_text = format_transcript(candidate_segments, use_seconds=True)
    if settings.enable_boundary_llm_pair_rerank:
        logger.debug("Using top-pair LLM rerank path for candidate=%s", ranked.candidate.id)
        llm_moment = _call_pair_rerank(
            ranked,
            transcript_text,
            context_summary,
            pairs,
            candidate_segments,
            profile,
        )
        if llm_moment is not None:
            return llm_moment

    logger.debug("Using heuristic top-pair fallback for candidate=%s", ranked.candidate.id)
    return {
        "rank": 1,
        "candidate_id": ranked.candidate.id,
        "start_time": selected_pair.start_time,
        "end_time": selected_pair.end_time,
        "viral_score": max(1.0, min(10.0, round(selected_pair.pair_score * 10.0, 2))),
        "reason": selected_pair.reason or "Selected from top-scoring boundary pair.",
        "selection_boundary_reason": selected_pair.reason,
    }


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


def _call_pair_rerank(
    ranked: RankedCandidate,
    transcript_text: str,
    context_summary: str,
    pairs,
    segments: list[RichSegment] | list[Segment],
    profile: str,
) -> Moment | None:
    user_msg = PAIR_USER_TEMPLATE.format(
        context_summary=context_summary,
        source=ranked.candidate.source,
        profile=profile,
        anchor_time=f"{ranked.candidate.anchor_time:.2f}",
        rank_score=f"{ranked.rank_score:.2f}",
        rationale=", ".join(ranked.rationale) or "none",
        transcript=transcript_text,
        pair_summaries=_pair_summaries(pairs, segments),
    )
    content: str | None = None
    if settings.gemini_api_key:
        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=settings.gemini_api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            response = client.chat.completions.create(
                model=settings.gemini_model,
                messages=[
                    {"role": "system", "content": PAIR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=180,
            )
            content = (response.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.debug("Gemini boundary pair rerank failed: %s", exc)

    if not content:
        try:
            client = ollama.Client(host=settings.ollama_host)
            response = client.chat(
                model=settings.ollama_model,
                messages=[
                    {"role": "system", "content": PAIR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            )
            content = response["message"]["content"].strip()
        except Exception as exc:
            logger.debug("Ollama boundary pair rerank failed: %s", exc)
            return None

    selected_pair = _parse_pair_selection(content, pairs)
    if selected_pair is None:
        logger.debug("Pair rerank response invalid; falling back to top-scoring pair for candidate=%s", ranked.candidate.id)
        return None
    return {
        "rank": 1,
        "candidate_id": ranked.candidate.id,
        "start_time": selected_pair["start_time"],
        "end_time": selected_pair["end_time"],
        "viral_score": selected_pair["viral_score"],
        "reason": selected_pair["reason"],
        "selection_boundary_reason": selected_pair.get("selection_boundary_reason") or selected_pair["reason"],
    }


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


def _parse_pair_selection(content: str, pairs) -> dict | None:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", content)
        if not match:
            return None
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return None

    selected_pair_id = str(data.get("selected_pair_id") or "").strip()
    if not selected_pair_id:
        return None

    pair_map = {pair.id: pair for pair in pairs}
    selected = pair_map.get(selected_pair_id)
    if selected is None:
        return None

    try:
        viral_score = max(1.0, min(10.0, float(data.get("viral_score", selected.pair_score * 10.0))))
    except (TypeError, ValueError):
        viral_score = max(1.0, min(10.0, selected.pair_score * 10.0))

    reason = str(data.get("reason") or selected.reason or "Selected from top-scoring boundary pair.")
    return {
        "start_time": selected.start_time,
        "end_time": selected.end_time,
        "viral_score": round(viral_score, 2),
        "reason": reason,
        "selection_boundary_reason": selected.reason,
    }


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


def _pair_summaries(pairs, segments: list[RichSegment] | list[Segment]) -> str:
    lines: list[str] = []
    for pair in pairs[:3]:
        window = [
            seg for seg in segments
            if _seg_end(seg) > pair.start_time and _seg_start(seg) < pair.end_time
        ]
        opening_text = _seg_text(window[0]) if window else ""
        closing_text = _seg_text(window[-1]) if window else ""
        lines.append(
            (
                f"- {pair.id}: {pair.start_time:.2f}-{pair.end_time:.2f} "
                f"(duration={pair.duration:.2f}s score={pair.pair_score:.2f}) "
                f"hook={pair.hook_score:.2f} payoff={pair.payoff_score:.2f} "
                f"narrative={pair.narrative_score:.2f} reason={pair.reason} "
                f"opening=\"{opening_text[:90]}\" closing=\"{closing_text[:90]}\""
            )
        )
    return "\n".join(lines)


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


def _selection_reasons(ranked: RankedCandidate, moment: Moment) -> list[str]:
    reasons: list[str] = []
    for reason in _feature_selection_reasons(ranked)[:2]:
        _append_unique_reason(reasons, reason)

    judge_reason = _judge_selection_reason(ranked)
    if judge_reason:
        _append_unique_reason(reasons, judge_reason)

    boundary_reason = _boundary_selection_reason(moment.get("selection_boundary_reason"))
    if boundary_reason:
        _append_unique_reason(reasons, boundary_reason)

    for reason in _rationale_selection_reasons(ranked.rationale):
        _append_unique_reason(reasons, reason)
        if len(reasons) >= _MAX_SELECTION_REASONS:
            break

    if not reasons:
        fallback_reason = str(moment.get("reason") or "").strip()
        if fallback_reason:
            _append_unique_reason(reasons, fallback_reason)

    return reasons[:_MAX_SELECTION_REASONS]


def _feature_selection_reasons(ranked: RankedCandidate) -> list[str]:
    features = ranked.features
    if features is None:
        return []

    reasons: list[str] = []
    if features.hook_strength >= 0.65:
        reasons.append("Der Einstieg hat einen starken Hook in den ersten Sekunden.")
    if features.payoff_score >= 4.0 or features.payoff_clarity_score >= 0.65:
        reasons.append("Der Clip enthält einen klaren Payoff oder eine eindeutige Reaktion.")
    if (
        features.standalone_score >= 0.6
        or (features.context_efficiency >= 0.65 and features.context_dependency_score < 0.45)
    ):
        reasons.append("Der Ausschnitt ist weitgehend ohne zusätzlichen Kontext verständlich.")
    if max(features.chat_peak, features.face_reaction_peak, features.motion_peak) >= 0.45:
        reasons.append("Zusätzliche Chat- oder Reaktionssignale bestätigen den Moment.")
    if features.topic_shift_strength >= 0.45:
        reasons.append("Ein Themenwechsel macht den Einstieg natürlich und sofort verständlich.")
    return reasons


def _rationale_selection_reasons(rationale: list[str]) -> list[str]:
    reasons: list[str] = []
    for tag in rationale:
        if tag in {"judge_reranked", "judge_llm", "ml_ranked", "overlap_risk", "slow_open_risk", "context_dependent", "weak_resolution_risk"}:
            continue
        if tag.startswith("profile_fit:"):
            profile_name = tag.split(":", 1)[1].strip() or "aktuellen"
            reasons.append(f"Der Moment passt gut zum Profil {profile_name}.")
            continue
        text = _RATIONALE_REASON_TEXT.get(tag)
        if text:
            reasons.append(text)
            continue
        if tag:
            reasons.append(f"Zusätzliches Ranking-Signal: {tag.replace('_', ' ')}.")
    return reasons


def _judge_selection_reason(ranked: RankedCandidate) -> str | None:
    judge = ranked.judge_result
    if judge is None:
        return None

    axis_scores = {
        "scroll_stop_power": judge.scroll_stop_power,
        "standalone_clarity": judge.standalone_clarity,
        "payoff_strength": judge.payoff_strength,
        "narrative_completeness": judge.narrative_completeness,
        "platform_fit_shortform": judge.platform_fit_shortform,
    }
    strongest = [
        _JUDGE_AXIS_LABELS[name]
        for name, score in sorted(axis_scores.items(), key=lambda item: item[1], reverse=True)
        if score >= 0.62
    ][:2]
    if strongest:
        if len(strongest) == 1:
            return f"Der Judge-Check bestätigt vor allem {strongest[0]}."
        return f"Der Judge-Check bestätigt vor allem {strongest[0]} und {strongest[1]}."

    reason = judge.judge_reason.strip()
    return f"Der Judge-Check hat den Clip zusätzlich bestätigt: {reason}" if reason else None


def _boundary_selection_reason(raw_reason: str | None) -> str | None:
    if not raw_reason:
        return None

    parts = [part.strip() for part in raw_reason.split(";") if part.strip()]
    start_part = next((part for part in parts if part in _BOUNDARY_START_REASON_TEXT), None)
    end_part = next((part for part in parts if part in _BOUNDARY_END_REASON_TEXT), None)

    lines: list[str] = []
    if start_part:
        lines.append(_BOUNDARY_START_REASON_TEXT[start_part])
    if end_part:
        lines.append(_BOUNDARY_END_REASON_TEXT[end_part])

    if not lines:
        cleaned = raw_reason.replace("penalties:", "Hinweise:")
        return f"Die finalen Schnittgrenzen wurden so gewählt: {cleaned}."
    return " ".join(lines)


def _append_unique_reason(reasons: list[str], reason: str) -> None:
    cleaned = reason.strip()
    if not cleaned:
        return
    normalized = cleaned.casefold()
    if any(existing.casefold() == normalized for existing in reasons):
        return
    reasons.append(cleaned)


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


def _apply_variant_spec(
    moment: Moment,
    candidate: Candidate,
    boundaries: list[float],
    spec: dict,
) -> Moment:
    adjusted = dict(moment)
    if not boundaries:
        return adjusted

    start_bias = float(spec.get("start_bias_sec", 0.0))
    end_bias = float(spec.get("end_bias_sec", 0.0))
    context_multiplier = float(spec.get("context_multiplier", 1.0))
    start_target = moment["start_time"] + start_bias
    end_target = moment["end_time"] + end_bias

    if context_multiplier > 1.0:
        start_target = min(start_target, candidate.anchor_time - _MIN_CONTEXT_BEFORE_ANCHOR * context_multiplier)
        end_target = max(end_target, candidate.anchor_time + _MIN_PAYOFF_AFTER_ANCHOR)
    elif context_multiplier < 1.0:
        start_target = max(start_target, candidate.anchor_time - _MIN_CONTEXT_BEFORE_ANCHOR * context_multiplier)

    adjusted["start_time"] = _boundary_at_or_before(boundaries, start_target, floor=boundaries[0])
    adjusted["end_time"] = _boundary_at_or_after(boundaries, end_target, ceiling=boundaries[-1])
    if adjusted["end_time"] <= adjusted["start_time"]:
        adjusted["start_time"] = boundaries[0]
        adjusted["end_time"] = boundaries[-1]
    return adjusted


def _variant_hook_delta(variant_key: str) -> float:
    return {
        "aggressive_hook": 0.08,
        "reaction_first": 0.06,
        "quote_first": 0.03,
        "context_balanced": -0.02,
    }.get(variant_key, 0.0)


def _variant_retention_delta(variant_key: str) -> float:
    return {
        "context_balanced": 0.07,
        "balanced": 0.03,
        "quote_first": 0.02,
        "aggressive_hook": -0.01,
    }.get(variant_key, 0.0)


def _seg_start(seg: RichSegment | Segment) -> float:
    return seg.start if isinstance(seg, RichSegment) else seg[0]


def _seg_end(seg: RichSegment | Segment) -> float:
    return seg.end if isinstance(seg, RichSegment) else seg[1]


def _seg_text(seg: RichSegment | Segment) -> str:
    return seg.text if isinstance(seg, RichSegment) else seg[2]


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
