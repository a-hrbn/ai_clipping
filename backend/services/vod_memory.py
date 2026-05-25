from __future__ import annotations

import re

from services.timeline_types import Candidate, Chapter, MemorySpan
from services.transcriber import RichSegment

_TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")
_TOPIC_SHIFT_TOKENS = {
    "anyway", "aber", "anyways", "bro", "listen", "okay", "ok", "alright", "wait", "now", "next",
}
_SETUP_TOKENS = {
    "if", "when", "bet", "watch", "wait", "remember", "listen", "truth", "guess", "challenge",
    "prediction", "deal", "promise", "question", "warum", "wieso", "wenn", "warte",
}
_PAYOFF_TOKENS = {
    "there", "won", "lost", "worked", "happened", "look", "crazy", "insane", "no", "way", "finally",
    "boom", "clip", "laugh", "holy", "bro", "genau", "krass", "fertig",
}
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "so", "to", "of", "in", "on", "it", "is", "this", "that",
    "he", "she", "they", "we", "you", "i", "ich", "du", "er", "sie", "es", "und", "oder", "aber",
    "das", "der", "die", "ein", "eine", "ist", "war", "mit", "for", "at", "be", "are", "was",
}
_MAX_LOOKBACK_SEC = 180.0
_MIN_RETRIEVAL_RELEVANCE = 0.2


def build_memory_spans(
    segments: list[RichSegment],
    chapters: list[Chapter] | None = None,
) -> list[MemorySpan]:
    if not segments:
        return []

    spans: list[MemorySpan] = []
    spans.extend(_topic_spans(segments, chapters))
    spans.extend(_speaker_spans(segments))
    setup_spans = _setup_spans(segments)
    payoff_spans = _payoff_spans(segments)
    spans.extend(setup_spans)
    spans.extend(payoff_spans)
    spans.extend(_callback_spans(segments, setup_spans, payoff_spans))
    return sorted(spans, key=lambda span: (span.start, span.end, span.kind, span.id))


def candidate_memory_context(
    candidate: Candidate,
    segments: list[RichSegment],
    spans: list[MemorySpan],
) -> dict[str, object]:
    window = [seg for seg in segments if seg.end > candidate.start and seg.start < candidate.end]
    window_text = " ".join(seg.text.strip() for seg in window if seg.text.strip())
    window_tokens = _tokenize(window_text)
    window_speakers = {seg.speaker for seg in window if seg.speaker != "UNKNOWN"}

    prior_spans = [
        span for span in spans
        if span.start < candidate.start - 1.0
        and candidate.start - span.start <= _MAX_LOOKBACK_SEC
        and _span_allowed_for_candidate(span, candidate)
    ]
    overlapping_spans = [
        span for span in spans
        if span.end > candidate.start
        and span.start < candidate.end
        and _span_allowed_for_candidate(span, candidate)
    ]
    relevant_spans = sorted(
        prior_spans + [span for span in overlapping_spans if span not in prior_spans],
        key=lambda span: _memory_relevance(span, candidate, window_tokens, window_speakers),
        reverse=True,
    )
    retrieved = [
        span for span in relevant_spans
        if _memory_relevance(span, candidate, window_tokens, window_speakers) > _MIN_RETRIEVAL_RELEVANCE
    ][:4]

    active_topic = next(
        (
            span for span in retrieved
            if span.kind == "topic_span"
        ),
        next((span for span in reversed(prior_spans) if span.kind == "topic_span"), None),
    )
    setup_span = next(
        (
            span for span in retrieved
            if span.kind in {"setup_span", "open_loop_span"} and span.start < candidate.start - 1.0
        ),
        None,
    )
    if setup_span is None:
        setup_span = next(
            (
                span for span in sorted(
                    [
                        span for span in prior_spans
                        if span.kind in {"setup_span", "open_loop_span"}
                        and _memory_relevance(span, candidate, window_tokens, window_speakers) > _MIN_RETRIEVAL_RELEVANCE
                    ],
                    key=lambda item: (item.confidence, item.end),
                    reverse=True,
                )
            ),
            None,
        )
    payoff_span = next((span for span in overlapping_spans if span.kind == "payoff_span"), None)
    callback_span = next((span for span in retrieved if span.kind == "callback_span"), None)
    if callback_span is None:
        callback_span = next(
            (
                span for span in sorted(
                    [
                        span for span in prior_spans
                        if span.kind == "callback_span"
                        and _memory_relevance(span, candidate, window_tokens, window_speakers) > _MIN_RETRIEVAL_RELEVANCE
                    ],
                    key=lambda item: (item.confidence, item.end),
                    reverse=True,
                )
            ),
            None,
        )

    minimum_required_setup_start: float | None = None
    if setup_span is not None:
        minimum_required_setup_start = float(setup_span.start)
    if callback_span is not None:
        callback_start = float(callback_span.start)
        minimum_required_setup_start = callback_start if minimum_required_setup_start is None else min(minimum_required_setup_start, callback_start)
    if minimum_required_setup_start is not None and minimum_required_setup_start >= float(candidate.start) - 1.0:
        minimum_required_setup_start = None

    setup_depth_score = 0.0
    if setup_span is not None and minimum_required_setup_start is not None and candidate.start > setup_span.start:
        setup_depth_score = _clamp01((candidate.start - setup_span.start) / 35.0)

    callback_strength = _clamp01(_memory_relevance(callback_span, candidate, window_tokens, window_speakers)) if callback_span else 0.0
    topic_continuity_score = _topic_continuity(active_topic, window_tokens)
    speaker_context_clarity = _speaker_context_clarity(window_speakers, retrieved)
    arc_payoff_match = _arc_payoff_match(setup_span, payoff_span, callback_span, candidate)
    retrieved_count = float(len(retrieved))

    memory_dependency_score = _clamp01(
        setup_depth_score * 0.35
        + callback_strength * 0.35
        + max(0.0, 1.0 - topic_continuity_score) * 0.15
        + max(0.0, 1.0 - speaker_context_clarity) * 0.15
    )

    story_stage_score = 0.2
    if setup_span is not None:
        story_stage_score = 0.45
    if payoff_span is not None:
        story_stage_score = 0.8
    if callback_span is not None and payoff_span is not None:
        story_stage_score = 1.0

    context_summary = _context_summary(active_topic, setup_span, payoff_span, callback_span)
    return {
        "retrieved_spans": [span.to_dict() for span in retrieved],
        "active_topic": active_topic.summary if active_topic else "",
        "relevant_setup": setup_span.summary if setup_span else "",
        "why_this_is_a_payoff": payoff_span.summary if payoff_span else (callback_span.summary if callback_span else ""),
        "speaker_roles": sorted(window_speakers),
        "minimum_required_setup_start": round(minimum_required_setup_start, 2) if minimum_required_setup_start is not None else None,
        "context_summary": context_summary,
        "setup_depth_score": round(setup_depth_score, 3),
        "callback_strength": round(callback_strength, 3),
        "arc_payoff_match": round(arc_payoff_match, 3),
        "topic_continuity_score": round(topic_continuity_score, 3),
        "speaker_context_clarity": round(speaker_context_clarity, 3),
        "memory_dependency_score": round(memory_dependency_score, 3),
        "story_stage_score": round(story_stage_score, 3),
        "required_setup_lead_sec": round(max(candidate.start - minimum_required_setup_start, 0.0), 3) if minimum_required_setup_start is not None else 0.0,
        "retrieved_memory_span_count": round(retrieved_count, 3),
    }


def _topic_spans(segments: list[RichSegment], chapters: list[Chapter] | None) -> list[MemorySpan]:
    spans: list[MemorySpan] = []
    if chapters:
        for chapter in chapters:
            chapter_segments = [seg for seg in segments if seg.end > chapter.start and seg.start < chapter.end]
            if not chapter_segments:
                continue
            summary = _summarize_text(" ".join(seg.text for seg in chapter_segments))
            spans.append(
                MemorySpan(
                    id=f"topic:{chapter.id}",
                    kind="topic_span",
                    start=float(chapter.start),
                    end=float(chapter.end),
                    confidence=round(_clamp01(0.45 + chapter.confidence * 0.45), 3),
                    summary=summary,
                    entities=_top_entities(chapter_segments),
                    speaker_ids=_speakers_for_segments(chapter_segments),
                    metadata={"chapter_id": chapter.id, "chapter_label": chapter.label},
                )
            )
        return spans

    current = [segments[0]]
    for seg in segments[1:]:
        if _starts_new_topic(current[-1], seg):
            spans.append(_topic_span_from_segments(current, len(spans)))
            current = [seg]
        else:
            current.append(seg)
    if current:
        spans.append(_topic_span_from_segments(current, len(spans)))
    return spans


def _topic_span_from_segments(segments: list[RichSegment], index: int) -> MemorySpan:
    text = " ".join(seg.text for seg in segments)
    return MemorySpan(
        id=f"topic:auto:{index}",
        kind="topic_span",
        start=float(segments[0].start),
        end=float(segments[-1].end),
        confidence=round(_clamp01(0.35 + min(len(segments), 5) * 0.1), 3),
        summary=_summarize_text(text),
        entities=_top_entities(segments),
        speaker_ids=_speakers_for_segments(segments),
    )


def _speaker_spans(segments: list[RichSegment]) -> list[MemorySpan]:
    spans: list[MemorySpan] = []
    current = [segments[0]]
    for seg in segments[1:]:
        if seg.speaker != current[-1].speaker:
            spans.append(_speaker_span_from_segments(current, len(spans)))
            current = [seg]
        else:
            current.append(seg)
    if current:
        spans.append(_speaker_span_from_segments(current, len(spans)))
    return spans


def _speaker_span_from_segments(segments: list[RichSegment], index: int) -> MemorySpan:
    speaker = segments[0].speaker or "UNKNOWN"
    summary = _summarize_text(" ".join(seg.text for seg in segments))
    return MemorySpan(
        id=f"speaker:{index}",
        kind="speaker_turn_span",
        start=float(segments[0].start),
        end=float(segments[-1].end),
        confidence=0.7 if speaker != "UNKNOWN" else 0.35,
        summary=f"{speaker}: {summary}" if summary else speaker,
        entities=_top_entities(segments),
        speaker_ids=[speaker] if speaker != "UNKNOWN" else [],
    )


def _setup_spans(segments: list[RichSegment]) -> list[MemorySpan]:
    spans: list[MemorySpan] = []
    for index, seg in enumerate(segments):
        text = seg.text.strip().lower()
        tokens = _tokenize(text)
        setup_score = 0.0
        if text.endswith("?"):
            setup_score += 0.35
        if tokens & _SETUP_TOKENS:
            setup_score += 0.35
        if seg.speaker != "UNKNOWN":
            setup_score += 0.1
        if seg.hype_score >= 1.8:
            setup_score += 0.12
        if seg.arc_score >= 2.0:
            setup_score += 0.12
        if setup_score < 0.45:
            continue
        lead_end = min(seg.end + 20.0, segments[min(index + 2, len(segments) - 1)].end)
        spans.append(
            MemorySpan(
                id=f"setup:{index}",
                kind="setup_span",
                start=float(seg.start),
                end=float(lead_end),
                confidence=round(_clamp01(setup_score), 3),
                summary=_summarize_text(seg.text),
                entities=_top_entities([seg]),
                speaker_ids=[seg.speaker] if seg.speaker != "UNKNOWN" else [],
                metadata={"anchor_index": index},
            )
        )
        spans.append(
            MemorySpan(
                id=f"open_loop:{index}",
                kind="open_loop_span",
                start=float(seg.start),
                end=float(min(seg.start + 45.0, seg.end + 45.0)),
                confidence=round(_clamp01(setup_score * 0.85), 3),
                summary=f"Offene Schleife: {_summarize_text(seg.text)}",
                entities=_top_entities([seg]),
                speaker_ids=[seg.speaker] if seg.speaker != "UNKNOWN" else [],
                related_span_ids=[f"setup:{index}"],
                metadata={"anchor_index": index},
            )
        )
    return spans


def _payoff_spans(segments: list[RichSegment]) -> list[MemorySpan]:
    spans: list[MemorySpan] = []
    for index, seg in enumerate(segments):
        text = seg.text.strip().lower()
        tokens = _tokenize(text)
        payoff_score = 0.0
        if "audio_laughter" in seg.emotions or "laughter" in seg.emotions:
            payoff_score += 0.3
        if "exclamation" in seg.emotions:
            payoff_score += 0.15
        if seg.chat_spike_score >= 0.35:
            payoff_score += 0.15
        if max(seg.max_face_reaction, seg.max_motion_score) >= 0.35:
            payoff_score += 0.15
        if seg.hype_score >= 3.0:
            payoff_score += 0.15
        if tokens & _PAYOFF_TOKENS:
            payoff_score += 0.1
        if payoff_score < 0.45:
            continue
        spans.append(
            MemorySpan(
                id=f"payoff:{index}",
                kind="payoff_span",
                start=float(seg.start),
                end=float(seg.end),
                confidence=round(_clamp01(payoff_score), 3),
                summary=_summarize_text(seg.text),
                entities=_top_entities([seg]),
                speaker_ids=[seg.speaker] if seg.speaker != "UNKNOWN" else [],
                metadata={"anchor_index": index},
            )
        )
    return spans


def _callback_spans(
    segments: list[RichSegment],
    setup_spans: list[MemorySpan],
    payoff_spans: list[MemorySpan],
) -> list[MemorySpan]:
    spans: list[MemorySpan] = []
    for payoff in payoff_spans:
        payoff_tokens = set(payoff.entities)
        if not payoff_tokens:
            continue
        match = None
        best_score = 0.0
        for setup in setup_spans:
            if setup.start >= payoff.start:
                continue
            setup_lead_sec = payoff.start - setup.start
            setup_gap_sec = payoff.start - setup.end
            if setup_lead_sec > _MAX_LOOKBACK_SEC or setup_gap_sec > _MAX_LOOKBACK_SEC:
                continue
            overlap = len(set(setup.entities) & payoff_tokens)
            if overlap <= 0:
                continue
            proximity = _clamp01(1.0 - ((payoff.start - setup.end) / 120.0))
            if overlap < 2 and proximity < 0.35:
                continue
            score = overlap * 0.45 + proximity * 0.55
            if score > best_score:
                best_score = score
                match = setup
        if match is None or best_score < 0.55:
            continue
        segments_between = [seg for seg in segments if seg.end > match.start and seg.start < payoff.end]
        spans.append(
            MemorySpan(
                id=f"callback:{match.id}:{payoff.id}",
                kind="callback_span",
                start=float(match.start),
                end=float(payoff.end),
                confidence=round(_clamp01(best_score), 3),
                summary=f"Callback von \"{match.summary}\" zu \"{payoff.summary}\"",
                entities=sorted(set(match.entities) | set(payoff.entities))[:6],
                speaker_ids=sorted(set(match.speaker_ids) | set(payoff.speaker_ids)),
                related_span_ids=[match.id, payoff.id],
                metadata={"setup_id": match.id, "payoff_id": payoff.id, "segment_count": len(segments_between)},
            )
        )
    return spans


def _memory_relevance(
    span: MemorySpan | None,
    candidate: Candidate,
    window_tokens: set[str],
    window_speakers: set[str],
) -> float:
    if span is None:
        return 0.0
    span_tokens = set(span.entities)
    token_overlap = _jaccard(span_tokens, window_tokens)
    span_speakers = set(span.speaker_ids)
    speaker_overlap = _jaccard(span_speakers, window_speakers)
    temporal_anchor = span.start if span.kind in {"callback_span", "setup_span", "open_loop_span"} else span.end
    temporal = _clamp01(1.0 - (max(candidate.start - temporal_anchor, 0.0) / _MAX_LOOKBACK_SEC))
    kind_bonus = {
        "callback_span": 0.28,
        "setup_span": 0.22,
        "open_loop_span": 0.2,
        "payoff_span": 0.24,
        "topic_span": 0.16,
        "speaker_turn_span": 0.08,
    }.get(span.kind, 0.0)
    return _clamp01(token_overlap * 0.35 + speaker_overlap * 0.15 + temporal * 0.3 + span.confidence * 0.2 + kind_bonus)


def _span_allowed_for_candidate(span: MemorySpan, candidate: Candidate) -> bool:
    if span.kind in {"callback_span", "setup_span", "open_loop_span"}:
        return candidate.start - span.start <= _MAX_LOOKBACK_SEC
    return True


def _topic_continuity(topic_span: MemorySpan | None, window_tokens: set[str]) -> float:
    if topic_span is None:
        return 0.0
    return _clamp01(_jaccard(set(topic_span.entities), window_tokens) * 0.75 + topic_span.confidence * 0.25)


def _speaker_context_clarity(window_speakers: set[str], spans: list[MemorySpan]) -> float:
    if not window_speakers:
        return 0.0
    span_speakers = set()
    for span in spans:
        span_speakers.update(span.get("speaker_ids", []) if isinstance(span, dict) else span.speaker_ids)
    return _clamp01(_jaccard(window_speakers, span_speakers) + 0.2)


def _arc_payoff_match(
    setup_span: MemorySpan | None,
    payoff_span: MemorySpan | None,
    callback_span: MemorySpan | None,
    candidate: Candidate,
) -> float:
    if payoff_span is None:
        return 0.0
    score = payoff_span.confidence * 0.55
    if setup_span is not None and setup_span.start < candidate.start:
        score += 0.25
    if callback_span is not None:
        score += 0.2
    return _clamp01(score)


def _context_summary(
    active_topic: MemorySpan | None,
    setup_span: MemorySpan | None,
    payoff_span: MemorySpan | None,
    callback_span: MemorySpan | None,
) -> str:
    parts: list[str] = []
    if active_topic is not None and active_topic.summary:
        parts.append(f"Thema: {active_topic.summary}")
    if setup_span is not None and setup_span.summary:
        parts.append(f"Setup: {setup_span.summary}")
    if payoff_span is not None and payoff_span.summary:
        parts.append(f"Payoff: {payoff_span.summary}")
    elif callback_span is not None and callback_span.summary:
        parts.append(f"Callback: {callback_span.summary}")
    return " | ".join(parts) if parts else "No prior context available."


def _starts_new_topic(prev: RichSegment, current: RichSegment) -> bool:
    prev_tokens = _tokenize(prev.text)
    current_tokens = _tokenize(current.text)
    if current.speaker != prev.speaker and current.start - prev.end >= 1.5:
        return True
    if current.start - prev.end >= 4.5:
        return True
    if current_tokens & _TOPIC_SHIFT_TOKENS and _jaccard(prev_tokens, current_tokens) < 0.15:
        return True
    if _jaccard(prev_tokens, current_tokens) < 0.1 and current.has_scene_change:
        return True
    return False


def _tokenize(text: str) -> set[str]:
    return {token for token in _TOKEN_RE.findall((text or "").lower()) if len(token) >= 3 and token not in _STOPWORDS}


def _top_entities(segments: list[RichSegment], limit: int = 6) -> list[str]:
    counts: dict[str, int] = {}
    for seg in segments:
        for token in _tokenize(seg.text):
            counts[token] = counts.get(token, 0) + 1
    return [token for token, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def _speakers_for_segments(segments: list[RichSegment]) -> list[str]:
    return sorted({seg.speaker for seg in segments if seg.speaker != "UNKNOWN"})


def _summarize_text(text: str, max_words: int = 10) -> str:
    words = [word for word in text.strip().split() if word]
    if not words:
        return ""
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).strip() + "..."


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(len(left | right), 1)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
