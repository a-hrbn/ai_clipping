from __future__ import annotations

from services.timeline_types import Candidate
from services.transcriber import RichSegment

_MIN_DURATION = 8.0
_MAX_HOOK_DELAY = 3.0
_MIN_CONTEXT_BEFORE_ANCHOR = 1.5
_MAX_EXTEND_SEC = 12.0


def normalize_candidates(
    candidates: list[Candidate],
    segments: list[RichSegment],
    max_duration: float,
) -> list[Candidate]:
    normalized: list[Candidate] = []
    for candidate in candidates:
        normalized_candidate = _normalize_candidate(candidate, segments, max_duration)
        if normalized_candidate is not None:
            normalized.append(normalized_candidate)
    return _dedupe_candidates(normalized)


def _normalize_candidate(
    candidate: Candidate,
    segments: list[RichSegment],
    max_duration: float,
) -> Candidate | None:
    if not segments:
        return None

    start_seg = _seg_at(segments, candidate.start)
    end_seg = _seg_at(segments, candidate.end)
    anchor_seg = _seg_at(segments, candidate.anchor_time)
    if start_seg is None or end_seg is None or anchor_seg is None:
        return None

    start = start_seg.start
    end = end_seg.end

    min_start = max(0.0, candidate.anchor_time - _MAX_HOOK_DELAY)
    max_start = max(0.0, candidate.anchor_time - _MIN_CONTEXT_BEFORE_ANCHOR)
    start = _nearest_boundary_at_or_after(segments, start, floor=min_start)
    start = _nearest_boundary_at_or_before(segments, start, ceiling=max_start)

    if end < candidate.anchor_time + 1.0:
        end = _nearest_end_at_or_after(segments, candidate.anchor_time + 1.0)
    end = _extend_to_sentence_end(segments, end, deadline=end + _MAX_EXTEND_SEC)

    duration = end - start
    if duration < _MIN_DURATION:
        end = _nearest_end_at_or_after(segments, start + _MIN_DURATION)
    if end - start > max_duration:
        end = _nearest_end_at_or_before(segments, start + max_duration)

    if end - start <= 0:
        return None

    return Candidate(
        id=candidate.id,
        start=round(start, 2),
        end=round(end, 2),
        source=candidate.source,
        anchor_time=round(candidate.anchor_time, 2),
        score_hint=candidate.score_hint,
        tags=list(candidate.tags),
    )


def _dedupe_candidates(candidates: list[Candidate]) -> list[Candidate]:
    kept: list[Candidate] = []
    by_score = sorted(candidates, key=lambda c: (c.score_hint, -(c.end - c.start)), reverse=True)
    for candidate in by_score:
        duplicate = False
        for existing in kept:
            if abs(candidate.anchor_time - existing.anchor_time) <= 4.0:
                duplicate = True
                break
            overlap = min(candidate.end, existing.end) - max(candidate.start, existing.start)
            if overlap <= 0:
                continue
            shorter = min(candidate.end - candidate.start, existing.end - existing.start)
            if shorter > 0 and overlap / shorter >= 0.7:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return sorted(kept, key=lambda candidate: candidate.start)


def _seg_at(segments: list[RichSegment], time: float) -> RichSegment | None:
    for seg in segments:
        if seg.start <= time <= seg.end:
            return seg
    if segments:
        return min(segments, key=lambda seg: abs(seg.start - time))
    return None


def _nearest_boundary_at_or_after(segments: list[RichSegment], current: float, floor: float) -> float:
    candidates = [seg.start for seg in segments if seg.start >= floor]
    if not candidates:
        return current
    return min(candidates, key=lambda value: abs(value - current))


def _nearest_boundary_at_or_before(segments: list[RichSegment], current: float, ceiling: float) -> float:
    candidates = [seg.start for seg in segments if seg.start <= ceiling]
    if not candidates:
        return current
    return max(candidates, key=lambda value: (value, -abs(value - current)))


def _nearest_end_at_or_after(segments: list[RichSegment], floor: float) -> float:
    candidates = [seg.end for seg in segments if seg.end >= floor]
    return candidates[0] if candidates else segments[-1].end


def _nearest_end_at_or_before(segments: list[RichSegment], ceiling: float) -> float:
    candidates = [seg.end for seg in segments if seg.end <= ceiling]
    return candidates[-1] if candidates else segments[0].end


def _extend_to_sentence_end(segments: list[RichSegment], current_end: float, deadline: float) -> float:
    for seg in segments:
        if seg.end < current_end:
            continue
        if seg.end > deadline:
            break
        if seg.text.strip().endswith((".", "!", "?")):
            return seg.end
    return current_end
