from __future__ import annotations

from services.timeline_types import Candidate, DetectedPeak, SegmentScore
from services.transcriber import RichSegment

_MIN_PEAK_SCORE = 3.0
_MIN_PROMINENCE = 0.75
_MIN_GAP_SEC = 18.0
_MAX_PEAKS = 18


def detect_peaks(
    scores: list[SegmentScore],
    *,
    min_peak_score: float = _MIN_PEAK_SCORE,
    min_prominence: float = _MIN_PROMINENCE,
    min_gap_sec: float = _MIN_GAP_SEC,
    max_peaks: int = _MAX_PEAKS,
) -> list[DetectedPeak]:
    if not scores:
        return []

    candidates: list[DetectedPeak] = []
    for idx, item in enumerate(scores):
        left = scores[idx - 1].smoothed_score if idx > 0 else float("-inf")
        right = scores[idx + 1].smoothed_score if idx + 1 < len(scores) else float("-inf")
        if item.smoothed_score < min_peak_score:
            continue
        if item.smoothed_score < left or item.smoothed_score < right:
            continue

        left_base_idx = _walk_base(scores, idx, direction=-1)
        right_base_idx = _walk_base(scores, idx, direction=1)
        base_level = max(scores[left_base_idx].smoothed_score, scores[right_base_idx].smoothed_score)
        prominence = round(item.smoothed_score - base_level, 3)
        if prominence < min_prominence:
            continue

        candidates.append(
            DetectedPeak(
                peak_time=round((item.start + item.end) / 2.0, 2),
                peak_score=item.smoothed_score,
                left_base_time=round(scores[left_base_idx].start, 2),
                right_base_time=round(scores[right_base_idx].end, 2),
                prominence=prominence,
                window_start=round(scores[left_base_idx].start, 2),
                window_end=round(scores[right_base_idx].end, 2),
                segment_index=item.segment_index,
            )
        )

    if not candidates:
        return _fallback_peaks(scores, min_gap_sec=min_gap_sec, max_peaks=max_peaks)

    candidates.sort(key=lambda peak: (peak.peak_score, peak.prominence), reverse=True)
    kept: list[DetectedPeak] = []
    for peak in candidates:
        if any(abs(peak.peak_time - other.peak_time) < min_gap_sec for other in kept):
            continue
        kept.append(peak)
        if len(kept) >= max_peaks:
            break
    return sorted(kept, key=lambda peak: peak.peak_time)


def build_peak_candidates(
    peaks: list[DetectedPeak],
    segments: list[RichSegment],
    *,
    lead_sec: float = 8.0,
    tail_sec: float = 16.0,
) -> list[Candidate]:
    if not peaks or not segments:
        return []

    candidates: list[Candidate] = []
    transcript_start = segments[0].start
    transcript_end = segments[-1].end
    for rank, peak in enumerate(peaks):
        start = max(transcript_start, min(peak.window_start, peak.peak_time - lead_sec))
        end = min(transcript_end, max(peak.window_end, peak.peak_time + tail_sec))
        candidates.append(
            Candidate(
                id=f"peak:{rank}:{int(peak.peak_time * 100)}",
                start=round(start, 2),
                end=round(end, 2),
                source="peak",
                anchor_time=peak.peak_time,
                score_hint=round(peak.peak_score + peak.prominence, 3),
                tags=["peak_detected"],
            )
        )
    return candidates


def _walk_base(scores: list[SegmentScore], peak_idx: int, *, direction: int) -> int:
    idx = peak_idx
    current = scores[peak_idx].smoothed_score
    while 0 <= idx + direction < len(scores):
        next_idx = idx + direction
        next_score = scores[next_idx].smoothed_score
        if next_score > current:
            break
        idx = next_idx
        current = next_score
    return idx


def _fallback_peaks(
    scores: list[SegmentScore],
    *,
    min_gap_sec: float,
    max_peaks: int,
) -> list[DetectedPeak]:
    ranked = sorted(scores, key=lambda item: item.virality_score, reverse=True)
    kept: list[DetectedPeak] = []
    for item in ranked:
        peak_time = round((item.start + item.end) / 2.0, 2)
        if any(abs(peak_time - other.peak_time) < min_gap_sec for other in kept):
            continue
        kept.append(
            DetectedPeak(
                peak_time=peak_time,
                peak_score=item.smoothed_score or item.virality_score,
                left_base_time=round(item.start, 2),
                right_base_time=round(item.end, 2),
                prominence=0.0,
                window_start=round(item.start, 2),
                window_end=round(item.end, 2),
                segment_index=item.segment_index,
            )
        )
        if len(kept) >= max_peaks:
            break
    return sorted(kept, key=lambda peak: peak.peak_time)
