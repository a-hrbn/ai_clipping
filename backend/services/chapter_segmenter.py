from __future__ import annotations

import re

from services.timeline_types import Chapter
from services.transcriber import RichSegment

_MIN_CHAPTER_SEGMENTS = 3
_MIN_CHAPTER_DURATION_SEC = 25.0
_MAX_CHAPTER_DURATION_SEC = 180.0
_PAUSE_BOUNDARY_SEC = 2.5
_CHAT_SHIFT_THRESHOLD = 0.45
_MOTION_SHIFT_THRESHOLD = 0.4
_TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")


def segment_chapters(segments: list[RichSegment]) -> list[Chapter]:
    if not segments:
        return []

    if len(segments) <= _MIN_CHAPTER_SEGMENTS:
        return [_make_chapter(segments, 0, len(segments) - 1, confidence=0.55)]

    chapters: list[Chapter] = []
    start_idx = 0
    for idx in range(1, len(segments)):
        boundary_score = _boundary_score(segments, idx)
        chapter_duration = segments[idx - 1].end - segments[start_idx].start
        chapter_segment_count = idx - start_idx
        strong_boundary = boundary_score >= 0.95
        force_split = chapter_duration >= _MAX_CHAPTER_DURATION_SEC and boundary_score >= 0.4
        enough_room = (
            chapter_duration >= _MIN_CHAPTER_DURATION_SEC
            and chapter_segment_count >= _MIN_CHAPTER_SEGMENTS
            and (len(segments) - idx) >= _MIN_CHAPTER_SEGMENTS
        )
        if strong_boundary and enough_room or force_split:
            chapters.append(_make_chapter(segments, start_idx, idx - 1, confidence=min(boundary_score, 1.0)))
            start_idx = idx

    chapters.append(_make_chapter(segments, start_idx, len(segments) - 1, confidence=0.6))
    return _merge_short_chapters(chapters)


def segments_for_chapter(segments: list[RichSegment], chapter: Chapter) -> list[RichSegment]:
    return [
        seg for seg in segments
        if seg.end > chapter.start and seg.start < chapter.end
    ]


def chapter_for_time(chapters: list[Chapter], time_sec: float) -> Chapter | None:
    if not chapters:
        return None
    for chapter in chapters:
        if chapter.start <= time_sec <= chapter.end:
            return chapter
    return min(
        chapters,
        key=lambda chapter: min(abs(time_sec - chapter.start), abs(time_sec - chapter.end)),
    )


def _merge_short_chapters(chapters: list[Chapter]) -> list[Chapter]:
    if not chapters:
        return []

    merged: list[Chapter] = [chapters[0]]
    for chapter in chapters[1:]:
        previous = merged[-1]
        if (previous.end - previous.start) < _MIN_CHAPTER_DURATION_SEC * 0.6:
            merged[-1] = Chapter(
                id=previous.id,
                start=previous.start,
                end=chapter.end,
                index=previous.index,
                label=_merge_label(previous.label, chapter.label),
                confidence=round(max(previous.confidence, chapter.confidence) * 0.9, 3),
            )
            continue
        merged.append(chapter)

    return [
        Chapter(
            id=f"chapter_{idx:02d}",
            start=round(chapter.start, 2),
            end=round(chapter.end, 2),
            index=idx,
            label=chapter.label,
            confidence=chapter.confidence,
        )
        for idx, chapter in enumerate(merged)
    ]


def _make_chapter(
    segments: list[RichSegment],
    start_idx: int,
    end_idx: int,
    *,
    confidence: float,
) -> Chapter:
    chapter_segments = segments[start_idx:end_idx + 1]
    return Chapter(
        id=f"chapter_{start_idx:02d}",
        start=round(chapter_segments[0].start, 2),
        end=round(chapter_segments[-1].end, 2),
        index=start_idx,
        label=_label_for_chapter(chapter_segments),
        confidence=round(max(min(confidence, 1.0), 0.0), 3),
    )


def _label_for_chapter(segments: list[RichSegment]) -> str:
    reaction_hits = sum(
        1 for seg in segments
        if {"laughter", "audio_laughter", "face_reaction", "exclamation"} & set(seg.emotions)
    )
    dialogue_hits = sum(1 for seg in segments if seg.speaker != "UNKNOWN")
    insight_hits = sum(1 for seg in segments if len(seg.text.split()) >= 10)
    if reaction_hits >= max(2, len(segments) // 3):
        return "reaction"
    if dialogue_hits >= max(2, len(segments) // 2):
        return "dialogue"
    if insight_hits >= max(2, len(segments) // 2):
        return "discussion"
    return "mixed"


def _merge_label(left: str, right: str) -> str:
    if left == right:
        return left
    if "reaction" in {left, right}:
        return "reaction"
    if "dialogue" in {left, right}:
        return "dialogue"
    return "mixed"


def _boundary_score(segments: list[RichSegment], idx: int) -> float:
    prev = segments[idx - 1]
    curr = segments[idx]
    score = 0.0

    pause = max(0.0, curr.start - prev.end)
    if pause >= _PAUSE_BOUNDARY_SEC:
        score += 0.45

    if prev.has_scene_change or curr.has_scene_change:
        score += 0.35

    if prev.speaker != "UNKNOWN" and curr.speaker != "UNKNOWN" and prev.speaker != curr.speaker:
        score += 0.2

    if abs(curr.chat_spike_score - prev.chat_spike_score) >= _CHAT_SHIFT_THRESHOLD:
        score += 0.15

    if abs(curr.visual_activity - prev.visual_activity) >= _MOTION_SHIFT_THRESHOLD:
        score += 0.1

    if _lexical_distance(prev.text, curr.text) >= 0.9:
        score += 0.35

    return score


def _lexical_distance(left: str, right: str) -> float:
    left_tokens = set(_TOKEN_RE.findall(left.lower()))
    right_tokens = set(_TOKEN_RE.findall(right.lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)
    return 1.0 - overlap
