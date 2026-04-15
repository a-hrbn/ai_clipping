from __future__ import annotations

from services.timeline_types import BoundaryPoint, Candidate
from services.transcriber import RichSegment

_PAUSE_BREAK_SEC = 1.0
_KIND_PRIORITY = {
    "sentence_end": 7,
    "sentence_start": 6,
    "speaker_turn": 5,
    "scene_cut": 4,
    "pause_break": 3,
    "segment_end": 2,
    "segment_start": 1,
}


def build_boundary_points(
    segments: list[RichSegment],
    candidate: Candidate,
    *,
    context_lookback_sec: float = 18.0,
    context_lookahead_sec: float = 18.0,
) -> list[BoundaryPoint]:
    if not segments:
        return []

    window_start = max(segments[0].start, candidate.start - context_lookback_sec)
    window_end = min(segments[-1].end, candidate.end + context_lookahead_sec)
    relevant_indexes = [
        idx
        for idx, seg in enumerate(segments)
        if seg.end >= window_start and seg.start <= window_end
    ]
    if not relevant_indexes:
        return []

    by_time: dict[float, BoundaryPoint] = {}
    for idx in relevant_indexes:
        seg = segments[idx]
        prev_seg = segments[idx - 1] if idx > 0 else None
        next_seg = segments[idx + 1] if idx + 1 < len(segments) else None

        pause_before = max(0.0, seg.start - prev_seg.end) if prev_seg is not None else 0.0
        pause_after = max(0.0, next_seg.start - seg.end) if next_seg is not None else 0.0
        speaker_turn = (
            prev_seg is not None
            and prev_seg.speaker != "UNKNOWN"
            and seg.speaker != "UNKNOWN"
            and prev_seg.speaker != seg.speaker
        )
        scene_cut = bool(seg.has_scene_change or (prev_seg.has_scene_change if prev_seg is not None else False))
        sentence_start = (
            prev_seg is None
            or _ends_sentence(prev_seg.text)
            or pause_before >= _PAUSE_BREAK_SEC
            or speaker_turn
            or scene_cut
        )

        _merge_boundary(
            by_time,
            time=seg.start,
            kind="segment_start",
            segment_index=idx,
            speaker=seg.speaker,
            text_left=prev_seg.text if prev_seg is not None else "",
            text_right=seg.text,
            pause_before_sec=pause_before,
            pause_after_sec=0.0,
            chat_delta=_delta(seg, prev_seg, "chat_spike_score"),
            hype_delta=_delta(seg, prev_seg, "hype_score"),
            visual_delta=_delta(seg, prev_seg, "visual_activity"),
            face_reaction_delta=_delta(seg, prev_seg, "max_face_reaction"),
        )
        if sentence_start:
            _merge_boundary(
                by_time,
                time=seg.start,
                kind="sentence_start",
                segment_index=idx,
                speaker=seg.speaker,
                text_left=prev_seg.text if prev_seg is not None else "",
                text_right=seg.text,
                is_sentence_start=True,
                pause_before_sec=pause_before,
            )
        if speaker_turn:
            _merge_boundary(
                by_time,
                time=seg.start,
                kind="speaker_turn",
                segment_index=idx,
                speaker=seg.speaker,
                text_left=prev_seg.text if prev_seg is not None else "",
                text_right=seg.text,
                is_speaker_turn=True,
                pause_before_sec=pause_before,
            )
        if scene_cut:
            _merge_boundary(
                by_time,
                time=seg.start,
                kind="scene_cut",
                segment_index=idx,
                speaker=seg.speaker,
                text_left=prev_seg.text if prev_seg is not None else "",
                text_right=seg.text,
                is_scene_cut=True,
                pause_before_sec=pause_before,
            )
        if pause_before >= _PAUSE_BREAK_SEC:
            _merge_boundary(
                by_time,
                time=seg.start,
                kind="pause_break",
                segment_index=idx,
                speaker=seg.speaker,
                text_left=prev_seg.text if prev_seg is not None else "",
                text_right=seg.text,
                is_pause_break=True,
                pause_before_sec=pause_before,
            )

        _merge_boundary(
            by_time,
            time=seg.end,
            kind="segment_end",
            segment_index=idx,
            speaker=seg.speaker,
            text_left=seg.text,
            text_right=next_seg.text if next_seg is not None else "",
            pause_before_sec=0.0,
            pause_after_sec=pause_after,
            chat_delta=_delta(next_seg, seg, "chat_spike_score"),
            hype_delta=_delta(next_seg, seg, "hype_score"),
            visual_delta=_delta(next_seg, seg, "visual_activity"),
            face_reaction_delta=_delta(next_seg, seg, "max_face_reaction"),
        )
        if _ends_sentence(seg.text):
            _merge_boundary(
                by_time,
                time=seg.end,
                kind="sentence_end",
                segment_index=idx,
                speaker=seg.speaker,
                text_left=seg.text,
                text_right=next_seg.text if next_seg is not None else "",
                is_sentence_end=True,
                pause_after_sec=pause_after,
            )

    points = sorted(by_time.values(), key=lambda item: (item.time, -_KIND_PRIORITY.get(item.kind, 0)))
    return [
        point
        for point in points
        if candidate.start - context_lookback_sec <= point.time <= candidate.end + context_lookahead_sec
    ]


def _merge_boundary(
    by_time: dict[float, BoundaryPoint],
    *,
    time: float,
    kind: str,
    segment_index: int,
    speaker: str,
    text_left: str,
    text_right: str,
    is_sentence_end: bool = False,
    is_sentence_start: bool = False,
    is_speaker_turn: bool = False,
    is_scene_cut: bool = False,
    is_pause_break: bool = False,
    pause_before_sec: float = 0.0,
    pause_after_sec: float = 0.0,
    chat_delta: float = 0.0,
    hype_delta: float = 0.0,
    visual_delta: float = 0.0,
    face_reaction_delta: float = 0.0,
) -> None:
    key = round(time, 2)
    existing = by_time.get(key)
    if existing is None:
        by_time[key] = BoundaryPoint(
            id=f"boundary:{segment_index}:{int(key * 100)}",
            time=key,
            kind=kind,
            segment_index=segment_index,
            speaker=speaker,
            text_left=text_left.strip(),
            text_right=text_right.strip(),
            is_sentence_end=is_sentence_end,
            is_sentence_start=is_sentence_start,
            is_speaker_turn=is_speaker_turn,
            is_scene_cut=is_scene_cut,
            is_pause_break=is_pause_break,
            pause_before_sec=round(max(pause_before_sec, 0.0), 3),
            pause_after_sec=round(max(pause_after_sec, 0.0), 3),
            chat_delta=round(chat_delta, 3),
            hype_delta=round(hype_delta, 3),
            visual_delta=round(visual_delta, 3),
            face_reaction_delta=round(face_reaction_delta, 3),
        )
        return

    if _KIND_PRIORITY.get(kind, 0) > _KIND_PRIORITY.get(existing.kind, 0):
        existing.kind = kind
    existing.speaker = speaker or existing.speaker
    existing.text_left = existing.text_left or text_left.strip()
    existing.text_right = existing.text_right or text_right.strip()
    existing.is_sentence_end = existing.is_sentence_end or is_sentence_end
    existing.is_sentence_start = existing.is_sentence_start or is_sentence_start
    existing.is_speaker_turn = existing.is_speaker_turn or is_speaker_turn
    existing.is_scene_cut = existing.is_scene_cut or is_scene_cut
    existing.is_pause_break = existing.is_pause_break or is_pause_break
    existing.pause_before_sec = round(max(existing.pause_before_sec, pause_before_sec), 3)
    existing.pause_after_sec = round(max(existing.pause_after_sec, pause_after_sec), 3)
    existing.chat_delta = round(existing.chat_delta if abs(existing.chat_delta) >= abs(chat_delta) else chat_delta, 3)
    existing.hype_delta = round(existing.hype_delta if abs(existing.hype_delta) >= abs(hype_delta) else hype_delta, 3)
    existing.visual_delta = round(existing.visual_delta if abs(existing.visual_delta) >= abs(visual_delta) else visual_delta, 3)
    existing.face_reaction_delta = round(
        existing.face_reaction_delta if abs(existing.face_reaction_delta) >= abs(face_reaction_delta) else face_reaction_delta,
        3,
    )


def _ends_sentence(text: str) -> bool:
    text = text.strip()
    return bool(text) and text.endswith((".", "!", "?"))


def _delta(current: RichSegment | None, previous: RichSegment | None, attr: str) -> float:
    left = float(getattr(previous, attr, 0.0) or 0.0) if previous is not None else 0.0
    right = float(getattr(current, attr, 0.0) or 0.0) if current is not None else 0.0
    return right - left
