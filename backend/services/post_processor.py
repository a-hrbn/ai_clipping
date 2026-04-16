"""
Post-processing of LLM-returned clip moments.

Two passes run after the LLM returns candidate clips:

1. snap_boundaries  — aligns start/end timestamps to real segment boundaries
                      and extends to the nearest sentence ending so clips never
                      cut mid-word or mid-thought.

2. enforce_diversity — ensures clips are spread across the video by enforcing
                       a minimum gap between any two clips. Prevents the same
                       viral peak from appearing as multiple near-identical clips.
"""

from __future__ import annotations
import re
from services.transcriber import RichSegment

# Words that strongly indicate the clip starts mid-thought.
# A clip opening with any of these feels contextless to the viewer.
_MID_THOUGHT_RE = re.compile(
    r"^(and |but |so |because |then |also |or |if |when |which |who |where |"
    r"he |she |they |it |that |this |those |these )",
    re.IGNORECASE,
)

# Maximum seconds we allow boundary snapping to extend a clip end.
_MAX_EXTEND_SEC = 12.0

# Maximum seconds we allow boundary snapping to pull a clip start backward.
_MAX_RETRACT_SEC = 8.0

# Minimum clip length to preserve setup -> payoff continuity.
_MIN_DURATION_SEC = 8.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def snap_boundaries(
    moments: list[dict],
    segments: list[RichSegment],
) -> list[dict]:
    """
    Align each clip's start/end to real segment boundaries.

    Start snapping:
      - Move start_time to the beginning of the containing segment.
      - If that segment starts mid-thought (conjunction / pronoun opening),
        try the *next* complete-sentence segment instead (up to _MAX_RETRACT_SEC
        worth of shift).  Never move forward past the original peak area.

    End snapping:
      - Move end_time to the end of the containing segment.
      - If that segment does not end with a sentence boundary (.!?), extend
        forward to the next segment that does (up to _MAX_EXTEND_SEC).

    Duration guard: never produce a non-positive clip or a clip longer than 90 s.
    """
    snapped = []
    for m in moments:
        m = dict(m)  # work on a copy

        start_seg = _seg_at(segments, m["start_time"])
        end_seg   = _seg_at(segments, m["end_time"])

        if start_seg:
            # Snap to segment boundary
            m["start_time"] = start_seg.start

            # If opening is mid-thought, try to pull the start backward to the
            # previous clean sentence so we keep the missing context instead of
            # cutting even more off the front.
            if _is_mid_thought(start_seg.text):
                better = _find_clean_start(
                    segments, start_seg,
                    max_retract_sec=_MAX_RETRACT_SEC,
                )
                if better:
                    m["start_time"] = better.start

        if end_seg:
            m["end_time"] = end_seg.end

            # Extend to sentence boundary if the segment ends mid-sentence
            if not _ends_sentence(end_seg.text):
                extended = _find_sentence_end(
                    segments, end_seg,
                    max_extend_sec=_MAX_EXTEND_SEC,
                )
                if extended:
                    m["end_time"] = extended.end

        # Duration guard
        duration = m["end_time"] - m["start_time"]
        if duration <= 0 and end_seg:
            m["end_time"] = max(end_seg.end, m["start_time"] + 1.0)
        if 0 < duration < _MIN_DURATION_SEC:
            m = _extend_min_duration(m, segments, min_duration_sec=_MIN_DURATION_SEC)
        if duration > 90.0:
            m["end_time"] = m["start_time"] + 90.0

        snapped.append(m)

    return snapped


def enforce_diversity(
    moments: list[dict],
    min_gap_sec: float = 45.0,
) -> list[dict]:
    """
    Greedily keep only clips that are sufficiently spread apart in time.

    Process clips from highest viral_score to lowest.  A clip is dropped if
    it overlaps or is within `min_gap_sec` of any already-kept clip.

    If the strict gap would leave fewer than 3 clips, the gap is relaxed to
    15 s (just enough to prevent actual overlap) and the selection is retried.

    Surviving clips are re-ranked by start_time so rank == chronological order.
    """
    result = _apply_min_gap(moments, min_gap_sec)
    if len(result) < 3 and min_gap_sec > 15.0:
        result = _apply_min_gap(moments, 15.0)
    # Re-rank chronologically
    for i, m in enumerate(sorted(result, key=lambda m: m["start_time"]), 1):
        m["rank"] = i
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_min_gap(moments: list[dict], min_gap_sec: float) -> list[dict]:
    """Core greedy selection: highest score first, enforce minimum time gap."""
    by_score = sorted(moments, key=lambda m: m["viral_score"], reverse=True)
    kept: list[dict] = []

    for m in by_score:
        too_close = False
        for k in kept:
            # Gap = time between the two clips (negative = overlap)
            gap = (
                max(m["start_time"], k["start_time"])
                - min(m["end_time"],   k["end_time"])
            )
            if gap < min_gap_sec:
                too_close = True
                break
        if not too_close:
            kept.append(m)

    return kept


def _seg_at(segments: list[RichSegment], time: float) -> RichSegment | None:
    """Return the segment whose time range contains `time`."""
    for seg in segments:
        if seg.start <= time <= seg.end:
            return seg
    # Fallback: nearest segment by start distance
    if segments:
        return min(segments, key=lambda s: abs(s.start - time))
    return None


def _seg_index(segments: list[RichSegment], seg: RichSegment) -> int:
    # Identity first: segments are stored by reference across the pipeline, so
    # `is` works when the list hasn't been rebuilt. Fall back to an `isclose`
    # match on start to tolerate minor float drift from upstream mutations.
    for i, s in enumerate(segments):
        if s is seg:
            return i
    for i, s in enumerate(segments):
        if abs(s.start - seg.start) < 1e-6:
            return i
    return -1


def _is_mid_thought(text: str) -> bool:
    return bool(_MID_THOUGHT_RE.match(text.strip()))


def _ends_sentence(text: str) -> bool:
    return bool(text.strip()) and text.strip()[-1] in ".!?"


def _find_clean_start(
    segments: list[RichSegment],
    current: RichSegment,
    max_retract_sec: float,
) -> RichSegment | None:
    """
    Walk backward from `current` to find a better clip opening within
    `max_retract_sec` of the current start.

    Prefer:
    1. a previous segment that ends a sentence, then start at the next segment,
    2. otherwise the earliest nearby segment that does not itself open mid-thought.
    """
    idx = _seg_index(segments, current)
    if idx < 0:
        return None

    earliest_start = current.start - max_retract_sec
    candidate: RichSegment | None = None

    for back_idx in range(idx - 1, -1, -1):
        s = segments[back_idx]
        if s.start < earliest_start:
            break
        if _ends_sentence(s.text):
            next_idx = back_idx + 1
            if next_idx <= idx:
                return segments[next_idx]
        if not _is_mid_thought(s.text):
            candidate = s

    if candidate is not None:
        return candidate

    return None


def _find_sentence_end(
    segments: list[RichSegment],
    current: RichSegment,
    max_extend_sec: float,
) -> RichSegment | None:
    """
    Walk forward from `current` to find the next segment that ends with .!?
    within `max_extend_sec` of the current end.
    """
    idx = _seg_index(segments, current)
    if idx < 0:
        return None
    deadline = current.end + max_extend_sec
    for s in segments[idx + 1:]:
        if s.end > deadline:
            break
        if _ends_sentence(s.text):
            return s
    return None


def _extend_min_duration(
    moment: dict,
    segments: list[RichSegment],
    min_duration_sec: float,
) -> dict:
    current = dict(moment)
    start_seg = _seg_at(segments, current["start_time"])
    end_seg = _seg_at(segments, current["end_time"])
    if start_seg is None or end_seg is None:
        return current

    start_idx = _seg_index(segments, start_seg)
    end_idx = _seg_index(segments, end_seg)
    if start_idx < 0 or end_idx < 0:
        return current

    while current["end_time"] - current["start_time"] < min_duration_sec:
        expanded = False
        if start_idx > 0:
            start_idx -= 1
            current["start_time"] = segments[start_idx].start
            expanded = True
        if current["end_time"] - current["start_time"] >= min_duration_sec:
            break
        if end_idx < len(segments) - 1:
            end_idx += 1
            current["end_time"] = segments[end_idx].end
            expanded = True
        if not expanded:
            break

    return current
