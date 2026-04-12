from __future__ import annotations

from services.timeline_types import Candidate, Chapter
from services.transcriber import RichSegment
from services import chapter_segmenter

_DENSE_TARGET_DURATIONS = (9.0, 16.0, 28.0, 45.0)
_DENSE_PER_DURATION = 12


def generate_candidates(
    segments: list[RichSegment],
    chapters: list[Chapter] | None = None,
) -> list[Candidate]:
    return generate_semantic_candidates(segments, chapters=chapters)


def generate_semantic_candidates(
    segments: list[RichSegment],
    chapters: list[Chapter] | None = None,
) -> list[Candidate]:
    if chapters:
        candidates: list[Candidate] = []
        for chapter in chapters:
            chapter_segments = chapter_segmenter.segments_for_chapter(segments, chapter)
            candidates.extend(_generate_semantic_candidates_for_segments(chapter_segments))
        return candidates
    return _generate_semantic_candidates_for_segments(segments)


def _generate_semantic_candidates_for_segments(segments: list[RichSegment]) -> list[Candidate]:
    candidates: list[Candidate] = []
    candidates.extend(_generate_quote_candidates(segments))
    candidates.extend(_generate_tension_build_candidates(segments))
    candidates.extend(_generate_narrative_reveal_candidates(segments))
    candidates.extend(_generate_controversy_candidates(segments))
    return candidates


def generate_fallback_candidates(
    segments: list[RichSegment],
    chapters: list[Chapter] | None = None,
) -> list[Candidate]:
    if chapters:
        candidates: list[Candidate] = []
        for chapter in chapters:
            chapter_segments = chapter_segmenter.segments_for_chapter(segments, chapter)
            candidates.extend(_generate_fallback_candidates_for_segments(chapter_segments))
        return candidates
    return _generate_fallback_candidates_for_segments(segments)


def _generate_fallback_candidates_for_segments(segments: list[RichSegment]) -> list[Candidate]:
    candidates: list[Candidate] = []
    candidates.extend(_generate_hype_peak_candidates(segments))
    candidates.extend(_generate_chat_sync_candidates(segments))
    candidates.extend(_generate_reaction_candidates(segments))
    candidates.extend(_generate_dialogue_candidates(segments))
    candidates.extend(_generate_dense_window_candidates(segments))
    return candidates


def merge_prioritized_candidates(
    primary: list[Candidate],
    supplemental: list[Candidate],
    *,
    supplemental_source_cap: int = 4,
) -> list[Candidate]:
    merged: list[Candidate] = []
    supplemental_counts: dict[str, int] = {}

    for candidate in primary:
        if _is_duplicate_candidate(candidate, merged):
            continue
        merged.append(candidate)

    for candidate in supplemental:
        if supplemental_counts.get(candidate.source, 0) >= supplemental_source_cap:
            continue
        if _is_duplicate_candidate(candidate, merged):
            continue
        merged.append(candidate)
        supplemental_counts[candidate.source] = supplemental_counts.get(candidate.source, 0) + 1

    return sorted(merged, key=lambda candidate: (candidate.start, candidate.anchor_time))


def _generate_hype_peak_candidates(segments: list[RichSegment]) -> list[Candidate]:
    peaks = sorted(
        [seg for seg in segments if seg.hype_score >= 3.0],
        key=lambda seg: seg.hype_score,
        reverse=True,
    )
    candidates: list[Candidate] = []
    for idx, seg in enumerate(peaks[:16]):
        candidates.append(_window_candidate(seg, idx, "hype_peak", lead=10.0, tail=18.0))
    return candidates


def _generate_chat_sync_candidates(segments: list[RichSegment]) -> list[Candidate]:
    peaks = sorted(
        [seg for seg in segments if seg.chat_spike_score >= 0.45],
        key=lambda seg: seg.chat_spike_score,
        reverse=True,
    )
    candidates: list[Candidate] = []
    for idx, seg in enumerate(peaks[:12]):
        candidates.append(_window_candidate(seg, idx, "chat_sync", lead=8.0, tail=16.0))
    return candidates


def _generate_reaction_candidates(segments: list[RichSegment]) -> list[Candidate]:
    candidates: list[Candidate] = []
    reaction_tags = {"audio_laughter", "laughter", "face_reaction", "high_motion", "exclamation"}
    found = 0
    for idx, seg in enumerate(segments):
        if found >= 12:
            break
        if reaction_tags.isdisjoint(set(seg.emotions)):
            continue
        if seg.hype_score < 2.0 and seg.max_face_reaction < 0.35 and seg.chat_spike_score < 0.35:
            continue
        candidates.append(_window_candidate(seg, found, "reaction", lead=7.0, tail=15.0))
        found += 1
    return candidates


def _generate_dialogue_candidates(segments: list[RichSegment]) -> list[Candidate]:
    candidates: list[Candidate] = []
    if len(segments) < 3:
        return candidates
    found = 0
    for i in range(1, len(segments) - 1):
        if found >= 10:
            break
        window = segments[max(0, i - 1): min(len(segments), i + 2)]
        speakers = [seg.speaker for seg in window if seg.speaker != "UNKNOWN"]
        if len(set(speakers)) < 2:
            continue
        speaker_switches = sum(
            1 for left, right in zip(window, window[1:])
            if left.speaker != "UNKNOWN" and right.speaker != "UNKNOWN" and left.speaker != right.speaker
        )
        if speaker_switches < 2:
            continue
        anchor = window[len(window) // 2]
        candidates.append(_window_candidate(anchor, found, "dialogue", lead=8.0, tail=12.0))
        found += 1
    return candidates


def _generate_quote_candidates(segments: list[RichSegment]) -> list[Candidate]:
    from services.semantic_candidates import find_semantic_candidates
    sem = find_semantic_candidates(segments)
    candidates: list[Candidate] = []
    # Use embedding hits when available, fall back to regex heuristic
    if "quote" in sem:
        for found, idx in enumerate(sem["quote"]):
            seg = segments[idx]
            if len(seg.text.strip().split()) < 4:
                continue
            candidates.append(_window_candidate(seg, found, "quote", lead=5.0, tail=10.0))
        return candidates
    # Regex fallback
    found = 0
    for seg in segments:
        if found >= 10:
            break
        if len(seg.text.strip().split()) < 4:
            continue
        if not _is_quote_like(seg):
            continue
        candidates.append(_window_candidate(seg, found, "quote", lead=5.0, tail=10.0))
        found += 1
    return candidates


def _window_candidate(
    anchor: RichSegment,
    idx: int,
    source: str,
    *,
    lead: float,
    tail: float,
) -> Candidate:
    return Candidate(
        id=f"{source}:{idx}:{int(anchor.start * 100)}",
        start=max(0.0, anchor.start - lead),
        end=max(anchor.end, anchor.start + 1.0) + tail,
        source=source,
        anchor_time=(anchor.start + anchor.end) / 2.0,
        score_hint=round(anchor.hype_score + anchor.arc_score + anchor.chat_spike_score * 2.0, 3),
        tags=sorted(set(anchor.emotions)),
    )


def _is_quote_like(seg: RichSegment) -> bool:
    """Regex fallback — used only when sentence-transformers is unavailable."""
    text = seg.text.strip()
    if text.endswith("?") or text.endswith("!"):
        return True
    if seg.arc_score >= 2.5 or seg.hype_score >= 3.5:
        return True
    lower = text.lower()
    starters = ("ich", "i ", "listen", "bro", "das", "this", "why", "how")
    return lower.startswith(starters)


def _is_duplicate_candidate(candidate: Candidate, existing: list[Candidate]) -> bool:
    for other in existing:
        if abs(candidate.anchor_time - other.anchor_time) <= 4.0:
            return True
        overlap = min(candidate.end, other.end) - max(candidate.start, other.start)
        if overlap <= 0:
            continue
        shorter = min(candidate.end - candidate.start, other.end - other.start)
        if shorter > 0 and overlap / shorter >= 0.7:
            return True
    return False


# ---------------------------------------------------------------------------
# Narrative-driven generators (embedding-based)
# ---------------------------------------------------------------------------

def _generate_tension_build_candidates(segments: list[RichSegment]) -> list[Candidate]:
    """
    Finds moments of rising tension before an explosion.

    Two complementary signals:
    1. Semantic: segment text similar to "something is about to happen" patterns.
    2. Heuristic: hype slope — current hype is medium, next 2 segments are
       significantly higher (the rise hasn't peaked yet).

    Window: long tail (25 s) to capture the payoff that follows the build-up.
    """
    from services.semantic_candidates import find_semantic_candidates

    candidates: list[Candidate] = []
    found_ids: set[int] = set()

    # --- Semantic hits ---
    sem = find_semantic_candidates(segments)
    for idx in sem.get("tension_build", []):
        seg = segments[idx]
        candidates.append(_window_candidate(seg, idx, "tension_build", lead=5.0, tail=25.0))
        found_ids.add(idx)

    # --- Heuristic slope detection (complementary, no duplicate) ---
    for i, seg in enumerate(segments):
        if i in found_ids:
            continue
        if len(candidates) >= 12:
            break
        # Skip if already at peak or nearly silent
        if seg.hype_score < 1.2 or seg.hype_score > 4.5:
            continue
        future = segments[i + 1: i + 4]
        if not future:
            continue
        max_future_hype = max(s.hype_score for s in future)
        # Must rise at least 2.5 points above current
        if max_future_hype - seg.hype_score >= 2.5:
            candidates.append(_window_candidate(seg, i, "tension_build", lead=5.0, tail=25.0))
            found_ids.add(i)

    return candidates


def _generate_narrative_reveal_candidates(segments: list[RichSegment]) -> list[Candidate]:
    """
    Finds twists, reveals, and information drops that reframe what came before.

    Purely semantic — audio signals don't reliably detect reveals (they're often
    delivered calmly, then followed by a reaction).

    Window: long lead (15 s) to include the context the reveal reframes.
    """
    from services.semantic_candidates import find_semantic_candidates

    sem = find_semantic_candidates(segments)
    candidates: list[Candidate] = []
    for found, idx in enumerate(sem.get("narrative_reveal", [])):
        seg = segments[idx]
        candidates.append(_window_candidate(seg, found, "narrative_reveal", lead=15.0, tail=18.0))
    return candidates


def _generate_controversy_candidates(segments: list[RichSegment]) -> list[Candidate]:
    """
    Finds genuine disagreement, hot-takes, and confrontation — goes well beyond
    the regex contradiction_score which only catches words like "aber"/"but".

    Window: moderate lead to include the setup, longer tail for the reaction.
    """
    from services.semantic_candidates import find_semantic_candidates

    sem = find_semantic_candidates(segments)
    candidates: list[Candidate] = []
    for found, idx in enumerate(sem.get("controversy", [])):
        seg = segments[idx]
        candidates.append(_window_candidate(seg, found, "controversy", lead=8.0, tail=20.0))
    return candidates


def _generate_dense_window_candidates(segments: list[RichSegment]) -> list[Candidate]:
    if len(segments) < 2:
        return []

    candidates: list[Candidate] = []
    used_ranges: list[tuple[float, float]] = []
    for target_duration in _DENSE_TARGET_DURATIONS:
        proposals = _dense_window_proposals(segments, target_duration)
        for rank, proposal in enumerate(proposals[:_DENSE_PER_DURATION]):
            start, end, anchor, coarse_score, tags = proposal
            if _range_overlaps_dense_candidate(start, end, used_ranges):
                continue
            candidates.append(
                Candidate(
                    id=f"dense_window:{int(target_duration)}:{rank}:{int(start * 100)}",
                    start=round(start, 2),
                    end=round(end, 2),
                    source="dense_window",
                    anchor_time=round(anchor.start + (anchor.end - anchor.start) / 2.0, 2),
                    score_hint=round(coarse_score, 3),
                    tags=tags,
                )
            )
            used_ranges.append((start, end))
    return candidates


def _dense_window_proposals(
    segments: list[RichSegment],
    target_duration: float,
) -> list[tuple[float, float, RichSegment, float, list[str]]]:
    proposals: list[tuple[float, float, RichSegment, float, list[str]]] = []
    step_sec = max(target_duration * 0.35, 4.0)
    last_start = max(segments[-1].end - target_duration, segments[0].start)
    cursor = segments[0].start
    while cursor <= last_start + 0.01:
        end = cursor + target_duration
        window = [seg for seg in segments if seg.end > cursor and seg.start < end]
        if len(window) >= 2:
            anchor = max(window, key=_anchor_signal)
            coarse_score = _dense_coarse_score(window, target_duration)
            if coarse_score >= 2.2:
                tags = sorted({emotion for seg in window for emotion in seg.emotions})
                proposals.append((cursor, end, anchor, coarse_score, tags))
        cursor += step_sec

    proposals.sort(key=lambda item: item[3], reverse=True)
    return proposals


def _dense_coarse_score(window: list[RichSegment], target_duration: float) -> float:
    max_hype = max(seg.hype_score for seg in window)
    mean_hype = sum(seg.hype_score for seg in window) / len(window)
    arc_peak = max(seg.arc_score for seg in window)
    chat_peak = max(seg.chat_spike_score for seg in window)
    motion_peak = max(seg.max_motion_score for seg in window)
    face_peak = max(seg.max_face_reaction for seg in window)
    laughter_hits = sum(
        1 for seg in window if "audio_laughter" in seg.emotions or "laughter" in seg.emotions
    )
    exclamation_hits = sum(1 for seg in window if "exclamation" in seg.emotions)
    speaker_switches = sum(
        1 for left, right in zip(window, window[1:])
        if left.speaker != "UNKNOWN" and right.speaker != "UNKNOWN" and left.speaker != right.speaker
    )
    start_hype = window[0].hype_score
    end_hype = window[-1].hype_score
    duration_bonus = 0.25 if target_duration <= 20.0 else 0.0
    return (
        max_hype * 0.34
        + mean_hype * 0.18
        + arc_peak * 0.26
        + chat_peak * 2.6
        + max(motion_peak, face_peak) * 2.2
        + min(laughter_hits, 3) * 0.45
        + min(exclamation_hits, 3) * 0.3
        + min(speaker_switches, 4) * 0.18
        + max(end_hype - start_hype, 0.0) * 0.2
        + duration_bonus
    )


def _anchor_signal(seg: RichSegment) -> float:
    return (
        seg.hype_score
        + seg.arc_score * 0.8
        + seg.chat_spike_score * 2.0
        + max(seg.max_face_reaction, seg.max_motion_score) * 2.2
    )


def _range_overlaps_dense_candidate(
    start: float,
    end: float,
    ranges: list[tuple[float, float]],
) -> bool:
    for other_start, other_end in ranges:
        overlap = min(end, other_end) - max(start, other_start)
        shorter = min(end - start, other_end - other_start)
        if shorter > 0 and overlap / shorter >= 0.75:
            return True
    return False
