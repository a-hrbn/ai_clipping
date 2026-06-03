from __future__ import annotations

from services.timeline_types import Candidate, Chapter
from services.transcriber import RichSegment
from services import chapter_segmenter, clip_strategy

_DENSE_TARGET_DURATIONS = (9.0, 16.0, 28.0, 45.0)
_DENSE_PER_DURATION = 12
_DENSE_INVENTORY_DURATIONS = (12.0, 20.0, 30.0, 45.0, 60.0)
_DENSE_INVENTORY_STRIDE_SEC = 5.0


def generate_candidates(
    segments: list[RichSegment],
    chapters: list[Chapter] | None = None,
    strategy: dict | None = None,
) -> list[Candidate]:
    return generate_semantic_candidates(segments, chapters=chapters, strategy=strategy)


def generate_semantic_candidates(
    segments: list[RichSegment],
    chapters: list[Chapter] | None = None,
    strategy: dict | None = None,
) -> list[Candidate]:
    normalized_strategy = clip_strategy.normalize_strategy(strategy)
    if chapters:
        candidates: list[Candidate] = []
        for chapter in chapters:
            chapter_segments = chapter_segmenter.segments_for_chapter(segments, chapter)
            candidates.extend(_generate_semantic_candidates_for_segments(chapter_segments, normalized_strategy))
        return _filter_candidates(candidates, normalized_strategy)
    return _filter_candidates(_generate_semantic_candidates_for_segments(segments, normalized_strategy), normalized_strategy)


def _generate_semantic_candidates_for_segments(segments: list[RichSegment], strategy: dict | None = None) -> list[Candidate]:
    candidates: list[Candidate] = []
    candidates.extend(_generate_quote_candidates(segments))
    candidates.extend(_generate_chat_velocity_candidates(segments))
    candidates.extend(_generate_tension_build_candidates(segments))
    candidates.extend(_generate_narrative_reveal_candidates(segments))
    candidates.extend(_generate_controversy_candidates(segments))
    candidates.extend(_generate_goal_prompt_candidates(segments, strategy))
    return candidates


def generate_fallback_candidates(
    segments: list[RichSegment],
    chapters: list[Chapter] | None = None,
    strategy: dict | None = None,
) -> list[Candidate]:
    normalized_strategy = clip_strategy.normalize_strategy(strategy)
    if chapters:
        candidates: list[Candidate] = []
        for chapter in chapters:
            chapter_segments = chapter_segmenter.segments_for_chapter(segments, chapter)
            candidates.extend(_generate_fallback_candidates_for_segments(chapter_segments, normalized_strategy))
        return _filter_candidates(candidates, normalized_strategy)
    return _filter_candidates(_generate_fallback_candidates_for_segments(segments, normalized_strategy), normalized_strategy)


def generate_dense_window_candidates(
    segments: list[RichSegment],
    chapters: list[Chapter] | None = None,
    strategy: dict | None = None,
) -> list[Candidate]:
    normalized_strategy = clip_strategy.normalize_strategy(strategy)
    if chapters:
        candidates: list[Candidate] = []
        for chapter in chapters:
            chapter_segments = chapter_segmenter.segments_for_chapter(segments, chapter)
            candidates.extend(_generate_dense_window_candidates(chapter_segments))
        return _filter_candidates(candidates, normalized_strategy)
    return _filter_candidates(_generate_dense_window_candidates(segments), normalized_strategy)


def generate_dense_candidate_inventory(
    segments: list[RichSegment],
    chapters: list[Chapter] | None = None,
    strategy: dict | None = None,
    *,
    durations: tuple[float, ...] | list[float] | None = None,
    stride_sec: float = _DENSE_INVENTORY_STRIDE_SEC,
    max_candidates: int = 0,
) -> list[Candidate]:
    """Generate broad overlapping whole-VOD windows before heuristic caps."""
    normalized_strategy = clip_strategy.normalize_strategy(strategy)
    clean_durations = tuple(
        sorted({float(duration) for duration in (durations or _DENSE_INVENTORY_DURATIONS) if float(duration) > 0})
    )
    if len(segments) < 1 or not clean_durations:
        return []

    stride = max(float(stride_sec or _DENSE_INVENTORY_STRIDE_SEC), 1.0)
    ranges = _dense_inventory_ranges(segments, chapters, normalized_strategy)
    candidates: list[Candidate] = []
    seen_ids: set[str] = set()
    for range_index, (range_start, range_end) in enumerate(ranges):
        for duration in clean_durations:
            latest_start = max(range_start, range_end - duration)
            start = range_start
            window_index = 0
            while start <= latest_start + 0.001:
                end = min(range_end, start + duration)
                if end - start >= min(duration * 0.55, 8.0):
                    candidate = _dense_inventory_candidate(
                        segments,
                        start=start,
                        end=end,
                        duration=duration,
                        range_index=range_index,
                        window_index=window_index,
                    )
                    if candidate.id not in seen_ids and clip_strategy.candidate_in_time_range(
                        candidate.start,
                        candidate.end,
                        normalized_strategy,
                    ):
                        candidates.append(candidate)
                        seen_ids.add(candidate.id)
                start += stride
                window_index += 1

    candidates.sort(key=lambda candidate: (-candidate.score_hint, candidate.start, candidate.end))
    if max_candidates and max_candidates > 0 and len(candidates) > max_candidates:
        candidates = candidates[:max_candidates]
    return sorted(candidates, key=lambda candidate: (candidate.start, candidate.end, candidate.anchor_time))


def _generate_fallback_candidates_for_segments(segments: list[RichSegment], strategy: dict | None = None) -> list[Candidate]:
    candidates: list[Candidate] = []
    candidates.extend(_generate_hype_peak_candidates(segments))
    candidates.extend(_generate_chat_sync_candidates(segments))
    candidates.extend(_generate_reaction_candidates(segments))
    candidates.extend(_generate_clean_reaction_candidates(segments))
    candidates.extend(_generate_dialogue_candidates(segments))
    candidates.extend(_generate_hot_take_candidates(segments))
    candidates.extend(_generate_story_payoff_candidates(segments))
    candidates.extend(_generate_visual_reveal_candidates(segments))
    candidates.extend(_generate_audio_burst_candidates(segments))
    candidates.extend(_generate_silence_break_candidates(segments))
    candidates.extend(_generate_dense_window_candidates(segments))
    return candidates


def merge_prioritized_candidates(
    primary: list[Candidate],
    supplemental: list[Candidate],
    *,
    supplemental_source_cap: int = 4,
    source_cap_overrides: dict[str, int] | None = None,
    preserve_supplemental_ids: set[str] | None = None,
    preserve_context_variants: bool = False,
    max_context_variants_per_region: int = 2,
    strategy: dict | None = None,
) -> list[Candidate]:
    normalized_strategy = clip_strategy.normalize_strategy(strategy)
    merged: list[Candidate] = []
    supplemental_counts: dict[str, int] = {}
    source_caps = clip_strategy.source_caps(normalized_strategy)
    preserved_ids = {str(candidate_id) for candidate_id in (preserve_supplemental_ids or set())}
    primary_ids: set[str] = set()
    for source, cap in (source_cap_overrides or {}).items():
        source_caps[str(source)] = max(int(cap), 0)

    for candidate in primary:
        if not clip_strategy.candidate_in_time_range(candidate.start, candidate.end, normalized_strategy):
            continue
        if _is_duplicate_candidate(
            candidate,
            merged,
            preserve_context_variants=preserve_context_variants,
            max_context_variants_per_region=max_context_variants_per_region,
        ):
            continue
        merged.append(candidate)
        primary_ids.add(candidate.id)

    for candidate in supplemental:
        if not clip_strategy.candidate_in_time_range(candidate.start, candidate.end, normalized_strategy):
            continue
        is_preserved = candidate.id in preserved_ids
        effective_cap = source_caps.get(candidate.source, supplemental_source_cap)
        if not is_preserved and supplemental_counts.get(candidate.source, 0) >= effective_cap:
            continue
        if (
            not is_preserved
            and _is_duplicate_supplemental_candidate(
                candidate,
                merged,
                primary_ids,
                preserve_context_variants=preserve_context_variants,
                max_context_variants_per_region=max_context_variants_per_region,
            )
        ):
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


def _generate_chat_velocity_candidates(segments: list[RichSegment]) -> list[Candidate]:
    scored = []
    for seg in segments:
        score = _chat_velocity_candidate_score(seg)
        if score >= 0.38:
            scored.append((score, seg))
    peaks = [seg for _score, seg in sorted(scored, key=lambda item: item[0], reverse=True)]
    candidates: list[Candidate] = []
    for idx, seg in enumerate(peaks[:16]):
        candidates.append(_window_candidate(seg, idx, "chat_velocity", lead=7.0, tail=18.0))
    return candidates


def _chat_velocity_candidate_score(seg: RichSegment) -> float:
    derivative_score = min(max(float(seg.chat_rate_derivative or 0.0), 0.0) / 2.0, 1.0)
    return max(
        float(seg.chat_spike_score or 0.0),
        float(seg.chat_ema_crossing_score or 0.0),
        derivative_score,
    )


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


def _generate_clean_reaction_candidates(segments: list[RichSegment]) -> list[Candidate]:
    candidates: list[Candidate] = []
    ranked = sorted(
        [
            seg for seg in segments
            if (
                seg.max_face_reaction >= 0.45
                or seg.chat_spike_score >= 0.5
                or "audio_laughter" in seg.emotions
                or "laughter" in seg.emotions
            )
        ],
        key=lambda seg: (
            max(seg.max_face_reaction, seg.chat_spike_score)
            + seg.hype_score * 0.25
            + (0.2 if "audio_laughter" in seg.emotions or "laughter" in seg.emotions else 0.0)
        ),
        reverse=True,
    )
    for idx, seg in enumerate(ranked[:10]):
        candidates.append(_window_candidate(seg, idx, "clean_reaction", lead=5.5, tail=12.0))
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


def _generate_hot_take_candidates(segments: list[RichSegment]) -> list[Candidate]:
    candidates: list[Candidate] = []
    patterns = (
        "i think", "i'm telling", "listen", "bro", "that's why", "truth is",
        "ich finde", "ich sag", "das ding ist", "ehrlich", "eigentlich",
    )
    found = 0
    for seg in segments:
        if found >= 10:
            break
        text = seg.text.strip().lower()
        words = len(text.split())
        if words < 5 or words > 28:
            continue
        if not any(pattern in text for pattern in patterns) and not text.endswith(("!", "?")):
            continue
        if seg.hype_score < 1.5 and seg.arc_score < 2.0 and seg.chat_spike_score < 0.3:
            continue
        candidates.append(_window_candidate(seg, found, "hot_take", lead=6.0, tail=13.0))
        found += 1
    return candidates


def _generate_story_payoff_candidates(segments: list[RichSegment]) -> list[Candidate]:
    candidates: list[Candidate] = []
    for idx in range(1, len(segments)):
        if len(candidates) >= 10:
            break
        seg = segments[idx]
        history = segments[max(0, idx - 3):idx]
        if not history:
            continue
        setup_hype = sum(item.hype_score for item in history) / len(history)
        payoff_signal = (
            seg.hype_score
            + seg.arc_score
            + seg.chat_spike_score * 2.2
            + max(seg.max_face_reaction, seg.max_motion_score) * 2.0
        )
        if payoff_signal < 5.2:
            continue
        if seg.hype_score - setup_hype < 1.8 and seg.arc_score < 3.0:
            continue
        candidates.append(_window_candidate(seg, idx, "story_payoff", lead=12.0, tail=18.0))
    return candidates


def _generate_visual_reveal_candidates(segments: list[RichSegment]) -> list[Candidate]:
    ranked = sorted(
        [
            seg for seg in segments
            if seg.has_scene_change or seg.max_motion_score >= 0.5 or seg.visual_activity >= 2.4
        ],
        key=lambda seg: (
            (0.4 if seg.has_scene_change else 0.0)
            + seg.max_motion_score * 0.8
            + seg.max_face_reaction * 0.7
            + min(seg.visual_activity / 6.0, 1.0) * 0.5
        ),
        reverse=True,
    )
    return [
        _window_candidate(seg, idx, "visual_reveal", lead=6.0, tail=12.0)
        for idx, seg in enumerate(ranked[:10])
    ]


def _generate_audio_burst_candidates(segments: list[RichSegment]) -> list[Candidate]:
    ranked = sorted(
        [
            seg for seg in segments
            if seg.laughter_prob >= 0.35 or seg.max_onset_rate >= 3.0 or seg.max_pitch_hz >= 280.0
        ],
        key=lambda seg: (
            seg.laughter_prob * 1.2
            + min(seg.max_onset_rate / 8.0, 1.0) * 0.8
            + min(seg.max_pitch_hz / 600.0, 1.0) * 0.5
            + seg.hype_score * 0.15
        ),
        reverse=True,
    )
    return [
        _window_candidate(seg, idx, "audio_burst", lead=5.0, tail=13.0)
        for idx, seg in enumerate(ranked[:10])
    ]


def _generate_silence_break_candidates(segments: list[RichSegment]) -> list[Candidate]:
    candidates: list[Candidate] = []
    for idx in range(1, len(segments)):
        prev = segments[idx - 1]
        current = segments[idx]
        if len(candidates) >= 8:
            break
        if prev.mean_rms_db <= 0.0 and prev.hype_score > 1.5:
            continue
        if current.hype_score - prev.hype_score < 2.2:
            continue
        if max(current.max_motion_score, current.max_face_reaction, current.chat_spike_score) < 0.35:
            continue
        candidates.append(_window_candidate(current, idx, "silence_break", lead=4.0, tail=12.0))
    return candidates


def _generate_goal_prompt_candidates(segments: list[RichSegment], strategy: dict | None) -> list[Candidate]:
    normalized = clip_strategy.normalize_strategy(strategy)
    if not normalized["keywords"]:
        return []
    ranked: list[tuple[float, RichSegment]] = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        relevance = clip_strategy.text_relevance_score(text, normalized)
        if relevance <= 0.0:
            continue
        score = relevance + seg.hype_score * 0.08 + seg.chat_spike_score * 0.4 + seg.max_face_reaction * 0.35
        ranked.append((score, seg))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [
        _window_candidate(seg, idx, "goal_prompt", lead=6.0, tail=14.0)
        for idx, (_score, seg) in enumerate(ranked[:10])
    ]


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
    lead, tail = _adaptive_window(anchor, lead, tail)
    return Candidate(
        id=f"{source}:{idx}:{int(anchor.start * 100)}",
        start=max(0.0, anchor.start - lead),
        end=max(anchor.end, anchor.start + 1.0) + tail,
        source=source,
        anchor_time=(anchor.start + anchor.end) / 2.0,
        score_hint=round(anchor.hype_score + anchor.arc_score + anchor.chat_spike_score * 2.0, 3),
        tags=sorted(set(anchor.emotions)),
    )


def _adaptive_window(anchor: RichSegment, lead: float, tail: float) -> tuple[float, float]:
    hook_urgency = min(
        1.0,
        max(
            anchor.chat_spike_score,
            anchor.max_face_reaction,
            min(anchor.hype_score / 6.0, 1.0),
            1.0 if "exclamation" in anchor.emotions else 0.0,
            0.85 if anchor.text.strip().endswith(("!", "?")) else 0.0,
        ),
    )
    story_weight = min(max(anchor.arc_score / 5.0, 0.0), 1.0)
    adaptive_lead = lead * (1.15 - 0.45 * hook_urgency) + story_weight * 2.5
    adaptive_tail = (
        tail * (1.05 + 0.25 * story_weight - 0.15 * hook_urgency)
        + anchor.chat_spike_score * 1.5
        + anchor.max_face_reaction * 1.5
    )
    return round(max(3.0, adaptive_lead), 2), round(max(6.0, adaptive_tail), 2)


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


def _is_duplicate_candidate(
    candidate: Candidate,
    existing: list[Candidate],
    *,
    preserve_context_variants: bool = False,
    max_context_variants_per_region: int = 2,
) -> bool:
    for other in existing:
        if abs(candidate.anchor_time - other.anchor_time) <= 4.0:
            if preserve_context_variants and _context_variant_should_survive(
                candidate,
                other,
                existing,
                max_context_variants_per_region=max_context_variants_per_region,
            ):
                continue
            return True
        overlap = min(candidate.end, other.end) - max(candidate.start, other.start)
        if overlap <= 0:
            continue
        shorter = min(candidate.end - candidate.start, other.end - other.start)
        if shorter > 0 and overlap / shorter >= 0.7:
            if preserve_context_variants and _context_variant_should_survive(
                candidate,
                other,
                existing,
                max_context_variants_per_region=max_context_variants_per_region,
            ):
                continue
            return True
    return False


def _is_duplicate_supplemental_candidate(
    candidate: Candidate,
    existing: list[Candidate],
    primary_ids: set[str],
    *,
    preserve_context_variants: bool = False,
    max_context_variants_per_region: int = 2,
) -> bool:
    for other in existing:
        if not _candidate_windows_overlap_like_duplicates(candidate, other):
            continue
        if preserve_context_variants and _context_variant_should_survive(
            candidate,
            other,
            existing,
            max_context_variants_per_region=max_context_variants_per_region,
        ):
            continue
        if other.id in primary_ids:
            return True
        if candidate.source == other.source:
            return True
        if _candidate_windows_are_effectively_identical(candidate, other):
            return True
    return False


def _context_variant_should_survive(
    candidate: Candidate,
    other: Candidate,
    existing: list[Candidate],
    *,
    max_context_variants_per_region: int,
) -> bool:
    if max_context_variants_per_region <= 0:
        return False
    if _candidate_windows_are_effectively_identical(candidate, other):
        return False

    overlapping = [item for item in existing if _candidate_windows_overlap_like_duplicates(candidate, item)]
    if len(overlapping) >= max_context_variants_per_region:
        best_existing = max((_context_variant_pre_score(item) for item in overlapping), default=0.0)
        return _context_variant_pre_score(candidate) >= best_existing + 0.14

    candidate_score = _context_variant_pre_score(candidate)
    other_score = _context_variant_pre_score(other)
    if candidate_score < 0.56:
        return False
    if candidate_score >= other_score + 0.08:
        return True

    candidate_duration = candidate.end - candidate.start
    other_duration = other.end - other.start
    candidate_lead = candidate.anchor_time - candidate.start
    other_lead = other.anchor_time - other.start
    return (
        candidate.source != other.source
        and candidate_duration >= min(other_duration + 8.0, other_duration * 1.45)
        and candidate_lead >= other_lead + 4.0
    )


def _context_variant_pre_score(candidate: Candidate) -> float:
    duration = max(candidate.end - candidate.start, 0.0)
    lead = max(candidate.anchor_time - candidate.start, 0.0)
    source_prior = {
        "quote": 0.84,
        "narrative_reveal": 0.82,
        "tension_build": 0.78,
        "controversy": 0.74,
        "story_payoff": 0.78,
        "goal_prompt": 0.74,
        "chat_velocity": 0.78,
        "dense_window": 0.52,
        "dense_inventory": 0.44,
    }.get(candidate.source, 0.58)
    duration_score = min(duration / 28.0, 1.0) * 0.7 + (1.0 - min(abs(duration - 32.0) / 55.0, 1.0)) * 0.3
    setup_score = min(lead / 10.0, 1.0)
    signal_score = min(max(candidate.score_hint, 0.0) / 7.0, 1.0)
    tag_score = 0.0
    tags = set(candidate.tags or [])
    if tags.intersection({"question", "exclamation", "audio_laughter", "laughter"}):
        tag_score = 0.15
    return min(
        1.0,
        source_prior * 0.34
        + duration_score * 0.22
        + setup_score * 0.22
        + signal_score * 0.14
        + tag_score,
    )


def _candidate_windows_overlap_like_duplicates(candidate: Candidate, other: Candidate) -> bool:
    if abs(candidate.anchor_time - other.anchor_time) <= 4.0:
        return True
    overlap = min(candidate.end, other.end) - max(candidate.start, other.start)
    if overlap <= 0:
        return False
    shorter = min(candidate.end - candidate.start, other.end - other.start)
    return shorter > 0 and overlap / shorter >= 0.7


def _candidate_windows_are_effectively_identical(candidate: Candidate, other: Candidate) -> bool:
    overlap = min(candidate.end, other.end) - max(candidate.start, other.start)
    if overlap <= 0:
        return False
    candidate_duration = max(candidate.end - candidate.start, 0.001)
    other_duration = max(other.end - other.start, 0.001)
    shorter = min(candidate_duration, other_duration)
    duration_delta = abs(candidate_duration - other_duration)
    return (
        abs(candidate.anchor_time - other.anchor_time) <= 2.0
        and shorter > 0
        and overlap / shorter >= 0.92
        and duration_delta <= 3.0
    )


def _filter_candidates(candidates: list[Candidate], strategy: dict | None) -> list[Candidate]:
    normalized = clip_strategy.normalize_strategy(strategy)
    filtered = [
        candidate
        for candidate in candidates
        if clip_strategy.candidate_in_time_range(candidate.start, candidate.end, normalized)
    ]
    return sorted(filtered, key=lambda candidate: (candidate.start, candidate.anchor_time))


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


def _dense_inventory_ranges(
    segments: list[RichSegment],
    chapters: list[Chapter] | None,
    strategy: dict,
) -> list[tuple[float, float]]:
    time_start = float(strategy.get("time_range_start_sec", 0.0) or 0.0)
    time_end = strategy.get("time_range_end_sec")
    ranges: list[tuple[float, float]] = []
    if chapters:
        for chapter in chapters:
            start = max(float(chapter.start), time_start)
            end = float(chapter.end)
            if time_end is not None:
                end = min(end, float(time_end))
            if end > start:
                ranges.append((start, end))
    else:
        start = max(float(segments[0].start), time_start)
        end = float(segments[-1].end)
        if time_end is not None:
            end = min(end, float(time_end))
        if end > start:
            ranges.append((start, end))
    return ranges


def _dense_inventory_candidate(
    segments: list[RichSegment],
    *,
    start: float,
    end: float,
    duration: float,
    range_index: int,
    window_index: int,
) -> Candidate:
    window = [seg for seg in segments if seg.end > start and seg.start < end]
    if window:
        anchor = max(window, key=_anchor_signal)
        anchor_time = anchor.start + (anchor.end - anchor.start) / 2.0
        score_hint = _dense_coarse_score(window, duration)
        tags = sorted({emotion for seg in window for emotion in seg.emotions})
    else:
        anchor_time = start + (end - start) / 2.0
        score_hint = 0.0
        tags = []
    return Candidate(
        id=f"dense_inventory:{int(round(duration))}:{range_index}:{window_index}:{int(start * 100)}",
        start=round(max(0.0, start), 3),
        end=round(max(end, start + 1.0), 3),
        source="dense_inventory",
        anchor_time=round(anchor_time, 3),
        score_hint=round(score_hint, 3),
        tags=tags,
    )


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
