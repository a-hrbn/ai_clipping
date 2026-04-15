from __future__ import annotations

import re

from services.timeline_types import BoundaryFeatures, BoundaryPoint, Candidate
from services.transcriber import RichSegment

_MID_THOUGHT_RE = re.compile(
    r"^(and |but |so |because |then |also |or |if |when |which |who |where |"
    r"he |she |they |it |that |this |those |these |und |aber |also |dann |weil )",
    re.IGNORECASE,
)
_PRONOUN_HEAVY_RE = re.compile(
    r"\b(he|she|they|it|that|this|those|these|er|sie|das|dieser|diese)\b",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")


def build_boundary_features(
    candidate: Candidate,
    segments: list[RichSegment],
    boundaries: list[BoundaryPoint],
) -> list[BoundaryFeatures]:
    results: list[BoundaryFeatures] = []
    for boundary in boundaries:
        results.append(_build_feature(candidate, segments, boundary, role="start"))
        results.append(_build_feature(candidate, segments, boundary, role="end"))
    return results


def _build_feature(
    candidate: Candidate,
    segments: list[RichSegment],
    boundary: BoundaryPoint,
    *,
    role: str,
) -> BoundaryFeatures:
    anchor_distance = abs(boundary.time - candidate.anchor_time)
    left_distance = max(boundary.time - candidate.start, 0.0)
    right_distance = max(candidate.end - boundary.time, 0.0)
    after_window = _segments_between(segments, boundary.time, boundary.time + 3.0)
    before_window = _segments_between(segments, boundary.time - 3.0, boundary.time)
    late_tail = _segments_between(segments, max(candidate.anchor_time, boundary.time - 4.0), boundary.time)

    speaker_transition = 1.0 if boundary.is_speaker_turn else 0.0
    scene_transition = 1.0 if boundary.is_scene_cut else 0.0
    pause_strength = min(max(boundary.pause_before_sec, boundary.pause_after_sec) / 2.5, 1.0)
    lexical_break = _lexical_break(boundary.text_left, boundary.text_right)
    mid_thought_risk = _mid_thought_risk(boundary.text_right, boundary.text_left, role=role)
    context_dependency_risk = min(
        1.0,
        mid_thought_risk * 0.7
        + (0.25 if _PRONOUN_HEAVY_RE.search(boundary.text_right) else 0.0)
        + (0.2 if not boundary.is_sentence_start and role == "start" else 0.0),
    )
    hook_proximity = _hook_signal(after_window)
    reaction_tail = _reaction_signal(late_tail or before_window)
    payoff_capture = _reaction_signal(_segments_between(segments, candidate.anchor_time, boundary.time))

    if role == "start":
        clean_open = min(
            1.0,
            (0.35 if boundary.is_sentence_start else 0.0)
            + (0.20 if speaker_transition else 0.0)
            + (0.15 if scene_transition else 0.0)
            + (0.15 if pause_strength >= 0.4 else 0.0)
            + (0.20 if hook_proximity >= 0.35 else 0.0)
            + (0.15 if lexical_break >= 0.5 else 0.0),
        )
        closure = 0.0
        duration_prior = _duration_prior(candidate.anchor_time - boundary.time)
    else:
        clean_open = 0.0
        closure = min(
            1.0,
            (0.40 if boundary.is_sentence_end else 0.0)
            + (0.15 if speaker_transition else 0.0)
            + (0.10 if scene_transition else 0.0)
            + (0.15 if pause_strength >= 0.35 else 0.0)
            + (0.25 if reaction_tail >= 0.35 else 0.0)
            + (0.15 if lexical_break >= 0.5 else 0.0),
        )
        duration_prior = _duration_prior(boundary.time - candidate.anchor_time)

    return BoundaryFeatures(
        boundary_id=boundary.id,
        candidate_id=candidate.id,
        role=role,
        anchor_distance_sec=round(anchor_distance, 3),
        candidate_left_distance_sec=round(left_distance, 3),
        candidate_right_distance_sec=round(right_distance, 3),
        clean_open_score=round(clean_open, 3),
        closure_score=round(closure, 3),
        mid_thought_risk=round(mid_thought_risk, 3),
        context_dependency_risk=round(context_dependency_risk, 3),
        hook_proximity_score=round(hook_proximity, 3),
        payoff_capture_score=round(payoff_capture, 3),
        speaker_transition_score=round(speaker_transition, 3),
        scene_transition_score=round(scene_transition, 3),
        pause_strength_score=round(pause_strength, 3),
        lexical_break_score=round(lexical_break, 3),
        reaction_tail_score=round(reaction_tail, 3),
        duration_prior_score=round(duration_prior, 3),
    )


def _segments_between(segments: list[RichSegment], start: float, end: float) -> list[RichSegment]:
    return [seg for seg in segments if seg.end > start and seg.start < end]


def _mid_thought_risk(text_right: str, text_left: str, *, role: str) -> float:
    score = 0.0
    right = text_right.strip()
    left = text_left.strip()
    if role == "start":
        if _MID_THOUGHT_RE.match(right):
            score += 0.65
        if right and not right[:1].isupper():
            score += 0.15
        if left and not left.endswith((".", "!", "?")):
            score += 0.15
    else:
        if left and not left.endswith((".", "!", "?")):
            score += 0.55
        if right and _MID_THOUGHT_RE.match(right):
            score += 0.1
    return min(score, 1.0)


def _lexical_break(text_left: str, text_right: str) -> float:
    left_tokens = set(_TOKEN_RE.findall(text_left.lower()))
    right_tokens = set(_TOKEN_RE.findall(text_right.lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)
    return 1.0 - overlap


def _hook_signal(segments: list[RichSegment]) -> float:
    if not segments:
        return 0.0
    score = 0.0
    max_hype = max((seg.hype_score for seg in segments), default=0.0)
    score += min(max_hype / 6.0, 0.45)
    if any("exclamation" in seg.emotions or "question" in seg.emotions for seg in segments):
        score += 0.2
    if any(seg.chat_spike_score >= 0.35 for seg in segments):
        score += 0.15
    if any(seg.max_face_reaction >= 0.35 for seg in segments):
        score += 0.15
    return min(score, 1.0)


def _reaction_signal(segments: list[RichSegment]) -> float:
    if not segments:
        return 0.0
    score = 0.0
    max_hype = max((seg.hype_score for seg in segments), default=0.0)
    score += min(max_hype / 7.0, 0.3)
    if any({"laughter", "audio_laughter", "face_reaction", "exclamation"} & set(seg.emotions) for seg in segments):
        score += 0.35
    if any(seg.chat_spike_score >= 0.35 for seg in segments):
        score += 0.15
    if any(seg.max_face_reaction >= 0.3 for seg in segments):
        score += 0.15
    if any(seg.text.strip().endswith((".", "!", "?")) for seg in segments):
        score += 0.05
    return min(score, 1.0)


def _duration_prior(delta_sec: float) -> float:
    if delta_sec <= 0:
        return 0.0
    if 1.5 <= delta_sec <= 8.0:
        return 1.0
    if delta_sec < 1.5:
        return max(delta_sec / 1.5, 0.0)
    if delta_sec <= 14.0:
        return max(1.0 - ((delta_sec - 8.0) / 8.0), 0.0)
    return 0.0
