from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from services.transcriber import RichSegment

_TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")

_RAGE_TERMS = {
    "fuck",
    "fucking",
    "bro",
    "trash",
    "cheating",
    "cheater",
    "stupid",
    "bad",
    "mad",
    "rage",
    "tilted",
}
_CLUTCH_TERMS = {
    "clutch",
    "win",
    "won",
    "1v1",
    "1v2",
    "last",
    "round",
    "overtime",
    "champion",
    "kill",
    "dead",
}
_FUNNY_REACTION_TERMS = {
    "what",
    "wait",
    "no",
    "way",
    "crazy",
    "insane",
    "laugh",
    "funny",
    "bro",
    "chat",
}
_TRASH_TALK_TERMS = {
    "bad",
    "trash",
    "dog",
    "free",
    "owned",
    "washed",
    "terrible",
    "garbage",
    "easy",
}
_CHAT_SHOCK_TERMS = {
    "chat",
    "look",
    "wait",
    "what",
    "no",
    "way",
    "guys",
}
_RANKED_TENSION_TERMS = {
    "rank",
    "ranked",
    "elo",
    "champ",
    "diamond",
    "plat",
    "match",
    "round",
    "game",
}
_FAILED_SETUP_TERMS = {
    "failed",
    "missed",
    "threw",
    "throw",
    "sold",
    "lost",
    "died",
    "wrong",
}
_PERSONA_TERMS = {
    "jynxzi",
    "junko",
    "chat",
    "stream",
    "fanmail",
    "daddy",
    "brother",
}
_JYNXZI_CONTEXT_SCORE_KEYS = {
    "rage",
    "clutch",
    "funny_reaction",
    "trash_talk_payoff",
    "chat_shock",
    "ranked_tension",
    "failed_setup",
    "streamer_persona_beat",
}


@dataclass(frozen=True)
class ClipTaxonomy:
    clip_type: str
    scores: dict[str, float]


def classify_segments(
    segments: Sequence[RichSegment],
    *,
    source: str = "",
    fallback_type: str = "general",
) -> ClipTaxonomy:
    """Classify a candidate window into a Jynxzi-specific viral clip bucket."""

    text = " ".join(str(seg.text or "") for seg in segments)
    tokens = _TOKEN_RE.findall(text.lower())
    token_set = set(tokens)
    source = str(source or "").lower()
    max_hype = max((float(getattr(seg, "hype_score", 0.0) or 0.0) for seg in segments), default=0.0)
    max_chat = max((float(getattr(seg, "chat_spike_score", 0.0) or 0.0) for seg in segments), default=0.0)
    max_motion = max((float(getattr(seg, "max_motion_score", 0.0) or 0.0) for seg in segments), default=0.0)
    max_face = max((float(getattr(seg, "max_face_reaction", 0.0) or 0.0) for seg in segments), default=0.0)
    laughter = sum(1 for seg in segments if {"laughter", "audio_laughter"} & set(getattr(seg, "emotions", []) or []))
    exclamation = sum(1 for seg in segments if "!" in str(getattr(seg, "text", "") or "") or "exclamation" in set(getattr(seg, "emotions", []) or []))
    question = sum(1 for seg in segments if "?" in str(getattr(seg, "text", "") or "") or "question" in set(getattr(seg, "emotions", []) or []))

    scores = _score_text_context(
        token_set,
        source=source,
        max_hype=max_hype,
        max_chat=max_chat,
        max_motion=max_motion,
        max_face=max_face,
        laughter=laughter,
        exclamation=exclamation,
        question=question,
    )
    return _taxonomy_from_scores(scores, fallback_type=fallback_type)


def classify_text(
    text: str,
    *,
    source: str = "",
    fallback_type: str = "general",
) -> ClipTaxonomy:
    """Classify free text such as short titles or matched transcript excerpts."""

    tokens = _TOKEN_RE.findall(str(text or "").lower())
    scores = _score_text_context(set(tokens), source=source)
    return _taxonomy_from_scores(scores, fallback_type=fallback_type)


def score_keys() -> set[str]:
    return set(_JYNXZI_CONTEXT_SCORE_KEYS)


def _taxonomy_from_scores(scores: dict[str, float], *, fallback_type: str) -> ClipTaxonomy:
    clip_type, score = max(scores.items(), key=lambda item: (item[1], item[0]))
    if score <= 0.18:
        clip_type = fallback_type
    return ClipTaxonomy(
        clip_type=clip_type,
        scores={key: round(value, 6) for key, value in scores.items()},
    )


def best_type_from_feature_row(row: Mapping[str, object]) -> str:
    value = str(row.get("clip_type") or row.get("jynxzi_clip_type") or "").strip()
    if value:
        return value
    scores = {
        key[:-6]: _as_float(value)
        for key, value in row.items()
        if str(key).endswith("_score")
        and str(key)[:-6] in _JYNXZI_CONTEXT_SCORE_KEYS
    }
    if not scores:
        return "general"
    return max(scores.items(), key=lambda item: (item[1], item[0]))[0]


def _term_score(tokens: set[str], terms: set[str]) -> float:
    if not tokens:
        return 0.0
    hits = sum(1 for term in terms if term in tokens)
    return min(hits / 3.0, 1.0)


def _score_text_context(
    token_set: set[str],
    *,
    source: str = "",
    max_hype: float = 0.0,
    max_chat: float = 0.0,
    max_motion: float = 0.0,
    max_face: float = 0.0,
    laughter: int = 0,
    exclamation: int = 0,
    question: int = 0,
) -> dict[str, float]:
    hype = min(max_hype / 6.0, 1.0)
    return {
        "rage": _clamp01(_term_score(token_set, _RAGE_TERMS) * 0.45 + hype * 0.35 + exclamation * 0.08),
        "clutch": _clamp01(_term_score(token_set, _CLUTCH_TERMS) * 0.5 + max_motion * 0.2 + hype * 0.2 + (0.2 if "game_event" in source else 0.0)),
        "funny_reaction": _clamp01(_term_score(token_set, _FUNNY_REACTION_TERMS) * 0.35 + laughter * 0.18 + max_face * 0.25 + question * 0.08),
        "trash_talk_payoff": _clamp01(_term_score(token_set, _TRASH_TALK_TERMS) * 0.55 + exclamation * 0.08 + hype * 0.2),
        "chat_shock": _clamp01(_term_score(token_set, _CHAT_SHOCK_TERMS) * 0.35 + max_chat * 0.45 + question * 0.08 + exclamation * 0.05),
        "ranked_tension": _clamp01(_term_score(token_set, _RANKED_TENSION_TERMS) * 0.45 + hype * 0.25 + max_motion * 0.15),
        "failed_setup": _clamp01(_term_score(token_set, _FAILED_SETUP_TERMS) * 0.55 + exclamation * 0.08 + max_face * 0.18),
        "streamer_persona_beat": _clamp01(_term_score(token_set, _PERSONA_TERMS) * 0.45 + max_chat * 0.2 + laughter * 0.12),
    }


def _as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))
