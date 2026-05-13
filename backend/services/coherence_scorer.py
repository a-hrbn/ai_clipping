from __future__ import annotations

import re

from services.transcriber import RichSegment

_MID_THOUGHT_OPENERS = {
    "and", "but", "so", "because", "then", "that", "this", "it", "he", "she",
    "they", "we", "you", "also", "like", "yeah", "nah", "no", "okay",
}
_CONTEXT_REFERENCES = {
    "that", "this", "it", "there", "here", "him", "her", "them", "those",
    "these", "again", "before", "earlier", "same",
}
_PAYOFF_MARKERS = {
    "no way", "what", "why", "how", "finally", "actually", "because",
    "wait", "look", "watch", "bro", "insane", "crazy", "happened",
}


def score_window(window: list[RichSegment]) -> dict[str, float]:
    text = " ".join(seg.text.strip() for seg in window if seg.text.strip())
    if not text:
        return {
            "coherence_score": 0.0,
            "context_dependency_score": 1.0,
            "weak_resolution_penalty": 1.0,
        }

    tokens = _tokens(text)
    opening_tokens = _tokens(" ".join(seg.text.strip() for seg in window[:2]))
    closing_text = " ".join(seg.text.strip() for seg in window[-2:])
    starts_mid_thought = bool(opening_tokens and opening_tokens[0] in _MID_THOUGHT_OPENERS)
    has_question_or_exclamation = any(mark in text for mark in ("?", "!"))
    has_payoff = any(marker in text.lower() for marker in _PAYOFF_MARKERS) or has_question_or_exclamation
    context_refs = sum(1 for token in tokens if token in _CONTEXT_REFERENCES)
    context_ref_ratio = context_refs / max(len(tokens), 1)
    closes_cleanly = closing_text.rstrip().endswith((".", "!", "?")) or any(
        token in closing_text.lower() for token in ("that's it", "there it is", "let's go", "gg", "done")
    )
    speaker_count = len({seg.speaker for seg in window if str(seg.speaker or "UNKNOWN") != "UNKNOWN"})

    context_dependency = min(
        1.0,
        context_ref_ratio * 3.5
        + (0.28 if starts_mid_thought else 0.0)
        + (0.12 if speaker_count > 2 else 0.0),
    )
    weak_resolution = min(
        1.0,
        (0.35 if not closes_cleanly else 0.0)
        + (0.35 if not has_payoff else 0.0)
        + (0.15 if len(tokens) < 8 else 0.0),
    )
    coherence = max(
        0.0,
        min(
            1.0,
            0.55
            + (0.16 if has_payoff else -0.18)
            + (0.14 if closes_cleanly else -0.16)
            - context_dependency * 0.38
            - weak_resolution * 0.24
            - (0.12 if starts_mid_thought else 0.0),
        ),
    )
    return {
        "coherence_score": round(coherence, 3),
        "context_dependency_score": round(context_dependency, 3),
        "weak_resolution_penalty": round(weak_resolution, 3),
    }


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())
