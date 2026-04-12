from __future__ import annotations

import json
import logging
import re

import ollama

from config import settings
from services.transcriber import RichSegment, format_transcript

logger = logging.getLogger(__name__)

_SURPRISE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\b(no way|what|wtf|unbelievable|crazy|insane|niemals|krass)\b",
        r"\b(i can'?t believe|ich kann'?s nicht glauben)\b",
    ]
]
_CONTRADICTION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\b(but|however|eigentlich|aber|doch|trotzdem)\b",
        r"\b(i thought .* but|ich dachte .* aber)\b",
    ]
]
_QUOTE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"^[A-ZÄÖÜ].{18,}[.!?]$",
        r"\b(remember|listen|truth is|the thing is|merke dir|das ding ist)\b",
    ]
]

# ---------------------------------------------------------------------------
# Regex baseline (fast, language-limited)
# ---------------------------------------------------------------------------

def quoteability_score(window: list[RichSegment]) -> float:
    if not window:
        return 0.0
    best = 0.0
    for seg in window:
        text = seg.text.strip()
        score = 0.0
        if any(pattern.search(text) for pattern in _QUOTE_PATTERNS):
            score += 0.6
        words = len(text.split())
        if 6 <= words <= 20:
            score += 0.2
        if seg.speaker != "UNKNOWN":
            score += 0.1
        if text.endswith(("!", "?", ".")):
            score += 0.1
        best = max(best, min(score, 1.0))
    return round(best, 3)


def surprise_score(window: list[RichSegment]) -> float:
    if not window:
        return 0.0
    score = 0.0
    for seg in window:
        text = seg.text.strip()
        if any(pattern.search(text) for pattern in _SURPRISE_PATTERNS):
            score += 0.35
        if "exclamation" in seg.emotions:
            score += 0.2
        if seg.chat_spike_score >= 0.5:
            score += 0.15
        if seg.max_face_reaction >= 0.35:
            score += 0.15
    return round(min(score, 1.0), 3)


def contradiction_score(window: list[RichSegment]) -> float:
    if not window:
        return 0.0
    score = 0.0
    speakers = {seg.speaker for seg in window if seg.speaker != "UNKNOWN"}
    for seg in window:
        text = seg.text.strip()
        if any(pattern.search(text) for pattern in _CONTRADICTION_PATTERNS):
            score += 0.3
    if len(speakers) >= 2:
        score += 0.2
    return round(min(score, 1.0), 3)


def standalone_score(window: list[RichSegment], hook_delay_sec: float) -> float:
    if not window:
        return 0.0
    first = window[0].text.strip()
    score = 0.3 if first and not first.lower().startswith(("and ", "but ", "so ", "weil ", "aber ")) else 0.0
    if hook_delay_sec <= 2.0:
        score += 0.3
    if window[-1].text.strip().endswith((".", "!", "?")):
        score += 0.2
    if len(window) >= 2:
        score += 0.1
    return round(min(score, 1.0), 3)


def semantic_density(window: list[RichSegment]) -> float:
    if not window:
        return 0.0
    text = " ".join(seg.text.strip() for seg in window if seg.text.strip())
    words = len(text.split())
    if words == 0:
        return 0.0
    marker_hits = (
        sum(1 for pattern in _SURPRISE_PATTERNS if pattern.search(text))
        + sum(1 for pattern in _CONTRADICTION_PATTERNS if pattern.search(text))
        + sum(1 for pattern in _QUOTE_PATTERNS if pattern.search(text))
    )
    return round(min((marker_hits * 8) / words, 1.0), 3)


# ---------------------------------------------------------------------------
# LLM-backed scoring (optional second pass, language-agnostic)
# ---------------------------------------------------------------------------

# The transcript passed to the LLM uses format_transcript() so the model sees
# all enriched signals inline: [HYPE:X] [CHAT:X] [VIS:X] [ARC:X].
# The LLM returns five viral-focused dimensions which are then remapped to the
# internal field names used throughout the pipeline (quoteability, surprise,
# contradiction, standalone, density) before blending with the regex baseline.
#
# LLM key → internal key:
#   virality  → quoteability   (replaces the narrow "quoteable line" concept)
#   surprise  → surprise       (unchanged)
#   conflict  → contradiction  (broadens "stance reversal" to include drama/tension)
#   standalone→ standalone     (unchanged)
#   hook      → density        (replaces "info density" with "opening grabs attention")

_LLM_PROMPT = """\
You are evaluating a stream or video clip for viral potential on TikTok and YouTube Shorts.
The content may be any format: gaming, Just Chatting, IRL, podcast, interview, comedy, etc.

The transcript includes enriched signals per line:
  [HYPE:X]  — composite excitement score (laughter + energy + pitch spike + chat)
  [CHAT:X]  — chat message-rate spike, 0=flat 1=huge spike (crowd-sourced engagement signal)
  [VIS:X]   — visual activity level (motion, face reactions)
  [ARC:X]   — emotional arc quality of this window (build-up → peak → payoff)

Score these five dimensions, each a float 0.0–1.0:
  virality  — overall: would a viewer immediately share this clip with a friend?
              High = genuine emotion, insight, humor, drama, shock, or "did he just say that?" moment.
  surprise  — how unexpected the outcome, reaction, or statement is; reversals, reveals, things going wrong.
  conflict  — tension, confrontation, heated argument, emotional stakes, or chaos (any content type).
  standalone— clip makes complete sense without watching the stream beforehand.
  hook      — the clip opens with something that grabs attention in the first 2 seconds;
              no slow lead-in, no mid-sentence start, immediate payoff or intrigue.

Return ONLY valid JSON, no markdown, no explanation:
{{"virality": X, "surprise": X, "conflict": X, "standalone": X, "hook": X}}

Transcript:
{text}

JSON:"""

# Maps LLM output keys → internal pipeline keys (CandidateFeatures field names).
# "hook" now correctly maps to hook_strength (was wrongly mapped to density before).
# semantic_density is regex-only — the LLM hook dimension is a better signal for
# "does the opening grab attention?" than information density.
_LLM_KEY_MAP = {
    "virality":   "quoteability",
    "surprise":   "surprise",
    "conflict":   "contradiction",
    "standalone": "standalone",
    "hook":       "hook_strength",
}


def _call_llm(text: str, provider: str | None = None) -> dict[str, float] | None:
    """Score the five viral dimensions with an optional provider preference."""
    preferred = (provider or "auto").strip().lower()
    if preferred == "ollama":
        return _call_ollama_semantic(text)
    if preferred == "groq":
        if settings.groq_api_key:
            return _call_groq(text)
        return _call_ollama_semantic(text)
    if settings.groq_api_key:
        return _call_groq(text)
    return _call_ollama_semantic(text)


def _call_groq(text: str) -> dict[str, float] | None:
    """Ask Groq (Llama 3.3 70B) to score the five viral dimensions."""
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": _LLM_PROMPT.format(text=text)}],
            temperature=0.0,
            max_tokens=80,
        )
        content = response.choices[0].message.content.strip()
        match = re.search(r"\{[^}]+\}", content, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return {k: max(0.0, min(float(v), 1.0)) for k, v in data.items() if isinstance(v, (int, float))}
    except Exception as exc:
        logger.warning("Groq semantic scoring failed: %s — falling back to Ollama", exc)
        return _call_ollama_semantic(text)
    return None


def _call_ollama_semantic(text: str) -> dict[str, float] | None:
    """Ask Ollama to score the five viral dimensions. Returns None on any failure."""
    try:
        client = ollama.Client(host=settings.ollama_host)
        response = client.chat(
            model=settings.ollama_model,
            messages=[{"role": "user", "content": _LLM_PROMPT.format(text=text)}],
        )
        content = response["message"]["content"].strip()
        match = re.search(r"\{[^}]+\}", content, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return {k: max(0.0, min(float(v), 1.0)) for k, v in data.items() if isinstance(v, (int, float))}
    except Exception as exc:
        logger.debug("Ollama semantic scoring failed: %s", exc)
    return None


def score_window(
    window: list[RichSegment],
    hook_delay_sec: float,
    *,
    use_llm: bool = True,
    llm_provider: str | None = None,
) -> dict[str, float]:
    """
    Compute semantic scores for a candidate window.

    Returns keys: quoteability, surprise, contradiction, standalone, density,
    hook_strength, llm_used.

    Strategy:
    - Compute regex baseline first (fast, always available).
    - If Groq/Ollama reachable, call LLM and blend: final = 0.4 * regex + 0.6 * llm.
    - "hook_strength" comes exclusively from the LLM (regex has no equivalent);
      falls back to a heuristic based on hook_delay_sec when LLM unavailable.
    - "density" stays regex-only — the LLM "hook" dimension is a stronger signal
      for hook quality than for information density.
    """
    regex_scores = {
        "quoteability": quoteability_score(window),
        "surprise": surprise_score(window),
        "contradiction": contradiction_score(window),
        "standalone": standalone_score(window, hook_delay_sec),
        "density": semantic_density(window),
    }

    transcript_text = format_transcript(window, use_seconds=True)
    if not transcript_text.strip():
        regex_scores["hook_strength"] = max(0.0, 1.0 - min(hook_delay_sec, 5.0) / 5.0)
        regex_scores["llm_used"] = False
        return regex_scores

    if not use_llm:
        regex_scores["hook_strength"] = max(0.0, 1.0 - min(hook_delay_sec, 5.0) / 5.0)
        regex_scores["llm_used"] = False
        return regex_scores

    llm_raw = _call_llm(transcript_text, provider=llm_provider)
    if llm_raw is None:
        regex_scores["hook_strength"] = max(0.0, 1.0 - min(hook_delay_sec, 5.0) / 5.0)
        regex_scores["llm_used"] = False
        return regex_scores

    # Blend LLM scores into regex baseline for quoteability/surprise/contradiction/standalone
    blended: dict[str, float] = {}
    for internal_key, regex_val in regex_scores.items():
        llm_key = next((k for k, v in _LLM_KEY_MAP.items() if v == internal_key), None)
        llm_val = llm_raw.get(llm_key, regex_val) if llm_key else regex_val
        blended[internal_key] = round(0.4 * regex_val + 0.6 * llm_val, 3)

    # hook_strength: LLM-only (no regex equivalent)
    blended["hook_strength"] = round(float(llm_raw.get("hook", 0.5)), 3)
    blended["llm_used"] = True

    return blended


def score_window_llm(
    window: list[RichSegment],
    hook_delay_sec: float,
) -> dict[str, float]:
    return score_window(window, hook_delay_sec, use_llm=True)
