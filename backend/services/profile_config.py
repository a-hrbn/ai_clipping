"""
Clip profile configuration.

Built-in profiles are defined in CLIP_PROFILES.  Custom or overriding profiles
can be loaded from a JSON file whose path is set via settings.profiles_path
(e.g. ``PROFILES_PATH=/etc/ai_clipping/profiles.json``).

File format — a JSON object mapping profile name → weight dict:
    {
        "esports": {
            "chat_weight": 1.3,
            "visual_weight": 1.2,
            "dialogue_weight": 0.7,
            "quote_weight": 0.6,
            "surprise_weight": 1.2,
            "hook_weight": 1.2
        }
    }

Values from the file are merged on top of the built-in defaults, so only the
keys that differ need to be specified.  New profile names are added; existing
profile names are overridden entirely.
"""
from __future__ import annotations

import json
import logging
import pathlib

logger = logging.getLogger(__name__)

_REQUIRED_KEYS = {
    "chat_weight", "visual_weight", "dialogue_weight",
    "quote_weight", "surprise_weight", "hook_weight",
}

CLIP_PROFILES: dict[str, dict[str, float]] = {
    # Balanced profile — suitable for any content type when the format is unknown.
    # All signals weighted equally; no bias toward reaction-heavy or dialogue-heavy content.
    "general": {
        "chat_weight": 1.0,
        "visual_weight": 1.0,
        "dialogue_weight": 1.0,
        "quote_weight": 1.0,
        "surprise_weight": 1.0,
        "hook_weight": 1.0,
        "semantic_weight": 1.0,
    },
    # Gaming streams: reaction-driven moments, chat spikes, hype peaks.
    # Emphasises audience reaction and visual energy; quotes matter less.
    "gaming": {
        "chat_weight": 1.2,
        "visual_weight": 1.1,
        "dialogue_weight": 0.8,
        "quote_weight": 0.7,
        "surprise_weight": 1.1,
        "hook_weight": 1.1,
        "semantic_weight": 0.7,  # clip virality driven by reaction, not content depth
    },
    # Just Chatting / IRL: conversation, personal stories, emotional moments.
    # Dialogue and quotes carry more weight; raw hype signals matter less.
    "just_chatting": {
        "chat_weight": 1.0,
        "visual_weight": 0.8,
        "dialogue_weight": 1.2,
        "quote_weight": 1.1,
        "surprise_weight": 1.0,
        "hook_weight": 1.0,
        "semantic_weight": 1.4,  # quoteable/surprising moments are the viral unit here
    },
    # Jynxzi test profile: mixed streamer content spanning gaming, reactions,
    # community/fan-mail segments, and loud conversational beats. More balanced
    # than gaming, but still rewards hook and surprise strongly.
    "jynxzi": {
        "chat_weight": 1.0,
        "visual_weight": 1.0,
        "dialogue_weight": 1.1,
        "quote_weight": 1.05,
        "surprise_weight": 1.2,
        "hook_weight": 1.15,
        "semantic_weight": 1.1,
    },
    # IRL / outdoor / travel streams: visual environment changes, spontaneous
    # reactions, real-world events. Balance visual and dialogue signals.
    "irl": {
        "chat_weight": 0.9,
        "visual_weight": 1.1,
        "dialogue_weight": 1.0,
        "quote_weight": 0.9,
        "surprise_weight": 1.1,
        "hook_weight": 1.0,
        "semantic_weight": 0.9,
    },
    # Podcast / interview: insight-driven, measured delivery, quoteable lines.
    # Chat and visual signals are sparse; quotes and dialogue are paramount.
    "podcast": {
        "chat_weight": 0.5,
        "visual_weight": 0.6,
        "dialogue_weight": 1.1,
        "quote_weight": 1.3,
        "surprise_weight": 1.0,
        "hook_weight": 0.95,
        "semantic_weight": 1.5,  # insight depth is the only virality lever in this format
    },
}


def _load_custom_profiles(path: str | None) -> None:
    """Merge profiles from a JSON file into CLIP_PROFILES (in-place)."""
    if not path:
        return
    p = pathlib.Path(path)
    if not p.exists():
        logger.warning("profiles_path '%s' does not exist — using built-in profiles only", path)
        return
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Top-level value must be a JSON object")
        # Optional heuristic weight overrides
        if "heuristic_weights" in data:
            for key, val in data["heuristic_weights"].items():
                if key in HEURISTIC_WEIGHTS:
                    HEURISTIC_WEIGHTS[key] = float(val)
                    logger.info("Heuristic weight '%s' overridden to %s", key, val)
        if "streamer_heuristic_overrides" in data:
            for streamer, overrides in data["streamer_heuristic_overrides"].items():
                if isinstance(overrides, dict):
                    STREAMER_HEURISTIC_OVERRIDES[streamer.lower()] = {
                        k: float(v) for k, v in overrides.items() if k in HEURISTIC_WEIGHTS
                    }
                    logger.info("Streamer heuristic overrides loaded for '%s'", streamer)
        # Profile overrides / additions
        for name, weights in data.items():
            if name == "heuristic_weights":
                continue
            missing = _REQUIRED_KEYS - set(weights.keys())
            if missing:
                logger.warning("Profile '%s' in %s is missing keys %s — skipping", name, path, missing)
                continue
            CLIP_PROFILES[name] = {k: float(v) for k, v in weights.items()}
            logger.info("Loaded custom profile '%s' from %s", name, path)
    except Exception as exc:
        logger.error("Failed to load profiles from '%s': %s", path, exc)


def _init() -> None:
    """Load custom profiles at import time (deferred to avoid circular imports)."""
    try:
        from config import settings
        _load_custom_profiles(getattr(settings, "profiles_path", None))
    except Exception:
        pass


_init()


# ---------------------------------------------------------------------------
# Heuristic scoring weights
# ---------------------------------------------------------------------------
# These coefficients control the relative importance of each signal in
# the heuristic rank formula inside ranker.py.  They can be overridden via
# the same JSON config file under the key ``"heuristic_weights"``.

HEURISTIC_WEIGHTS: dict[str, float] = {
    "hook":               0.20,
    "payoff":             0.20,
    "payoff_clarity":     0.06,
    "hype":               0.18,
    "arc":                0.09,   # reduced: arc structure matters less than content quality
    "chat":               0.11,   # raised: chat spike is a reliable virality proxy on Twitch
    "visual":             0.05,
    "dialogue":           0.05,
    "semantic":           0.08,   # raised: quoteable/surprising content is a core viral signal
    "context_efficiency": 0.03,
    "profile_fit":        0.06,   # raised: profile fit should meaningfully influence ranking
    "chapter_coherence":  0.05,   # reward candidates that stay semantically inside a coherent chapter
    "topic_shift":        0.04,   # topic pivots near a hook can be strong short-form entry points
    "chapter_density":    0.03,   # dense chapters tend to contain more clip-worthy beats
    "trend_prior":        0.05,   # lightweight prior from common viral hooks/topics
    "overlap_penalty":   -0.04,   # penalize candidates that heavily overlap nearby alternatives
    "slow_open_penalty":  0.00,   # disabled in serving; kept for config compatibility
    "context_dependency_penalty": -0.08,  # punish clips that need prior stream context
    "weak_resolution_penalty": -0.06,  # punish clips without a clean ending/payoff
    "context_cost":      -0.05,   # penalty — negative keeps the sign clear
}


# Maps a known streamer name (lowercase) to its primary content-type profile.
# Used by Ranker._load_artifact() as the second level of the fallback chain:
#   streamer artifact → content-type artifact → heuristic only
# Add a new entry here whenever a streamer model is trained.
STREAMER_CONTENT_TYPE_MAP: dict[str, str] = {
    "jynxzi": "jynxzi",
    # Examples for future streamers:
    # "shroud":    "gaming",
    # "hasanabi":  "just_chatting",
    # "xqc":       "gaming",
}

# Per-streamer heuristic weight overrides.
# Only keys that differ from HEURISTIC_WEIGHTS need to be specified — the rest
# are inherited from the base weights.  Applied on top of content-type profiles,
# so they express streamer-level quirks that profile categories can't capture.
#
# Examples (uncomment and adjust when data supports it):
#   "hasanabi": {"semantic": 0.14, "arc": 0.07},   # controversy/takes drive virality
#   "shroud":   {"visual": 0.10, "hype": 0.20},    # mechanical skill clips, less chat
#   "xqc":      {"hype": 0.22, "chat": 0.13},      # chaos + chat explosion is the brand
STREAMER_HEURISTIC_OVERRIDES: dict[str, dict[str, float]] = {
    "jynxzi": {"hook": 0.22, "surprise_via_semantic": 0.10},
}


def get_content_type_for_streamer(streamer: str) -> str | None:
    """Return the content-type profile for a known streamer, or None if unknown."""
    return STREAMER_CONTENT_TYPE_MAP.get(streamer.lower().strip())


def get_heuristic_weights() -> dict[str, float]:
    return dict(HEURISTIC_WEIGHTS)


def get_effective_heuristic_weights(profile: str) -> dict[str, float]:
    """Return heuristic weights merged with any streamer-specific overrides.

    Falls back to the base HEURISTIC_WEIGHTS when no override exists for the
    given profile name.  Only keys present in HEURISTIC_WEIGHTS are eligible
    for override — unknown keys in the override dict are silently ignored.
    """
    base = dict(HEURISTIC_WEIGHTS)
    overrides = STREAMER_HEURISTIC_OVERRIDES.get(profile.lower().strip(), {})
    ignored = sorted(key for key in overrides if key not in base)
    if ignored:
        logger.warning(
            "Ignoring unknown heuristic override keys for profile '%s': %s",
            profile,
            ignored,
        )
    for key, value in overrides.items():
        if key in base:
            base[key] = float(value)
    return base


def get_profile(name: str) -> dict[str, float]:
    profile = CLIP_PROFILES.get(name)
    if profile is None:
        logger.warning(
            "Unknown clip profile '%s' — falling back to 'general'. "
            "Available profiles: %s",
            name,
            sorted(CLIP_PROFILES.keys()),
        )
        return CLIP_PROFILES["general"]
    return profile
