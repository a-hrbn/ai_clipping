from __future__ import annotations

import re
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_DEFAULT_VARIANT_COUNT = 1

_PLATFORM_DURATION_TARGETS: dict[str, tuple[float, float, float]] = {
    "general": (12.0, 45.0, 24.0),
    "tiktok": (10.0, 30.0, 18.0),
    "youtube_shorts": (14.0, 45.0, 28.0),
    "instagram_reels": (10.0, 35.0, 20.0),
    "linkedin": (18.0, 50.0, 30.0),
}

_DEFAULT_SOURCE_CAPS: dict[str, int] = {
    "quote": 4,
    "tension_build": 4,
    "narrative_reveal": 4,
    "controversy": 4,
    "hype_peak": 4,
    "chat_sync": 4,
    "reaction": 4,
    "clean_reaction": 4,
    "dialogue": 4,
    "hot_take": 4,
    "story_payoff": 4,
    "dense_window": 3,
    "visual_reveal": 4,
    "audio_burst": 4,
    "silence_break": 3,
    "goal_prompt": 5,
}

_GOAL_SOURCE_BONUS: dict[str, dict[str, int]] = {
    "reaction": {"reaction": 3, "clean_reaction": 3, "chat_sync": 2, "audio_burst": 2},
    "quote": {"quote": 3, "hot_take": 2, "dialogue": 2, "controversy": 2},
    "controversy": {"controversy": 3, "dialogue": 2, "hot_take": 2, "tension_build": 2},
    "story": {"story_payoff": 3, "narrative_reveal": 2, "dense_window": 1},
    "action": {"visual_reveal": 3, "audio_burst": 2, "hype_peak": 2},
}

_PLATFORM_FEATURE_BIAS: dict[str, dict[str, float]] = {
    "general": {},
    "tiktok": {"hook": 1.2, "visual": 1.15, "context": 0.85, "payoff": 1.1},
    "youtube_shorts": {"hook": 1.05, "visual": 1.0, "context": 1.0, "payoff": 1.05},
    "instagram_reels": {"hook": 1.15, "visual": 1.1, "context": 0.9, "payoff": 1.05},
    "linkedin": {"hook": 0.95, "visual": 0.85, "context": 1.15, "payoff": 1.0},
}

_VARIANT_LIBRARY: dict[str, dict[str, Any]] = {
    "balanced": {
        "key": "balanced",
        "label": "Balanced",
        "start_bias_sec": 0.0,
        "end_bias_sec": 0.0,
        "context_multiplier": 1.0,
    },
    "aggressive_hook": {
        "key": "aggressive_hook",
        "label": "Aggressive Hook",
        "start_bias_sec": 1.25,
        "end_bias_sec": -0.5,
        "context_multiplier": 0.78,
    },
    "context_balanced": {
        "key": "context_balanced",
        "label": "Context Balanced",
        "start_bias_sec": -1.5,
        "end_bias_sec": 0.25,
        "context_multiplier": 1.2,
    },
    "reaction_first": {
        "key": "reaction_first",
        "label": "Reaction First",
        "start_bias_sec": 0.8,
        "end_bias_sec": 0.5,
        "context_multiplier": 0.82,
    },
    "quote_first": {
        "key": "quote_first",
        "label": "Quote First",
        "start_bias_sec": -0.4,
        "end_bias_sec": 0.0,
        "context_multiplier": 0.92,
    },
}


def normalize_strategy(strategy: dict | None) -> dict[str, Any]:
    raw = dict(strategy or {})
    must_include = _coerce_list(raw.get("must_include"))
    exclude = _coerce_list(raw.get("exclude"))
    goal = _clean_text(raw.get("goal"))
    topic = _clean_text(raw.get("topic"))
    tone = _clean_text(raw.get("tone"))
    speaker = _clean_text(raw.get("speaker"))
    platform = _normalize_platform(raw.get("platform"))
    clip_style = _normalize_style(raw.get("clip_style"))
    variant_count = _coerce_variant_count(raw.get("variant_count"))
    time_range = _normalize_time_range(raw.get("time_range"))
    archetype_judge = _coerce_optional_bool(raw.get("archetype_judge"))
    return {
        "platform": platform,
        "goal": goal,
        "tone": tone,
        "topic": topic,
        "must_include": must_include,
        "exclude": exclude,
        "speaker": speaker,
        "time_range": time_range,
        "clip_style": clip_style,
        "variant_count": variant_count,
        # Per-job Archetype-Judge opt-out (#105): None inherits the global default,
        # False turns the judge off for this job, True forces it on. The Conductor
        # resolves it against ``moment_finder_archetype_judge_enabled``.
        "archetype_judge": archetype_judge,
        "goal_family": infer_goal_family(goal, clip_style, topic),
        "keywords": strategy_keywords(
            {
                "goal": goal,
                "tone": tone,
                "topic": topic,
                "must_include": must_include,
                "speaker": speaker,
            }
        ),
        "exclude_keywords": _tokenize(" ".join(exclude)),
    }


def strategy_keywords(strategy: dict | None) -> set[str]:
    normalized = dict(strategy or {})
    tokens: set[str] = set()
    for value in (
        normalized.get("goal"),
        normalized.get("tone"),
        normalized.get("topic"),
        normalized.get("speaker"),
    ):
        tokens.update(_tokenize(str(value or "")))
    for value in normalized.get("must_include", []) or []:
        tokens.update(_tokenize(str(value)))
    return {token for token in tokens if len(token) >= 3}


def infer_goal_family(goal: str, clip_style: str, topic: str = "") -> str:
    haystack = " ".join(part for part in (goal, clip_style, topic) if part).lower()
    if any(token in haystack for token in ("reaction", "rage", "funny", "laugh", "clip reaction")):
        return "reaction"
    if any(token in haystack for token in ("quote", "take", "opinion", "insight", "statement")):
        return "quote"
    if any(token in haystack for token in ("controvers", "argument", "debate", "drama", "heated")):
        return "controversy"
    if any(token in haystack for token in ("story", "reveal", "payoff", "moment", "arc")):
        return "story"
    if any(token in haystack for token in ("goal", "kill", "play", "action", "highlight", "sports")):
        return "action"
    return "general"


def source_caps(strategy: dict | None) -> dict[str, int]:
    normalized = normalize_strategy(strategy)
    caps = dict(_DEFAULT_SOURCE_CAPS)
    for source, bonus in _GOAL_SOURCE_BONUS.get(normalized["goal_family"], {}).items():
        caps[source] = caps.get(source, 3) + bonus
    if normalized["platform"] in {"tiktok", "instagram_reels"}:
        for source in ("reaction", "clean_reaction", "visual_reveal", "audio_burst"):
            caps[source] = caps.get(source, 3) + 1
    if normalized["clip_style"] == "quote_first":
        for source in ("quote", "hot_take", "dialogue"):
            caps[source] = caps.get(source, 3) + 1
    return caps


def text_relevance_score(text: str, strategy: dict | None) -> float:
    normalized = normalize_strategy(strategy)
    keywords: set[str] = normalized["keywords"]
    if not keywords:
        return 0.0
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    overlap = len(tokens & keywords)
    base = min(overlap / max(len(keywords), 1), 1.0)
    penalty = 0.0
    exclude_overlap = len(tokens & normalized["exclude_keywords"])
    if exclude_overlap:
        penalty += min(exclude_overlap * 0.25, 0.75)
    return max(0.0, min(base - penalty, 1.0))


def candidate_in_time_range(start: float, end: float, strategy: dict | None) -> bool:
    normalized = normalize_strategy(strategy)
    time_range = normalized["time_range"]
    if not time_range:
        return True
    range_start = float(time_range.get("start_sec", 0.0) or 0.0)
    range_end = time_range.get("end_sec")
    if range_end is not None and start >= float(range_end):
        return False
    if end <= range_start:
        return False
    return True


def platform_biases(strategy: dict | None) -> dict[str, float]:
    normalized = normalize_strategy(strategy)
    return dict(_PLATFORM_FEATURE_BIAS.get(normalized["platform"], _PLATFORM_FEATURE_BIAS["general"]))


def platform_duration_target(strategy: dict | None) -> tuple[float, float, float]:
    normalized = normalize_strategy(strategy)
    low, high, ideal = _PLATFORM_DURATION_TARGETS.get(
        normalized["platform"],
        _PLATFORM_DURATION_TARGETS["general"],
    )
    if normalized["clip_style"] == "aggressive_hook":
        return max(low - 2.0, 8.0), max(high - 6.0, 20.0), max(ideal - 6.0, 14.0)
    if normalized["clip_style"] == "context_balanced":
        return low + 2.0, high + 6.0, ideal + 4.0
    return low, high, ideal


def build_variant_specs(strategy: dict | None) -> list[dict[str, Any]]:
    normalized = normalize_strategy(strategy)
    requested = normalized["clip_style"]
    variant_count = normalized["variant_count"]
    ordered_keys = [requested]
    if requested != "balanced":
        ordered_keys.append("balanced")
    if normalized["goal_family"] == "reaction":
        ordered_keys.append("reaction_first")
    elif normalized["goal_family"] == "quote":
        ordered_keys.append("quote_first")
    else:
        ordered_keys.append("aggressive_hook")
    if normalized["platform"] in {"youtube_shorts", "linkedin"}:
        ordered_keys.append("context_balanced")
    unique_keys: list[str] = []
    for key in ordered_keys:
        if key not in unique_keys and key in _VARIANT_LIBRARY:
            unique_keys.append(key)
    return [
        dict(_VARIANT_LIBRARY[key], platform=normalized["platform"])
        for key in unique_keys[:variant_count]
    ]


def variant_label(strategy: dict | None, variant_key: str) -> str:
    normalized = normalize_strategy(strategy)
    label = _VARIANT_LIBRARY.get(variant_key, _VARIANT_LIBRARY["balanced"])["label"]
    platform = normalized["platform"]
    if platform == "general":
        return label
    return f"{label} · {platform.replace('_', ' ').title()}"


def _normalize_platform(value: Any) -> str:
    lowered = _clean_text(value).lower().replace(" ", "_")
    if lowered in {"", "auto"}:
        return "general"
    if lowered in {"shorts", "youtube"}:
        return "youtube_shorts"
    if lowered in {"reels", "instagram"}:
        return "instagram_reels"
    return lowered if lowered in _PLATFORM_DURATION_TARGETS else "general"


def _normalize_style(value: Any) -> str:
    lowered = _clean_text(value).lower().replace(" ", "_")
    return lowered if lowered in _VARIANT_LIBRARY else "balanced"


def _normalize_time_range(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    start = _coerce_float(value.get("start_sec"))
    end = _coerce_float(value.get("end_sec"))
    if start is None and end is None:
        return None
    if start is not None and end is not None and end <= start:
        return None
    result: dict[str, float] = {}
    if start is not None:
        result["start_sec"] = start
    if end is not None:
        result["end_sec"] = end
    return result


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = _clean_text(value)
        return [cleaned] if cleaned else []
    if isinstance(value, list):
        return [item for item in (_clean_text(part) for part in value) if item]
    return []


def _coerce_variant_count(value: Any) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError):
        count = _DEFAULT_VARIANT_COUNT
    return max(1, min(count, 4))


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_bool(value: Any) -> bool | None:
    """Tri-state per-job toggle: ``None`` (absent → inherit the global default),
    else the explicit truthiness. The Conductor reads this to opt a single job out
    of (or into) the Archetype-Judge without a code change."""
    if value is None:
        return None
    return bool(value)


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))
