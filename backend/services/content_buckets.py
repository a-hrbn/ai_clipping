from __future__ import annotations


CONTENT_BUCKET_FEATURE_PREFIX = "content_bucket__"
CONTENT_BUCKET_CHAT_INTERACTION_PREFIX = "content_chat__"
TOP_CHAT_INTERACTION_FEATURES: tuple[str, ...] = (
    "chat_spike_coverage",
    "chat_message_rate_peak",
    "chat_density_peak",
    "chat_peak",
    "chat_post_hook_peak",
)


def bucket_from_source(source: str) -> str:
    normalized = str(source or "").strip().lower()
    if normalized in {"quote", "hot_take", "dialogue", "controversy"}:
        return "quote_driven"
    if normalized in {"reaction", "clean_reaction", "chat_sync"}:
        return "reaction"
    if normalized in {"visual_reveal", "audio_burst", "hype_peak"}:
        return "visual_action"
    if normalized in {"story_payoff", "narrative_reveal", "dense_window", "tension_build"}:
        return "story"
    return "general"


def bucket_from_row(row: dict, features: dict | None = None) -> str:
    existing = str(row.get("content_bucket") or "").strip().lower()
    if existing:
        return existing
    feature_values = features if isinstance(features, dict) else dict(row.get("features") or {})
    return bucket_from_source(str(row.get("candidate_source") or feature_values.get("source") or ""))


def bucket_from_feature(feature) -> str:
    return bucket_from_source(str(getattr(feature, "source", "") or ""))


def feature_value(name: str, feature, bucket: str | None = None) -> float:
    if name.startswith(CONTENT_BUCKET_FEATURE_PREFIX):
        expected = name[len(CONTENT_BUCKET_FEATURE_PREFIX):]
        return 1.0 if (bucket or bucket_from_feature(feature)) == expected else 0.0
    if name.startswith(CONTENT_BUCKET_CHAT_INTERACTION_PREFIX):
        remainder = name[len(CONTENT_BUCKET_CHAT_INTERACTION_PREFIX):]
        expected, sep, base_name = remainder.partition("__")
        if not sep or (bucket or bucket_from_feature(feature)) != expected:
            return 0.0
        return float(getattr(feature, base_name, 0.0) or 0.0)
    return float(getattr(feature, name, 0.0) or 0.0)
