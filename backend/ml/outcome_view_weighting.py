from __future__ import annotations

from typing import Any


VIEW_WEIGHT_VERSION = 1


def view_count(row: dict[str, Any]) -> float:
    for key in ("views", "repost_views", "view_count"):
        value = _as_float(row.get(key))
        if value > 0.0:
            return value
    return 0.0


def outcome_view_tier(views: float) -> str:
    value = max(float(views or 0.0), 0.0)
    if value >= 1_000_000:
        return "elite_viral"
    if value >= 500_000:
        return "breakout_viral"
    if value >= 100_000:
        return "viral"
    if value >= 25_000:
        return "strong"
    if value > 0:
        return "clipped"
    return "unknown"


def outcome_view_sample_weight(views: float) -> float:
    tier = outcome_view_tier(views)
    if tier == "elite_viral":
        return 3.0
    if tier == "breakout_viral":
        return 2.5
    if tier == "viral":
        return 2.0
    if tier == "strong":
        return 1.75
    if tier == "clipped":
        return 1.5
    return 1.0


def outcome_view_rank_target(views: float) -> int:
    tier = outcome_view_tier(views)
    if tier in {"elite_viral", "breakout_viral"}:
        return 5
    if tier == "viral":
        return 4
    return 3


def annotate_cross_platform_view_weight(row: dict[str, Any]) -> dict[str, Any]:
    copied = dict(row)
    views = view_count(copied)
    tier = outcome_view_tier(views)
    weight = outcome_view_sample_weight(views)
    copied["views"] = views
    copied["outcome_view_tier"] = tier
    copied["outcome_view_sample_weight"] = weight
    copied["outcome_view_weight_version"] = VIEW_WEIGHT_VERSION
    copied["rank_target"] = max(int(copied.get("rank_target") or 0), outcome_view_rank_target(views))
    copied["ordinal_label"] = max(int(copied.get("ordinal_label") or 0), min(outcome_view_rank_target(views), 4))
    copied["label_quality_class"] = (
        "cross_platform_viral_positive"
        if tier in {"viral", "breakout_viral", "elite_viral"}
        else "cross_platform_clipped_positive"
    )
    copied["sample_weight"] = max(_as_float(copied.get("sample_weight")), weight)
    return copied


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
