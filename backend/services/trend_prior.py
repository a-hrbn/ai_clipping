from __future__ import annotations

import json
import pathlib
import re
from functools import lru_cache

from config import settings
from services.transcriber import RichSegment

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_DEFAULT_MANIFEST_PATH = pathlib.Path(__file__).resolve().parents[1] / "data" / "hook_patterns.json"


def score_candidate_window(
    window: list[RichSegment],
    profile: str = "general",
) -> dict[str, object]:
    if not settings.enable_trend_prior or not window:
        return _empty_result()

    manifest = _load_manifest(settings.trend_prior_data_path or str(_DEFAULT_MANIFEST_PATH))
    if manifest is None:
        return _empty_result()

    opening_text = _opening_text(window)
    full_text = " ".join(seg.text.strip() for seg in window if seg.text.strip())
    hook_matches = _match_groups(_groups_for(manifest, profile, "hook_patterns"), opening_text)
    topic_matches = _match_groups(_groups_for(manifest, profile, "topic_patterns"), full_text)

    hook_score = _aggregate_score(hook_matches)
    topic_score = _aggregate_score(topic_matches)
    prior_score = _clamp01(0.62 * hook_score + 0.38 * topic_score)

    return {
        "trend_hook_score": round(hook_score, 3),
        "trend_topic_score": round(topic_score, 3),
        "trend_prior_score": round(prior_score, 3),
        "matched_labels": [match["label"] for match in hook_matches + topic_matches],
    }


@lru_cache(maxsize=4)
def _load_manifest(path_str: str) -> dict | None:
    path = pathlib.Path(path_str)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _groups_for(manifest: dict, profile: str, key: str) -> list[dict]:
    global_groups = list(manifest.get("global", {}).get(key, []) or [])
    profile_groups = list(manifest.get("profiles", {}).get(profile, {}).get(key, []) or [])
    return global_groups + profile_groups


def _match_groups(groups: list[dict], text: str) -> list[dict]:
    normalized = _normalize_text(text)
    if not normalized:
        return []

    matched: list[dict] = []
    for group in groups:
        patterns = [_normalize_text(str(pattern)) for pattern in group.get("patterns", [])]
        if any(pattern and pattern in normalized for pattern in patterns):
            matched.append(
                {
                    "label": str(group.get("label") or "pattern"),
                    "weight": _clamp01(float(group.get("weight", 0.0))),
                }
            )
    matched.sort(key=lambda item: item["weight"], reverse=True)
    return matched


def _aggregate_score(matches: list[dict]) -> float:
    if not matches:
        return 0.0
    top_weights = [float(match["weight"]) for match in matches[:2]]
    if len(top_weights) == 1:
        return _clamp01(top_weights[0])
    return _clamp01(top_weights[0] + top_weights[1] * 0.45)


def _opening_text(window: list[RichSegment]) -> str:
    opening_segments = window[:2]
    return " ".join(seg.text.strip() for seg in opening_segments if seg.text.strip())


def _normalize_text(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(text.lower()))


def _empty_result() -> dict[str, object]:
    return {
        "trend_hook_score": 0.0,
        "trend_topic_score": 0.0,
        "trend_prior_score": 0.0,
        "matched_labels": [],
    }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
