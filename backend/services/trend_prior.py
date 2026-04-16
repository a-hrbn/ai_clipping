from __future__ import annotations

import json
import pathlib
import re
from functools import lru_cache
from math import sqrt

from config import settings
from services.transcriber import RichSegment
from services import clip_strategy

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_DEFAULT_MANIFEST_PATH = pathlib.Path(__file__).resolve().parents[1] / "data" / "hook_patterns.json"
_DEFAULT_TRANSCRIPTS_DIR = pathlib.Path(__file__).resolve().parents[2] / "transcripts"
_FEEDBACK_MEMORY_LIMIT = 200


def score_candidate_window(
    window: list[RichSegment],
    profile: str = "general",
    strategy: dict | None = None,
) -> dict[str, object]:
    normalized_strategy = clip_strategy.normalize_strategy(strategy)
    if not settings.enable_trend_prior or not window:
        return _empty_result()

    manifest = _load_manifest(settings.trend_prior_data_path or str(_DEFAULT_MANIFEST_PATH))
    opening_text = _opening_text(window)
    full_text = " ".join(seg.text.strip() for seg in window if seg.text.strip())
    hook_matches = _match_groups(_groups_for(manifest, profile, "hook_patterns"), opening_text) if manifest else []
    topic_matches = _match_groups(_groups_for(manifest, profile, "topic_patterns"), full_text) if manifest else []

    hook_score = _aggregate_score(hook_matches)
    topic_score = _aggregate_score(topic_matches)
    dynamic_memory = _load_dynamic_memory(
        pathlib.Path(settings.storage_dir),
        _DEFAULT_TRANSCRIPTS_DIR,
    )
    dynamic = _dynamic_scores(dynamic_memory, opening_text, full_text, normalized_strategy)
    goal_alignment_score = clip_strategy.text_relevance_score(full_text, normalized_strategy)
    prior_score = _clamp01(
        0.45 * hook_score
        + 0.2 * topic_score
        + 0.2 * dynamic["dynamic_trend_score"]
        + 0.15 * goal_alignment_score
    )
    outcome_prior_score = _clamp01(
        0.45 * dynamic["dynamic_trend_score"]
        + 0.3 * dynamic["outcome_prior_score"]
        + 0.25 * goal_alignment_score
    )

    return {
        "trend_hook_score": round(hook_score, 3),
        "trend_topic_score": round(topic_score, 3),
        "trend_prior_score": round(prior_score, 3),
        "dynamic_trend_score": round(dynamic["dynamic_trend_score"], 3),
        "outcome_prior_score": round(outcome_prior_score, 3),
        "goal_alignment_score": round(goal_alignment_score, 3),
        "matched_labels": [match["label"] for match in hook_matches + topic_matches],
        "matched_feedback_labels": dynamic["matched_feedback_labels"],
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


@lru_cache(maxsize=2)
def _load_dynamic_memory(storage_dir: pathlib.Path, transcripts_dir: pathlib.Path) -> list[dict]:
    feedback_dir = storage_dir / "feedback"
    snapshots_path = feedback_dir / "feedback_updates.jsonl"
    if not snapshots_path.exists():
        return []

    rows: list[dict] = []
    for line in snapshots_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not _is_positive_feedback_row(row):
            continue
        clip_texts = _extract_feedback_texts(row, transcripts_dir)
        if not clip_texts:
            continue
        platform = str(row.get("publish_platform") or "general").strip().lower() or "general"
        rows.append(
            {
                "platform": platform,
                "opening_tokens": _normalize_text(clip_texts["opening"]).split(),
                "full_tokens": _normalize_text(clip_texts["full"]).split(),
                "score": _feedback_row_score(row),
                "labels": _feedback_row_labels(row),
            }
        )
    rows.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return rows[:_FEEDBACK_MEMORY_LIMIT]


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
        "dynamic_trend_score": 0.0,
        "outcome_prior_score": 0.0,
        "goal_alignment_score": 0.0,
        "matched_labels": [],
        "matched_feedback_labels": [],
    }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _dynamic_scores(memory_rows: list[dict], opening_text: str, full_text: str, strategy: dict) -> dict[str, object]:
    opening_tokens = set(_normalize_text(opening_text).split())
    full_tokens = set(_normalize_text(full_text).split())
    if not opening_tokens and not full_tokens:
        return {
            "dynamic_trend_score": 0.0,
            "outcome_prior_score": 0.0,
            "matched_feedback_labels": [],
        }

    platform = str(strategy.get("platform") or "general")
    best_score = 0.0
    best_outcome = 0.0
    labels: list[str] = []
    for row in memory_rows:
        if row.get("platform") not in {platform, "general"}:
            continue
        opening_overlap = _token_overlap_score(opening_tokens, set(row.get("opening_tokens", [])))
        full_overlap = _token_overlap_score(full_tokens, set(row.get("full_tokens", [])))
        if opening_overlap <= 0 and full_overlap <= 0:
            continue
        weighted = _clamp01(opening_overlap * 0.6 + full_overlap * 0.4)
        score = _clamp01(weighted * float(row.get("score", 0.0)))
        outcome = _clamp01(weighted * sqrt(max(float(row.get("score", 0.0)), 0.0)))
        if score > best_score:
            best_score = score
            best_outcome = outcome
            labels = list(row.get("labels", []))
    return {
        "dynamic_trend_score": best_score,
        "outcome_prior_score": best_outcome,
        "matched_feedback_labels": labels[:4],
    }


def _token_overlap_score(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    if overlap <= 0:
        return 0.0
    return _clamp01(overlap / max(min(len(left), len(right)), 1))


def _is_positive_feedback_row(row: dict) -> bool:
    if not bool(row.get("selected")):
        return False
    return _feedback_row_score(row) >= 0.55


def _feedback_row_score(row: dict) -> float:
    rating = str(row.get("feedback_rating") or "").strip().lower()
    if rating == "good":
        return 1.0
    if rating == "medium":
        return 0.55
    if rating == "bad":
        return 0.0
    return 0.5


def _feedback_row_labels(row: dict) -> list[str]:
    rating = str(row.get("feedback_rating") or "").strip().lower()
    return [rating] if rating in {"good", "medium", "bad"} else []


def _extract_feedback_texts(row: dict, transcripts_dir: pathlib.Path) -> dict[str, str] | None:
    job_id = str(row.get("job_id") or "").strip()
    if not job_id:
        return None
    transcript_path = transcripts_dir / f"{job_id}.jsonl"
    if not transcript_path.exists():
        return None
    start_time = _as_float(row.get("final_start"))
    end_time = _as_float(row.get("final_end"))
    if end_time <= start_time:
        return None
    segments = _read_transcript_segments(transcript_path)
    window = [
        segment for segment in segments
        if float(segment.get("end", 0.0)) > start_time and float(segment.get("start", 0.0)) < end_time
    ]
    if not window:
        return None
    opening = " ".join(str(segment.get("text") or "").strip() for segment in window[:2] if str(segment.get("text") or "").strip())
    full = " ".join(str(segment.get("text") or "").strip() for segment in window if str(segment.get("text") or "").strip())
    return {"opening": opening, "full": full}


def _read_transcript_segments(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
