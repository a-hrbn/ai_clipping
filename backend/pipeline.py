import asyncio
import json
import logging
import os
import pathlib
import re
import shutil
import subprocess
import uuid
from dataclasses import asdict, fields, is_dataclass, replace
from types import SimpleNamespace
from config import settings
from job_manager import is_job_cancelled, update_job
from services import (
    downloader,
    audio_extractor,
    transcriber,
    analyzer,
    clipper,
    audio_features,
    video_features,
    scorer,
    post_processor,
    chat_analyzer,
    candidate_generators,
    candidate_normalizer,
    feature_builder,
    selector,
    feedback_store,
    peak_detector,
    clip_judge,
    keyframe_sampler,
    chapter_segmenter,
    clip_strategy,
    candidate_recall_report,
    clip_taxonomy,
    context_patterns,
    high_recall_protection,
)
from services.timeline_types import Candidate, RankedCandidate
from services.transcriber import RichSegment

logger = logging.getLogger(__name__)

_TWITCH_VOD_RE = re.compile(r"twitch\.tv/videos/(\d+)")
_AUDIO_CACHE_VERSION = 1
_FEATURE_FRAME_CACHE_VERSION = 1
_CROSS_PLATFORM_CONTEXT_TEXT_FIELDS = (
    "repost_title",
    "short_title",
    "video_title",
    "title",
    "matched_cross_platform_title",
    "transcript_repost_excerpt",
    "transcript_vod_excerpt",
    "description",
    "repost_description",
)


class PipelineCancelledError(RuntimeError):
    pass


def _raise_if_cancelled(job_id: str) -> None:
    if is_job_cancelled(job_id):
        raise PipelineCancelledError(f"Job {job_id} cancelled by client")


def _frontend_cache_enabled_for_job(job_id: str) -> bool:
    if bool(getattr(settings, "frontend_job_cache_enabled", False)):
        return True
    job_id = str(job_id or "")
    return job_id.startswith(("twitch_vod", "judge_backfill_"))


def _transcript_cache_enabled_for_job(job_id: str, url: str | None) -> bool:
    if _frontend_cache_enabled_for_job(job_id):
        return True
    if not bool(getattr(settings, "frontend_transcript_cache_enabled", True)):
        return False
    return bool(_source_twitch_vod_id(url))


def _feature_cache_enabled_for_job(job_id: str, url: str | None) -> bool:
    if _frontend_cache_enabled_for_job(job_id):
        return True
    if not bool(getattr(settings, "frontend_feature_cache_enabled", True)):
        return False
    return bool(_source_twitch_vod_id(url))


def _path_is_relative_to(path: pathlib.Path, parent: pathlib.Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
    except ValueError:
        return False
    return True


def _cleanup_audio_artifacts(audio_path: pathlib.Path | None) -> None:
    if audio_path is None:
        return
    try:
        audio_path.unlink(missing_ok=True)
    except OSError:
        logger.warning("pipeline_audio_cleanup_failed path=%s", audio_path, exc_info=True)
    chunks_dir = audio_path.parent / "chunks"
    if chunks_dir.exists():
        try:
            shutil.rmtree(chunks_dir)
        except OSError:
            logger.warning("pipeline_audio_chunks_cleanup_failed path=%s", chunks_dir, exc_info=True)


def _cleanup_serving_source_media(
    *,
    job_id: str,
    video_path: pathlib.Path | None,
    audio_path: pathlib.Path | None,
) -> None:
    if _frontend_cache_enabled_for_job(job_id):
        return

    job_dir = pathlib.Path(settings.storage_dir) / job_id
    _cleanup_audio_artifacts(audio_path)

    if video_path is None or not _path_is_relative_to(video_path, job_dir):
        return

    try:
        video_path.unlink(missing_ok=True)
        logger.info("pipeline_source_video_deleted job_id=%s path=%s", job_id, video_path)
    except OSError:
        logger.warning(
            "pipeline_source_video_cleanup_failed job_id=%s path=%s",
            job_id,
            video_path,
            exc_info=True,
        )


def _transcript_cache_source(url: str | None, video_path: pathlib.Path | None) -> str | None:
    if url:
        twitch_vod = re.search(r"twitch\.tv/videos/(\d+)", url.strip().lower())
        if twitch_vod:
            return f"twitch_vod:{twitch_vod.group(1)}"
        return f"url:{url.strip()}"
    if video_path is not None:
        try:
            return f"path:{video_path.resolve()}"
        except OSError:
            return f"path:{video_path}"
    return None


def _feature_cache_source(
    *,
    job_id: str,
    url: str | None,
    video_path: pathlib.Path | None,
) -> str | None:
    if not _feature_cache_enabled_for_job(job_id, url):
        return None
    source = _transcript_cache_source(url, video_path)
    if not source:
        return None
    if not _frontend_cache_enabled_for_job(job_id) and not source.startswith("twitch_vod:"):
        return None
    return source


def _source_twitch_vod_id(url: str | None) -> str:
    if not url:
        return ""
    match = _TWITCH_VOD_RE.search(str(url).strip().lower())
    return match.group(1) if match else ""


def _cross_platform_positive_rows_path(profile: str) -> pathlib.Path:
    configured = str(getattr(settings, "serving_cross_platform_positive_rows_path", "") or "").strip()
    if configured:
        path = pathlib.Path(configured).expanduser()
        if not path.is_absolute():
            path = pathlib.Path(__file__).resolve().parent / path
        return path
    safe_profile = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(profile or "general").strip().lower())
    return pathlib.Path(__file__).resolve().parent / "ml" / "artifacts" / f"{safe_profile}_cross_platform_fingerprint_rows.jsonl"


def _load_cross_platform_positive_windows(
    profile: str,
    source_url: str | None,
    *,
    force_load: bool = False,
) -> tuple[list[dict], dict]:
    vod_id = _source_twitch_vod_id(source_url)
    enabled = bool(getattr(settings, "serving_cross_platform_recall_enabled", False))
    active = enabled or force_load
    path = _cross_platform_positive_rows_path(profile)
    summary = {
        "enabled": enabled,
        "active": active,
        "diagnostic_only": bool(force_load and not enabled),
        "vod_id": vod_id,
        "path": str(path),
        "rows": 0,
        "matched_rows": 0,
        "status": "disabled" if not active else "not_loaded",
    }
    if not active:
        return [], summary
    if not vod_id:
        summary["status"] = "missing_vod_id"
        return [], summary
    if not path.exists():
        summary["status"] = "missing_file"
        return [], summary
    rows: list[dict] = []
    matched: list[dict] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)
            if str(row.get("vod_id") or "") == vod_id or vod_id in str(row.get("vod_url") or ""):
                matched.append(row)
    except (OSError, json.JSONDecodeError) as exc:
        summary.update({"rows": len(rows), "status": "read_error", "error": str(exc)})
        return [], summary
    summary.update({"rows": len(rows), "matched_rows": len(matched), "status": "loaded"})
    return matched, summary


def _best_cross_platform_positive_match(start: float, end: float, positives: list[dict]) -> dict:
    best = {"overlap_ratio": 0.0, "row": None}
    for row in positives:
        try:
            pos_start = float(row["window_start"])
            pos_end = float(row["window_end"])
        except (KeyError, TypeError, ValueError):
            continue
        overlap = max(0.0, min(float(end), pos_end) - max(float(start), pos_start))
        if overlap <= 0.0:
            continue
        candidate_duration = max(float(end) - float(start), 0.001)
        positive_duration = max(pos_end - pos_start, 0.001)
        ratio = max(overlap / candidate_duration, overlap / positive_duration)
        if ratio > float(best["overlap_ratio"]):
            best = {"overlap_ratio": ratio, "row": row}
    return best


def _cross_platform_recall_targets(candidates, positives: list[dict], *, profile: str = "jynxzi") -> dict[str, dict]:
    if not positives:
        return {}
    min_overlap = float(getattr(settings, "serving_cross_platform_recall_overlap_ratio", 0.45) or 0.45)
    targets: dict[str, dict] = {}
    for candidate in candidates:
        match = _best_cross_platform_positive_match(candidate.start, candidate.end, positives)
        if float(match["overlap_ratio"] or 0.0) < min_overlap:
            continue
        row = match["row"] or {}
        context = _cross_platform_context_from_row(row, profile=profile)
        targets[candidate.id] = {
            "positive_overlap_ratio": round(float(match["overlap_ratio"] or 0.0), 6),
            "matched_cross_platform_candidate_id": row.get("candidate_id") or "",
            "matched_cross_platform_source_url": row.get("source_url") or row.get("repost_source_url") or "",
            "matched_cross_platform_title": context["title"],
            "matched_cross_platform_context_text": context["text"],
            "matched_cross_platform_clip_type": context["clip_type"],
            "matched_cross_platform_similarity": row.get("transcript_similarity") or row.get("confidence") or 0.0,
            "matched_cross_platform_views": row.get("repost_views") or 0.0,
            "matched_cross_platform_window_start": row.get("window_start"),
            "matched_cross_platform_window_end": row.get("window_end"),
            "matched_cross_platform_pattern": context["pattern"],
        }
    return targets


def _cross_platform_context_from_row(row: dict, *, profile: str = "jynxzi") -> dict[str, object]:
    title = _first_nonempty_text(
        row,
        ("repost_title", "short_title", "video_title", "title", "matched_cross_platform_title"),
    )
    text_parts = [
        str(row.get(key) or "").strip()
        for key in _CROSS_PLATFORM_CONTEXT_TEXT_FIELDS
        if str(row.get(key) or "").strip()
    ]
    text = " ".join(dict.fromkeys(text_parts))
    taxonomy = clip_taxonomy.classify_text(text, fallback_type="general")
    pattern = context_patterns.score_text(text, profile=profile)
    return {
        "title": title,
        "text": text[:500],
        "clip_type": taxonomy.clip_type,
        "pattern": {
            "pattern_id": pattern.pattern_id,
            "label": pattern.label,
            "score": pattern.score,
            "support": pattern.support,
            "views_prior": pattern.views_prior,
            "confidence": pattern.confidence,
            "similarity_margin": pattern.similarity_margin,
        },
    }


def _first_nonempty_text(row: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _cross_platform_recall_preserve_ids(
    candidates,
    positives: list[dict],
    *,
    max_per_positive: int = 3,
) -> set[str]:
    """Pick a small set of candidate ids that must not be deduped away.

    The full target set can include many overlapping inventory windows for the
    same repost. For merge preservation we only need the strongest few per
    positive window so known-good supervision does not flood the serving funnel.
    """

    if not positives or max_per_positive <= 0:
        return set()
    min_overlap = float(getattr(settings, "serving_cross_platform_recall_overlap_ratio", 0.45) or 0.45)
    grouped: dict[tuple[str, str, str], list[tuple[float, float, float, str]]] = {}
    for candidate in candidates:
        match = _best_cross_platform_positive_match(candidate.start, candidate.end, positives)
        overlap = float(match["overlap_ratio"] or 0.0)
        if overlap < min_overlap:
            continue
        row = match["row"] or {}
        key = (
            str(row.get("repost_source_url") or row.get("source_url") or ""),
            str(row.get("window_start") or ""),
            str(row.get("window_end") or ""),
        )
        duration = max(float(candidate.end) - float(candidate.start), 0.001)
        grouped.setdefault(key, []).append(
            (
                overlap,
                float(getattr(candidate, "score_hint", 0.0) or 0.0),
                -duration,
                str(candidate.id),
            )
        )
    preserved: set[str] = set()
    for rows in grouped.values():
        rows.sort(reverse=True)
        preserved.update(candidate_id for *_scores, candidate_id in rows[:max_per_positive])
    return preserved


_CONTEXT_VARIANT_SOURCES = {
    "quote",
    "narrative_reveal",
    "tension_build",
    "controversy",
    "story_payoff",
    "goal_prompt",
    "chat_velocity",
    "dense_window",
    "dense_inventory",
}
_CONTEXT_SETUP_TERMS = (
    "because",
    "listen",
    "let me",
    "here's",
    "heres",
    "this is",
    "what happened",
    "finds out",
    "reacts",
    "sent",
    "asked",
    "told",
    "first",
)
_CONTEXT_PAYOFF_TERMS = (
    "what",
    "wait",
    "no way",
    "oh my",
    "bro",
    "laugh",
    "hilarious",
    "insane",
    "crazy",
    "banned",
    "cheated",
    "won",
    "lost",
    "revealed",
    "finally",
)


def _add_context_variant_candidates(
    base_candidates: list[Candidate],
    variant_candidates: list[Candidate],
    segments: list[RichSegment],
    *,
    max_duration: float,
    max_new_candidates: int = 96,
    max_variants_per_region: int = 3,
) -> tuple[list[Candidate], dict[str, object]]:
    """Add the best setup/payoff variants that were lost during generic merge.

    This is label-free: it only inspects candidate shape and local transcript
    text. It exists because a short dense window can suppress a better semantic
    setup window for the same moment before Proposal/Judge ever sees it.
    """

    if not variant_candidates or not segments:
        return base_candidates, {"enabled": True, "added": 0, "candidate_pool": 0}

    existing_ids = {candidate.id for candidate in base_candidates}
    pool = [
        candidate
        for candidate in variant_candidates
        if candidate.id not in existing_ids and candidate.source in _CONTEXT_VARIANT_SOURCES
    ]
    if not pool:
        return base_candidates, {"enabled": True, "added": 0, "candidate_pool": 0}

    normalized_pool = candidate_normalizer.normalize_candidates(pool, segments, max_duration)
    scored = [
        (_context_variant_score(candidate, segments), candidate)
        for candidate in normalized_pool
        if candidate.id not in existing_ids
    ]
    scored = [(score, candidate) for score, candidate in scored if score >= 0.58]
    scored.sort(key=lambda item: (item[0], item[1].score_hint, item[1].end - item[1].start), reverse=True)

    represented_candidates = list(base_candidates)
    overlapping_best_by_bucket: dict[int, tuple[float, Candidate]] = {}
    novel_best_by_bucket: dict[int, tuple[float, Candidate]] = {}
    for score, candidate in scored:
        bucket = int(float(candidate.anchor_time) // 120.0)
        overlaps = [
            existing
            for existing in represented_candidates
            if _candidate_region_overlap(candidate, existing) >= 0.48
        ]
        if overlaps:
            existing = overlapping_best_by_bucket.get(bucket)
            if existing is None or score > existing[0]:
                overlapping_best_by_bucket[bucket] = (score, candidate)
            continue
        existing = novel_best_by_bucket.get(bucket)
        if existing is None or score > existing[0]:
            novel_best_by_bucket[bucket] = (score, candidate)

    overlap_budget = max(max_new_candidates // 2, 1)
    overlap_ordered = sorted(overlapping_best_by_bucket.values(), key=lambda item: int(float(item[1].anchor_time) // 120.0))
    novel_ordered = sorted(novel_best_by_bucket.values(), key=lambda item: int(float(item[1].anchor_time) // 120.0))
    ordered_scored = [
        *overlap_ordered[:overlap_budget],
        *novel_ordered,
        *overlap_ordered[overlap_budget:],
    ]

    result = list(base_candidates)
    added: list[dict[str, object]] = []
    for score, candidate in ordered_scored:
        if len(added) >= max_new_candidates:
            break
        overlaps = [existing for existing in result if _candidate_region_overlap(candidate, existing) >= 0.48]
        if any(_candidate_effectively_identical(candidate, existing) for existing in overlaps):
            continue
        if overlaps:
            region_scores = [_context_variant_score(existing, segments) for existing in overlaps]
            region_count = len(overlaps)
            best_region_score = max(region_scores or [0.0])
            if region_count >= max_variants_per_region and score < best_region_score + 0.12:
                continue
            if score < best_region_score + 0.04 and not _candidate_has_more_setup(candidate, overlaps):
                continue
        elif score < 0.68:
            continue
        result.append(candidate)
        added.append(
            {
                "candidate_id": candidate.id,
                "source": candidate.source,
                "start": round(float(candidate.start), 3),
                "end": round(float(candidate.end), 3),
                "score": round(float(score), 6),
                "overlap_candidates": len(overlaps),
            }
        )
        existing_ids.add(candidate.id)

    return sorted(result, key=lambda candidate: (candidate.start, candidate.anchor_time)), {
        "enabled": True,
        "candidate_pool": len(pool),
        "normalized_pool": len(normalized_pool),
        "scored_pool": len(scored),
        "added": len(added),
        "added_candidates": added[:25],
    }


def _context_variant_score(candidate: Candidate, segments: list[RichSegment]) -> float:
    duration = max(float(candidate.end) - float(candidate.start), 0.0)
    lead = max(float(candidate.anchor_time) - float(candidate.start), 0.0)
    tail = max(float(candidate.end) - float(candidate.anchor_time), 0.0)
    window_segments = _segments_overlapping(segments, candidate.start, candidate.end)
    text = " ".join(str(getattr(segment, "text", "") or "") for segment in window_segments).lower()
    words = re.findall(r"[a-z0-9']+", text)
    token_score = min(len(words) / 42.0, 1.0)
    source_score = {
        "quote": 0.78,
        "narrative_reveal": 0.84,
        "tension_build": 0.80,
        "controversy": 0.76,
        "story_payoff": 0.82,
        "goal_prompt": 0.76,
        "chat_velocity": 0.82,
        "dense_window": 0.48,
        "dense_inventory": 0.42,
    }.get(candidate.source, 0.52)
    setup_score = min(lead / 12.0, 1.0)
    payoff_capture = min(tail / 10.0, 1.0)
    duration_score = 1.0 - min(abs(duration - 32.0) / 54.0, 1.0)
    setup_term_score = 1.0 if any(term in text for term in _CONTEXT_SETUP_TERMS) else 0.0
    payoff_term_score = 1.0 if any(term in text for term in _CONTEXT_PAYOFF_TERMS) else 0.0
    question_score = 1.0 if "?" in text else 0.0
    signal_score = min(max(float(candidate.score_hint or 0.0), 0.0) / 8.0, 1.0)
    return max(
        0.0,
        min(
            1.0,
            source_score * 0.22
            + setup_score * 0.18
            + payoff_capture * 0.14
            + duration_score * 0.12
            + token_score * 0.12
            + setup_term_score * 0.08
            + payoff_term_score * 0.08
            + question_score * 0.03
            + signal_score * 0.03,
        ),
    )


def _segments_overlapping(segments: list[RichSegment], start: float, end: float) -> list[RichSegment]:
    return [
        segment
        for segment in segments
        if min(float(getattr(segment, "end", 0.0)), end) - max(float(getattr(segment, "start", 0.0)), start) > 0
    ]


def _candidate_region_overlap(left: Candidate, right: Candidate) -> float:
    start = max(float(left.start), float(right.start))
    end = min(float(left.end), float(right.end))
    overlap = max(0.0, end - start)
    if overlap <= 0:
        return 0.0
    left_duration = max(float(left.end) - float(left.start), 0.001)
    right_duration = max(float(right.end) - float(right.start), 0.001)
    return max(overlap / left_duration, overlap / right_duration)


def _candidate_effectively_identical(left: Candidate, right: Candidate) -> bool:
    return (
        abs(float(left.anchor_time) - float(right.anchor_time)) <= 2.0
        and abs(float(left.start) - float(right.start)) <= 1.5
        and abs(float(left.end) - float(right.end)) <= 1.5
    )


def _candidate_has_more_setup(candidate: Candidate, existing: list[Candidate]) -> bool:
    candidate_lead = float(candidate.anchor_time) - float(candidate.start)
    candidate_duration = float(candidate.end) - float(candidate.start)
    for other in existing:
        other_lead = float(other.anchor_time) - float(other.start)
        other_duration = float(other.end) - float(other.start)
        if candidate_lead >= other_lead + 4.0 and candidate_duration >= other_duration + 8.0:
            return True
    return False


def _promote_cross_platform_recall_candidates(candidates, positives: list[dict], *, max_candidates: int) -> tuple[list, dict]:
    targets = _cross_platform_recall_targets(candidates, positives)
    summary = {
        "enabled": bool(positives),
        "uncapped_candidate_windows": len(candidates),
        "positive_recall_candidate_windows": len(targets),
        "positive_recall_candidates_promoted": 0,
        "max_candidates": max_candidates,
    }
    if not positives or max_candidates <= 0 or len(candidates) <= max_candidates:
        return list(candidates), summary

    target_ids = set(targets)
    recall_candidates = [candidate for candidate in candidates if candidate.id in target_ids]
    remaining_slots = max(max_candidates - len(recall_candidates), 0)
    selected_non_targets = [
        candidate for candidate in sorted(
            [candidate for candidate in candidates if candidate.id not in target_ids],
            key=lambda candidate: (-float(candidate.score_hint), float(candidate.start), float(candidate.end), candidate.id),
        )[:remaining_slots]
    ]
    previous_cap_ids = {
        candidate.id for candidate in sorted(
            candidates,
            key=lambda candidate: (-float(candidate.score_hint), float(candidate.start), float(candidate.end), candidate.id),
        )[:max_candidates]
    }
    promoted = max(0, len([candidate for candidate in recall_candidates if candidate.id not in previous_cap_ids]))
    summary["positive_recall_candidates_promoted"] = promoted
    selected = [*recall_candidates, *selected_non_targets]
    return sorted(selected, key=lambda candidate: (candidate.start, candidate.end, candidate.anchor_time)), summary



def _apply_cross_platform_feature_boosts(features, targets: dict[str, dict]) -> dict:
    if not targets:
        return {"targets": 0, "boosted_features": 0}
    if not bool(getattr(settings, "serving_cross_platform_feature_boosts_enabled", False)):
        return {
            "targets": len(targets),
            "boosted_features": 0,
            "status": "disabled_oracle_feature_boosts",
        }
    boosted = 0
    for feature in features:
        target = targets.get(feature.candidate_id)
        if not target:
            continue
        boosted += 1
        overlap = float(target.get("positive_overlap_ratio") or 0.0)
        similarity = float(target.get("matched_cross_platform_similarity") or 0.0)
        views = float(target.get("matched_cross_platform_views") or 0.0)
        feature.cross_platform_recall_target = True
        feature.cross_platform_positive_overlap_ratio = max(feature.cross_platform_positive_overlap_ratio, overlap)
        feature.cross_platform_positive_similarity = max(feature.cross_platform_positive_similarity, similarity)
        feature.cross_platform_positive_views = max(feature.cross_platform_positive_views, views)
        feature.outcome_prior_score = max(float(feature.outcome_prior_score), min(1.0, 0.55 + overlap * 0.35))
        feature.projected_retention_score = max(float(feature.projected_retention_score), min(1.0, 0.55 + overlap * 0.25))
        feature.platform_relevance_score = max(float(feature.platform_relevance_score), 0.65)
        clip_type = str(target.get("matched_cross_platform_clip_type") or "").strip()
        if clip_type and clip_type != "general":
            feature.clip_type = clip_type
        pattern = target.get("matched_cross_platform_pattern") or {}
        pattern_confidence = float(pattern.get("confidence") or 0.0)
        if pattern_confidence > 0.0:
            feature.context_pattern_score = max(float(feature.context_pattern_score), float(pattern.get("score") or 0.0))
            feature.context_pattern_support = max(float(feature.context_pattern_support), float(pattern.get("support") or 0.0))
            feature.context_pattern_views_prior = max(float(feature.context_pattern_views_prior), float(pattern.get("views_prior") or 0.0))
            feature.context_pattern_confidence = max(float(feature.context_pattern_confidence), pattern_confidence)
            feature.context_pattern_similarity_margin = max(float(feature.context_pattern_similarity_margin), float(pattern.get("similarity_margin") or 0.0))
            if pattern_confidence >= 0.25:
                feature.context_pattern_id = str(pattern.get("pattern_id") or feature.context_pattern_id or "")
                feature.context_pattern_label = str(pattern.get("label") or feature.context_pattern_label or "")
    return {"targets": len(targets), "boosted_features": boosted}


def _probe_media_duration_sec(video_path: pathlib.Path) -> float | None:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        return None
    return duration if duration > 0 else None


def _load_complete_cached_transcript(
    *,
    cache_source: str | None,
    video_path: pathlib.Path,
    progress_cb,
) -> list[transcriber.RichSegment] | None:
    duration_sec = _probe_media_duration_sec(video_path)
    if duration_sec is None:
        return None
    return transcriber.load_cached_transcript(
        cache_source,
        expected_duration_sec=duration_sec,
        progress_cb=progress_cb,
    )


def _should_extract_full_visual_features(video_path: pathlib.Path) -> bool:
    max_duration_sec = float(settings.pipeline_full_visual_feature_max_duration_sec or 0.0)
    if max_duration_sec <= 0:
        return True
    duration_sec = _probe_media_duration_sec(video_path)
    if duration_sec is None or duration_sec <= max_duration_sec:
        return True
    logger.info(
        "pipeline_full_visual_features_skipped path=%s duration_sec=%.3f max_duration_sec=%.3f",
        video_path,
        duration_sec,
        max_duration_sec,
    )
    return False


def _feature_cache_path(cache_source: str | None, kind: str) -> pathlib.Path | None:
    digest = transcriber.stable_cache_key_for_source(cache_source)
    if not digest:
        return None
    safe_kind = re.sub(r"[^a-zA-Z0-9_.-]+", "_", kind.strip().lower())
    return pathlib.Path(settings.storage_dir) / "_feature_cache" / digest / f"{safe_kind}_frames.json"


def _audio_cache_path(cache_source: str | None) -> pathlib.Path | None:
    digest = transcriber.stable_cache_key_for_source(cache_source)
    if not digest:
        return None
    return pathlib.Path(settings.storage_dir) / "_audio_cache" / digest / "audio.wav"


def _audio_cache_metadata_path(audio_cache_path: pathlib.Path) -> pathlib.Path:
    return audio_cache_path.with_name("audio_metadata.json")


def _feature_cache_metadata(kind: str, cache_source: str, video_path: pathlib.Path | None) -> dict:
    metadata: dict[str, object] = {
        "version": _FEATURE_FRAME_CACHE_VERSION,
        "kind": kind,
        "source": cache_source,
    }
    if video_path is not None:
        duration_sec = _probe_media_duration_sec(video_path)
        metadata["duration_sec"] = round(duration_sec, 3) if duration_sec is not None else None
    if kind == "video":
        metadata["runtime_summary"] = video_features.runtime_summary()
        metadata["full_visual_feature_max_duration_sec"] = float(
            settings.pipeline_full_visual_feature_max_duration_sec or 0.0
        )
    return metadata


def _audio_cache_metadata(cache_source: str, video_path: pathlib.Path | None) -> dict:
    metadata: dict[str, object] = {
        "version": _AUDIO_CACHE_VERSION,
        "source": cache_source,
        "sample_rate_hz": 16000,
        "channels": 1,
        "codec": "pcm_s16le",
    }
    if video_path is not None:
        duration_sec = _probe_media_duration_sec(video_path)
        metadata["duration_sec"] = round(duration_sec, 3) if duration_sec is not None else None
    return metadata


def _metadata_matches(cached: object, expected: dict) -> bool:
    if not isinstance(cached, dict):
        return False
    for key, value in expected.items():
        if cached.get(key) != value:
            return False
    return True


def _frame_from_payload(frame_cls, payload: object):
    if not isinstance(payload, dict):
        return None
    if is_dataclass(frame_cls):
        allowed_fields = {field.name for field in fields(frame_cls)}
        values = {key: value for key, value in payload.items() if key in allowed_fields}
        try:
            return frame_cls(**values)
        except TypeError:
            return None
    return None


def _json_safe_cache_value(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, pathlib.Path):
        return str(value)
    if is_dataclass(value):
        return _json_safe_cache_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe_cache_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_cache_value(item) for item in value]

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _json_safe_cache_value(tolist())
        except (TypeError, ValueError):
            pass

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe_cache_value(item())
        except (TypeError, ValueError):
            pass

    return str(value)


def _link_or_copy_cache_file(source: pathlib.Path, target: pathlib.Path) -> pathlib.Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        if source.resolve(strict=False) == target.resolve(strict=False):
            return target
    except OSError:
        pass
    target.unlink(missing_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
    return target


def _job_audio_cache_target(job_id: str) -> pathlib.Path:
    return pathlib.Path(settings.storage_dir) / job_id / "audio.wav"


def _load_audio_cache(
    cache_source: str | None,
    *,
    video_path: pathlib.Path,
    job_id: str,
) -> pathlib.Path | None:
    path = _audio_cache_path(cache_source)
    if path is None or not path.exists():
        return None
    metadata_path = _audio_cache_metadata_path(path)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if path.stat().st_size <= 44:
            return None
    except (OSError, json.JSONDecodeError):
        logger.warning("pipeline_audio_cache_read_failed path=%s", path, exc_info=True)
        return None
    expected = _audio_cache_metadata(str(cache_source), video_path)
    if not _metadata_matches(metadata, expected):
        logger.info("pipeline_audio_cache_stale source=%s path=%s", cache_source, path)
        return None
    target = _job_audio_cache_target(job_id)
    try:
        _link_or_copy_cache_file(path, target)
    except OSError:
        logger.warning("pipeline_audio_cache_link_failed source=%s path=%s", cache_source, path, exc_info=True)
        return None
    logger.info("pipeline_audio_cache_hit job_id=%s source=%s path=%s", job_id, cache_source, path)
    return target


def _write_audio_cache(
    cache_source: str | None,
    *,
    video_path: pathlib.Path,
    audio_path: pathlib.Path,
) -> None:
    path = _audio_cache_path(cache_source)
    if path is None:
        return
    try:
        if not audio_path.exists() or audio_path.stat().st_size <= 44:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        write_token = f"{os.getpid()}.{uuid.uuid4().hex}"
        tmp_path = path.with_suffix(path.suffix + f".tmp.{write_token}")
        _link_or_copy_cache_file(audio_path, tmp_path)
        tmp_path.replace(path)
        metadata = _audio_cache_metadata(str(cache_source), video_path)
        metadata_path = _audio_cache_metadata_path(path)
        tmp_metadata_path = metadata_path.with_suffix(metadata_path.suffix + f".tmp.{write_token}")
        tmp_metadata_path.write_text(
            json.dumps(_json_safe_cache_value(metadata), ensure_ascii=True),
            encoding="utf-8",
        )
        tmp_metadata_path.replace(metadata_path)
        logger.info(
            "pipeline_audio_cache_written source=%s path=%s size_bytes=%s",
            cache_source,
            path,
            path.stat().st_size,
        )
    except (OSError, TypeError, ValueError):
        logger.warning("pipeline_audio_cache_write_failed source=%s path=%s", cache_source, path, exc_info=True)


def _extract_audio_cached(
    video_path: pathlib.Path,
    job_id: str,
    *,
    cache_source: str | None,
) -> pathlib.Path:
    cached = _load_audio_cache(cache_source, video_path=video_path, job_id=job_id) if cache_source else None
    if cached is not None:
        return cached
    logger.info("pipeline_audio_cache_miss job_id=%s source=%s", job_id, cache_source)
    audio_path = audio_extractor.extract(video_path, job_id)
    if cache_source:
        _write_audio_cache(cache_source, video_path=video_path, audio_path=audio_path)
    return audio_path


def _load_feature_frame_cache(cache_source: str | None, kind: str, frame_cls, metadata: dict) -> list | None:
    path = _feature_cache_path(cache_source, kind)
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("pipeline_feature_cache_read_failed kind=%s path=%s", kind, path, exc_info=True)
        return None
    if not isinstance(payload, dict):
        return None
    if not _metadata_matches(payload.get("metadata"), metadata):
        logger.info("pipeline_feature_cache_stale kind=%s path=%s", kind, path)
        return None
    raw_frames = payload.get("frames")
    if not isinstance(raw_frames, list):
        return None
    frames = []
    for raw_frame in raw_frames:
        frame = _frame_from_payload(frame_cls, raw_frame)
        if frame is None:
            logger.info("pipeline_feature_cache_invalid kind=%s path=%s", kind, path)
            return None
        frames.append(frame)
    logger.info(
        "pipeline_feature_cache_hit kind=%s source=%s frames=%s",
        kind,
        cache_source,
        len(frames),
    )
    return frames


def _write_feature_frame_cache(cache_source: str | None, kind: str, frames: list, metadata: dict) -> None:
    path = _feature_cache_path(cache_source, kind)
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": _json_safe_cache_value(metadata),
            "frames": [asdict(frame) if is_dataclass(frame) else dict(frame) for frame in frames],
        }
        payload["frames"] = _json_safe_cache_value(payload["frames"])
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
        tmp_path.replace(path)
        logger.info(
            "pipeline_feature_cache_written kind=%s source=%s frames=%s path=%s",
            kind,
            cache_source,
            len(frames),
            path,
        )
    except (OSError, TypeError, ValueError):
        logger.warning("pipeline_feature_cache_write_failed kind=%s path=%s", kind, path, exc_info=True)


def _extract_audio_features_cached(
    video_path: pathlib.Path,
    *,
    audio_path: pathlib.Path | None,
    job_id: str,
    cache_source: str | None,
    audio_cache_source: str | None = None,
) -> tuple[list[audio_features.AudioFeatureFrame], pathlib.Path | None]:
    metadata = _feature_cache_metadata("audio", cache_source, video_path) if cache_source else {}
    cached = (
        _load_feature_frame_cache(cache_source, "audio", audio_features.AudioFeatureFrame, metadata)
        if cache_source
        else None
    )
    if cached is not None:
        return cached, audio_path
    logger.info("pipeline_feature_cache_miss kind=audio source=%s", cache_source)
    if audio_path is None:
        audio_path = _extract_audio_cached(
            video_path,
            job_id,
            cache_source=audio_cache_source if audio_cache_source is not None else cache_source,
        )
    frames = audio_features.extract(audio_path)
    if cache_source:
        _write_feature_frame_cache(cache_source, "audio", frames, metadata)
    return frames, audio_path


def _extract_video_features_cached(
    video_path: pathlib.Path,
    *,
    cache_source: str | None,
    allow_extract: bool = True,
) -> list[video_features.VideoFeatureFrame]:
    metadata = _feature_cache_metadata("video", cache_source, video_path) if cache_source else {}
    cached = (
        _load_feature_frame_cache(cache_source, "video", video_features.VideoFeatureFrame, metadata)
        if cache_source
        else None
    )
    if cached is not None:
        return cached
    if not allow_extract:
        logger.info("pipeline_feature_cache_skip kind=video reason=full_visual_feature_limit source=%s", cache_source)
        return []
    logger.info("pipeline_feature_cache_miss kind=video source=%s", cache_source)
    frames = video_features.extract(video_path)
    if cache_source:
        _write_feature_frame_cache(cache_source, "video", frames, metadata)
    return frames


def _write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def _build_review_candidate_rows(
    feedback_events: list[feedback_store.CandidateFeedback],
    candidate_trace_rows: list[dict],
    *,
    limit: int,
) -> list[dict]:
    trace_by_id = {
        str(row.get("candidate_id") or "").strip(): row
        for row in candidate_trace_rows
        if str(row.get("candidate_id") or "").strip()
    }
    rows: list[dict] = []
    seen: set[str] = set()
    for event in feedback_events:
        payload = event.to_dict()
        candidate_id = str(payload.get("candidate_id") or "").strip()
        if not candidate_id or candidate_id in seen:
            continue
        seen.add(candidate_id)
        start = _optional_float_value(payload.get("final_start" if payload.get("selected") else "proposed_start"))
        end = _optional_float_value(payload.get("final_end" if payload.get("selected") else "proposed_end"))
        if start is None:
            start = _optional_float_value(payload.get("proposed_start"))
        if end is None:
            end = _optional_float_value(payload.get("proposed_end"))
        if start is None or end is None or end <= start:
            continue

        trace = trace_by_id.get(candidate_id, {})
        rows.append(
            {
                "candidate_id": candidate_id,
                "source": trace.get("source"),
                "clip_type": trace.get("clip_type"),
                "start_time": round(start, 3),
                "end_time": round(end, 3),
                "duration": round(end - start, 3),
                "rank_score": payload.get("rank_score"),
                "pre_judge_rank": trace.get("pre_judge_rank"),
                "judge_score": payload.get("judge_score"),
                "selected": bool(payload.get("selected")),
                "rejected_reason": payload.get("rejected_reason"),
                "rejection_source": payload.get("rejection_source"),
                "judge_reason": payload.get("judge_reason"),
                "feedback_rating": payload.get("feedback_rating"),
                "feedback_tags": list(payload.get("feedback_tags") or []),
                "feedback_notes": payload.get("feedback_notes"),
            }
        )

    rows.sort(key=_review_candidate_sort_key)
    return rows[: max(int(limit or 0), 0)]


def _review_candidate_sort_key(row: dict) -> tuple:
    selected_rank = 0 if bool(row.get("selected")) else 1
    judged_rank = 0 if row.get("judge_score") not in (None, "") else 1
    pre_judge_rank = row.get("pre_judge_rank")
    try:
        pre_judge_sort = int(pre_judge_rank)
    except (TypeError, ValueError):
        pre_judge_sort = 10_000
    return (
        selected_rank,
        judged_rank,
        -_float_value(row.get("judge_score")),
        pre_judge_sort,
        -_float_value(row.get("rank_score")),
        _float_value(row.get("start_time")),
    )


def _optional_float_value(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_value(value) -> float:
    parsed = _optional_float_value(value)
    return parsed if parsed is not None else 0.0


async def _await_with_optional_timeout(awaitable, timeout_sec: float | None):
    if timeout_sec is None or timeout_sec <= 0:
        return await asyncio.shield(awaitable)
    return await asyncio.wait_for(
        asyncio.shield(awaitable),
        timeout=timeout_sec,
    )


async def _await_clip_judge_or_fallback(
    awaitable,
    *,
    timeout_sec: float | None,
    fallback_candidates: list,
    job_id: str,
) -> tuple[list, bool]:
    try:
        return await _await_with_optional_timeout(awaitable, timeout_sec), False
    except asyncio.TimeoutError:
        fallback = list(fallback_candidates)
        logger.warning(
            json.dumps(
                {
                    "event": "clip_judge_stage_degraded",
                    "job_id": job_id,
                    "reason": "timeout",
                    "timeout_sec": float(timeout_sec or 0.0),
                    "fail_fast": bool(getattr(settings, "clip_judge_fail_fast", False)),
                    "fallback": "pre_judge_ranked_candidates",
                    "fallback_candidates": len(fallback),
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )
        return fallback, True


def _build_judge_recall_reserved_order(
    ranked_candidates: list[RankedCandidate],
    recall_signal_rows: list[dict] | tuple[dict, ...],
    judge_top_k: int,
) -> tuple[list[RankedCandidate], dict]:
    """Reserve some judge slots for deeper candidates with strong recall signals."""

    total_candidates = len(ranked_candidates)
    judge_limit = min(max(int(judge_top_k or 0), 0), total_candidates)
    enabled = bool(getattr(settings, "judge_recall_reserve_enabled", True))
    summary = {
        "enabled": enabled,
        "judge_top_k": judge_limit,
        "normal_slots": judge_limit,
        "reserve_slots": 0,
        "pool_k": judge_limit,
        "selected": 0,
        "selected_candidates": [],
        "min_score": float(getattr(settings, "judge_recall_reserve_min_score", 0.12) or 0.0),
    }
    if not enabled or judge_limit <= 0:
        return list(ranked_candidates), summary

    reserve_slots = min(
        max(int(getattr(settings, "judge_recall_reserve_slots", 0) or 0), 0),
        judge_limit,
    )
    if reserve_slots <= 0:
        return list(ranked_candidates), summary

    normal_slots = max(judge_limit - reserve_slots, 0)
    pool_k = min(
        max(int(getattr(settings, "judge_recall_reserve_pool_k", judge_limit) or judge_limit), judge_limit),
        total_candidates,
    )
    min_score = float(getattr(settings, "judge_recall_reserve_min_score", 0.12) or 0.0)
    recall_signal_by_id = {
        _base_candidate_id(str(row.get("candidate_id") or row.get("id") or "")): row
        for row in recall_signal_rows
        if row.get("candidate_id") or row.get("id")
    }
    normal = list(ranked_candidates[:normal_slots])
    normal_ids = {_base_candidate_id(item.candidate.id) for item in normal}
    scored: list[tuple[float, int, RankedCandidate, dict]] = []
    for index, ranked in enumerate(ranked_candidates[normal_slots:pool_k], start=normal_slots + 1):
        candidate_id = _base_candidate_id(ranked.candidate.id)
        if candidate_id in normal_ids:
            continue
        recall_signal = recall_signal_by_id.get(candidate_id, {})
        protection = high_recall_protection.oracle_free_recall_score(ranked)
        reserve_score = _judge_recall_reserve_score(ranked, index, pool_k, protection)
        if reserve_score < min_score:
            continue
        scored.append((reserve_score, index, ranked, recall_signal))

    scored.sort(
        key=lambda item: (
            item[0],
            _safe_float(item[2].rank_score),
        ),
        reverse=True,
    )
    selected: list[tuple[float, int, RankedCandidate, dict]] = []
    for item in scored:
        if len(selected) >= reserve_slots:
            break
        ranked = item[2]
        if any(_candidate_overlap_ratio(ranked, selected_item[2]) > 0.65 for selected_item in selected):
            continue
        if any(_candidate_overlap_ratio(ranked, normal_item) > 0.65 for normal_item in normal):
            continue
        selected.append(item)
    if len(selected) < reserve_slots:
        selected_ids = {_base_candidate_id(item[2].candidate.id) for item in selected}
        for item in scored:
            if len(selected) >= reserve_slots:
                break
            candidate_id = _base_candidate_id(item[2].candidate.id)
            if candidate_id in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(candidate_id)

    reserve = [
        replace(
            ranked,
            rationale=[*ranked.rationale, "judge_recall_reserved"],
        )
        for _, _, ranked, _ in selected
    ]
    reserved_ids = {_base_candidate_id(item.candidate.id) for item in reserve}
    rest = [
        ranked
        for ranked in ranked_candidates
        if _base_candidate_id(ranked.candidate.id) not in normal_ids | reserved_ids
    ]
    summary.update(
        {
            "normal_slots": normal_slots,
            "reserve_slots": reserve_slots,
            "pool_k": pool_k,
            "selected": len(reserve),
            "selected_candidates": [
                {
                    "candidate_id": ranked.candidate.id,
                    "source": ranked.candidate.source,
                    "start": round(float(ranked.candidate.start), 3),
                    "end": round(float(ranked.candidate.end), 3),
                    "pre_judge_rank": index,
                    "pre_judge_rank_score": round(_safe_float(ranked.rank_score), 6),
                    "reserve_score": round(score, 6),
                    "high_recall_score": protection["score"],
                    "high_recall_reasons": list(protection["reasons"]),
                }
                for score, index, ranked, row in selected
                for protection in [high_recall_protection.oracle_free_recall_score(ranked)]
            ],
        }
    )
    return [*normal, *reserve, *rest], summary


def _judge_recall_reserve_score(
    ranked: RankedCandidate,
    pre_judge_rank: int,
    pool_k: int,
    protection: dict[str, object] | None = None,
) -> float:
    features = ranked.features
    rank_score = min(max(_safe_float(ranked.rank_score) / 12.0, 0.0), 1.0)
    heuristic_score = min(max(_safe_float(ranked.heuristic_rank_score) / 12.0, 0.0), 1.0)
    score_hint = min(max(_safe_float(ranked.candidate.score_hint) / 10.0, 0.0), 1.0)
    rank_position = 1.0 - (max(pre_judge_rank - 1, 0) / max(pool_k, 1))
    feature_score = 0.0
    if features is not None:
        if bool(getattr(settings, "serving_cross_platform_feature_boosts_enabled", False)) and bool(
            getattr(features, "cross_platform_recall_target", False)
        ):
            overlap = min(max(_safe_float(getattr(features, "cross_platform_positive_overlap_ratio", 0.0)), 0.0), 1.0)
            similarity = min(max(_safe_float(getattr(features, "cross_platform_positive_similarity", 0.0)), 0.0), 1.0)
            feature_score += 4.0 + overlap * 1.5 + similarity * 0.5
        feature_score += 0.24 * min(max(_safe_float(features.payoff_score) / 10.0, 0.0), 1.0)
        context_completeness = min(max(_safe_float(getattr(features, "context_completeness_score", 0.0)), 0.0), 1.0)
        context_strength = min(max(_safe_float(getattr(features, "context_strength_score", 0.0)), 0.0), 1.0)
        setup_present = min(max(_safe_float(getattr(features, "setup_present_score", 0.0)), 0.0), 1.0)
        payoff_present = min(max(_safe_float(getattr(features, "payoff_present_score", 0.0)), 0.0), 1.0)
        pattern_score = min(max(_safe_float(getattr(features, "context_pattern_score", 0.0)), 0.0), 1.0)
        pattern_margin = min(max(_safe_float(getattr(features, "context_pattern_similarity_margin", 0.0)) / 0.08, 0.0), 1.0)
        feature_score += 0.55 * context_completeness
        feature_score += 0.38 * context_strength
        feature_score += 0.24 * setup_present
        feature_score += 0.24 * payoff_present
        feature_score += 0.26 * pattern_score
        feature_score += 0.18 * pattern_margin
        feature_score += 0.18 * min(max(_safe_float(features.projected_retention_score), 0.0), 1.0)
        feature_score += 0.16 * min(max(_safe_float(features.multimodal_proxy_score), 0.0), 1.0)
        feature_score += 0.16 * min(max(_safe_float(features.outcome_prior_score), 0.0), 1.0)
        feature_score += 0.10 * min(max(_safe_float(features.hook_strength) / 10.0, 0.0), 1.0)
        feature_score += 0.10 * min(max(_safe_float(features.standalone_score), 0.0), 1.0)
    protection_score = 0.0
    if isinstance(protection, dict):
        protection_score = min(max(_safe_float(protection.get("score")), 0.0), 1.0)
    source_bonus = 0.06 if ranked.candidate.source in {"dense_inventory", "dense_window", "dense"} else 0.0
    return (
        0.35 * rank_score
        + 0.20 * heuristic_score
        + 0.15 * score_hint
        + 0.08 * rank_position
        + 0.65 * protection_score
        + feature_score
        + source_bonus
    )


def _candidate_overlap_ratio(left: RankedCandidate, right: RankedCandidate) -> float:
    start = max(float(left.candidate.start), float(right.candidate.start))
    end = min(float(left.candidate.end), float(right.candidate.end))
    overlap = max(0.0, end - start)
    if overlap <= 0:
        return 0.0
    left_duration = max(float(left.candidate.end) - float(left.candidate.start), 0.001)
    right_duration = max(float(right.candidate.end) - float(right.candidate.start), 0.001)
    return max(overlap / left_duration, overlap / right_duration)


def _base_candidate_id(candidate_id: str) -> str:
    return candidate_id.split("::", 1)[0]


def _unique_candidates_by_id(candidates: list[Candidate]) -> list[Candidate]:
    unique: list[Candidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate_id = str(candidate.id)
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        unique.append(candidate)
    return unique


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


async def run_pipeline(
    job_id: str,
    url: str | None = None,
    video_path: pathlib.Path | None = None,
    profile: str | None = None,
    strategy: dict | None = None,
) -> None:
    effective_profile = profile or settings.clip_profile
    normalized_strategy = clip_strategy.normalize_strategy(strategy)
    cross_platform_positive_windows, cross_platform_positive_summary = _load_cross_platform_positive_windows(
        effective_profile,
        url,
    )
    diagnostic_positive_windows, diagnostic_positive_summary = _load_cross_platform_positive_windows(
        effective_profile,
        url,
        force_load=True,
    )
    loop = asyncio.get_running_loop()
    audio_path: pathlib.Path | None = None
    try:
        _raise_if_cancelled(job_id)
        if video_path is None:
            update_job(job_id, status="downloading", progress_pct=5)
            video_path = await loop.run_in_executor(None, downloader.download, url, job_id)
        else:
            update_job(job_id, status="downloading", progress_pct=19)

        def _transcription_progress(done_chunks: int, total_chunks: int, _chunk_path) -> None:
            # Reserve 30-55% for ASR progress.
            pct = 30 + int((done_chunks / max(total_chunks, 1)) * 25)
            update_job(job_id, status="transcribing", progress_pct=min(max(pct, 30), 55))

        _raise_if_cancelled(job_id)
        transcript_cache_source = (
            _transcript_cache_source(url, video_path)
            if _transcript_cache_enabled_for_job(job_id, url)
            else None
        )
        feature_cache_source = _feature_cache_source(
            job_id=job_id,
            url=url,
            video_path=video_path,
        )
        audio_cache_source = transcript_cache_source or feature_cache_source
        chat_path = None
        if settings.enable_chat_analysis:
            update_job(job_id, status="downloading_chat", progress_pct=28)
            chat_path = await loop.run_in_executor(
                None,
                chat_analyzer.download_chat,
                url,
                job_id,
                pathlib.Path(settings.storage_dir) / job_id,
            )

        update_job(job_id, status="transcribing", progress_pct=30)
        transcript = await loop.run_in_executor(
            None,
            lambda: (
                _load_complete_cached_transcript(
                    cache_source=transcript_cache_source,
                    video_path=video_path,
                    progress_cb=_transcription_progress,
                )
                if transcript_cache_source
                else None
            ),
        )
        transcript_cache_hit = transcript is not None
        if transcript_cache_hit:
            logger.info(
                "pipeline_transcript_cache_hit job_id=%s source=%s segments=%s",
                job_id,
                transcript_cache_source,
                len(transcript),
            )
        else:
            update_job(job_id, status="extracting_audio", progress_pct=20)
            audio_path = await loop.run_in_executor(
                None,
                lambda: _extract_audio_cached(
                    video_path,
                    job_id,
                    cache_source=audio_cache_source,
                ),
            )

            # Start non-transcription tasks in the background, but keep transcription
            # as its own visible phase so long uploads do not appear stuck.
            update_job(job_id, status="transcribing", progress_pct=30)
            transcript = await loop.run_in_executor(
                None,
                lambda: transcriber.transcribe_with_progress(
                    audio_path,
                    _transcription_progress,
                    cache_key=transcript_cache_source,
                    isolate_chunks=bool(settings.pipeline_isolate_transcription_chunks),
                    enable_diarization=bool(settings.pipeline_enable_diarization),
                ),
            )

        overlap_feature_extraction = bool(settings.pipeline_overlap_feature_extraction)
        extract_full_visual_features = _should_extract_full_visual_features(video_path)
        audio_features_task = None
        video_features_task = None
        if overlap_feature_extraction:
            audio_features_task = loop.run_in_executor(
                None,
                lambda: _extract_audio_features_cached(
                    video_path,
                    audio_path=audio_path,
                    job_id=job_id,
                    cache_source=feature_cache_source,
                    audio_cache_source=audio_cache_source,
                ),
            )
            if extract_full_visual_features or feature_cache_source:
                video_features_task = loop.run_in_executor(
                    None,
                    lambda: _extract_video_features_cached(
                        video_path,
                        cache_source=feature_cache_source,
                        allow_extract=extract_full_visual_features,
                    ),
                )

        _raise_if_cancelled(job_id)
        update_job(job_id, status="extracting_features", progress_pct=56)
        if overlap_feature_extraction:
            visual_frames = await video_features_task if video_features_task is not None else []
            audio_frames, audio_path = (
                await audio_features_task if audio_features_task is not None else ([], audio_path)
            )
        else:
            audio_frames, audio_path = await loop.run_in_executor(
                None,
                lambda: _extract_audio_features_cached(
                    video_path,
                    audio_path=audio_path,
                    job_id=job_id,
                    cache_source=feature_cache_source,
                    audio_cache_source=audio_cache_source,
                ),
            )
            _raise_if_cancelled(job_id)
            visual_frames = []
            if extract_full_visual_features or feature_cache_source:
                visual_frames = await loop.run_in_executor(
                    None,
                    lambda: _extract_video_features_cached(
                        video_path,
                        cache_source=feature_cache_source,
                        allow_extract=extract_full_visual_features,
                    ),
                )

        # Merge audio features, then visual features into transcript segments
        transcript = audio_features.enrich_segments(transcript, audio_frames)
        transcript = video_features.enrich_segments(transcript, visual_frames)

        # Speaker diarization — MFCC clustering (runs when pyannote is unavailable)
        # pyannote already ran inside transcriber.transcribe() when hf_token is set
        if settings.pipeline_enable_diarization and not settings.hf_token and audio_path is None:
            update_job(job_id, status="extracting_audio", progress_pct=64)
            audio_path = await loop.run_in_executor(
                None,
                lambda: _extract_audio_cached(
                    video_path,
                    job_id,
                    cache_source=audio_cache_source,
                ),
            )
        if audio_path is not None and settings.pipeline_enable_diarization and not settings.hf_token:
            update_job(job_id, status="diarizing", progress_pct=65)
            transcript = await loop.run_in_executor(
                None, audio_features.diarize_speakers, transcript, audio_path
            )

        # Chat-replay enrichment (best-effort — no-op when chat_path is None)
        transcript = chat_analyzer.enrich_segments(transcript, chat_path)

        # Attach composite hype scores before saving or analyzing
        _raise_if_cancelled(job_id)
        transcript = scorer.score_segments(transcript)
        chapters = await loop.run_in_executor(
            None,
            chapter_segmenter.segment_chapters,
            transcript,
        )
        score_timeline = scorer.build_score_timeline(transcript)
        detected_peaks = await loop.run_in_executor(None, peak_detector.detect_peaks, score_timeline)
        peak_candidates = await loop.run_in_executor(
            None,
            peak_detector.build_peak_candidates,
            detected_peaks,
            transcript,
        )

        update_job(job_id, status="analyzing", progress_pct=72)
        semantic_candidates = await loop.run_in_executor(
            None,
            candidate_generators.generate_semantic_candidates,
            transcript,
            chapters,
            normalized_strategy,
        )
        dense_candidates = []
        if settings.enable_whole_vod_dense_candidates:
            dense_candidates = await loop.run_in_executor(
                None,
                candidate_generators.generate_dense_window_candidates,
                transcript,
                chapters,
                normalized_strategy,
            )
        fallback_candidates = []
        if not peak_candidates:
            fallback_candidates = await loop.run_in_executor(
                None,
                candidate_generators.generate_fallback_candidates,
                transcript,
                chapters,
                normalized_strategy,
        )
        heuristic_merge_audit_rows: list[dict] = []
        dense_inventory_candidates = []
        cross_platform_dense_recall_summary: dict[str, object] = {"enabled": False, "added": 0}
        if bool(getattr(settings, "enable_dense_candidate_inventory", True)):
            dense_inventory_cap = max(int(settings.dense_candidate_inventory_max_candidates or 0), 0)
            dense_inventory_candidates = await loop.run_in_executor(
                None,
                lambda: candidate_generators.generate_dense_candidate_inventory(
                    transcript,
                    chapters,
                    normalized_strategy,
                    durations=_parse_float_list(settings.dense_candidate_inventory_durations_sec),
                    stride_sec=float(settings.dense_candidate_inventory_stride_sec),
                    max_candidates=dense_inventory_cap,
                ),
            )
            dense_inventory_candidates, cross_platform_dense_recall_summary = _promote_cross_platform_recall_candidates(
                dense_inventory_candidates,
                cross_platform_positive_windows,
                max_candidates=dense_inventory_cap,
            )
        dense_recall_preserve_ids = _cross_platform_recall_preserve_ids(
            dense_inventory_candidates,
            cross_platform_positive_windows,
        )
        ranker_recall_pool = [
            *semantic_candidates,
            *dense_candidates,
            *dense_inventory_candidates,
        ]
        merge_recall_carry_summary: dict[str, object] = {"enabled": False, "added": 0}
        recall_carry_candidates: list[Candidate] = []
        recall_carry_ids: set[str] = set()
        if bool(getattr(settings, "merge_recall_carry_forward_enabled", True)):
            recall_carry_candidates = await loop.run_in_executor(
                None,
                lambda: candidate_generators.select_temporal_recall_candidates(
                    ranker_recall_pool,
                    bucket_sec=float(getattr(settings, "merge_recall_carry_bucket_sec", 60.0) or 60.0),
                    per_source_bucket=max(
                        int(getattr(settings, "merge_recall_carry_per_source_bucket", 1) or 0),
                        0,
                    ),
                    max_candidates=max(
                        int(getattr(settings, "merge_recall_carry_max_candidates", 700) or 0),
                        0,
                    ),
                ),
            )
            recall_carry_ids = {str(candidate.id) for candidate in recall_carry_candidates}
            recall_carry_cap = max(len(recall_carry_candidates), 1)
            merge_recall_carry_summary = {
                "enabled": True,
                "candidate_pool": len(ranker_recall_pool),
                "selected_candidates": len(recall_carry_candidates),
                "added": 0,
                "output_candidates": 0,
                "bucket_sec": float(getattr(settings, "merge_recall_carry_bucket_sec", 60.0) or 60.0),
                "per_source_bucket": max(
                    int(getattr(settings, "merge_recall_carry_per_source_bucket", 1) or 0),
                    0,
                ),
                "max_candidates": max(
                    int(getattr(settings, "merge_recall_carry_max_candidates", 700) or 0),
                    0,
                ),
                "sample_candidate_ids": sorted(recall_carry_ids)[:25],
            }
        context_variant_summary: dict[str, object] = {"enabled": False, "added": 0}
        direct_ranker_source_cap = max(
            int(getattr(settings, "dense_candidate_inventory_source_cap", 700) or 0),
            0,
        )
        source_cap_overrides = {
            "dense_window": max(int(settings.whole_vod_dense_source_cap or 0), 0),
            "dense_inventory": direct_ranker_source_cap,
        }
        if recall_carry_candidates:
            recall_carry_cap = max(len(recall_carry_candidates), 1)
            for source in {candidate.source for candidate in recall_carry_candidates}:
                source_cap_overrides[source] = max(
                    int(source_cap_overrides.get(source, 0) or 0),
                    recall_carry_cap,
                )
        direct_supplemental_candidates = _unique_candidates_by_id(
            [
                *recall_carry_candidates,
                *semantic_candidates,
                *dense_candidates,
                *dense_inventory_candidates,
            ]
        )
        heuristic_candidates = await loop.run_in_executor(
            None,
            lambda: candidate_generators.merge_prioritized_candidates(
                peak_candidates or fallback_candidates,
                direct_supplemental_candidates,
                strategy=normalized_strategy,
                supplemental_source_cap=4,
                source_cap_overrides=source_cap_overrides,
                preserve_supplemental_ids=recall_carry_ids | dense_recall_preserve_ids,
                preserve_context_variants=True,
                max_context_variants_per_region=8 if recall_carry_candidates else 3,
                audit_rows=heuristic_merge_audit_rows,
                audit_stage="direct_ranker_merge",
            ),
        )
        if merge_recall_carry_summary.get("enabled"):
            merge_recall_carry_summary["added"] = sum(
                1 for candidate in heuristic_candidates if str(candidate.id) in recall_carry_ids
            )
            merge_recall_carry_summary["output_candidates"] = len(heuristic_candidates)
        candidates = heuristic_candidates
        normalized_candidates = await loop.run_in_executor(
            None,
            candidate_normalizer.normalize_candidates,
            candidates,
            transcript,
            settings.max_clip_duration,
        )
        candidate_features = await loop.run_in_executor(
            None,
            lambda: feature_builder.build_features(
                normalized_candidates,
                transcript,
                effective_profile,
                False,
                None,
                None,
                chapters,
                normalized_strategy,
                video_path=video_path,
            ),
        )
        normalized_recall_targets = _cross_platform_recall_targets(
            normalized_candidates,
            cross_platform_positive_windows,
            profile=effective_profile,
        )
        cross_platform_feature_boost_summary = _apply_cross_platform_feature_boosts(
            candidate_features,
            normalized_recall_targets,
        )
        ranked_candidates = await loop.run_in_executor(
            None,
            selector.rank_candidates,
            normalized_candidates,
            candidate_features,
            effective_profile,
            normalized_strategy,
        )
        shortlist_top_k = min(max(settings.shortlist_face_reaction_top_k, 0), len(ranked_candidates))
        llm_top_k = min(max(settings.shortlist_llm_top_k, 0), len(ranked_candidates))
        if shortlist_top_k > 0:
            transcript = await loop.run_in_executor(
                None,
                video_features.refine_face_reactions,
                transcript,
                video_path,
                [ranked.candidate for ranked in ranked_candidates[:shortlist_top_k]],
                settings.shortlist_face_reaction_sample_sec,
            )
        if shortlist_top_k > 0 or llm_top_k > 0:
            llm_candidate_ids = {
                ranked.candidate.id for ranked in ranked_candidates[:llm_top_k]
            } if llm_top_k > 0 else set()
            candidate_features = await loop.run_in_executor(
                None,
                lambda: feature_builder.build_features(
                    normalized_candidates,
                    transcript,
                    effective_profile,
                    True,
                    llm_candidate_ids,
                    None,
                    chapters,
                    normalized_strategy,
                    video_path=video_path,
                ),
            )
            cross_platform_feature_boost_summary = _apply_cross_platform_feature_boosts(
                candidate_features,
                normalized_recall_targets,
            )
            ranked_candidates = await loop.run_in_executor(
                None,
                selector.rank_candidates,
                normalized_candidates,
                candidate_features,
                effective_profile,
                normalized_strategy,
            )
        ranked_candidates_pre_judge = list(ranked_candidates)
        judged_ranked_candidates = list(ranked_candidates)
        judge_top_k = (
            len(ranked_candidates)
            if settings.judge_all_candidates
            else min(max(settings.shortlist_judge_top_k, 0), len(ranked_candidates))
        )
        judge_input_candidates, judge_recall_reserve_summary = _build_judge_recall_reserved_order(
            ranked_candidates,
            [],
            judge_top_k,
        )
        job_dir = pathlib.Path(settings.storage_dir) / job_id
        debug_dir = job_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        sampled_keyframes: dict[str, list[keyframe_sampler.SampledKeyframe]] = {}
        judge_provider = (getattr(settings, "judge_provider", None) or "auto").strip().lower()
        clip_judge_stage_status = {
            "enabled": bool(settings.enable_clip_judge),
            "attempted": False,
            "status": "pending" if settings.enable_clip_judge and judge_top_k > 0 else "disabled",
            "reason": None,
            "error_type": None,
            "error": None,
            "candidates": len(ranked_candidates),
            "judge_top_k": judge_top_k,
            "judge_provider": judge_provider,
            "stage_timeout_sec": 0.0,
            "provider_request_timeout_sec": float(settings.clip_judge_timeout_sec or 0.0),
            "fail_fast": bool(getattr(settings, "clip_judge_fail_fast", False)),
            "provider_fallback_enabled": bool(getattr(settings, "judge_provider_fallback_enabled", True)),
            "sample_count": int(getattr(settings, "clip_judge_sample_count", 1) or 1),
            "uncertain_sample_count": int(getattr(settings, "clip_judge_uncertain_sample_count", 1) or 1),
            "cache_enabled": bool(getattr(settings, "clip_judge_cache_enabled", True)),
            "fallback": None,
            "fallback_candidates": 0,
            "processed_candidates": 0,
            "attempted_candidates": 0,
            "recall_filtered_candidates": 0,
            "judged_candidates": 0,
            "elapsed_sec": 0.0,
            "judge_input_candidates": len(judge_input_candidates),
            "judge_recall_reserve": judge_recall_reserve_summary,
            "cross_platform_recall": {
                **cross_platform_positive_summary,
                "normalized_recall_targets": len(normalized_recall_targets),
                **cross_platform_feature_boost_summary,
            },
        }

        def _clip_judge_progress(progress: dict[str, object]) -> None:
            processed = int(progress.get("processed_candidates") or 0)
            total = max(int(progress.get("total_candidates") or judge_top_k or 0), 1)
            clip_judge_stage_status.update(
                {
                    "status": "running",
                    "processed_candidates": processed,
                    "attempted_candidates": int(progress.get("attempted_candidates") or 0),
                    "judged_candidates": int(progress.get("judged_candidates") or 0),
                    "recall_filtered_candidates": int(progress.get("recall_filtered_candidates") or 0),
                    "current_candidate_id": progress.get("candidate_id"),
                    "current_candidate_start": progress.get("candidate_start"),
                    "current_candidate_end": progress.get("candidate_end"),
                    "elapsed_sec": float(progress.get("elapsed_sec") or 0.0),
                }
            )
            _write_json(debug_dir / "clip_judge_stage.json", clip_judge_stage_status)
            progress_fraction = min(max(processed / total, 0.0), 1.0)
            update_job(
                job_id,
                status="judging",
                progress_pct=76 + int(progress_fraction * 10),
            )

        if not settings.enable_clip_judge:
            clip_judge_stage_status.update(
                {
                    "status": "disabled",
                    "reason": "enable_clip_judge_false",
                }
            )
            logger.info(
                json.dumps(
                    {
                        "event": "clip_judge_stage_disabled",
                        "job_id": job_id,
                        "reason": "enable_clip_judge_false",
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                )
            )
        if settings.enable_clip_judge and judge_top_k > 0:
            update_job(job_id, status="judging", progress_pct=76)
            use_anthropic_two_tier = (
                bool(settings.judge_two_tier_enabled)
                and bool(settings.anthropic_api_key)
                and bool(settings.enable_multimodal_clip_judge)
                and int(settings.judge_topk_multimodal_k) > 0
                and judge_provider in {"", "auto", "anthropic", "anthropic_text", "anthropic_multimodal"}
            )
            clip_judge_stage_status.update(
                {
                    "attempted": True,
                    "status": "running",
                    "anthropic_two_tier": use_anthropic_two_tier,
                }
            )
            logger.info(
                json.dumps(
                    {
                        "event": "clip_judge_stage_start",
                        "job_id": job_id,
                        "candidates": len(ranked_candidates),
                        "judge_input_candidates": len(judge_input_candidates),
                        "judge_top_k": judge_top_k,
                        "judge_recall_reserve": judge_recall_reserve_summary,
                        "judge_provider": judge_provider,
                        "anthropic_two_tier": use_anthropic_two_tier,
                        "sample_count": int(getattr(settings, "clip_judge_sample_count", 1) or 1),
                        "uncertain_sample_count": int(getattr(settings, "clip_judge_uncertain_sample_count", 1) or 1),
                        "fail_fast": bool(getattr(settings, "clip_judge_fail_fast", False)),
                        "provider_fallback_enabled": bool(getattr(settings, "judge_provider_fallback_enabled", True)),
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                )
            )
            try:
                if use_anthropic_two_tier:
                    text_judge_future = loop.run_in_executor(
                        None,
                        lambda: clip_judge.rerank_candidates(
                            judge_input_candidates,
                            transcript,
                            effective_profile,
                            judge_top_k,
                            "anthropic",
                            None,
                            progress_cb=_clip_judge_progress,
                        ),
                    )
                    judged_ranked_candidates = await _await_with_optional_timeout(text_judge_future, None)
                    judge_degraded = False
                    if judge_degraded:
                        clip_judge_stage_status.update(
                            {
                                "status": "degraded",
                                "reason": "timeout",
                                "fallback": "pre_judge_ranked_candidates",
                                "fallback_candidates": len(ranked_candidates_pre_judge),
                            }
                        )
                    if not judge_degraded:
                        multimodal_top_k = min(
                            max(int(settings.judge_topk_multimodal_k), 0),
                            len(judged_ranked_candidates),
                        )
                        if multimodal_top_k > 0:
                            sampled_keyframes = await loop.run_in_executor(
                                None,
                                keyframe_sampler.sample_candidate_keyframes,
                                video_path,
                                judged_ranked_candidates[:multimodal_top_k],
                                debug_dir,
                                settings.judge_keyframes_per_candidate,
                            )
                            multimodal_judge_future = loop.run_in_executor(
                                None,
                                lambda: clip_judge.rerank_candidates(
                                    judged_ranked_candidates,
                                    transcript,
                                    effective_profile,
                                    multimodal_top_k,
                                    "anthropic_multimodal",
                                    sampled_keyframes,
                                    progress_cb=_clip_judge_progress,
                                ),
                            )
                            judged_ranked_candidates = await _await_with_optional_timeout(
                                multimodal_judge_future,
                                None,
                            )
                            _judge_degraded = False
                            if _judge_degraded:
                                clip_judge_stage_status.update(
                                    {
                                        "status": "degraded",
                                        "reason": "timeout",
                                        "fallback": "text_judged_candidates",
                                        "fallback_candidates": len(judged_ranked_candidates),
                                    }
                                )
                else:
                    multimodal_provider_allowed = judge_provider in {
                        "",
                        "auto",
                        "anthropic",
                        "anthropic_multimodal",
                        "gemini",
                    }
                    multimodal_provider_available = bool(
                        multimodal_provider_allowed and (settings.gemini_api_key or settings.anthropic_api_key)
                    )
                    if settings.enable_multimodal_clip_judge and multimodal_provider_available:
                        sampled_keyframes = await loop.run_in_executor(
                            None,
                            keyframe_sampler.sample_candidate_keyframes,
                            video_path,
                            judge_input_candidates[:judge_top_k],
                            debug_dir,
                            settings.judge_keyframes_per_candidate,
                        )
                    judge_future = loop.run_in_executor(
                        None,
                        lambda: clip_judge.rerank_candidates(
                            judge_input_candidates,
                            transcript,
                            effective_profile,
                            judge_top_k,
                            judge_provider,
                            sampled_keyframes,
                            progress_cb=_clip_judge_progress,
                        ),
                    )
                    judged_ranked_candidates = await _await_with_optional_timeout(judge_future, None)
                    _judge_degraded = False
                    if _judge_degraded:
                        clip_judge_stage_status.update(
                            {
                                "status": "degraded",
                                "reason": "timeout",
                                "fallback": "pre_judge_ranked_candidates",
                                "fallback_candidates": len(ranked_candidates_pre_judge),
                            }
                        )
            except clip_judge.ClipJudgeError as exc:
                clip_judge_stage_status.update(
                    {
                        "status": "failed" if bool(getattr(settings, "clip_judge_fail_fast", False)) else "degraded",
                        "reason": "llm_judge_failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "fallback": "pre_judge_ranked_candidates",
                        "fallback_candidates": len(ranked_candidates_pre_judge),
                    }
                )
                logger.error(
                    json.dumps(
                        {
                            "event": "clip_judge_stage_failed",
                            "job_id": job_id,
                            "reason": "llm_judge_failed",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "fail_fast": bool(getattr(settings, "clip_judge_fail_fast", False)),
                        },
                        ensure_ascii=True,
                        sort_keys=True,
                    )
                )
                if bool(getattr(settings, "clip_judge_fail_fast", False)):
                    raise
            else:
                judged_count = sum(
                    1 for item in judged_ranked_candidates if item.judge_result is not None
                )
                clip_judge_stage_status["judged_candidates"] = judged_count
                if clip_judge_stage_status.get("status") == "running":
                    clip_judge_stage_status["status"] = "ok" if judged_count > 0 else "degraded"
                    if judged_count <= 0:
                        clip_judge_stage_status.update(
                            {
                                "reason": "empty_judge_result",
                                "fallback": "pre_judge_ranked_candidates",
                                "fallback_candidates": len(ranked_candidates_pre_judge),
                            }
                        )
                logger.info(
                    json.dumps(
                        {
                            "event": "clip_judge_stage_done",
                            "job_id": job_id,
                            "judged_candidates": judged_count,
                            "candidates": len(judged_ranked_candidates),
                        },
                        ensure_ascii=True,
                        sort_keys=True,
                    )
                )
        _write_json(debug_dir / "clip_judge_stage.json", clip_judge_stage_status)
        quality_gate_result = await loop.run_in_executor(
            None,
            selector.apply_serving_quality_gate,
            judged_ranked_candidates,
        )
        logger.info(
            json.dumps(
                {
                    "event": "serving_quality_gate_done",
                    "job_id": job_id,
                    **quality_gate_result.summary,
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )
        final_top_k = 12
        judge_completed_with_results = (
            bool(settings.enable_clip_judge)
            and bool(clip_judge_stage_status.get("attempted"))
            and int(clip_judge_stage_status.get("judged_candidates") or 0) > 0
        )
        if judge_completed_with_results:
            final_selection_result = await loop.run_in_executor(
                None,
                lambda: selector.select_post_judge_final_candidates(
                    quality_gate_result.kept,
                    final_top_k,
                    strategy=normalized_strategy,
                ),
            )
        else:
            def _select_without_judge_results():
                selected = selector.select_diverse_ranked_candidates(
                    quality_gate_result.kept,
                    final_top_k,
                    strategy=normalized_strategy,
                )
                selected_judged = sum(1 for item in selected if item.judge_result is not None)
                return SimpleNamespace(
                    selected=selected,
                    summary={
                        "policy": "pre_judge_diverse_selection",
                        "reason": (
                            "clip_judge_disabled"
                            if not settings.enable_clip_judge
                            else "clip_judge_no_judged_results"
                        ),
                        "input_candidates": len(quality_gate_result.kept),
                        "selected_candidates": len(selected),
                        "selected_judged_candidates": selected_judged,
                        "selected_unjudged_candidates": len(selected) - selected_judged,
                        "fallback_unjudged_candidates": 0,
                        "unjudged_fallback_enabled": False,
                        "judge_only_final_selection": False,
                        "top_k": int(final_top_k),
                        "strategy_platform": normalized_strategy.get("platform"),
                        "selected": [
                            {
                                "candidate_id": item.candidate.id,
                                "source": item.candidate.source,
                                "start": round(float(item.candidate.start), 3),
                                "end": round(float(item.candidate.end), 3),
                                "rank_score": round(float(item.rank_score), 3),
                                "judge_score": (
                                    round(float(item.judge_result.judge_score), 3)
                                    if item.judge_result is not None
                                    else None
                                ),
                            }
                            for item in selected
                        ],
                    },
                )

            final_selection_result = await loop.run_in_executor(
                None,
                _select_without_judge_results,
            )
        ranked_candidates = final_selection_result.selected
        logger.info(
            json.dumps(
                {
                    "event": "post_judge_final_selection_done",
                    "job_id": job_id,
                    **final_selection_result.summary,
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )
        boundary_refinement_limit = len(ranked_candidates)
        refine_future = loop.run_in_executor(
            None,
            analyzer.refine_candidates,
            ranked_candidates,
            transcript,
            boundary_refinement_limit,
            effective_profile,
            normalized_strategy,
        )
        try:
            moments = await _await_with_optional_timeout(
                refine_future,
                settings.analyzer_refine_timeout_sec,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "analyzer.refine_candidates timed out after %.1fs; no moments will be produced for this job",
                settings.analyzer_refine_timeout_sec,
            )
            moments = []

        transcripts_dir = pathlib.Path(__file__).parents[1] / "transcripts"
        transcripts_dir.mkdir(exist_ok=True)
        (transcripts_dir / f"{job_id}.txt").write_text(
            transcriber.format_transcript(transcript), encoding="utf-8"
        )
        (transcripts_dir / f"{job_id}.jsonl").write_text(
            transcriber.format_jsonl(transcript), encoding="utf-8"
        )
        _write_jsonl(debug_dir / "segments.jsonl", _segment_records(transcript))
        _write_jsonl(debug_dir / "chapters.jsonl", [chapter.to_dict() for chapter in chapters])
        _write_jsonl(debug_dir / "segment_scores.jsonl", _score_records(score_timeline))
        _write_jsonl(debug_dir / "detected_peaks.jsonl", [peak.to_dict() for peak in detected_peaks])
        _write_jsonl(debug_dir / "candidates_peak.jsonl", [candidate.to_dict() for candidate in peak_candidates])
        _write_jsonl(debug_dir / "candidates_semantic.jsonl", [candidate.to_dict() for candidate in semantic_candidates])
        _write_jsonl(debug_dir / "candidates_dense.jsonl", [candidate.to_dict() for candidate in dense_candidates])
        _write_jsonl(
            debug_dir / "candidates_dense_inventory.jsonl",
            [candidate.to_dict() for candidate in dense_inventory_candidates],
        )
        _write_jsonl(debug_dir / "candidates_fallback.jsonl", [candidate.to_dict() for candidate in fallback_candidates])
        _write_jsonl(debug_dir / "candidates_merged.jsonl", [candidate.to_dict() for candidate in candidates])
        _write_jsonl(
            debug_dir / "candidate_merge_audit.jsonl",
            heuristic_merge_audit_rows,
        )
        _write_json(
            debug_dir / "candidate_merge_summary.json",
            {
                "heuristic_merge_audit_rows": len(heuristic_merge_audit_rows),
                "merge_recall_carry_forward": merge_recall_carry_summary,
            },
        )
        _write_jsonl(
            debug_dir / "candidates_normalized.jsonl",
            [candidate.to_dict() for candidate in normalized_candidates],
        )
        _write_jsonl(
            debug_dir / "candidate_features.jsonl",
            [feature.to_dict() for feature in candidate_features],
        )
        _write_jsonl(
            debug_dir / "ranked_candidates_pre_judge.jsonl",
            [ranked.to_dict() for ranked in ranked_candidates_pre_judge],
        )
        _write_jsonl(
            debug_dir / "judge_input_candidates.jsonl",
            [ranked.to_dict() for ranked in judge_input_candidates[:judge_top_k]],
        )
        (debug_dir / "judge_recall_reserve_summary.json").write_text(
            json.dumps(judge_recall_reserve_summary, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        _write_jsonl(
            debug_dir / "judged_candidates.jsonl",
            [
                ranked.to_dict()
                for ranked in judged_ranked_candidates
                if ranked.judge_result is not None
            ],
        )
        _write_jsonl(
            debug_dir / "serving_quality_gate.jsonl",
            quality_gate_result.audit_rows,
        )
        (debug_dir / "post_judge_final_selection.json").write_text(
            json.dumps(final_selection_result.summary, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        _write_jsonl(
            debug_dir / "judge_keyframes.jsonl",
            [
                frame.to_dict()
                for frames in sampled_keyframes.values()
                for frame in frames
            ],
        )
        _write_jsonl(
            debug_dir / "ranked_candidates.jsonl",
            [ranked.to_dict() for ranked in ranked_candidates],
        )
        _write_jsonl(
            debug_dir / "ranked_candidates_v2.jsonl",
            _ranked_score_records(
                ranked_candidates_pre_judge,
                judged_ranked_candidates,
                ranked_candidates,
            ),
        )
        # Post-process: align to sentence boundaries, then enforce clip diversity
        moments = post_processor.snap_boundaries(moments, transcript)
        moments = post_processor.enforce_diversity(moments)
        moments = await loop.run_in_executor(
            None,
            analyzer.expand_variants,
            moments,
            ranked_candidates,
            transcript,
            normalized_strategy,
        )
        feedback_events = feedback_store.build_pipeline_feedback(
            job_id,
            ranked_candidates,
            moments,
            quality_gate_result.audit_rows,
        )
        feedback_store.write_job_feedback(debug_dir, feedback_events)
        feedback_store.append_feedback_events(pathlib.Path(settings.storage_dir), feedback_events)
        (debug_dir / "final_moments.json").write_text(
            json.dumps(moments, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        recall_report = candidate_recall_report.build_candidate_recall_report(
            stage_candidates={
                "peak": peak_candidates,
                "semantic": semantic_candidates,
                "dense": dense_candidates,
                "dense_inventory": dense_inventory_candidates,
                "fallback": fallback_candidates,
                "merged": candidates,
                "normalized": normalized_candidates,
            },
            pre_judge_ranked=ranked_candidates_pre_judge,
            judged_ranked=judged_ranked_candidates,
            quality_gate_rows=quality_gate_result.audit_rows,
            final_ranked=ranked_candidates,
            final_moments=moments,
            judge_top_k=judge_top_k,
            judge_scope_ranked=judge_input_candidates,
            positive_windows=diagnostic_positive_windows,
            positive_overlap_ratio=float(getattr(settings, "serving_cross_platform_recall_overlap_ratio", 0.45) or 0.45),
        )
        candidate_trace_rows = candidate_recall_report.build_candidate_trace(
            stage_candidates={
                "peak": peak_candidates,
                "semantic": semantic_candidates,
                "dense": dense_candidates,
                "dense_inventory": dense_inventory_candidates,
                "fallback": fallback_candidates,
                "merged": candidates,
                "normalized": normalized_candidates,
            },
            pre_judge_ranked=ranked_candidates_pre_judge,
            judged_ranked=judged_ranked_candidates,
            quality_gate_rows=quality_gate_result.audit_rows,
            final_ranked=ranked_candidates,
            final_moments=moments,
            judge_top_k=judge_top_k,
            judge_scope_ranked=judge_input_candidates,
        )
        candidate_funnel_summary = candidate_recall_report.build_candidate_funnel_summary(
            recall_report,
            candidate_trace_rows,
        )
        candidate_funnel_summary["cross_platform_positive_diagnostics"] = diagnostic_positive_summary
        _write_json(debug_dir / "candidate_recall_report.json", recall_report)
        _write_jsonl(debug_dir / "candidate_trace.jsonl", candidate_trace_rows)
        _write_json(debug_dir / "candidate_funnel_summary.json", candidate_funnel_summary)
        review_candidate_rows = _build_review_candidate_rows(
            feedback_events,
            candidate_trace_rows,
            limit=int(getattr(settings, "frontend_review_preview_limit", 40) or 0),
        )
        if review_candidate_rows:
            update_job(job_id, status="clipping", progress_pct=86)
            _raise_if_cancelled(job_id)
            review_candidate_rows = await loop.run_in_executor(
                None,
                clipper.cut_review_clips,
                video_path,
                review_candidate_rows,
                job_id,
                int(getattr(settings, "frontend_review_preview_limit", 40) or 0),
            )
        _write_jsonl(debug_dir / "review_candidates.jsonl", review_candidate_rows)
        logger.info(
            json.dumps(
                {
                    "event": "candidate_recall_report_built",
                    "job_id": job_id,
                    "stage_counts": recall_report.get("stage_counts", {}),
                    "trace_rows": len(candidate_trace_rows),
                    "drop_stage_counts": candidate_funnel_summary.get("drop_stage_counts", {}),
                    "largest_losses": recall_report.get("largest_losses", [])[:3],
                    "positive_window_recall": recall_report.get("positive_window_recall", {}),
                    "cross_platform_positive_diagnostics": diagnostic_positive_summary,
                    "review_candidate_rows": len(review_candidate_rows),
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )

        update_job(job_id, status="clipping", progress_pct=88)
        _raise_if_cancelled(job_id)
        clips = await loop.run_in_executor(None, clipper.cut_clips, video_path, moments, job_id)

        _cleanup_audio_artifacts(audio_path)
        audio_path = None

        update_job(job_id, status="done", progress_pct=100, clips=clips)

        # Schedule cleanup of entire job directory after 1 hour
        asyncio.create_task(_cleanup_later(job_id))

    except (PipelineCancelledError, downloader.DownloadCancelledError):
        logger.info("Pipeline cancelled for job %s", job_id)
    except Exception as e:
        logger.exception("Pipeline failed for job %s", job_id)
        update_job(job_id, status="failed", error=str(e))
    finally:
        _cleanup_serving_source_media(
            job_id=job_id,
            video_path=video_path,
            audio_path=audio_path,
        )


async def _cleanup_later(job_id: str) -> None:
    await asyncio.sleep(3600)
    import pathlib
    job_dir = pathlib.Path(settings.storage_dir) / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)


def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in rows),
        encoding="utf-8",
    )


def _parse_float_list(value: str | list[float] | tuple[float, ...] | None) -> tuple[float, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raw_items = value.split(",")
    else:
        raw_items = list(value)
    parsed: list[float] = []
    for item in raw_items:
        try:
            parsed_value = float(str(item).strip())
        except ValueError:
            continue
        if parsed_value > 0:
            parsed.append(parsed_value)
    return tuple(parsed)


def _segment_records(transcript) -> list[dict]:
    lines = transcriber.format_jsonl(transcript).splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _score_records(score_timeline) -> list[dict]:
    return [item.to_dict() for item in score_timeline]


def _ranked_score_records(
    pre_judge: list,
    judged: list,
    final_ranked: list,
) -> list[dict]:
    judged_map = {item.candidate.id: item for item in judged}
    final_ids = {item.candidate.id for item in final_ranked}
    rows: list[dict] = []
    for item in pre_judge:
        judged_item = judged_map.get(item.candidate.id, item)
        rows.append(
            {
                "candidate_id": item.candidate.id,
                "source": item.candidate.source,
                "heuristic_rank_score": item.heuristic_rank_score,
                "trained_rank_score": item.trained_rank_score,
                "model_blend": item.model_blend,
                "pre_judge_rank_score": item.rank_score,
                "judged_rank_score": judged_item.rank_score,
                "trained_model_used": item.trained_model_used,
                "judge_used": judged_item.judge_result is not None,
                "selected_after_diversity": item.candidate.id in final_ids,
                "rationale": list(judged_item.rationale),
            }
        )
    rows.sort(key=lambda row: (float(row.get("judged_rank_score") or 0.0), row["candidate_id"]), reverse=True)
    return rows
