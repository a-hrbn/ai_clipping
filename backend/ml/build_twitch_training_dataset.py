from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass
import hashlib
import importlib
import json
import logging
import math
import os
import pathlib
import shutil
import sys
import time

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config import settings
from ml import collect_viral_clips, sample_background_negatives
from services import downloader, game_events as game_event_service
from services.game_events import GameEvent

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_POSITIVE_LABEL_CLASSES = {"viral_positive", "positive", "weak_positive"}
_SUPPLEMENTAL_CANDIDATE_LIMIT = 2
_SUPPLEMENTAL_CANDIDATE_WEIGHT = 0.65
_BOUNDARY_NEAR_MISS_SAMPLE_WEIGHT = 0.45
_BOUNDARY_NEAR_MISS_LIMIT = 6
_BACKGROUND_NEGATIVE_WINDOWS_PER_VOD = 12
_BACKGROUND_NEGATIVE_WINDOW_SEC = 30.0
_BACKGROUND_NEGATIVE_MAX_OVERLAP = 0.20
_BACKGROUND_NEGATIVE_QUIET_VERSION = 3
_BACKGROUND_QUIET_FEATURE_KEYS = (
    "chat_message_rate_mean",
    "chat_message_rate_peak",
    "max_hype",
    "face_reaction_peak",
    "laughter_count",
    "audio_pop_score",
    "chat_laugh_score",
)
_BACKGROUND_ABSOLUTE_QUIET_THRESHOLDS = {
    "chat_message_rate_mean": 0.35,
    "chat_message_rate_peak": 1.25,
    "max_hype": 4.0,
    "face_reaction_peak": 0.65,
    "laughter_count": 1.0,
    "audio_pop_score": 0.75,
    "chat_laugh_score": 0.55,
}
_BACKGROUND_RELATIVE_QUIET_MIN_MATCHES = 5
_BACKGROUND_RELATIVE_QUIET_FALLBACK_MIN_MATCHES = 3
_BACKGROUND_MAX_HYPE_P95_FACTOR = 0.60
_BACKGROUND_FACE_REACTION_HARD_MAX = 0.40
_PERMANENT_CHAT_MISS_REASONS = {
    "twitch_graphql_video_missing",
    "missing_rechat",
}
_MAX_BOUNDARY_NEGATIVE_SHARE = 0.20
_MAX_VOD_NEGATIVE_ROWS_PER_POSITIVE = 8
_MAX_CANDIDATE_SOURCE_SHARE = 0.40
_DATASET_BUILD_LLM_PROVIDER = (os.environ.get("TRAINING_DATASET_LLM_PROVIDER") or "none").strip() or "none"
_VOD_CHECKPOINT_VERSION = 6
_VOD_FEATURE_CACHE_VERSION = 1
_CHAT_ENRICHMENT_FINGERPRINT_VERSION = 5
_CHAT_TARGET_NEGATIVE_LABEL_CLASSES = {
    "hard_negative",
    "background_negative",
    "auto_outcome_background_negative",
    "auto_outcome_hard_negative",
    "boundary_near_miss_negative",
}
_CHAT_NEGATIVE_ONLY_TARGET_LIMIT = 16
_WINDOW_SCOPED_TOP_VIEWED_SOURCE = "top_viewed_twitch_30d"
_WINDOW_SCOPED_TOP_VIEWED_KIND = "window_scoped_top_viewed"
_WINDOW_SCOPED_EXTENDED_KIND = "extended_window_scoped"
_WINDOW_SCOPED_CONTEXT_VERSION = 3
_BOUNDARY_PERTURBATION_SPECS = (
    ("start_late_small", 0.8, 0.0),
    ("start_late_large", 1.8, 0.0),
    ("start_early_small", -0.8, 0.0),
    ("end_early_small", 0.0, -0.8),
    ("end_early_large", 0.0, -1.8),
    ("end_late_small", 0.0, 1.2),
    ("both_tight", 0.8, -1.0),
    ("both_loose", -1.0, 1.2),
)


def _local_game_events_for_source(
    vod_id: str,
    *,
    source_start_sec: float,
    source_end_sec: float,
) -> list[GameEvent]:
    """Load cached game events and rebase them to the local scoped source timeline."""
    if not vod_id:
        return []
    try:
        cached_events = game_event_service.detect_events(vod_id)
    except Exception:
        return []
    if not cached_events:
        return []
    local_events: list[GameEvent] = []
    for event in cached_events:
        if event.end < source_start_sec or event.start > source_end_sec:
            continue
        local_start = max(0.0, float(event.start) - source_start_sec)
        local_end = max(local_start, float(event.end) - source_start_sec)
        local_events.append(
            GameEvent(
                vod_id=event.vod_id,
                start=round(local_start, 3),
                end=round(local_end, 3),
                event_type=event.event_type,
                confidence=event.confidence,
                source=event.source,
            )
        )
    return local_events


@dataclass(frozen=True)
class VodFeatureContext:
    transcript: list
    audio_signal: object
    audio_sample_rate: int
    cache_dir: pathlib.Path
    source_start_sec: float = 0.0
    source_end_sec: float = 0.0
    source_kind: str = "full_vod"


@dataclass(frozen=True)
class WindowScopedSourcePlan:
    scoped_kind: str
    expanded_start: float
    expanded_end: float
    cache_dir: pathlib.Path
    scoped_video_path: pathlib.Path
    metadata_path: pathlib.Path
    metadata_payload: dict[str, object]
    uses_local_vod_source: bool

def _dataset_build_fingerprint(profile: str) -> str:
    # Fingerprint the full dataset build, including cheap postprocessing. This is
    # useful for audits, but raw VOD checkpoints use _raw_vod_rows_fingerprint().
    payload = _raw_vod_rows_fingerprint_payload(profile)
    payload.update({
        "ranker_positive_dedupe": 1,
        "max_vod_negative_rows_per_positive": _MAX_VOD_NEGATIVE_ROWS_PER_POSITIVE,
    })
    normalized = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _raw_vod_rows_fingerprint(profile: str) -> str:
    payload = _raw_vod_rows_fingerprint_payload(profile)
    normalized = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _raw_vod_rows_fingerprint_payload(profile: str) -> dict[str, object]:
    # Fingerprint only the expensive row-generation path. Dataset-level
    # postprocessing such as dedupe, balancing, and source capping runs after all
    # VOD rows are loaded and must not force a VOD re-download.
    from services import feature_builder

    return {
        "profile": profile,
        "feature_build_version": int(getattr(feature_builder, "FEATURE_BUILD_VERSION", 1)),
        "llm_provider": _DATASET_BUILD_LLM_PROVIDER,
        "chat_enrichment": {
            "version": _CHAT_ENRICHMENT_FINGERPRINT_VERSION,
            "enabled": bool(settings.enable_chat_analysis),
            "timeout_sec": int(settings.chat_download_timeout_sec),
        },
        "training_vod_max_height": _resolved_training_vod_max_height(),
        "window_scoped_context": {
            "version": _WINDOW_SCOPED_CONTEXT_VERSION,
            "top_viewed_kind": _WINDOW_SCOPED_TOP_VIEWED_KIND,
            "extended_kind": _WINDOW_SCOPED_EXTENDED_KIND,
            "extended_enabled": bool(settings.training_vod_extended_window_scoped),
            "extended_context_sec": _resolved_extended_window_context_sec(),
            "extended_group_max_sec": _resolved_extended_window_group_max_sec(),
        },
        "supplemental_candidate_limit": _SUPPLEMENTAL_CANDIDATE_LIMIT,
        "supplemental_candidate_weight": _SUPPLEMENTAL_CANDIDATE_WEIGHT,
        "background_negatives": {
            "quiet_version": _BACKGROUND_NEGATIVE_QUIET_VERSION,
            "windows_per_vod": _BACKGROUND_NEGATIVE_WINDOWS_PER_VOD,
            "window_sec": _BACKGROUND_NEGATIVE_WINDOW_SEC,
            "max_overlap": _BACKGROUND_NEGATIVE_MAX_OVERLAP,
            "relative_quiet_min_matches": _BACKGROUND_RELATIVE_QUIET_MIN_MATCHES,
            "relative_quiet_fallback_min_matches": _BACKGROUND_RELATIVE_QUIET_FALLBACK_MIN_MATCHES,
            "max_hype_p95_factor": _BACKGROUND_MAX_HYPE_P95_FACTOR,
            "face_reaction_hard_max": _BACKGROUND_FACE_REACTION_HARD_MAX,
            "boundary_max_negative_share": _MAX_BOUNDARY_NEGATIVE_SHARE,
        },
        "boundary_near_miss_limit": _BOUNDARY_NEAR_MISS_LIMIT,
        "boundary_near_miss_sample_weight": _BOUNDARY_NEAR_MISS_SAMPLE_WEIGHT,
        "boundary_perturbation_specs": list(_BOUNDARY_PERTURBATION_SPECS),
    }


def _legacy_vod_checkpoint_fingerprints(profile: str) -> set[str]:
    payload = _raw_vod_rows_fingerprint_payload(profile)
    payload["ranker_positive_dedupe"] = 1
    normalized = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    legacy_before_postprocessing_split = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return {
        legacy_before_postprocessing_split,
        _dataset_build_fingerprint(profile),
    }


def _vod_feature_cache_root() -> pathlib.Path:
    root = pathlib.Path(settings.storage_dir).resolve().parent / "vod_feature_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _vod_feature_cache_dir(vod_id: str) -> pathlib.Path:
    path = _vod_feature_cache_root() / str(vod_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _source_video_signature(vod_path: pathlib.Path) -> dict[str, object]:
    stat = vod_path.stat()
    return {
        "path": str(vod_path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _stable_vod_source_signature(vod_id: str, vod_path: pathlib.Path) -> dict[str, object]:
    cache_path = downloader._cache_path_for_url(f"https://www.twitch.tv/videos/{vod_id}", namespace="twitch")
    try:
        source_stat = vod_path.stat()
    except OSError:
        source_stat = None
    if cache_path.exists() and source_stat is not None:
        try:
            cache_stat = cache_path.stat()
        except OSError:
            cache_stat = None
        if cache_stat is not None and int(cache_stat.st_size) == int(source_stat.st_size):
            return {
                "kind": "twitch_download_cache",
                "vod_id": str(vod_id),
                "size": int(cache_stat.st_size),
            }
    return {
        "kind": "local_source",
        **_source_video_signature(vod_path),
    }


def _audio_cache_payload(vod_id: str, vod_path: pathlib.Path) -> dict[str, object]:
    return {
        "version": _VOD_FEATURE_CACHE_VERSION,
        "kind": "vod_audio",
        "vod_id": str(vod_id),
        "source": _stable_vod_source_signature(vod_id, vod_path),
        "audio_codec": "pcm_s16le",
        "audio_sample_rate": 16000,
        "audio_channels": 1,
    }


def _audio_cache_metadata_path(cache_dir: pathlib.Path) -> pathlib.Path:
    return cache_dir / "audio.metadata.json"


def _audio_cache_is_usable(audio_path: pathlib.Path, metadata_path: pathlib.Path, payload: dict[str, object]) -> bool:
    if not audio_path.exists():
        return False
    if not metadata_path.exists():
        # Existing caches predate metadata. Keep them usable for compatibility;
        # the current run will write metadata after accepting the file.
        return True
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return metadata == payload


def _transcript_cache_payload(vod_id: str) -> dict[str, object]:
    return {
        "version": _VOD_FEATURE_CACHE_VERSION,
        "kind": "transcript",
        "vod_id": str(vod_id),
        "whisper_model": str(settings.whisper_model),
        "whisper_language": str(settings.whisper_language or "").strip().lower(),
    }


def _window_feature_cache_payload(
    vod_id: str,
    vod_path: pathlib.Path,
    *,
    window_start: float,
    window_end: float,
) -> dict[str, object]:
    from services import video_features

    runtime = video_features.runtime_summary()
    return {
        "version": _VOD_FEATURE_CACHE_VERSION,
        "kind": "window_modalities",
        "vod_id": str(vod_id),
        "source": _stable_vod_source_signature(vod_id, vod_path),
        "window_start": round(float(window_start), 3),
        "window_end": round(float(window_end), 3),
        "video_expression_sample_sec": int(max(settings.video_expression_sample_sec, 1)),
        "video_face_detector": str(runtime.get("face_detector") or ""),
        "video_deepface_status": str(runtime.get("deepface_status") or ""),
    }


def _cache_key(payload: dict[str, object]) -> str:
    normalized = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _transcript_cache_path(cache_dir: pathlib.Path, payload: dict[str, object]) -> pathlib.Path:
    return cache_dir / f"transcript_{_cache_key(payload)}.json"


def _window_feature_cache_path(cache_dir: pathlib.Path, payload: dict[str, object]) -> pathlib.Path:
    return cache_dir / "windows" / f"{_cache_key(payload)}.json"


def _cached_chat_path(cache_dir: pathlib.Path) -> pathlib.Path | None:
    chat_dir = cache_dir / "chat"
    if not chat_dir.exists():
        return None
    candidates = sorted(
        (
            path
            for path in chat_dir.glob("chat*.json*")
            if path.name != "chat.metadata.json" and "metadata" not in path.name
        ),
        key=lambda path: path.stat().st_mtime_ns if path.exists() else 0,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _normalize_chat_target_offsets(target_offsets: list[float] | tuple[float, ...] | None) -> list[int]:
    if not target_offsets:
        return []
    normalized: set[int] = set()
    for value in target_offsets:
        try:
            offset = int(float(value))
        except (TypeError, ValueError):
            continue
        normalized.add(max(offset, 0))
    return sorted(normalized)


def _read_chat_cache_metadata(cache_dir: pathlib.Path) -> dict[str, object]:
    path = _chat_cache_metadata_path(cache_dir)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_chat_fetch_metadata(chat_path: pathlib.Path) -> dict[str, object]:
    metadata_path = chat_path.with_name(f"{chat_path.stem}.fetch_metadata.json")
    if not metadata_path.exists():
        return {}
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_chat_fetch_metadata_dir(path: pathlib.Path) -> dict[str, object]:
    metadata_path = path / "chat.twitch_rechat.fetch_metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _chat_messages(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(raw, list):
        return [row for row in raw if isinstance(row, dict)]
    if not isinstance(raw, dict):
        return []
    events = raw.get("events")
    if not isinstance(events, list):
        return []
    rows: list[dict] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        try:
            t = float(event.get("tStartMs", 0.0)) / 1000.0
        except (TypeError, ValueError):
            continue
        segments = event.get("segs")
        if not isinstance(segments, list):
            continue
        text = "".join(
            str(segment.get("utf8", ""))
            for segment in segments
            if isinstance(segment, dict)
        ).strip()
        if text:
            rows.append({"t": t, "text": text, "emotes": []})
    return rows


def _chat_time_stats(path: pathlib.Path) -> dict[str, object]:
    messages = _chat_messages(path)
    times: list[float] = []
    for message in messages:
        try:
            times.append(float(message.get("t", 0.0)))
        except (TypeError, ValueError):
            continue
    return {
        "message_count": len(messages),
        "first_t": round(min(times), 3) if times else None,
        "last_t": round(max(times), 3) if times else None,
    }


def _chat_dedupe_key(message: dict) -> tuple[str, float, str]:
    try:
        t = round(float(message.get("t", 0.0)), 3)
    except (TypeError, ValueError):
        t = 0.0
    return (str(message.get("id") or ""), t, str(message.get("text") or ""))


def _merge_chat_files(primary_path: pathlib.Path, refill_path: pathlib.Path, output_path: pathlib.Path) -> pathlib.Path:
    seen: set[tuple[str, float, str]] = set()
    merged: list[dict] = []
    for source in (primary_path, refill_path):
        for message in _chat_messages(source):
            key = _chat_dedupe_key(message)
            if key in seen:
                continue
            seen.add(key)
            merged.append(message)
    merged.sort(key=lambda row: float(row.get("t", 0.0) or 0.0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged, ensure_ascii=True), encoding="utf-8")
    return output_path


def _metadata_offsets(metadata: dict[str, object], key: str) -> list[int]:
    values = metadata.get(key)
    if not isinstance(values, list):
        return []
    return _normalize_chat_target_offsets(values)


def _missing_chat_target_offsets(cache_dir: pathlib.Path, requested_offsets: list[int]) -> list[int]:
    if not requested_offsets:
        return []
    metadata = _read_chat_cache_metadata(cache_dir)
    covered = set(_metadata_offsets(metadata, "covered_target_offsets"))
    if not covered:
        covered = set(_metadata_offsets(metadata, "target_offsets"))
    return [offset for offset in requested_offsets if offset not in covered]


def _chat_cache_has_complete_coverage(cache_dir: pathlib.Path) -> bool:
    if _cached_chat_path(cache_dir) is None:
        return False
    metadata = _read_chat_cache_metadata(cache_dir)
    coverage_status = str(metadata.get("coverage_status") or "")
    if coverage_status == "complete":
        return True
    if coverage_status != "unscoped":
        return False
    return not _metadata_offsets(metadata, "missing_target_offsets")


def _chat_cache_has_permanent_miss(cache_dir: pathlib.Path, requested_offsets: list[int]) -> bool:
    metadata = _read_chat_cache_metadata(cache_dir)
    reason = str(metadata.get("reason") or "")
    if reason not in _PERMANENT_CHAT_MISS_REASONS:
        return False
    missing_offsets = set(_metadata_offsets(metadata, "missing_target_offsets"))
    if requested_offsets:
        return set(requested_offsets).issubset(missing_offsets)
    return bool(metadata.get("unavailable") or missing_offsets or not _cached_chat_path(cache_dir))


def _chat_cache_metadata_path(cache_dir: pathlib.Path) -> pathlib.Path:
    return cache_dir / "chat" / "chat.metadata.json"


def _write_rebased_chat_file(
    chat_path: pathlib.Path,
    output_path: pathlib.Path,
    *,
    source_start_sec: float,
    source_end_sec: float,
) -> pathlib.Path | None:
    messages: list[dict] = []
    for message in _chat_messages(chat_path):
        try:
            absolute_t = float(message.get("t", 0.0))
        except (TypeError, ValueError):
            continue
        if absolute_t < source_start_sec or absolute_t > source_end_sec:
            continue
        copied = dict(message)
        copied["t"] = round(max(absolute_t - source_start_sec, 0.0), 3)
        messages.append(copied)
    if not messages:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(messages, ensure_ascii=True), encoding="utf-8")
    return output_path


def _write_chat_cache_metadata(
    cache_dir: pathlib.Path,
    *,
    vod_url: str | None,
    job_id: str,
    chat_path: pathlib.Path,
    cache_status: str,
    target_offsets: list[float] | tuple[float, ...] | list[int] | None = None,
    covered_target_offsets: list[float] | tuple[float, ...] | list[int] | None = None,
    missing_target_offsets: list[float] | tuple[float, ...] | list[int] | None = None,
    reason: str = "",
) -> None:
    try:
        stat = chat_path.stat()
    except OSError:
        size = 0
        mtime_ns = 0
    else:
        size = int(stat.st_size)
        mtime_ns = int(stat.st_mtime_ns)
    requested = _normalize_chat_target_offsets(target_offsets)
    covered = _normalize_chat_target_offsets(covered_target_offsets)
    missing = _normalize_chat_target_offsets(missing_target_offsets)
    stats = _chat_time_stats(chat_path)
    payload = {
            "version": _VOD_FEATURE_CACHE_VERSION,
            "kind": "vod_chat",
            "vod_id": str(cache_dir.name),
            "vod_url": str(vod_url or ""),
            "job_id": str(job_id),
            "cache_status": str(cache_status),
            "reason": str(reason or ""),
            "chat_path": str(chat_path),
            "size": size,
            "mtime_ns": mtime_ns,
            "cached_at_unix": round(time.time(), 3),
            "message_count": int(stats["message_count"]),
            "first_t": stats["first_t"],
            "last_t": stats["last_t"],
            "coverage_status": "complete" if requested and not missing else ("partial" if missing else "unscoped"),
            "partial": bool(missing),
            "target_offsets": requested,
            "covered_target_offsets": covered,
            "missing_target_offsets": missing,
    }
    _write_json_atomic(_chat_cache_metadata_path(cache_dir), payload)


def _write_chat_cache_miss_metadata(
    cache_dir: pathlib.Path,
    *,
    vod_url: str | None,
    job_id: str,
    cache_status: str,
    reason: str,
    target_offsets: list[float] | tuple[float, ...] | list[int] | None = None,
    missing_target_offsets: list[float] | tuple[float, ...] | list[int] | None = None,
) -> None:
    requested = _normalize_chat_target_offsets(target_offsets)
    missing = _normalize_chat_target_offsets(missing_target_offsets)
    _write_json_atomic(
        _chat_cache_metadata_path(cache_dir),
        {
            "version": _VOD_FEATURE_CACHE_VERSION,
            "kind": "vod_chat",
            "vod_id": str(cache_dir.name),
            "vod_url": str(vod_url or ""),
            "job_id": str(job_id),
            "cache_status": str(cache_status),
            "reason": str(reason),
            "chat_path": "",
            "size": 0,
            "mtime_ns": 0,
            "cached_at_unix": round(time.time(), 3),
            "message_count": 0,
            "first_t": None,
            "last_t": None,
            "coverage_status": "partial" if missing else "unscoped",
            "partial": bool(missing),
            "target_offsets": requested,
            "covered_target_offsets": [],
            "missing_target_offsets": missing,
        },
    )


def _load_or_download_chat(
    vod_url: str | None,
    *,
    job_id: str,
    cache_dir: pathlib.Path,
    target_offsets: list[float] | None = None,
) -> pathlib.Path | None:
    vod_id = str(cache_dir.name)
    requested_offsets = _normalize_chat_target_offsets(target_offsets)
    logger.info(
        "training_vod_chat_cache_lookup vod_id=%s job_id=%s cache_dir=%s target_offsets=%s",
        vod_id,
        job_id,
        cache_dir / "chat",
        len(requested_offsets),
    )
    cached_path = _cached_chat_path(cache_dir)
    if _chat_cache_has_permanent_miss(cache_dir, requested_offsets):
        logger.info(
            "training_vod_chat_download_skip vod_id=%s job_id=%s reason=permanent_chat_unavailable target_offsets=%s",
            vod_id,
            job_id,
            len(requested_offsets),
        )
        return cached_path
    if cached_path is not None:
        missing_offsets = _missing_chat_target_offsets(cache_dir, requested_offsets)
        if not missing_offsets:
            covered_offsets = requested_offsets or _metadata_offsets(
                _read_chat_cache_metadata(cache_dir),
                "covered_target_offsets",
            )
            _write_chat_cache_metadata(
                cache_dir,
                vod_url=vod_url,
                job_id=job_id,
                chat_path=cached_path,
                cache_status="hit",
                target_offsets=requested_offsets,
                covered_target_offsets=covered_offsets,
                missing_target_offsets=[],
            )
            logger.info(
                "training_vod_chat_cache_hit vod_id=%s job_id=%s chat_path=%s target_offsets=%s",
                vod_id,
                job_id,
                cached_path,
                len(requested_offsets),
            )
            return cached_path
        logger.info(
            "training_vod_chat_cache_partial vod_id=%s job_id=%s chat_path=%s target_offsets=%s missing_offsets=%s first_missing_offset=%s last_missing_offset=%s",
            vod_id,
            job_id,
            cached_path,
            len(requested_offsets),
            len(missing_offsets),
            missing_offsets[0],
            missing_offsets[-1],
        )
    if not vod_url:
        _write_chat_cache_miss_metadata(
            cache_dir,
            vod_url=vod_url,
            job_id=job_id,
            cache_status="skipped",
            reason="missing_vod_url",
            target_offsets=requested_offsets,
            missing_target_offsets=requested_offsets,
        )
        logger.info(
            "training_vod_chat_download_skip vod_id=%s job_id=%s reason=missing_vod_url",
            vod_id,
            job_id,
        )
        return None

    from services import chat_analyzer

    chat_dir = cache_dir / "chat"
    download_offsets = [float(offset) for offset in (missing_offsets if cached_path is not None else requested_offsets)]
    if not download_offsets and target_offsets:
        download_offsets = [float(offset) for offset in requested_offsets]
    download_target_offsets = download_offsets or target_offsets
    download_dir = chat_dir
    if cached_path is not None and download_offsets:
        download_dir = chat_dir / f"refill_{int(time.time() * 1000)}"
    logger.info(
        "training_vod_chat_download_start vod_id=%s job_id=%s vod_url=%s cache_dir=%s target_offsets=%s refill=%s",
        vod_id,
        job_id,
        vod_url,
        download_dir,
        len(download_offsets if download_offsets else (target_offsets or [])),
        cached_path is not None,
    )
    chat_path = chat_analyzer.download_chat(
        vod_url,
        f"{job_id}:chat_cache",
        download_dir,
        target_offsets=download_target_offsets,
    )
    if chat_path is None:
        fetch_metadata = _read_chat_fetch_metadata_dir(download_dir)
        unavailable_reason = str(fetch_metadata.get("reason") or "download_returned_none_see_twitch_chat_logs")
        if cached_path is not None:
            covered_offsets = sorted(set(requested_offsets) - set(missing_offsets))
            _write_chat_cache_metadata(
                cache_dir,
                vod_url=vod_url,
                job_id=job_id,
                chat_path=cached_path,
                cache_status="partial",
                target_offsets=requested_offsets,
                covered_target_offsets=covered_offsets,
                missing_target_offsets=missing_offsets,
                reason=unavailable_reason,
            )
            logger.info(
                "training_vod_chat_refill_unavailable vod_id=%s job_id=%s vod_url=%s missing_offsets=%s reason=%s",
                vod_id,
                job_id,
                vod_url,
                len(missing_offsets),
                unavailable_reason,
            )
            return cached_path
        _write_chat_cache_miss_metadata(
            cache_dir,
            vod_url=vod_url,
            job_id=job_id,
            cache_status="unavailable",
            reason=unavailable_reason,
            target_offsets=requested_offsets,
            missing_target_offsets=requested_offsets,
        )
        logger.info(
            "training_vod_chat_download_unavailable vod_id=%s job_id=%s vod_url=%s reason=%s",
            vod_id,
            job_id,
            vod_url,
            unavailable_reason,
        )
    else:
        fetch_metadata = _read_chat_fetch_metadata(chat_path)
        fetch_covered_offsets = _metadata_offsets(fetch_metadata, "covered_target_offsets")
        fetch_missing_offsets = _metadata_offsets(fetch_metadata, "missing_target_offsets")
        if requested_offsets and not fetch_covered_offsets and not fetch_missing_offsets:
            fetch_covered_offsets = download_offsets or requested_offsets
        if cached_path is not None:
            merged_path = chat_dir / "chat.twitch_rechat.json"
            chat_path = _merge_chat_files(cached_path, chat_path, merged_path)
            shutil.rmtree(download_dir, ignore_errors=True)
            previous_covered = set(requested_offsets) - set(missing_offsets)
            covered_offsets = sorted(previous_covered | set(fetch_covered_offsets))
            missing_after_refill = [offset for offset in requested_offsets if offset not in set(covered_offsets)]
            cache_status = "refilled" if not missing_after_refill else "partial"
            logger.info(
                "training_vod_chat_refill_success vod_id=%s job_id=%s chat_path=%s requested_missing_offsets=%s remaining_missing_offsets=%s",
                vod_id,
                job_id,
                chat_path,
                len(missing_offsets),
                len(missing_after_refill),
            )
        else:
            covered_offsets = fetch_covered_offsets
            missing_after_refill = fetch_missing_offsets or [
                offset for offset in requested_offsets if offset not in set(covered_offsets)
            ]
            cache_status = "downloaded" if not missing_after_refill else "partial"
        _write_chat_cache_metadata(
            cache_dir,
            vod_url=vod_url,
            job_id=job_id,
            chat_path=chat_path,
            cache_status=cache_status,
            target_offsets=requested_offsets,
            covered_target_offsets=covered_offsets,
            missing_target_offsets=missing_after_refill,
        )
        logger.info(
            "training_vod_chat_download_success vod_id=%s job_id=%s chat_path=%s cache_dir=%s cache_status=%s",
            vod_id,
            job_id,
            chat_path,
            chat_dir,
            cache_status,
        )
    return chat_path


def _serialize_rich_segment(segment) -> dict[str, object]:
    return {
        "start": float(segment.start),
        "end": float(segment.end),
        "text": str(segment.text),
        "speaker": str(getattr(segment, "speaker", "UNKNOWN")),
        "emotions": list(getattr(segment, "emotions", []) or []),
        "words_per_second": float(getattr(segment, "words_per_second", 0.0) or 0.0),
        "laughter_prob": float(getattr(segment, "laughter_prob", 0.0) or 0.0),
        "max_rms_db": float(getattr(segment, "max_rms_db", 0.0) or 0.0),
        "mean_rms_db": float(getattr(segment, "mean_rms_db", 0.0) or 0.0),
        "max_pitch_hz": float(getattr(segment, "max_pitch_hz", 0.0) or 0.0),
        "max_onset_rate": float(getattr(segment, "max_onset_rate", 0.0) or 0.0),
        "hype_score": float(getattr(segment, "hype_score", 0.0) or 0.0),
        "arc_score": float(getattr(segment, "arc_score", 0.0) or 0.0),
        "max_motion_score": float(getattr(segment, "max_motion_score", 0.0) or 0.0),
        "mean_face_size": float(getattr(segment, "mean_face_size", 0.0) or 0.0),
        "max_face_count": int(getattr(segment, "max_face_count", 0) or 0),
        "max_face_reaction": float(getattr(segment, "max_face_reaction", 0.0) or 0.0),
        "has_scene_change": bool(getattr(segment, "has_scene_change", False)),
        "visual_activity": float(getattr(segment, "visual_activity", 0.0) or 0.0),
        "chat_spike_score": float(getattr(segment, "chat_spike_score", 0.0) or 0.0),
        "chat_message_count": int(getattr(segment, "chat_message_count", 0) or 0),
        "chat_message_rate": float(getattr(segment, "chat_message_rate", 0.0) or 0.0),
        "chat_density_score": float(getattr(segment, "chat_density_score", 0.0) or 0.0),
        "chat_emote_score": float(getattr(segment, "chat_emote_score", 0.0) or 0.0),
    }


def _deserialize_rich_segment(payload: dict[str, object]):
    from services.transcriber import RichSegment

    return RichSegment(
        start=float(payload.get("start") or 0.0),
        end=float(payload.get("end") or 0.0),
        text=str(payload.get("text") or ""),
        speaker=str(payload.get("speaker") or "UNKNOWN"),
        emotions=list(payload.get("emotions") or []),
        words_per_second=float(payload.get("words_per_second") or 0.0),
        laughter_prob=float(payload.get("laughter_prob") or 0.0),
        max_rms_db=float(payload.get("max_rms_db") or 0.0),
        mean_rms_db=float(payload.get("mean_rms_db") or 0.0),
        max_pitch_hz=float(payload.get("max_pitch_hz") or 0.0),
        max_onset_rate=float(payload.get("max_onset_rate") or 0.0),
        hype_score=float(payload.get("hype_score") or 0.0),
        arc_score=float(payload.get("arc_score") or 0.0),
        max_motion_score=float(payload.get("max_motion_score") or 0.0),
        mean_face_size=float(payload.get("mean_face_size") or 0.0),
        max_face_count=int(payload.get("max_face_count") or 0),
        max_face_reaction=float(payload.get("max_face_reaction") or 0.0),
        has_scene_change=bool(payload.get("has_scene_change") or False),
        visual_activity=float(payload.get("visual_activity") or 0.0),
        chat_spike_score=float(payload.get("chat_spike_score") or 0.0),
        chat_message_count=int(payload.get("chat_message_count") or 0),
        chat_message_rate=float(payload.get("chat_message_rate") or 0.0),
        chat_density_score=float(payload.get("chat_density_score") or 0.0),
        chat_emote_score=float(payload.get("chat_emote_score") or 0.0),
    )


def _load_transcript_cache(path: pathlib.Path) -> tuple[dict[str, object], list] | None:
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    metadata = cached.get("metadata")
    if not isinstance(metadata, dict):
        return None
    try:
        transcript = [_deserialize_rich_segment(item) for item in list(cached.get("transcript") or [])]
    except (TypeError, ValueError):
        return None
    return metadata, transcript


def _transcript_cache_metadata_matches(metadata: dict[str, object], vod_id: str) -> bool:
    if int(metadata.get("version") or 0) != _VOD_FEATURE_CACHE_VERSION:
        return False
    if str(metadata.get("kind") or "") != "transcript":
        return False
    if str(metadata.get("vod_id") or "") != str(vod_id):
        return False
    if str(metadata.get("whisper_model") or "") != str(settings.whisper_model):
        return False
    cached_language = metadata.get("whisper_language")
    if cached_language is not None:
        current_language = str(settings.whisper_language or "").strip().lower()
        return str(cached_language or "").strip().lower() == current_language
    return True


def _load_compatible_transcript_cache(cache_dir: pathlib.Path, vod_id: str) -> tuple[pathlib.Path, list] | None:
    paths = sorted(
        cache_dir.glob("transcript_*.json"),
        key=lambda path: path.stat().st_mtime_ns if path.exists() else 0,
        reverse=True,
    )
    for path in paths:
        cached = _load_transcript_cache(path)
        if cached is None:
            continue
        metadata, transcript = cached
        if _transcript_cache_metadata_matches(metadata, vod_id):
            return path, transcript
    return None


def _prune_compatible_transcript_caches(
    cache_dir: pathlib.Path,
    *,
    keep_path: pathlib.Path,
    vod_id: str,
) -> None:
    keep_path = keep_path.resolve()
    for path in cache_dir.glob("transcript_*.json"):
        try:
            if path.resolve() == keep_path:
                continue
        except OSError:
            continue
        cached = _load_transcript_cache(path)
        if cached is None:
            continue
        metadata, _transcript = cached
        if not _transcript_cache_metadata_matches(metadata, vod_id):
            continue
        try:
            path.unlink(missing_ok=True)
        except OSError:
            continue


def _write_transcript_cache(
    path: pathlib.Path,
    payload: dict[str, object],
    transcript: list,
) -> None:
    _write_json_atomic(
        path,
        {
            "metadata": payload,
            "transcript": [_serialize_rich_segment(segment) for segment in transcript],
        },
    )


def _write_json_atomic(path: pathlib.Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    tmp_path.replace(path)


def _prune_transcript_chunks(cache_dir: pathlib.Path) -> None:
    chunks_dir = cache_dir / "chunks"
    if chunks_dir.exists():
        shutil.rmtree(chunks_dir, ignore_errors=True)


def _serialize_frame(frame) -> dict[str, object]:
    return {
        key: _json_scalar(value)
        for key, value in vars(frame).items()
    }


def _json_scalar(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            return 0.0
        return value
    item = getattr(value, "item", None)
    if callable(item):
        scalar = item()
        if scalar is value:
            return value
        return _json_scalar(scalar)
    return value


def _load_or_build_vod_feature_context(
    vod_id: str,
    *,
    vod_path: pathlib.Path,
    job_id: str,
    vod_url: str | None = None,
    chat_target_offsets: list[float] | None = None,
    cache_dir: pathlib.Path | None = None,
    chat_cache_dir: pathlib.Path | None = None,
    source_start_sec: float = 0.0,
    source_end_sec: float = 0.0,
    source_kind: str = "full_vod",
) -> VodFeatureContext:
    audio_extractor = importlib.import_module("services.audio_extractor")
    from services import chat_analyzer, transcriber
    import librosa

    cache_dir = cache_dir or _vod_feature_cache_dir(vod_id)
    chat_cache_dir = chat_cache_dir or cache_dir
    audio_path = cache_dir / "audio.wav"
    audio_payload = _audio_cache_payload(vod_id, vod_path)
    audio_metadata_path = _audio_cache_metadata_path(cache_dir)
    if not _audio_cache_is_usable(audio_path, audio_metadata_path, audio_payload):
        audio_path.unlink(missing_ok=True)
        audio_extractor.extract(
            vod_path,
            f"{job_id}:vod_audio_cache",
            output_path=audio_path,
        )
    _write_json_atomic(audio_metadata_path, audio_payload)

    transcript_payload = _transcript_cache_payload(vod_id)
    transcript_path = _transcript_cache_path(cache_dir, transcript_payload)
    transcript: list
    if transcript_path.exists():
        cached = _load_transcript_cache(transcript_path)
        if cached is not None:
            transcript = cached[1]
            _prune_compatible_transcript_caches(cache_dir, keep_path=transcript_path, vod_id=vod_id)
            _prune_transcript_chunks(cache_dir)
        else:
            compatible = _load_compatible_transcript_cache(cache_dir, vod_id)
            if compatible is None:
                transcript = transcriber.transcribe(audio_path)
            else:
                _compatible_path, transcript = compatible
            _write_transcript_cache(transcript_path, transcript_payload, transcript)
            _prune_compatible_transcript_caches(cache_dir, keep_path=transcript_path, vod_id=vod_id)
            _prune_transcript_chunks(cache_dir)
    else:
        compatible = _load_compatible_transcript_cache(cache_dir, vod_id)
        if compatible is None:
            transcript = transcriber.transcribe(audio_path)
        else:
            _compatible_path, transcript = compatible
        _write_transcript_cache(transcript_path, transcript_payload, transcript)
        _prune_compatible_transcript_caches(cache_dir, keep_path=transcript_path, vod_id=vod_id)
        _prune_transcript_chunks(cache_dir)

    chat_path = _load_or_download_chat(
        vod_url,
        job_id=job_id,
        cache_dir=chat_cache_dir,
        target_offsets=chat_target_offsets,
    )
    if chat_path is not None and float(source_start_sec) > 0.0:
        rebased_chat_path = _write_rebased_chat_file(
            chat_path,
            cache_dir / "chat" / "chat.local.json",
            source_start_sec=float(source_start_sec),
            source_end_sec=float(source_end_sec or 0.0),
        )
        chat_path = rebased_chat_path
    if chat_path is not None:
        transcript = chat_analyzer.enrich_segments(transcript, chat_path)
        chat_segment_count = sum(
            1
            for segment in transcript
            if (
                abs(float(getattr(segment, "chat_spike_score", 0.0) or 0.0)) > 1e-9
                or abs(float(getattr(segment, "chat_density_score", 0.0) or 0.0)) > 1e-9
            )
        )
        logger.info(
            "training_vod_chat_enrichment_applied vod_id=%s job_id=%s chat_path=%s segments=%s segments_with_chat=%s",
            vod_id,
            job_id,
            chat_path,
            len(transcript),
            chat_segment_count,
        )
    else:
        logger.info(
            "training_vod_chat_enrichment_skipped vod_id=%s job_id=%s reason=no_chat_path",
            vod_id,
            job_id,
        )

    audio_signal, audio_sample_rate = librosa.load(str(audio_path), sr=None, mono=True)
    return VodFeatureContext(
        transcript=transcript,
        audio_signal=audio_signal,
        audio_sample_rate=int(audio_sample_rate),
        cache_dir=cache_dir,
        source_start_sec=float(source_start_sec),
        source_end_sec=float(source_end_sec or 0.0),
        source_kind=str(source_kind or "full_vod"),
    )


def _is_window_scoped_top_viewed_label(label: dict) -> bool:
    if str(label.get("manual_gold_review_source") or "").strip() != _WINDOW_SCOPED_TOP_VIEWED_SOURCE:
        return False
    if str(label.get("label_class") or "") not in _POSITIVE_LABEL_CLASSES:
        return False
    return bool(label.get("vod_id") and label.get("vod_url"))


def _should_use_window_scoped_label_contexts(labels: list[dict]) -> bool:
    return bool(labels) and all(_is_window_scoped_top_viewed_label(label) for label in labels)


def _can_use_extended_window_scoped_label(label: dict) -> bool:
    if not settings.training_vod_extended_window_scoped:
        return False
    if not bool(label.get("vod_id") and label.get("vod_url")):
        return False
    try:
        start = float(label.get("window_start"))
        end = float(label.get("window_end"))
    except (TypeError, ValueError):
        return False
    return math.isfinite(start) and math.isfinite(end) and end > start


def _should_use_extended_window_scoped_label_contexts(labels: list[dict]) -> bool:
    return bool(labels) and all(_can_use_extended_window_scoped_label(label) for label in labels)


def _window_scoped_context_kind_for_labels(labels: list[dict]) -> str | None:
    if _should_use_window_scoped_label_contexts(labels):
        return _WINDOW_SCOPED_TOP_VIEWED_KIND
    if _should_use_extended_window_scoped_label_contexts(labels):
        return _WINDOW_SCOPED_EXTENDED_KIND
    return None


def _should_use_scoped_label_contexts(labels: list[dict]) -> bool:
    return _window_scoped_context_kind_for_labels(labels) is not None


def _resolved_extended_window_context_sec() -> int:
    try:
        return max(int(settings.training_vod_extended_context_sec or 0), 0)
    except (TypeError, ValueError):
        return 0


def _resolved_training_vod_max_height() -> int:
    try:
        return max(int(settings.training_vod_max_height or 0), 0)
    except (TypeError, ValueError):
        return 0


def _resolved_extended_window_group_max_sec() -> int:
    try:
        return max(int(settings.training_vod_extended_group_max_sec or 0), 1)
    except (TypeError, ValueError):
        return 900


def _should_prefetch_window_scoped_group_sources() -> bool:
    return bool(settings.training_vod_group_source_prefetch)


def _scoped_training_window(label: dict, source_kind: str) -> tuple[float, float]:
    expanded_start, expanded_end = _expanded_training_window(label)
    if source_kind != _WINDOW_SCOPED_EXTENDED_KIND:
        return expanded_start, expanded_end
    context_sec = float(_resolved_extended_window_context_sec())
    if context_sec <= 0.0:
        return expanded_start, expanded_end
    label_start = float(label.get("window_start") or expanded_start)
    label_end = float(label.get("window_end") or expanded_end)
    duration = float(label.get("vod_duration_seconds") or 0.0)
    scoped_start = max(0.0, min(expanded_start, label_start - context_sec))
    scoped_end = max(expanded_end, label_end + context_sec)
    if duration > 0.0:
        scoped_end = min(scoped_end, duration)
    return (
        round(scoped_start, 2),
        round(max(scoped_end, scoped_start + 1.0), 2),
    )


def _window_scoped_label_cache_dir(
    vod_id: str,
    label: dict,
    *,
    expanded_start: float,
    expanded_end: float,
    source_kind: str,
    cache_scope: dict[str, object] | None = None,
) -> pathlib.Path:
    clip_ids = sorted(str(item) for item in (label.get("clip_ids") or []) if str(item).strip())
    payload = {
        "version": _WINDOW_SCOPED_CONTEXT_VERSION,
        "vod_id": str(vod_id),
        "window_start": round(float(label.get("window_start") or 0.0), 2),
        "window_end": round(float(label.get("window_end") or 0.0), 2),
        "expanded_start": round(float(expanded_start), 2),
        "expanded_end": round(float(expanded_end), 2),
        "clip_ids": clip_ids,
        "source": str(label.get("manual_gold_review_source") or ""),
        "source_kind": str(source_kind),
        "training_vod_max_height": _resolved_training_vod_max_height(),
    }
    if cache_scope:
        payload["cache_scope"] = cache_scope
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    path = _vod_feature_cache_dir(vod_id) / "scoped_windows" / digest
    path.mkdir(parents=True, exist_ok=True)
    return path


def _scoped_source_metadata_path(cache_dir: pathlib.Path) -> pathlib.Path:
    return cache_dir / "source.metadata.json"


def _scoped_source_metadata(
    vod_id: str,
    *,
    expanded_start: float,
    expanded_end: float,
    source_kind: str,
    source_signature: dict[str, object],
) -> dict[str, object]:
    return {
        "version": _WINDOW_SCOPED_CONTEXT_VERSION,
        "kind": str(source_kind),
        "vod_id": str(vod_id),
        "expanded_start": round(float(expanded_start), 3),
        "expanded_end": round(float(expanded_end), 3),
        "source": source_signature,
    }


def _scoped_source_is_usable(source_path: pathlib.Path, metadata_path: pathlib.Path, payload: dict[str, object]) -> bool:
    if not source_path.exists() or not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return metadata == payload


def _window_scoped_source_plan_is_usable(plan: WindowScopedSourcePlan) -> bool:
    return _scoped_source_is_usable(plan.scoped_video_path, plan.metadata_path, plan.metadata_payload)


def _window_scoped_source_plan(
    vod_id: str,
    label: dict,
    *,
    vod_path: pathlib.Path | None = None,
    source_kind: str = _WINDOW_SCOPED_TOP_VIEWED_KIND,
    scoped_window: tuple[float, float] | None = None,
    cache_scope: dict[str, object] | None = None,
    source_signature: dict[str, object] | None = None,
) -> WindowScopedSourcePlan:
    scoped_kind = str(source_kind or _WINDOW_SCOPED_TOP_VIEWED_KIND)
    expanded_start, expanded_end = scoped_window or _scoped_training_window(label, scoped_kind)
    cache_dir = _window_scoped_label_cache_dir(
        vod_id,
        label,
        expanded_start=expanded_start,
        expanded_end=expanded_end,
        source_kind=scoped_kind,
        cache_scope=cache_scope,
    )
    scoped_video_path = cache_dir / "source.mp4"
    metadata_path = _scoped_source_metadata_path(cache_dir)
    if vod_path is not None:
        metadata_payload = _scoped_source_metadata(
            vod_id,
            expanded_start=expanded_start,
            expanded_end=expanded_end,
            source_kind=f"{scoped_kind}:cut_source",
            source_signature=_stable_vod_source_signature(vod_id, vod_path),
        )
        uses_local_vod_source = True
    else:
        metadata_payload = _scoped_source_metadata(
            vod_id,
            expanded_start=expanded_start,
            expanded_end=expanded_end,
            source_kind=f"{scoped_kind}:range_source",
            source_signature=source_signature or _window_scoped_range_source_signature(label),
        )
        uses_local_vod_source = False
    return WindowScopedSourcePlan(
        scoped_kind=scoped_kind,
        expanded_start=float(expanded_start),
        expanded_end=float(expanded_end),
        cache_dir=cache_dir,
        scoped_video_path=scoped_video_path,
        metadata_path=metadata_path,
        metadata_payload=metadata_payload,
        uses_local_vod_source=uses_local_vod_source,
    )


def _load_or_build_window_scoped_source(
    vod_id: str,
    label: dict,
    *,
    vod_path: pathlib.Path | None = None,
    job_id: str,
    source_kind: str = _WINDOW_SCOPED_TOP_VIEWED_KIND,
    scoped_window: tuple[float, float] | None = None,
    cache_scope: dict[str, object] | None = None,
    source_signature: dict[str, object] | None = None,
) -> WindowScopedSourcePlan:
    plan = _window_scoped_source_plan(
        vod_id,
        label,
        vod_path=vod_path,
        source_kind=source_kind,
        scoped_window=scoped_window,
        cache_scope=cache_scope,
        source_signature=source_signature,
    )
    if plan.uses_local_vod_source:
        if not _scoped_source_is_usable(plan.scoped_video_path, plan.metadata_path, plan.metadata_payload):
            plan.scoped_video_path.unlink(missing_ok=True)
            collect_viral_clips.cut_video_window(
                vod_path,
                plan.scoped_video_path,
                plan.expanded_start,
                plan.expanded_end,
            )
            _write_json_atomic(plan.metadata_path, plan.metadata_payload)
    else:
        _load_or_download_window_scoped_vod_range(
            vod_id,
            label,
            plan.scoped_video_path,
            plan.metadata_path,
            metadata_payload=plan.metadata_payload,
            job_id=job_id,
            expanded_start=plan.expanded_start,
            expanded_end=plan.expanded_end,
        )
    return plan


def _load_or_build_window_scoped_feature_context(
    vod_id: str,
    label: dict,
    *,
    vod_path: pathlib.Path | None = None,
    job_id: str,
    source_kind: str = _WINDOW_SCOPED_TOP_VIEWED_KIND,
    scoped_window: tuple[float, float] | None = None,
    context_labels: list[dict] | None = None,
    cache_scope: dict[str, object] | None = None,
    source_signature: dict[str, object] | None = None,
) -> tuple[pathlib.Path, VodFeatureContext]:
    source_plan = _load_or_build_window_scoped_source(
        vod_id,
        label,
        vod_path=vod_path,
        job_id=job_id,
        source_kind=source_kind,
        scoped_window=scoped_window,
        cache_scope=cache_scope,
        source_signature=source_signature,
    )

    context = _load_or_build_vod_feature_context(
        vod_id,
        vod_path=source_plan.scoped_video_path,
        job_id=job_id,
        vod_url=str(label.get("vod_url") or ""),
        chat_target_offsets=_chat_target_offsets_for_labels(context_labels or [label]),
        cache_dir=source_plan.cache_dir,
        chat_cache_dir=_vod_feature_cache_dir(vod_id),
        source_start_sec=source_plan.expanded_start,
        source_end_sec=source_plan.expanded_end,
        source_kind=source_plan.scoped_kind,
    )
    return source_plan.scoped_video_path, context


def _window_scoped_range_source_signature(label: dict) -> dict[str, object]:
    clip_ids = sorted(str(item) for item in (label.get("clip_ids") or []) if str(item).strip())
    return {
        "vod_url": str(label.get("vod_url") or ""),
        "clip_ids": clip_ids,
        "window_start": round(float(label.get("window_start") or 0.0), 3),
        "window_end": round(float(label.get("window_end") or 0.0), 3),
    }


def _extended_window_group_signature(group_labels: list[dict]) -> dict[str, object]:
    windows = [
        {
            "window_start": round(float(label.get("window_start") or 0.0), 3),
            "window_end": round(float(label.get("window_end") or 0.0), 3),
            "label_class": str(label.get("label_class") or ""),
            "clip_ids": sorted(str(item) for item in (label.get("clip_ids") or []) if str(item).strip()),
        }
        for label in group_labels
    ]
    digest = hashlib.sha256(
        json.dumps(windows, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    first = group_labels[0] if group_labels else {}
    return {
        "kind": "extended_window_scoped_group",
        "vod_url": str(first.get("vod_url") or ""),
        "label_count": len(group_labels),
        "windows_digest": digest,
    }


def _load_or_download_window_scoped_vod_range(
    vod_id: str,
    label: dict,
    scoped_video_path: pathlib.Path,
    metadata_path: pathlib.Path,
    *,
    metadata_payload: dict[str, object],
    job_id: str,
    expanded_start: float,
    expanded_end: float,
) -> pathlib.Path:
    if _scoped_source_is_usable(scoped_video_path, metadata_path, metadata_payload):
        return scoped_video_path
    vod_url = str(label.get("vod_url") or "").strip()
    if not vod_url:
        raise ValueError(f"Cannot build scoped source for VOD {vod_id}: missing vod_url")

    from services import downloader

    scoped_video_path.unlink(missing_ok=True)
    source_job_id = f"{job_id}:vod_range"
    logger.info(
        "training_vod_window_scoped_range_download vod_id=%s job_id=%s vod_url=%s start=%.3f end=%.3f output=%s",
        vod_id,
        source_job_id,
        vod_url,
        float(expanded_start),
        float(expanded_end),
        scoped_video_path,
    )
    try:
        downloaded_path = downloader.download_range(
            vod_url,
            source_job_id,
            start_sec=float(expanded_start),
            end_sec=float(expanded_end),
        )
        scoped_video_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = scoped_video_path.with_suffix(f"{scoped_video_path.suffix}.tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        shutil.move(str(downloaded_path), str(tmp_path))
        tmp_path.replace(scoped_video_path)
    finally:
        shutil.rmtree(pathlib.Path(settings.storage_dir) / source_job_id, ignore_errors=True)
    _write_json_atomic(metadata_path, metadata_payload)
    return scoped_video_path


def _slice_transcript_for_window(
    transcript: list,
    *,
    window_start: float,
    window_end: float,
) -> list:
    sliced: list = []
    for segment in transcript:
        if float(segment.end) <= window_start or float(segment.start) >= window_end:
            continue
        payload = _serialize_rich_segment(segment)
        payload["start"] = max(float(segment.start), window_start) - window_start
        payload["end"] = min(float(segment.end), window_end) - window_start
        if float(payload["end"]) <= float(payload["start"]):
            continue
        sliced.append(_deserialize_rich_segment(payload))
    return sliced


def _load_or_build_window_modalities(
    vod_id: str,
    *,
    vod_path: pathlib.Path,
    vod_context: VodFeatureContext,
    window_start: float,
    window_end: float,
) -> tuple[list, list]:
    from services import audio_features, video_features
    from services.audio_features import AudioFeatureFrame
    from services.video_features import VideoFeatureFrame

    payload = _window_feature_cache_payload(
        vod_id,
        vod_path,
        window_start=window_start,
        window_end=window_end,
    )
    cache_path = _window_feature_cache_path(vod_context.cache_dir, payload)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            audio_frames = [
                AudioFeatureFrame(**dict(item))
                for item in list(cached.get("audio_frames") or [])
            ]
            visual_frames = [
                VideoFeatureFrame(**dict(item))
                for item in list(cached.get("visual_frames") or [])
            ]
            return audio_frames, visual_frames
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            cache_path.unlink(missing_ok=True)

    audio_frames = audio_features.extract_window_from_signal(
        vod_context.audio_signal,
        vod_context.audio_sample_rate,
        start_sec=window_start,
        end_sec=window_end,
    )
    visual_frames = video_features.extract(
        vod_path,
        start_sec=window_start,
        end_sec=window_end,
    )
    _write_json_atomic(
        cache_path,
        {
            "metadata": payload,
            "audio_frames": [_serialize_frame(frame) for frame in audio_frames],
            "visual_frames": [_serialize_frame(frame) for frame in visual_frames],
        },
    )
    return audio_frames, visual_frames


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a Twitch VOD-window training dataset from labeled clip windows.")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", default="general")
    parser.add_argument("--work-dir", default=str(BACKEND_ROOT / "tmp" / "training_dataset_build"))
    parser.add_argument("--include-short-form", default="", help="Optional supplemental JSONL dataset.")
    args = parser.parse_args()

    labels = _read_jsonl(pathlib.Path(args.labels))
    if not labels:
        print("No Twitch label rows found.")
        return 1

    rows = build_dataset(labels, args.profile, pathlib.Path(args.work_dir))
    if args.include_short_form:
        rows.extend(_read_jsonl(pathlib.Path(args.include_short_form)))
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows), encoding="utf-8")
    print(json.dumps(dataset_summary(rows), ensure_ascii=True))
    return 0


def build_dataset(labels: list[dict], profile: str, work_dir: pathlib.Path) -> list[dict]:
    from services import semantic_candidates, transcriber, video_features

    rows: list[dict] = []
    work_dir.mkdir(parents=True, exist_ok=True)
    vod_workers = _resolved_training_vod_workers()
    label_workers = _resolved_training_label_workers()
    semantic_provider = _semantic_provider_summary(_DATASET_BUILD_LLM_PROVIDER)
    logger.info(
        "training_dataset_runtime profile=%s whisper_model=%s llm_provider=%s semantic_model=%s face_detector=%s deepface=%s vod_workers=%s label_workers=%s",
        profile,
        transcriber.runtime_summary()["whisper_model"],
        semantic_provider,
        semantic_candidates._MODEL_NAME,
        video_features.runtime_summary()["face_detector"],
        "enabled" if video_features.runtime_summary()["deepface_enabled"] else "disabled",
        vod_workers,
        label_workers,
    )
    labels_by_vod: dict[str, list[dict]] = {}
    for row in labels:
        if row.get("label_class") == "ambiguous":
            continue
        labels_by_vod.setdefault(str(row["vod_id"]), []).append(row)

    vod_items = list(labels_by_vod.items())
    if vod_workers > 1 and len(vod_items) > 1:
        return _build_dataset_parallel(vod_items, profile, work_dir, max_workers=vod_workers)

    prefetch_future: concurrent.futures.Future[pathlib.Path] | None = None
    prefetched_vod_id: str | None = None
    prefetched_job_id: str | None = None

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        for position, (vod_id, vod_rows) in enumerate(vod_items):
            job_id = f"twitch_vod_{vod_id}"
            vod_dir = work_dir / job_id
            download_dir = pathlib.Path(settings.storage_dir) / job_id
            prefetched_download_dir: pathlib.Path | None = None
            context_index = _build_vod_context_index(vod_rows)
            vod_checkpoint_rows: list[dict] = []
            vod_dir.mkdir(parents=True, exist_ok=True)
            try:
                checkpoint_rows = _load_vod_checkpoint(work_dir, profile, vod_id, vod_rows)
                if checkpoint_rows is not None:
                    logger.info("training_vod_checkpoint_hit %s", vod_rows[0]["vod_url"])
                    rows.extend(checkpoint_rows)
                    prefetch_future, prefetched_vod_id, prefetched_job_id = _schedule_prefetch(
                        executor,
                        vod_items,
                        position + 1,
                        profile=profile,
                        work_dir=work_dir,
                    )
                    continue

                if _should_use_scoped_label_contexts(vod_rows):
                    label_rows = _build_rows_for_vod_labels(
                        vod_id,
                        vod_rows,
                        vod_path=None,
                        vod_dir=vod_dir,
                        job_id=job_id,
                        profile=profile,
                        context_index=context_index,
                    )
                    rows.extend(label_rows)
                    vod_checkpoint_rows.extend(label_rows)
                    if vod_checkpoint_rows and _can_write_vod_checkpoint(vod_id, vod_checkpoint_rows):
                        _write_vod_checkpoint(work_dir, profile, vod_id, vod_rows, vod_checkpoint_rows)
                    prefetch_future, prefetched_vod_id, prefetched_job_id = _schedule_prefetch(
                        executor,
                        vod_items,
                        position + 1,
                        profile=profile,
                        work_dir=work_dir,
                    )
                    continue

                if prefetch_future is not None and prefetched_vod_id == vod_id:
                    try:
                        vod_path = prefetch_future.result()
                        if prefetched_job_id:
                            prefetched_download_dir = pathlib.Path(settings.storage_dir) / prefetched_job_id
                        logger.info("training_vod_prefetch_ready %s", vod_rows[0]["vod_url"])
                    except Exception as exc:
                        logger.warning("training_vod_prefetch_failed %s: %s", vod_rows[0]["vod_url"], exc)
                        logger.info("training_vod_prefetch_fallback %s", vod_rows[0]["vod_url"])
                        vod_path = _materialize_vod(vod_rows[0]["vod_url"], job_id)
                else:
                    vod_path = _materialize_vod(vod_rows[0]["vod_url"], job_id)

                prefetch_future, prefetched_vod_id, prefetched_job_id = _schedule_prefetch(
                    executor,
                    vod_items,
                    position + 1,
                    profile=profile,
                    work_dir=work_dir,
                )

                label_rows = _build_rows_for_vod_labels(
                    vod_id,
                    vod_rows,
                    vod_path=vod_path,
                    vod_dir=vod_dir,
                    job_id=job_id,
                    profile=profile,
                    context_index=context_index,
                )
                rows.extend(label_rows)
                vod_checkpoint_rows.extend(label_rows)
                if vod_checkpoint_rows and _can_write_vod_checkpoint(vod_id, vod_checkpoint_rows):
                    _write_vod_checkpoint(work_dir, profile, vod_id, vod_rows, vod_checkpoint_rows)
            finally:
                shutil.rmtree(vod_dir, ignore_errors=True)
                shutil.rmtree(download_dir, ignore_errors=True)
                if prefetched_download_dir is not None and prefetched_download_dir != download_dir:
                    shutil.rmtree(prefetched_download_dir, ignore_errors=True)
    return _prepare_ranker_rows(rows)


def _build_dataset_parallel(
    vod_items: list[tuple[str, list[dict]]],
    profile: str,
    work_dir: pathlib.Path,
    *,
    max_workers: int,
) -> list[dict]:
    resolved_workers = max(1, min(int(max_workers), len(vod_items)))
    rows: list[dict] = []
    scheduled_results: list[tuple[str, list[dict] | concurrent.futures.Future[list[dict]]]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=resolved_workers) as executor, \
         concurrent.futures.ThreadPoolExecutor(max_workers=resolved_workers) as download_executor:
        download_futures: dict[str, concurrent.futures.Future[pathlib.Path] | None] = {}
        for vod_id, vod_rows in vod_items:
            checkpoint_rows = _load_vod_checkpoint(work_dir, profile, vod_id, vod_rows)
            if checkpoint_rows is not None:
                logger.info("training_vod_checkpoint_hit %s", vod_rows[0]["vod_url"])
                scheduled_results.append(("checkpoint", checkpoint_rows))
                continue

            job_id = f"twitch_vod_{vod_id}"
            if _should_use_scoped_label_contexts(vod_rows):
                download_futures[vod_id] = None
            else:
                download_futures[vod_id] = download_executor.submit(
                    _materialize_vod,
                    str(vod_rows[0]["vod_url"]),
                    job_id,
                )
            scheduled_results.append((
                "future",
                executor.submit(
                    _build_rows_for_vod,
                    vod_id,
                    vod_rows,
                    profile,
                    work_dir,
                    vod_path_future=download_futures[vod_id],
                ),
            ))

        for kind, result in scheduled_results:
            if kind == "checkpoint":
                rows.extend(result)
            else:
                rows.extend(result.result())
    return _prepare_ranker_rows(rows)


def _build_rows_for_vod(
    vod_id: str,
    vod_rows: list[dict],
    profile: str,
    work_dir: pathlib.Path,
    *,
    vod_path_future: concurrent.futures.Future[pathlib.Path] | None = None,
) -> list[dict]:
    job_id = f"twitch_vod_{vod_id}"
    vod_dir = work_dir / job_id
    download_dir = pathlib.Path(settings.storage_dir) / job_id
    context_index = _build_vod_context_index(vod_rows)
    vod_checkpoint_rows: list[dict] = []
    vod_dir.mkdir(parents=True, exist_ok=True)
    try:
        checkpoint_rows = _load_vod_checkpoint(work_dir, profile, vod_id, vod_rows)
        if checkpoint_rows is not None:
            logger.info("training_vod_checkpoint_hit %s", vod_rows[0]["vod_url"])
            return checkpoint_rows

        if _should_use_scoped_label_contexts(vod_rows):
            vod_path = None
        elif vod_path_future is not None:
            try:
                vod_path = vod_path_future.result()
                logger.info("training_vod_parallel_prefetch_ready %s", vod_rows[0]["vod_url"])
            except Exception as exc:
                logger.warning("training_vod_parallel_prefetch_failed %s: %s", vod_rows[0]["vod_url"], exc)
                logger.info("training_vod_parallel_prefetch_fallback %s", vod_rows[0]["vod_url"])
                vod_path = _materialize_vod(vod_rows[0]["vod_url"], job_id)
        else:
            vod_path = _materialize_vod(vod_rows[0]["vod_url"], job_id)
        vod_checkpoint_rows.extend(
            _build_rows_for_vod_labels(
                vod_id,
                vod_rows,
                vod_path=vod_path,
                vod_dir=vod_dir,
                job_id=job_id,
                profile=profile,
                context_index=context_index,
            )
        )

        if vod_checkpoint_rows and _can_write_vod_checkpoint(vod_id, vod_checkpoint_rows):
            _write_vod_checkpoint(work_dir, profile, vod_id, vod_rows, vod_checkpoint_rows)
        return vod_checkpoint_rows
    finally:
        shutil.rmtree(vod_dir, ignore_errors=True)
        shutil.rmtree(download_dir, ignore_errors=True)


def _chat_target_offsets_for_labels(labels: list[dict]) -> list[float]:
    offsets: set[int] = set()
    target_labels = _chat_target_labels_for_offsets(labels)
    skipped = len(labels) - len(target_labels)
    if skipped:
        logger.info(
            "training_vod_chat_target_labels_filtered labels=%s target_labels=%s skipped=%s",
            len(labels),
            len(target_labels),
            skipped,
        )
    if not target_labels:
        return []
    for label in target_labels:
        start, end = _expanded_training_window(label)
        padded_start = max(float(start) - 5.0, 0.0)
        padded_end = max(float(end) + 5.0, padded_start)
        current = int(padded_start)
        final = int(padded_end)
        while current <= final:
            offsets.add(current)
            current += 10
        offsets.add(final)
    for window in _background_windows_for_labels(labels):
        current = int(max(window.start - 5.0, 0.0))
        final = int(max(window.end + 5.0, current))
        while current <= final:
            offsets.add(current)
            current += 10
        offsets.add(final)
    return [float(offset) for offset in sorted(offsets)]


def _background_windows_for_labels(labels: list[dict]) -> list[sample_background_negatives.BackgroundWindow]:
    if not labels:
        return []
    vod_id = str(labels[0].get("vod_id") or "")
    duration = _vod_duration_seconds(labels)
    blocked = [_label_window(label) for label in labels if _label_window(label) is not None]
    return sample_background_negatives.sample_background_windows(
        vod_id,
        duration_seconds=duration,
        blocked_windows=blocked,
        count=_BACKGROUND_NEGATIVE_WINDOWS_PER_VOD,
        window_seconds=_BACKGROUND_NEGATIVE_WINDOW_SEC,
        max_overlap_ratio=_BACKGROUND_NEGATIVE_MAX_OVERLAP,
    )


def _vod_duration_seconds(labels: list[dict]) -> float:
    values = [
        float(label.get("vod_duration_seconds") or 0.0)
        for label in labels
        if float(label.get("vod_duration_seconds") or 0.0) > 0
    ]
    if values:
        return max(values)
    return max((float(label.get("window_end") or 0.0) for label in labels), default=0.0)


def _label_window(label: dict) -> tuple[float, float] | None:
    start = float(label.get("window_start") or 0.0)
    end = float(label.get("window_end") or start)
    if end <= start:
        return None
    return start, end


def _chat_target_labels_for_offsets(labels: list[dict]) -> list[dict]:
    positives = [
        label
        for label in labels
        if str(label.get("label_class") or "") in _POSITIVE_LABEL_CLASSES
    ]
    negative_candidates = [
        label
        for label in labels
        if str(label.get("label_class") or "") in _CHAT_TARGET_NEGATIVE_LABEL_CLASSES
    ]
    if not positives:
        return sorted(
            _rank_chat_target_negative_labels(negative_candidates, [])[:_CHAT_NEGATIVE_ONLY_TARGET_LIMIT],
            key=lambda label: (
                float(label.get("window_start") or 0.0),
                float(label.get("window_end") or 0.0),
                str(label.get("label_class") or ""),
            ),
        )

    negative_limit = len(positives) * _MAX_VOD_NEGATIVE_ROWS_PER_POSITIVE
    selected_negatives = _rank_chat_target_negative_labels(
        negative_candidates,
        positives,
    )[:negative_limit]
    return sorted(
        [*positives, *selected_negatives],
        key=lambda label: (
            float(label.get("window_start") or 0.0),
            float(label.get("window_end") or 0.0),
            str(label.get("label_class") or ""),
        ),
    )


def _rank_chat_target_negative_labels(negative_labels: list[dict], positive_labels: list[dict]) -> list[dict]:
    return sorted(
        negative_labels,
        key=lambda label: (
            _distance_to_nearest_label(label, positive_labels),
            float(label.get("window_start") or 0.0),
            float(label.get("window_end") or 0.0),
            str(label.get("candidate_id") or label.get("clip_id") or ""),
        ),
    )


def _distance_to_nearest_label(label: dict, positive_labels: list[dict]) -> float:
    if not positive_labels:
        return 0.0
    center = _label_window_center(label)
    return min(abs(center - _label_window_center(positive)) for positive in positive_labels)


def _label_window_center(label: dict) -> float:
    start = float(label.get("window_start") or 0.0)
    end = float(label.get("window_end") or start)
    return (start + end) / 2.0


def _build_rows_for_vod_labels(
    vod_id: str,
    vod_rows: list[dict],
    *,
    vod_path: pathlib.Path | None,
    vod_dir: pathlib.Path,
    job_id: str,
    profile: str,
    context_index: dict,
) -> list[dict]:
    ordered_labels = sorted(vod_rows, key=lambda item: float(item["window_start"]))
    if not ordered_labels:
        return []
    source_kind = _window_scoped_context_kind_for_labels(ordered_labels)
    if source_kind is not None:
        return _build_rows_for_window_scoped_vod_labels(
            vod_id,
            ordered_labels,
            vod_path=vod_path,
            vod_dir=vod_dir,
            job_id=job_id,
            profile=profile,
            context_index=context_index,
            source_kind=source_kind,
        )
    if vod_path is None:
        raise ValueError(f"Cannot build full-VOD rows for {vod_id}: missing vod_path")
    vod_feature_context = _load_or_build_vod_feature_context(
        vod_id,
        vod_path=vod_path,
        job_id=job_id,
        vod_url=str(ordered_labels[0].get("vod_url") or ""),
        chat_target_offsets=_chat_target_offsets_for_labels(ordered_labels),
    )
    total_labels = len(ordered_labels)
    resolved_workers = min(_resolved_training_label_workers(), total_labels)
    logger.info(
        "training_vod_label_phase_start vod_id=%s labels=%s label_workers=%s",
        vod_id,
        total_labels,
        resolved_workers,
    )
    logger.info(
        "training_vod_label_window_source vod_id=%s mode=%s labels=%s",
        vod_id,
        "direct_vod",
        total_labels,
    )

    started_at = time.perf_counter()
    if resolved_workers <= 1:
        rows: list[dict] = []
        progress_interval = _label_progress_interval(total_labels)
        for index, label in enumerate(ordered_labels, start=1):
            rows.extend(
                _process_vod_label(
                    vod_id,
                    index,
                    label,
                    vod_path=vod_path,
                    vod_dir=vod_dir,
                    vod_feature_context=vod_feature_context,
                    job_id=job_id,
                    profile=profile,
                    context_index=context_index,
                )
            )
            if _should_log_label_progress(index, total_labels, progress_interval):
                logger.info(
                    "training_vod_label_progress vod_id=%s completed=%s/%s rows=%s elapsed_sec=%.1f mode=sequential",
                    vod_id,
                    index,
                    total_labels,
                    len(rows),
                    time.perf_counter() - started_at,
                )
        logger.info(
            "training_vod_label_phase_done vod_id=%s completed=%s rows=%s elapsed_sec=%.1f",
            vod_id,
            total_labels,
            len(rows),
            time.perf_counter() - started_at,
        )
        background_rows = _build_background_negative_rows_for_vod(
            vod_id,
            ordered_labels,
            rows,
            vod_path=vod_path,
            vod_dir=vod_dir,
            vod_feature_context=vod_feature_context,
            job_id=job_id,
            profile=profile,
            context_index=context_index,
        )
        rows.extend(background_rows)
        return rows

    progress_interval = _label_progress_interval(total_labels)
    futures: dict[concurrent.futures.Future[list[dict]], int] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=resolved_workers) as executor:
        for index, label in enumerate(ordered_labels, start=1):
            future = executor.submit(
                _process_vod_label,
                vod_id,
                index,
                label,
                vod_path=vod_path,
                vod_dir=vod_dir,
                vod_feature_context=vod_feature_context,
                job_id=job_id,
                profile=profile,
                context_index=context_index,
            )
            futures[future] = index

        completed = 0
        row_results: dict[int, list[dict]] = {}
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            row_results[index] = future.result()
            completed += 1
            if _should_log_label_progress(completed, total_labels, progress_interval):
                logger.info(
                    "training_vod_label_progress vod_id=%s completed=%s/%s latest_window=%s rows=%s elapsed_sec=%.1f mode=parallel",
                    vod_id,
                    completed,
                    total_labels,
                    index,
                    sum(len(result) for result in row_results.values()),
                    time.perf_counter() - started_at,
                )

        rows: list[dict] = []
        for index in range(1, total_labels + 1):
            rows.extend(row_results.get(index, []))
        logger.info(
            "training_vod_label_phase_done vod_id=%s completed=%s rows=%s elapsed_sec=%.1f",
            vod_id,
            total_labels,
            len(rows),
            time.perf_counter() - started_at,
        )
        background_rows = _build_background_negative_rows_for_vod(
            vod_id,
            ordered_labels,
            rows,
            vod_path=vod_path,
            vod_dir=vod_dir,
            vod_feature_context=vod_feature_context,
            job_id=job_id,
            profile=profile,
            context_index=context_index,
        )
        rows.extend(background_rows)
        return rows


def _build_rows_for_window_scoped_vod_labels(
    vod_id: str,
    ordered_labels: list[dict],
    *,
    vod_path: pathlib.Path | None,
    vod_dir: pathlib.Path,
    job_id: str,
    profile: str,
    context_index: dict,
    source_kind: str = _WINDOW_SCOPED_TOP_VIEWED_KIND,
) -> list[dict]:
    if source_kind == _WINDOW_SCOPED_EXTENDED_KIND:
        return _build_rows_for_extended_window_scoped_vod_labels(
            vod_id,
            ordered_labels,
            vod_path=vod_path,
            vod_dir=vod_dir,
            job_id=job_id,
            profile=profile,
            context_index=context_index,
        )

    logger.info(
        "training_vod_label_phase_start vod_id=%s labels=%s label_workers=%s",
        vod_id,
        len(ordered_labels),
        1,
    )
    logger.info(
        "training_vod_label_window_source vod_id=%s mode=%s labels=%s",
        vod_id,
        source_kind,
        len(ordered_labels),
    )
    started_at = time.perf_counter()
    rows: list[dict] = []
    progress_interval = _label_progress_interval(len(ordered_labels))
    for index, label in enumerate(ordered_labels, start=1):
        scoped_job_id = f"{job_id}:window_scoped:{index}"
        try:
            scoped_vod_path, scoped_context = _load_or_build_window_scoped_feature_context(
                vod_id,
                label,
                vod_path=vod_path,
                job_id=scoped_job_id,
                source_kind=source_kind,
            )
        except Exception as exc:
            logger.warning(
                "training_vod_window_scoped_label_skip vod_id=%s index=%s/%s job_id=%s reason=%s",
                vod_id,
                index,
                len(ordered_labels),
                scoped_job_id,
                _compact_exception_message(exc),
            )
            if _is_unavailable_vod_download_error(exc):
                logger.warning(
                    "training_vod_window_scoped_vod_unavailable vod_id=%s skipped_remaining=%s reason=%s",
                    vod_id,
                    len(ordered_labels) - index + 1,
                    _compact_exception_message(exc),
                )
                break
            continue
        rows.extend(
            _process_vod_label(
                vod_id,
                index,
                label,
                vod_path=scoped_vod_path,
                vod_dir=vod_dir,
                vod_feature_context=scoped_context,
                job_id=job_id,
                profile=profile,
                context_index=context_index,
            )
        )
        if _should_log_label_progress(index, len(ordered_labels), progress_interval):
            logger.info(
                "training_vod_label_progress vod_id=%s completed=%s/%s rows=%s elapsed_sec=%.1f mode=window_scoped",
                vod_id,
                index,
                len(ordered_labels),
                len(rows),
                time.perf_counter() - started_at,
            )
    logger.info(
        "training_vod_label_phase_done vod_id=%s completed=%s rows=%s elapsed_sec=%.1f mode=window_scoped",
        vod_id,
        len(ordered_labels),
        len(rows),
        time.perf_counter() - started_at,
    )
    return rows


def _build_rows_for_extended_window_scoped_vod_labels(
    vod_id: str,
    ordered_labels: list[dict],
    *,
    vod_path: pathlib.Path | None,
    vod_dir: pathlib.Path,
    job_id: str,
    profile: str,
    context_index: dict,
) -> list[dict]:
    groups = _extended_window_scoped_label_groups(ordered_labels)
    logger.info(
        "training_vod_label_phase_start vod_id=%s labels=%s label_workers=%s",
        vod_id,
        len(ordered_labels),
        1,
    )
    logger.info(
        "training_vod_label_window_source vod_id=%s mode=%s labels=%s groups=%s group_max_sec=%s",
        vod_id,
        _WINDOW_SCOPED_EXTENDED_KIND,
        len(ordered_labels),
        len(groups),
        _resolved_extended_window_group_max_sec(),
    )
    started_at = time.perf_counter()
    rows: list[dict] = []
    completed = 0
    total_labels = len(ordered_labels)
    progress_interval = _label_progress_interval(total_labels)
    checkpoint_work_dir = vod_dir.parent
    group_args = [
        _extended_window_scoped_group_context_args(group_index, group)
        for group_index, group in enumerate(groups, start=1)
    ]
    prefetch_enabled = _should_prefetch_window_scoped_group_sources() and len(group_args) > 1
    prefetch_futures: dict[int, concurrent.futures.Future] = {}
    background_rows_added = 0

    def _schedule_source_prefetch(
        executor: concurrent.futures.Executor,
        next_group_index: int,
    ) -> None:
        if next_group_index > len(group_args) or next_group_index in prefetch_futures:
            return
        args = group_args[next_group_index - 1]
        if _load_extended_window_group_checkpoint(
            checkpoint_work_dir,
            profile,
            vod_id,
            next_group_index,
            args["group_labels"],
            scoped_window=args["scoped_window"],
            cache_scope=args["cache_scope"],
            source_signature=args["source_signature"],
        ) is not None:
            logger.info(
                "training_vod_window_scoped_group_prefetch_skip vod_id=%s group=%s/%s reason=group_checkpoint",
                vod_id,
                next_group_index,
                len(group_args),
            )
            return
        source_plan = _window_scoped_source_plan(
            vod_id,
            args["representative_label"],
            vod_path=vod_path,
            source_kind=_WINDOW_SCOPED_EXTENDED_KIND,
            scoped_window=args["scoped_window"],
            cache_scope=args["cache_scope"],
            source_signature=args["source_signature"],
        )
        if _window_scoped_source_plan_is_usable(source_plan):
            logger.info(
                "training_vod_window_scoped_group_prefetch_skip vod_id=%s group=%s/%s reason=cached",
                vod_id,
                next_group_index,
                len(group_args),
            )
            return
        logger.info(
            "training_vod_window_scoped_group_prefetch_start vod_id=%s group=%s/%s labels=%s start=%.3f end=%.3f",
            vod_id,
            next_group_index,
            len(group_args),
            len(args["group_labels"]),
            args["scoped_window"][0],
            args["scoped_window"][1],
        )
        prefetch_futures[next_group_index] = executor.submit(
            _load_or_build_window_scoped_source,
            vod_id,
            args["representative_label"],
            vod_path=vod_path,
            job_id=f"{job_id}:window_scoped_group:{next_group_index}:source_prefetch",
            source_kind=_WINDOW_SCOPED_EXTENDED_KIND,
            scoped_window=args["scoped_window"],
            cache_scope=args["cache_scope"],
            source_signature=args["source_signature"],
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as source_prefetch_executor:
        for group_index, group in enumerate(groups, start=1):
            args = group_args[group_index - 1]
            group_labels = args["group_labels"]
            representative_label = args["representative_label"]
            scoped_job_id = f"{job_id}:window_scoped_group:{group_index}"
            scoped_window = args["scoped_window"]
            cache_scope = args["cache_scope"]
            source_signature = args["source_signature"]
            logger.info(
                "training_vod_window_scoped_group_start vod_id=%s group=%s/%s labels=%s start=%.3f end=%.3f",
                vod_id,
                group_index,
                len(groups),
                len(group_labels),
                scoped_window[0],
                scoped_window[1],
            )
            prefetch_future = prefetch_futures.pop(group_index, None)
            checkpoint_rows = _load_extended_window_group_checkpoint(
                checkpoint_work_dir,
                profile,
                vod_id,
                group_index,
                group_labels,
                scoped_window=scoped_window,
                cache_scope=cache_scope,
                source_signature=source_signature,
            )
            if checkpoint_rows is not None:
                if prefetch_future is not None:
                    prefetch_future.cancel()
                rows.extend(checkpoint_rows)
                background_rows_added += sum(
                    1
                    for row in checkpoint_rows
                    if str(row.get("supervision_tier") or "") == "background_negative"
                )
                completed += len(group_labels)
                logger.info(
                    "training_vod_window_scoped_group_checkpoint_hit vod_id=%s group=%s/%s labels=%s rows=%s",
                    vod_id,
                    group_index,
                    len(groups),
                    len(group_labels),
                    len(checkpoint_rows),
                )
                if _should_log_label_progress(completed, total_labels, progress_interval):
                    logger.info(
                        "training_vod_label_progress vod_id=%s completed=%s/%s rows=%s elapsed_sec=%.1f mode=window_scoped groups=%s",
                        vod_id,
                        completed,
                        total_labels,
                        len(rows),
                        time.perf_counter() - started_at,
                        len(groups),
                    )
                continue

            try:
                if prefetch_future is not None:
                    try:
                        prefetch_future.result()
                        logger.info(
                            "training_vod_window_scoped_group_prefetch_ready vod_id=%s group=%s/%s",
                            vod_id,
                            group_index,
                            len(groups),
                        )
                    except Exception as prefetch_exc:
                        logger.warning(
                            "training_vod_window_scoped_group_prefetch_failed vod_id=%s group=%s/%s reason=%s",
                            vod_id,
                            group_index,
                            len(groups),
                            _compact_exception_message(prefetch_exc),
                        )
                elif prefetch_enabled:
                    _load_or_build_window_scoped_source(
                        vod_id,
                        representative_label,
                        vod_path=vod_path,
                        job_id=f"{scoped_job_id}:source",
                        source_kind=_WINDOW_SCOPED_EXTENDED_KIND,
                        scoped_window=scoped_window,
                        cache_scope=cache_scope,
                        source_signature=source_signature,
                    )
                if prefetch_enabled:
                    _schedule_source_prefetch(source_prefetch_executor, group_index + 1)
                scoped_vod_path, scoped_context = _load_or_build_window_scoped_feature_context(
                    vod_id,
                    representative_label,
                    vod_path=vod_path,
                    job_id=scoped_job_id,
                    source_kind=_WINDOW_SCOPED_EXTENDED_KIND,
                    scoped_window=scoped_window,
                    context_labels=group_labels,
                    cache_scope=cache_scope,
                    source_signature=source_signature,
                )
            except Exception as exc:
                logger.warning(
                    "training_vod_window_scoped_group_skip vod_id=%s group=%s/%s job_id=%s reason=%s",
                    vod_id,
                    group_index,
                    len(groups),
                    scoped_job_id,
                    _compact_exception_message(exc),
                )
                if _is_unavailable_vod_download_error(exc):
                    logger.warning(
                        "training_vod_window_scoped_vod_unavailable vod_id=%s skipped_remaining_groups=%s reason=%s",
                        vod_id,
                        len(groups) - group_index + 1,
                        _compact_exception_message(exc),
                    )
                    break
                completed += len(group_labels)
                continue

            group_rows: list[dict] = []
            for index, label in group["items"]:
                label_rows = _process_vod_label(
                    vod_id,
                    index,
                    label,
                    vod_path=scoped_vod_path,
                    vod_dir=vod_dir,
                    vod_feature_context=scoped_context,
                    job_id=job_id,
                    profile=profile,
                    context_index=context_index,
                )
                group_rows.extend(label_rows)
                rows.extend(label_rows)
                completed += 1
                if _should_log_label_progress(completed, total_labels, progress_interval):
                    logger.info(
                        "training_vod_label_progress vod_id=%s completed=%s/%s rows=%s elapsed_sec=%.1f mode=window_scoped groups=%s",
                        vod_id,
                        completed,
                        total_labels,
                        len(rows),
                        time.perf_counter() - started_at,
                        len(groups),
                    )
            if group_rows:
                remaining_background_rows = max(_BACKGROUND_NEGATIVE_WINDOWS_PER_VOD - background_rows_added, 0)
                if remaining_background_rows > 0:
                    background_rows = _build_background_negative_rows_for_vod(
                        vod_id,
                        group_labels,
                        group_rows,
                        vod_path=scoped_vod_path,
                        vod_dir=vod_dir,
                        vod_feature_context=scoped_context,
                        job_id=scoped_job_id,
                        profile=profile,
                        context_index=context_index,
                        count=remaining_background_rows,
                    )
                    if background_rows:
                        background_rows_added += len(background_rows)
                        group_rows.extend(background_rows)
                        rows.extend(background_rows)
            if group_rows and _can_write_vod_checkpoint(vod_id, group_rows):
                _write_extended_window_group_checkpoint(
                    checkpoint_work_dir,
                    profile,
                    vod_id,
                    group_index,
                    group_labels,
                    group_rows,
                    scoped_window=scoped_window,
                    cache_scope=cache_scope,
                    source_signature=source_signature,
                )

    logger.info(
        "training_vod_label_phase_done vod_id=%s completed=%s rows=%s elapsed_sec=%.1f mode=window_scoped groups=%s",
        vod_id,
        total_labels,
        len(rows),
        time.perf_counter() - started_at,
        len(groups),
    )
    return rows


def _extended_window_scoped_group_context_args(group_index: int, group: dict[str, object]) -> dict[str, object]:
    group_labels = [label for _index, label in group["items"]]
    scoped_window = (float(group["expanded_start"]), float(group["expanded_end"]))
    source_signature = _extended_window_group_signature(group_labels)
    cache_scope = {
        "kind": "extended_window_group",
        "group_index": group_index,
        "label_count": len(group_labels),
        "group_start": round(scoped_window[0], 3),
        "group_end": round(scoped_window[1], 3),
        "windows_digest": source_signature["windows_digest"],
    }
    return {
        "group_labels": group_labels,
        "representative_label": group_labels[0],
        "scoped_window": scoped_window,
        "cache_scope": cache_scope,
        "source_signature": source_signature,
    }


def _extended_window_scoped_label_groups(ordered_labels: list[dict]) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    max_group_sec = float(_resolved_extended_window_group_max_sec())
    for index, label in enumerate(ordered_labels, start=1):
        start, end = _scoped_training_window(label, _WINDOW_SCOPED_EXTENDED_KIND)
        if not groups:
            groups.append({"expanded_start": start, "expanded_end": end, "items": [(index, label)]})
            continue
        current = groups[-1]
        current_start = float(current["expanded_start"])
        current_end = float(current["expanded_end"])
        merged_start = min(current_start, start)
        merged_end = max(current_end, end)
        if start <= current_end and (merged_end - merged_start) <= max_group_sec:
            current["expanded_start"] = merged_start
            current["expanded_end"] = merged_end
            current["items"].append((index, label))
            continue
        groups.append({"expanded_start": start, "expanded_end": end, "items": [(index, label)]})
    return groups


def _is_unavailable_vod_download_error(exc: Exception) -> bool:
    message = _compact_exception_message(exc).lower()
    return (
        "video" in message
        and (
            "does not exist" in message
            or "is unavailable" in message
            or "not available" in message
            or "has been deleted" in message
        )
    )


def _compact_exception_message(exc: Exception, limit: int = 500) -> str:
    message = str(exc).replace("\n", " ").strip() or exc.__class__.__name__
    if len(message) <= limit:
        return message
    return message[: limit - 3] + "..."


def _label_progress_interval(total_labels: int) -> int:
    if total_labels <= 10:
        return 1
    if total_labels <= 50:
        return 5
    return 25


def _should_log_label_progress(completed: int, total_labels: int, interval: int) -> bool:
    return completed == 1 or completed == total_labels or completed % max(interval, 1) == 0


def _process_vod_label(
    vod_id: str,
    index: int,
    label: dict,
    *,
    vod_path: pathlib.Path,
    vod_dir: pathlib.Path,
    vod_feature_context: VodFeatureContext,
    job_id: str,
    profile: str,
    context_index: dict,
) -> list[dict]:
    clip_dir = vod_dir / f"window_{index:04d}"
    clip_dir.mkdir(parents=True, exist_ok=True)
    try:
        try:
            label_rows = _build_rows_for_label(
                label,
                vod_path=vod_path,
                clip_dir=clip_dir,
                vod_feature_context=vod_feature_context,
                job_prefix=f"{job_id}_{index}",
                profile=profile,
            )
        except Exception as exc:
            logger.warning(
                "Skipping VOD window %s[%s] %.2f-%.2f: %s",
                vod_id,
                index,
                float(label["window_start"]),
                float(label["window_end"]),
                exc,
            )
            return []
        finalized_rows = _finalize_label_rows(label_rows, label, context_index)
        source_kind = str(getattr(vod_feature_context, "source_kind", "") or "")
        if source_kind:
            for row in finalized_rows:
                provenance = dict(row.get("feature_provenance") or {})
                provenance["vod_feature_source_kind"] = source_kind
                row["feature_provenance"] = provenance
        return finalized_rows
    finally:
        shutil.rmtree(clip_dir, ignore_errors=True)


def _build_background_negative_rows_for_vod(
    vod_id: str,
    ordered_labels: list[dict],
    existing_rows: list[dict],
    *,
    vod_path: pathlib.Path,
    vod_dir: pathlib.Path,
    vod_feature_context: VodFeatureContext,
    job_id: str,
    profile: str,
    context_index: dict,
    count: int = _BACKGROUND_NEGATIVE_WINDOWS_PER_VOD,
) -> list[dict]:
    source_kind = str(getattr(vod_feature_context, "source_kind", "full_vod") or "full_vod")
    if source_kind not in {"full_vod", _WINDOW_SCOPED_EXTENDED_KIND}:
        return []
    source_offset = float(getattr(vod_feature_context, "source_start_sec", 0.0) or 0.0)
    source_duration = _background_sampling_duration_seconds(ordered_labels, vod_feature_context)
    if source_duration <= 0.0 or count <= 0:
        return []
    blocked_windows = [
        window
        for window in (_label_window(label) for label in ordered_labels)
        if window is not None
    ]
    blocked_windows.extend(
        window
        for window in (_row_window_tuple(row) for row in existing_rows)
        if window is not None
    )
    local_blocked_windows = [
        window
        for window in (_local_background_window(window, source_offset, source_duration) for window in blocked_windows)
        if window is not None
    ]
    candidate_windows = sample_background_negatives.sample_background_windows(
        vod_id,
        duration_seconds=source_duration,
        blocked_windows=local_blocked_windows,
        count=count,
        window_seconds=_BACKGROUND_NEGATIVE_WINDOW_SEC,
        max_overlap_ratio=_BACKGROUND_NEGATIVE_MAX_OVERLAP,
    )
    if not candidate_windows:
        logger.info(
            "training_vod_background_negative_diagnostics vod_id=%s source_kind=%s candidate_windows=0 quiet_filter_rejected=0 feature_build_failed=0 rows=0",
            vod_id,
            source_kind,
        )
        return []

    from services import audio_features, feature_builder, scorer, video_features
    from services.timeline_types import Candidate

    rows: list[dict] = []
    clip_dir = vod_dir / "background_negatives"
    clip_dir.mkdir(parents=True, exist_ok=True)
    base_label = dict(ordered_labels[0])
    quiet_thresholds = _compute_vod_quiet_thresholds(existing_rows)
    local_game_events = _local_game_events_for_source(
        vod_id,
        source_start_sec=source_offset,
        source_end_sec=source_offset + source_duration,
    )
    quiet_filter_rejected_count = 0
    quiet_fallback_candidates: list[tuple[tuple[int, float, float], dict[str, float], Candidate, object]] = []
    feature_build_failed_count = 0
    empty_transcript_count = 0
    for index, window in enumerate(candidate_windows, start=1):
        transcript = _slice_transcript_for_window(
            vod_feature_context.transcript,
            window_start=window.start,
            window_end=window.end,
        )
        if not transcript:
            empty_transcript_count += 1
            continue
        try:
            audio_frames, visual_frames = _load_or_build_window_modalities(
                vod_id,
                vod_path=vod_path,
                vod_context=vod_feature_context,
                window_start=window.start,
                window_end=window.end,
            )
        except Exception as exc:
            logger.warning(
                "training_vod_background_negative_skip vod_id=%s index=%s reason=%s",
                vod_id,
                index,
                _compact_exception_message(exc),
            )
            feature_build_failed_count += 1
            continue
        transcript = audio_features.enrich_segments(transcript, audio_frames)
        transcript = video_features.enrich_segments(transcript, visual_frames)
        transcript = scorer.score_segments(transcript)
        candidate = Candidate(
            id=f"{job_id}:background_negative:{index}:{int(window.start * 100)}",
            start=0.0,
            end=round(window.end - window.start, 2),
            source="background_negative",
            anchor_time=round((window.end - window.start) / 2.0, 2),
            score_hint=0.0,
            tags=["background_negative"],
        )
        feature_rows = feature_builder.build_features(
            [candidate],
            transcript,
            profile,
            llm_candidate_ids=set(),
            llm_provider="none",
            game_events=local_game_events,
            video_path=vod_path,
        )
        if not feature_rows:
            feature_build_failed_count += 1
            continue
        features = feature_rows[0]
        quiet_assessment = _background_quiet_assessment(features, quiet_thresholds)
        if not quiet_assessment["accepted"]:
            quiet_filter_rejected_count += 1
            fallback_score = _background_quiet_fallback_score(features, quiet_thresholds, quiet_assessment)
            if fallback_score is not None:
                quiet_fallback_candidates.append(
                    (
                        fallback_score,
                        {
                            "start": round(source_offset + window.start, 2),
                            "end": round(source_offset + window.end, 2),
                        },
                        candidate,
                        features,
                    )
                )
            logger.debug(
                "training_vod_background_negative_quiet_rejected vod_id=%s index=%s source_kind=%s values=%s thresholds=%s",
                vod_id,
                index,
                source_kind,
                _background_quiet_feature_values(features),
                quiet_thresholds,
            )
            continue
        absolute_start = round(source_offset + window.start, 2)
        absolute_end = round(source_offset + window.end, 2)
        rows.append(
            _background_negative_training_row(
                base_label,
                candidate,
                features,
                absolute_start=absolute_start,
                absolute_end=absolute_end,
                context_index=context_index,
                negative_source="random_mid_vod_background",
                supervision_reason_suffix=":background_sampler",
                sample_weight_scale=1.0,
            )
        )
    fallback_rows_added = 0
    if not rows and quiet_fallback_candidates:
        quiet_fallback_candidates.sort(key=lambda item: item[0])
        for _score, window, candidate, features in quiet_fallback_candidates[: max(1, min(2, count))]:
            rows.append(
                _background_negative_training_row(
                    base_label,
                    candidate,
                    features,
                    absolute_start=float(window["start"]),
                    absolute_end=float(window["end"]),
                    context_index=context_index,
                    negative_source="relative_quiet_fallback",
                    supervision_reason_suffix=":background_sampler:relative_quiet_fallback",
                    sample_weight_scale=0.6,
                )
            )
            fallback_rows_added += 1
        logger.info(
            "training_vod_background_negative_relative_fallback vod_id=%s source_kind=%s candidates=%s rows=%s",
            vod_id,
            source_kind,
            len(quiet_fallback_candidates),
            fallback_rows_added,
        )
    logger.info(
        "training_vod_background_negative_diagnostics vod_id=%s source_kind=%s candidate_windows=%s quiet_filter_rejected=%s feature_build_failed=%s empty_transcript=%s fallback_rows=%s rows=%s thresholds=%s",
        vod_id,
        source_kind,
        len(candidate_windows),
        quiet_filter_rejected_count,
        feature_build_failed_count,
        empty_transcript_count,
        fallback_rows_added,
        len(rows),
        quiet_thresholds,
    )
    if rows:
        logger.info(
            "training_vod_background_negatives_added vod_id=%s source_kind=%s candidates=%s rows=%s",
            vod_id,
            source_kind,
            len(candidate_windows),
            len(rows),
        )
    return rows


def _background_negative_training_row(
    base_label: dict,
    candidate,
    features,
    *,
    absolute_start: float,
    absolute_end: float,
    context_index: dict,
    negative_source: str,
    supervision_reason_suffix: str,
    sample_weight_scale: float,
) -> dict:
    from services.timeline_types import Candidate

    label = dict(base_label)
    label.update(
        {
            "window_start": float(absolute_start),
            "window_end": float(absolute_end),
            "label_class": "background_negative",
            "label_score": 0.0,
            "negative_source": negative_source,
            "label_source": "background_negative_sampler",
            "clip_ids": [],
        }
    )
    absolute_candidate = Candidate(
        id=candidate.id,
        start=absolute_start,
        end=absolute_end,
        source=candidate.source,
        anchor_time=round((absolute_start + absolute_end) / 2.0, 2),
        score_hint=0.0,
        tags=list(candidate.tags),
    )
    row = _training_row_from_label(
        label,
        absolute_candidate,
        features,
        sample_weight_scale=sample_weight_scale,
        dataset_source="twitch_vod_background_negative",
        supervision_reason_suffix=supervision_reason_suffix,
        window_origin="random_mid_vod_background",
        label_class_override="background_negative",
        label_value_override=0,
        supervision_tier="background_negative",
        rejected_reason_override="background_negative",
    )
    _apply_vod_context_features(row, label, context_index)
    row["feature_provenance"] = collect_viral_clips._feature_provenance(
        row["features"],
        has_vod_context=True,
    )
    return row


def _background_sampling_duration_seconds(labels: list[dict], vod_feature_context: VodFeatureContext) -> float:
    source_start = float(getattr(vod_feature_context, "source_start_sec", 0.0) or 0.0)
    source_end = float(getattr(vod_feature_context, "source_end_sec", 0.0) or 0.0)
    if source_end > source_start:
        return max(source_end - source_start, 0.0)
    return _vod_duration_seconds(labels)


def _local_background_window(
    window: tuple[float, float],
    source_offset: float,
    source_duration: float,
) -> tuple[float, float] | None:
    start, end = window
    local_start = max(float(start) - source_offset, 0.0)
    local_end = min(float(end) - source_offset, source_duration)
    if local_end <= local_start:
        return None
    return (round(local_start, 2), round(local_end, 2))


def _compute_vod_quiet_thresholds(rows: list[dict]) -> dict[str, float | str]:
    thresholds: dict[str, float | str] = {}
    for key in _BACKGROUND_QUIET_FEATURE_KEYS:
        values = sorted(
            _background_row_feature_value(row, key)
            for row in rows
            if _background_row_feature_value(row, key) is not None
        )
        if values:
            thresholds[key] = values[min(len(values) - 1, len(values) // 4)]
    max_hype_values = sorted(
        value
        for row in rows
        if (value := _background_row_feature_value(row, "max_hype")) is not None
    )
    if max_hype_values:
        thresholds["max_hype_p95"] = max_hype_values[min(len(max_hype_values) - 1, int(math.floor((len(max_hype_values) - 1) * 0.95)))]
    thresholds["mode"] = "relative" if any(key in thresholds for key in _BACKGROUND_QUIET_FEATURE_KEYS) else "absolute"
    return thresholds


def _background_row_feature_value(row: dict, key: str) -> float | None:
    features = row.get("features")
    if not isinstance(features, dict) or key not in features:
        return None
    try:
        value = float(features.get(key) or 0.0)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _background_quiet_feature_values(features) -> dict[str, float]:
    return {key: _feature_attr_float(features, key) for key in _BACKGROUND_QUIET_FEATURE_KEYS}


def _feature_attr_float(features, key: str) -> float:
    if isinstance(features, dict):
        raw = features.get(key, 0.0)
    else:
        raw = getattr(features, key, 0.0)
    try:
        value = float(raw or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def _background_features_are_quiet(features, thresholds: dict[str, float | str] | None = None) -> bool:
    return bool(_background_quiet_assessment(features, thresholds)["accepted"])


def _background_quiet_assessment(features, thresholds: dict[str, float | str] | None = None) -> dict[str, object]:
    thresholds = thresholds or {}
    if str(thresholds.get("mode") or "absolute") != "relative":
        accepted = all(
            _feature_attr_float(features, key) <= float(limit)
            for key, limit in _BACKGROUND_ABSOLUTE_QUIET_THRESHOLDS.items()
        )
        return {
            "accepted": accepted,
            "matches": len(_BACKGROUND_ABSOLUTE_QUIET_THRESHOLDS) if accepted else 0,
            "comparable": len(_BACKGROUND_ABSOLUTE_QUIET_THRESHOLDS),
            "hard_reject": not accepted,
        }

    matches = 0
    comparable = 0
    for key in _BACKGROUND_QUIET_FEATURE_KEYS:
        limit = thresholds.get(key)
        if not isinstance(limit, (int, float)):
            continue
        comparable += 1
        if _feature_attr_float(features, key) <= float(limit):
            matches += 1
    max_hype = _feature_attr_float(features, "max_hype")
    p95 = thresholds.get("max_hype_p95")
    hard_reasons: list[str] = []
    if isinstance(p95, (int, float)) and p95 > 0.0 and max_hype > float(p95) * _BACKGROUND_MAX_HYPE_P95_FACTOR:
        hard_reasons.append("max_hype")
    if _feature_attr_float(features, "face_reaction_peak") > _BACKGROUND_FACE_REACTION_HARD_MAX:
        hard_reasons.append("face_reaction")
    if comparable < _BACKGROUND_RELATIVE_QUIET_MIN_MATCHES:
        accepted = all(
            _feature_attr_float(features, key) <= float(limit)
            for key, limit in _BACKGROUND_ABSOLUTE_QUIET_THRESHOLDS.items()
        )
        return {
            "accepted": accepted,
            "matches": matches,
            "comparable": comparable,
            "hard_reject": not accepted,
            "hard_reasons": hard_reasons,
        }
    if hard_reasons:
        return {
            "accepted": False,
            "matches": matches,
            "comparable": comparable,
            "hard_reject": True,
            "hard_reasons": hard_reasons,
        }
    return {
        "accepted": matches >= _BACKGROUND_RELATIVE_QUIET_MIN_MATCHES,
        "matches": matches,
        "comparable": comparable,
        "hard_reject": False,
        "hard_reasons": [],
    }


def _background_quiet_fallback_score(
    features,
    thresholds: dict[str, float | str] | None,
    assessment: dict[str, object],
) -> tuple[int, float, float] | None:
    if str((thresholds or {}).get("mode") or "absolute") != "relative":
        return None
    matches = int(assessment.get("matches") or 0)
    comparable = int(assessment.get("comparable") or 0)
    if comparable <= 0 or matches < _BACKGROUND_RELATIVE_QUIET_FALLBACK_MIN_MATCHES:
        return None
    p95 = (thresholds or {}).get("max_hype_p95")
    if isinstance(p95, (int, float)) and p95 > 0.0:
        if _feature_attr_float(features, "max_hype") > float(p95):
            return None
    face_limit = max(_BACKGROUND_FACE_REACTION_HARD_MAX, float(_BACKGROUND_ABSOLUTE_QUIET_THRESHOLDS["face_reaction_peak"]))
    if _feature_attr_float(features, "face_reaction_peak") > face_limit:
        return None
    overflow = 0.0
    for key in _BACKGROUND_QUIET_FEATURE_KEYS:
        limit = (thresholds or {}).get(key)
        if not isinstance(limit, (int, float)) or float(limit) <= 0.0:
            continue
        value = _feature_attr_float(features, key)
        if value > float(limit):
            overflow += (value - float(limit)) / max(float(limit), 1e-6)
    max_hype = _feature_attr_float(features, "max_hype")
    return (-matches, overflow, max_hype)


def _row_window_tuple(row: dict) -> tuple[float, float] | None:
    start = float(row.get("window_start") or row.get("proxy_window_start") or 0.0)
    end = float(row.get("window_end") or row.get("proxy_window_end") or start)
    if end <= start:
        return None
    return start, end


def _finalize_label_rows(label_rows: list[dict], label: dict, context_index: dict) -> list[dict]:
    rows: list[dict] = []
    for row in label_rows:
        row.setdefault("label_class", str(label.get("label_class", "")))
        row.setdefault("label_score", float(label.get("label_score", 0.0)))
        row.setdefault("platform_source", label.get("platform_source", "twitch_clip"))
        row.setdefault("label_source", label.get("label_source", "twitch_clip_api"))
        row.setdefault("window_start", float(label.get("window_start", 0.0)))
        row.setdefault("window_end", float(label.get("window_end", 0.0)))
        row.setdefault("vod_id", str(label.get("vod_id", "")))
        row.setdefault("vod_duration_seconds", float(label.get("vod_duration_seconds", 0.0)))
        row.setdefault("clip_ids", list(label.get("clip_ids") or []))
        row.setdefault("sample_weight", sample_weight_for_label(str(row.get("label_class", ""))))
        row.setdefault("dataset_source", "twitch_vod_primary")
        row.setdefault("supervision_tier", "twitch_clip_proxy")
        row.setdefault("supervision_reason", str(row.get("label_source", "twitch_clip_api")))
        row.setdefault(
            "ordinal_label",
            collect_viral_clips._ordinal_label(str(row.get("label_class", "")), int(row.get("label", 0))),
        )
        _apply_vod_context_features(row, label, context_index)
        row["feature_provenance"] = collect_viral_clips._feature_provenance(
            row["features"],
            has_vod_context=True,
        )
        rows.append(row)
    return rows


def _schedule_prefetch(
    executor: concurrent.futures.ThreadPoolExecutor,
    vod_items: list[tuple[str, list[dict]]],
    next_index: int,
    *,
    profile: str,
    work_dir: pathlib.Path,
) -> tuple[concurrent.futures.Future[pathlib.Path] | None, str | None, str | None]:
    if not settings.training_vod_prefetch:
        logger.info("training_vod_prefetch_skip disabled")
        return None, None, None
    if next_index >= len(vod_items):
        logger.info("training_vod_prefetch_skip end_of_queue")
        return None, None, None

    next_vod_id, next_rows = vod_items[next_index]
    if _load_vod_checkpoint(work_dir, profile, next_vod_id, next_rows) is not None:
        logger.info("training_vod_prefetch_skip checkpoint_hit %s", next_rows[0]["vod_url"])
        return None, None, None
    source_kind = _window_scoped_context_kind_for_labels(next_rows)
    if source_kind is not None:
        logger.info("training_vod_prefetch_skip %s %s", source_kind, next_rows[0]["vod_url"])
        return None, None, None

    next_url = str(next_rows[0]["vod_url"])
    prefetch_job_id = f"twitch_vod_prefetch_{next_vod_id}"
    logger.info("training_vod_prefetch_start %s", next_url)
    return (
        executor.submit(_materialize_vod, next_url, prefetch_job_id),
        next_vod_id,
        prefetch_job_id,
    )


def _build_rows_for_label(
    label: dict,
    *,
    vod_path: pathlib.Path,
    clip_dir: pathlib.Path,
    vod_feature_context: VodFeatureContext,
    job_prefix: str,
    profile: str,
) -> list[dict]:
    from services import (
        audio_features, feature_builder, scorer, video_features,
    )
    from services.timeline_types import Candidate

    expanded_start, expanded_end, local_expanded_start, local_expanded_end = _local_context_training_window(
        label,
        vod_feature_context,
    )
    local_game_events = _local_game_events_for_source(
        str(label.get("vod_id") or ""),
        source_start_sec=expanded_start,
        source_end_sec=expanded_end,
    )
    transcript = _slice_transcript_for_window(
        vod_feature_context.transcript,
        window_start=local_expanded_start,
        window_end=local_expanded_end,
    )
    audio_frames, visual_frames = _load_or_build_window_modalities(
        str(label.get("vod_id") or ""),
        vod_path=vod_path,
        vod_context=vod_feature_context,
        window_start=local_expanded_start,
        window_end=local_expanded_end,
    )
    if not transcript:
        return []

    transcript = audio_features.enrich_segments(transcript, audio_frames)
    transcript = video_features.enrich_segments(transcript, visual_frames)
    transcript = scorer.score_segments(transcript)
    if transcript[-1].end - transcript[0].start < 5.0:
        return []

    proxy_candidate = _proxy_candidate(label, expanded_start, job_prefix)
    candidates = _refined_candidates_for_label(label, transcript, proxy_candidate, job_prefix)
    if not candidates:
        return []

    boundary_candidates = _boundary_perturbation_candidates(
        label,
        transcript,
        candidates[0],
        job_prefix,
    )
    all_candidates = list(candidates) + boundary_candidates

    feature_rows = feature_builder.build_features(
        all_candidates,
        transcript,
        profile,
        llm_candidate_ids={candidates[0].id},
        llm_provider=_DATASET_BUILD_LLM_PROVIDER,
        game_events=local_game_events,
        video_path=vod_path,
    )
    if not feature_rows:
        return []

    rows: list[dict] = []
    primary_feature_rows = feature_rows[: len(candidates)]
    boundary_feature_rows = feature_rows[len(candidates):]
    primary_dataset_source = str(label.get("training_dataset_source") or "twitch_vod_primary")
    primary_supervision_tier = str(label.get("training_supervision_tier") or "twitch_clip_proxy")
    primary_supervision_reason = str(label.get("training_supervision_reason") or "")
    for idx, (candidate, features) in enumerate(zip(candidates, primary_feature_rows)):
        absolute_candidate = _absolute_candidate(candidate, expanded_start)
        rows.append(
            _training_row_from_label(
                label,
                absolute_candidate,
                features,
                sample_weight_scale=1.0 if idx == 0 else _SUPPLEMENTAL_CANDIDATE_WEIGHT,
                dataset_source=primary_dataset_source if idx == 0 else "twitch_vod_candidate_supplemental",
                supervision_reason_suffix=primary_supervision_reason if idx == 0 else ":candidate_generator",
                window_origin="proxy_label_refined" if idx == 0 else "candidate_generator_overlap",
                supervision_tier=primary_supervision_tier if idx == 0 else "twitch_clip_proxy",
                reference_window=(absolute_candidate.start, absolute_candidate.end) if idx == 0 else None,
            )
        )

    if boundary_candidates and boundary_feature_rows:
        reference_candidate = _absolute_candidate(candidates[0], expanded_start)
        for candidate, features in zip(boundary_candidates, boundary_feature_rows):
            absolute_candidate = _absolute_candidate(candidate, expanded_start)
            perturbation_kind = next(
                (tag.split(":", 1)[1] for tag in candidate.tags if tag.startswith("boundary_perturbation:")),
                "boundary_near_miss",
            )
            rows.append(
                _training_row_from_label(
                    label,
                    absolute_candidate,
                    features,
                    sample_weight_scale=_BOUNDARY_NEAR_MISS_SAMPLE_WEIGHT,
                    dataset_source="twitch_vod_boundary_near_miss",
                    supervision_reason_suffix=f":boundary_near_miss:{perturbation_kind}",
                    window_origin="boundary_perturbed",
                    label_class_override="boundary_near_miss_negative",
                    label_value_override=0,
                    supervision_tier="boundary_near_miss",
                    rejected_reason_override="boundary_near_miss",
                    reference_window=(reference_candidate.start, reference_candidate.end),
                    perturbation_kind=perturbation_kind,
                )
            )
    return rows


def _extract_clip_modalities(
    source_video_path: pathlib.Path,
    audio_path: pathlib.Path,
    *,
    window_start: float,
    window_end: float,
) -> tuple[list, list, list]:
    from services import audio_features, transcriber, video_features

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        transcript_future = executor.submit(transcriber.transcribe, audio_path)
        audio_future = executor.submit(audio_features.extract, audio_path)
        visual_future = executor.submit(
            video_features.extract,
            source_video_path,
            start_sec=window_start,
            end_sec=window_end,
        )

        transcript = transcript_future.result()
        audio_frames = audio_future.result()
        visual_frames = visual_future.result()

    return transcript, audio_frames, visual_frames


def _resolved_training_vod_workers() -> int:
    try:
        return max(int(settings.training_vod_workers), 1)
    except (TypeError, ValueError):
        return 1


def _resolved_training_label_workers() -> int:
    try:
        return max(int(settings.training_label_workers), 1)
    except (TypeError, ValueError):
        return 1


def _materialize_vod(vod_url: str, job_id: str) -> pathlib.Path:
    logger.info("training_vod_direct_download %s", vod_url)
    return downloader.download(vod_url, job_id)


def _semantic_provider_summary(preferred: str) -> str:
    provider = (preferred or "auto").strip().lower() or "auto"
    has_groq = bool(settings.groq_api_key)
    if provider in {"none", "off", "disabled"}:
        return "none"
    if provider == "groq":
        if has_groq:
            return f"groq:{settings.groq_model}"
        return f"groq_unavailable->ollama:{settings.ollama_model}"
    if provider == "ollama":
        return f"ollama:{settings.ollama_model}"
    if has_groq:
        return f"auto->ollama:{settings.ollama_model}|fallback=groq:{settings.groq_model}"
    return f"auto->ollama:{settings.ollama_model}"


def _checkpoint_dir(work_dir: pathlib.Path) -> pathlib.Path:
    return work_dir / "_vod_checkpoints"


def _vod_checkpoint_path(work_dir: pathlib.Path, vod_id: str) -> pathlib.Path:
    return _checkpoint_dir(work_dir) / f"{vod_id}.json"


def _extended_window_group_checkpoint_path(
    work_dir: pathlib.Path,
    vod_id: str,
    group_index: int,
) -> pathlib.Path:
    return _checkpoint_dir(work_dir) / "extended_window_groups" / str(vod_id) / f"group_{group_index:04d}.json"


def _vod_checkpoint_signature(vod_rows: list[dict]) -> str:
    normalized = json.dumps(
        _checkpoint_label_signature_payload(vod_rows),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _extended_window_group_checkpoint_signature(
    group_labels: list[dict],
    *,
    scoped_window: tuple[float, float],
    cache_scope: dict[str, object],
    source_signature: dict[str, object],
) -> str:
    payload = {
        "kind": _WINDOW_SCOPED_EXTENDED_KIND,
        "labels": _checkpoint_label_signature_payload(group_labels),
        "scoped_window": [round(float(scoped_window[0]), 3), round(float(scoped_window[1]), 3)],
        "cache_scope": cache_scope,
        "source_signature": source_signature,
    }
    normalized = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _legacy_vod_checkpoint_label_signatures(vod_rows: list[dict]) -> set[str]:
    # Older checkpoints hashed the full label dict. Keep accepting those
    # checkpoints after non-feature review metadata is added to mixed VODs.
    normalized = json.dumps(
        sorted(
            (
                _checkpoint_label_legacy_payload(
                    row,
                    include_top_viewed_source=_should_use_window_scoped_label_contexts(vod_rows),
                )
                for row in vod_rows
            ),
            key=lambda row: (
                float(row.get("window_start") or 0.0),
                float(row.get("window_end") or 0.0),
                str(row.get("label_class") or ""),
            ),
        ),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {hashlib.sha256(normalized.encode("utf-8")).hexdigest()}


def _checkpoint_label_signature_payload(vod_rows: list[dict]) -> list[dict[str, object]]:
    include_top_viewed_source = _should_use_window_scoped_label_contexts(vod_rows)
    return sorted(
        (
            _checkpoint_label_payload(row, include_top_viewed_source=include_top_viewed_source)
            for row in vod_rows
        ),
        key=lambda row: (
            float(row.get("window_start") or 0.0),
            float(row.get("window_end") or 0.0),
            str(row.get("label_class") or ""),
            str(row.get("clip_ids") or ""),
        ),
    )


def _checkpoint_label_payload(row: dict, *, include_top_viewed_source: bool) -> dict[str, object]:
    payload: dict[str, object] = {
        "broadcaster": str(row.get("broadcaster") or ""),
        "vod_id": str(row.get("vod_id") or ""),
        "vod_url": str(row.get("vod_url") or ""),
        "window_start": round(float(row.get("window_start") or 0.0), 3),
        "window_end": round(float(row.get("window_end") or 0.0), 3),
        "label_class": str(row.get("label_class") or ""),
        "label_score": round(float(row.get("label_score") or 0.0), 6),
        "label_source": str(row.get("label_source") or ""),
        "platform_source": str(row.get("platform_source") or ""),
        "negative_source": str(row.get("negative_source") or ""),
        "quality_bucket": str(row.get("quality_bucket") or ""),
        "view_count": int(row.get("view_count") or 0),
        "clip_ids": sorted(str(item) for item in (row.get("clip_ids") or []) if str(item).strip()),
    }
    if include_top_viewed_source:
        payload["manual_gold_review_source"] = str(row.get("manual_gold_review_source") or "")
    return payload


def _checkpoint_label_legacy_payload(row: dict, *, include_top_viewed_source: bool) -> dict:
    ignored_metadata_fields = {
        "gold_candidate_status",
        "intended_gold_grade",
        "manual_gold_review",
        "manual_gold_review_notes",
    }
    if not include_top_viewed_source:
        ignored_metadata_fields.add("manual_gold_review_source")
    return {
        key: value
        for key, value in row.items()
        if key not in ignored_metadata_fields
    }


def _can_write_vod_checkpoint(vod_id: str, rows: list[dict] | None = None) -> bool:
    if not settings.enable_chat_analysis:
        return True
    try:
        cache_dir = _vod_feature_cache_dir(vod_id)
    except OSError as exc:
        logger.info(
            "training_vod_checkpoint_allow vod_id=%s reason=cache_dir_unavailable error=%s",
            vod_id,
            exc,
        )
        return True
    if _chat_cache_has_complete_coverage(cache_dir):
        return True
    if _rows_have_chat_features(rows or []):
        logger.info(
            "training_vod_checkpoint_allow vod_id=%s reason=scoped_chat_features_present",
            vod_id,
        )
        return True
    logger.info(
        "training_vod_checkpoint_skip vod_id=%s reason=chat_enabled_without_complete_chat_cache",
        vod_id,
    )
    return False


def _rows_have_chat_features(rows: list[dict]) -> bool:
    for row in rows:
        provenance = row.get("feature_provenance")
        if isinstance(provenance, dict) and bool(provenance.get("has_chat_features")):
            return True
        features = row.get("features")
        if isinstance(features, dict):
            if bool(features.get("has_chat_features")):
                return True
            for key in (
                "chat_message_rate_mean",
                "chat_message_rate_peak",
                "chat_hype_score",
                "chat_laugh_score",
                "chat_shock_score",
                "chat_sadge_score",
                "chat_velocity_score",
                "chat_rate_derivative_score",
                "chat_ema_crossing_score",
            ):
                try:
                    if abs(float(features.get(key) or 0.0)) > 1e-9:
                        return True
                except (TypeError, ValueError):
                    continue
    return False


def _load_vod_checkpoint(
    work_dir: pathlib.Path,
    profile: str,
    vod_id: str,
    vod_rows: list[dict],
) -> list[dict] | None:
    path = _vod_checkpoint_path(work_dir, vod_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if int(payload.get("version") or 0) != _VOD_CHECKPOINT_VERSION:
        return None
    if str(payload.get("profile") or "") != profile:
        return None
    if str(payload.get("vod_id") or "") != vod_id:
        return None
    raw_fingerprint = str(payload.get("raw_vod_rows_fingerprint") or "")
    if raw_fingerprint:
        if raw_fingerprint != _raw_vod_rows_fingerprint(profile):
            return None
    else:
        legacy_fingerprint = str(payload.get("dataset_build_fingerprint") or "")
        if legacy_fingerprint not in _legacy_vod_checkpoint_fingerprints(profile):
            return None
    label_signature = str(payload.get("label_signature") or "")
    if (
        label_signature != _vod_checkpoint_signature(vod_rows)
        and label_signature not in _legacy_vod_checkpoint_label_signatures(vod_rows)
    ):
        return None
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return None
    return rows


def _load_extended_window_group_checkpoint(
    work_dir: pathlib.Path,
    profile: str,
    vod_id: str,
    group_index: int,
    group_labels: list[dict],
    *,
    scoped_window: tuple[float, float],
    cache_scope: dict[str, object],
    source_signature: dict[str, object],
) -> list[dict] | None:
    path = _extended_window_group_checkpoint_path(work_dir, vod_id, group_index)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if int(payload.get("version") or 0) != _VOD_CHECKPOINT_VERSION:
        return None
    if str(payload.get("profile") or "") != profile:
        return None
    if str(payload.get("vod_id") or "") != vod_id:
        return None
    if int(payload.get("group_index") or 0) != int(group_index):
        return None
    if str(payload.get("checkpoint_kind") or "") != "extended_window_group_rows":
        return None
    if str(payload.get("raw_vod_rows_fingerprint") or "") != _raw_vod_rows_fingerprint(profile):
        return None
    expected_signature = _extended_window_group_checkpoint_signature(
        group_labels,
        scoped_window=scoped_window,
        cache_scope=cache_scope,
        source_signature=source_signature,
    )
    if str(payload.get("group_signature") or "") != expected_signature:
        return None
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return None
    return rows


def _write_vod_checkpoint(
    work_dir: pathlib.Path,
    profile: str,
    vod_id: str,
    vod_rows: list[dict],
    rows: list[dict],
) -> None:
    path = _vod_checkpoint_path(work_dir, vod_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": _VOD_CHECKPOINT_VERSION,
        "profile": profile,
        "vod_id": vod_id,
        "checkpoint_kind": "raw_vod_rows",
        "raw_vod_rows_fingerprint": _raw_vod_rows_fingerprint(profile),
        "dataset_build_fingerprint": _dataset_build_fingerprint(profile),
        "label_signature": _vod_checkpoint_signature(vod_rows),
        "rows": rows,
    }
    serialized = json.dumps(payload, ensure_ascii=True)
    path.write_text(serialized, encoding="utf-8")


def _write_extended_window_group_checkpoint(
    work_dir: pathlib.Path,
    profile: str,
    vod_id: str,
    group_index: int,
    group_labels: list[dict],
    rows: list[dict],
    *,
    scoped_window: tuple[float, float],
    cache_scope: dict[str, object],
    source_signature: dict[str, object],
) -> None:
    path = _extended_window_group_checkpoint_path(work_dir, vod_id, group_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": _VOD_CHECKPOINT_VERSION,
        "profile": profile,
        "vod_id": vod_id,
        "group_index": int(group_index),
        "checkpoint_kind": "extended_window_group_rows",
        "raw_vod_rows_fingerprint": _raw_vod_rows_fingerprint(profile),
        "group_signature": _extended_window_group_checkpoint_signature(
            group_labels,
            scoped_window=scoped_window,
            cache_scope=cache_scope,
            source_signature=source_signature,
        ),
        "scoped_window": [round(float(scoped_window[0]), 3), round(float(scoped_window[1]), 3)],
        "label_count": len(group_labels),
        "rows": rows,
    }
    serialized = json.dumps(payload, ensure_ascii=True)
    path.write_text(serialized, encoding="utf-8")


def sample_weight_for_label(label_class: str) -> float:
    return {
        "viral_positive": 2.5,
        "positive": 1.5,
        "weak_positive": 0.8,
        "hard_negative": 1.8,
        "background_negative": 1.0,
    }.get(label_class, 1.0)


def dataset_summary(rows: list[dict]) -> dict:
    by_class: dict[str, int] = {}
    vods: set[str] = set()
    positive_rows = 0
    negative_rows = 0
    for row in rows:
        by_class[row.get("label_class", "")] = by_class.get(row.get("label_class", ""), 0) + 1
        if row.get("vod_id"):
            vods.add(str(row["vod_id"]))
        if int(row.get("label", 0)) == 1:
            positive_rows += 1
        elif int(row.get("label", 0)) == 0:
            negative_rows += 1
    return {
        "rows": len(rows),
        "positive_rows": positive_rows,
        "negative_rows": negative_rows,
        "label_classes": by_class,
        "vod_count": len(vods),
        "candidate_source_mix": _candidate_source_mix_summary(rows),
        "duplicate_windows": _duplicate_window_summary(rows),
        "platform_source_drift": _platform_source_drift_summary(rows),
        "split_leak_check": _split_leak_summary(rows),
    }


def _candidate_source_mix_summary(rows: list[dict]) -> dict[str, object]:
    counts: dict[str, int] = {}
    for row in rows:
        source = str(row.get("candidate_source") or "").strip().lower() or "unknown"
        counts[source] = counts.get(source, 0) + 1
    total = sum(counts.values()) or 1
    shares = {
        source: round(count / total, 3)
        for source, count in sorted(counts.items())
    }
    peak_source = max(shares, key=shares.get) if shares else ""
    peak_share = float(shares.get(peak_source, 0.0)) if peak_source else 0.0
    return {
        "max_share": _MAX_CANDIDATE_SOURCE_SHARE,
        "peak_source": peak_source,
        "peak_share": round(peak_share, 3),
        "by_source": dict(sorted(counts.items())),
        "share_by_source": shares,
    }


def _duplicate_window_summary(rows: list[dict]) -> dict[str, object]:
    counts: dict[tuple[str, float, float], int] = {}
    for row in rows:
        vod_id = str(row.get("vod_id") or "").strip()
        if not vod_id:
            continue
        key = (
            vod_id,
            round(float(row.get("window_start") or 0.0), 2),
            round(float(row.get("window_end") or 0.0), 2),
        )
        counts[key] = counts.get(key, 0) + 1
    duplicates = [
        {
            "vod_id": vod_id,
            "window_start": window_start,
            "window_end": window_end,
            "count": count,
        }
        for (vod_id, window_start, window_end), count in counts.items()
        if count > 1
    ]
    duplicates.sort(key=lambda item: (-int(item["count"]), str(item["vod_id"]), float(item["window_start"])))
    return {
        "count": len(duplicates),
        "sample": duplicates[:10],
    }


def _platform_source_drift_summary(rows: list[dict]) -> dict[str, dict[str, object]]:
    by_source: dict[str, dict[str, object]] = {}
    for row in rows:
        source = str(row.get("platform_source") or "unknown").strip() or "unknown"
        bucket = by_source.setdefault(
            source,
            {
                "rows": 0,
                "positive_rows": 0,
                "negative_rows": 0,
                "label_classes": {},
            },
        )
        bucket["rows"] = int(bucket["rows"]) + 1
        if int(row.get("label", 0)) == 1:
            bucket["positive_rows"] = int(bucket["positive_rows"]) + 1
        elif int(row.get("label", 0)) == 0:
            bucket["negative_rows"] = int(bucket["negative_rows"]) + 1
        label_class = str(row.get("label_class") or "")
        classes = dict(bucket["label_classes"])
        classes[label_class] = classes.get(label_class, 0) + 1
        bucket["label_classes"] = classes

    for bucket in by_source.values():
        rows_count = int(bucket["rows"]) or 1
        bucket["positive_rate"] = round(int(bucket["positive_rows"]) / rows_count, 3)
        bucket["label_classes"] = dict(sorted(dict(bucket["label_classes"]).items()))
    return dict(sorted(by_source.items()))


def _split_leak_summary(rows: list[dict]) -> dict[str, object]:
    if len(rows) < 2:
        return {
            "strategy": "insufficient_rows",
            "unit": "",
            "train_rows": len(rows),
            "validation_rows": 0,
            "vod_overlap_count": 0,
            "vod_overlap_sample": [],
        }

    from ml import train_ranker

    split = train_ranker._split_rows(rows)
    train_rows = list(split.get("train") or [])
    validation_rows = list(split.get("validation") or [])
    train_vods = _row_identity_keys(train_rows)
    validation_vods = _row_identity_keys(validation_rows)
    overlap = sorted(train_vods & validation_vods)
    return {
        "strategy": str(split.get("strategy") or ""),
        "unit": str(split.get("unit") or ""),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "vod_overlap_count": len(overlap),
        "vod_overlap_sample": overlap[:10],
    }


def _row_identity_keys(rows: list[dict]) -> set[str]:
    identities: set[str] = set()
    for row in rows:
        for key in ("vod_id", "source_url", "job_id"):
            value = str(row.get(key) or "").strip()
            if value:
                identities.add(value)
                break
    return identities


def _entry_from_label(label: dict) -> dict[str, str]:
    label_class = str(label.get("label_class", ""))
    is_positive = label_class in _POSITIVE_LABEL_CLASSES
    return {
        "url": str(label["vod_url"]),
        "streamer": str(label.get("broadcaster", "")),
        "platform": "twitch",
        "language": "",
        "tags": "twitch,vod",
        "label": "1" if is_positive else "0",
        "notes": f"twitch_label={label_class}",
        "quality_bucket": str(label.get("quality_bucket", "")),
        "label_reason": str(label.get("label_source", "twitch_clip_api")),
        "viral_score": str(label.get("label_score", 0.0)),
        "view_count": str(label.get("view_count", 0)),
        "like_count": "0",
        "comment_count": "0",
        "duration": str(max(float(label["window_end"]) - float(label["window_start"]), 0.0)),
        "publish_timestamp": "0",
        "upload_date": "",
        "platform_source": str(label.get("platform_source", "twitch_clip")),
        "label_source": str(label.get("label_source", "twitch_clip_api")),
        "label_class": label_class,
        "label_score": str(label.get("label_score", 0.0)),
        "window_start": str(label["window_start"]),
        "window_end": str(label["window_end"]),
        "vod_id": str(label["vod_id"]),
        "clip_ids": ",".join(label.get("clip_ids") or []),
    }


def _build_vod_context_index(vod_rows: list[dict]) -> dict:
    ordered = sorted(vod_rows, key=lambda item: float(item["window_start"]))
    positives = [
        row for row in ordered
        if str(row.get("label_class", "")) in {"viral_positive", "positive", "weak_positive"}
    ]
    duration = max((float(row.get("window_end") or 0.0) for row in ordered), default=0.0)
    for row in ordered:
        duration = max(duration, float(row.get("vod_duration_seconds") or 0.0))
    return {"ordered": ordered, "positives": positives, "duration": duration}


def _apply_vod_context_features(row: dict, label: dict, context_index: dict) -> None:
    features = row.get("features", {})
    start = float(row.get("window_start") or label.get("window_start") or 0.0)
    end = float(row.get("window_end") or label.get("window_end") or 0.0)
    center = (start + end) / 2.0
    duration = float(context_index.get("duration") or end or 1.0)
    ordered = context_index.get("ordered") or []
    positives = context_index.get("positives") or []

    prev_row = None
    next_row = None
    for candidate in ordered:
        candidate_start = float(candidate.get("window_start") or 0.0)
        candidate_end = float(candidate.get("window_end") or 0.0)
        if candidate_end <= start:
            prev_row = candidate
        elif candidate_start >= end and next_row is None:
            next_row = candidate
            break

    nearby_positive_count = 0
    for candidate in positives:
        candidate_center = (float(candidate.get("window_start") or 0.0) + float(candidate.get("window_end") or 0.0)) / 2.0
        if abs(candidate_center - center) <= 300.0:
            nearby_positive_count += 1

    left_context = start - float(prev_row.get("window_end") or 0.0) if prev_row else start
    right_context = float(next_row.get("window_start") or duration) - end if next_row else max(duration - end, 0.0)

    features["vod_progress"] = round(center / max(duration, 1.0), 3)
    features["boundary_left_context_sec"] = round(max(left_context, 0.0), 3)
    features["boundary_right_context_sec"] = round(max(right_context, 0.0), 3)
    features["neighbor_positive_density"] = round(nearby_positive_count / 10.0, 3)
    features["clip_overlap_count"] = float(len(label.get("clip_ids") or []))


def _expanded_training_window(label: dict) -> tuple[float, float]:
    start = float(label.get("window_start") or 0.0)
    end = float(label.get("window_end") or start)
    duration = max(end - start, 1.0)
    if str(label.get("label_class") or "") in _POSITIVE_LABEL_CLASSES:
        lead = min(max(duration * 0.4, 6.0), 18.0)
        tail = min(max(duration * 0.45, 8.0), 22.0)
        return round(max(0.0, start - lead), 2), round(end + tail, 2)
    return round(start, 2), round(end, 2)


def _local_context_training_window(
    label: dict,
    vod_feature_context: VodFeatureContext,
) -> tuple[float, float, float, float]:
    expanded_start, expanded_end = _expanded_training_window(label)
    context_offset = float(getattr(vod_feature_context, "source_start_sec", 0.0) or 0.0)
    source_kind = str(getattr(vod_feature_context, "source_kind", "") or "")
    if source_kind == _WINDOW_SCOPED_EXTENDED_KIND:
        source_end = float(getattr(vod_feature_context, "source_end_sec", 0.0) or 0.0)
        if source_end > context_offset:
            return (
                round(context_offset, 2),
                round(source_end, 2),
                0.0,
                round(max(source_end - context_offset, 1.0), 2),
            )
    local_expanded_start = max(float(expanded_start) - context_offset, 0.0)
    local_expanded_end = max(float(expanded_end) - context_offset, local_expanded_start + 1.0)
    return expanded_start, expanded_end, local_expanded_start, local_expanded_end


def _proxy_candidate(label: dict, expanded_start: float, job_prefix: str):
    from services.timeline_types import Candidate

    local_start = max(float(label.get("window_start") or 0.0) - expanded_start, 0.0)
    local_end = max(float(label.get("window_end") or 0.0) - expanded_start, local_start + 1.0)
    return Candidate(
        id=f"{job_prefix}:proxy",
        start=round(local_start, 2),
        end=round(local_end, 2),
        source="twitch_label_proxy",
        anchor_time=round((local_start + local_end) / 2.0, 2),
        score_hint=0.0,
        tags=[],
    )


def _refined_candidates_for_label(label: dict, transcript: list, proxy_candidate, job_prefix: str):
    from services import candidate_generators

    label_class = str(label.get("label_class") or "")
    if label_class not in _POSITIVE_LABEL_CLASSES:
        return [proxy_candidate]

    primary = candidate_generators.generate_semantic_candidates(transcript)
    supplemental = candidate_generators.generate_fallback_candidates(transcript)
    generated = candidate_generators.merge_prioritized_candidates(primary, supplemental, supplemental_source_cap=3)
    overlapping = [candidate for candidate in generated if _matches_proxy_window(candidate, proxy_candidate)]
    if not overlapping:
        return [proxy_candidate]

    ranked = sorted(overlapping, key=lambda candidate: _candidate_alignment_score(candidate, proxy_candidate), reverse=True)
    selected = [_clone_candidate(ranked[0], job_prefix, 0)]
    for rank, candidate in enumerate(ranked[1:], start=1):
        if len(selected) > _SUPPLEMENTAL_CANDIDATE_LIMIT:
            break
        if _is_same_window(candidate, ranked[0]):
            continue
        selected.append(_clone_candidate(candidate, job_prefix, rank))
    return selected


def _candidate_alignment_score(candidate, proxy_candidate) -> tuple[float, float, float, float]:
    overlap_ratio = _overlap_ratio(candidate.start, candidate.end, proxy_candidate.start, proxy_candidate.end)
    duration = max(candidate.end - candidate.start, 1.0)
    proxy_duration = max(proxy_candidate.end - proxy_candidate.start, 1.0)
    duration_similarity = max(0.0, 1.0 - abs(duration - proxy_duration) / max(proxy_duration, 1.0))
    anchor_inside = 1.0 if proxy_candidate.start <= candidate.anchor_time <= proxy_candidate.end else 0.0
    return (
        round(overlap_ratio, 4),
        round(anchor_inside, 4),
        round(duration_similarity, 4),
        round(float(candidate.score_hint), 4),
    )


def _matches_proxy_window(candidate, proxy_candidate) -> bool:
    if proxy_candidate.start <= candidate.anchor_time <= proxy_candidate.end:
        return True
    return _overlap_ratio(candidate.start, candidate.end, proxy_candidate.start, proxy_candidate.end) >= 0.35


def _overlap_ratio(left_start: float, left_end: float, right_start: float, right_end: float) -> float:
    overlap = min(left_end, right_end) - max(left_start, right_start)
    if overlap <= 0:
        return 0.0
    shorter = min(left_end - left_start, right_end - right_start)
    if shorter <= 0:
        return 0.0
    return round(overlap / shorter, 4)


def _is_same_window(left, right) -> bool:
    return abs(left.start - right.start) <= 1.0 and abs(left.end - right.end) <= 1.0


def _clone_candidate(candidate, job_prefix: str, rank: int):
    from services.timeline_types import Candidate

    return Candidate(
        id=f"{job_prefix}:{candidate.source}:{rank}:{int(candidate.start * 100)}",
        start=round(candidate.start, 2),
        end=round(candidate.end, 2),
        source=candidate.source,
        anchor_time=round(candidate.anchor_time, 2),
        score_hint=float(candidate.score_hint),
        tags=list(candidate.tags),
    )


def _absolute_candidate(candidate, expanded_start: float):
    from services.timeline_types import Candidate

    return Candidate(
        id=candidate.id,
        start=round(expanded_start + candidate.start, 2),
        end=round(expanded_start + candidate.end, 2),
        source=candidate.source,
        anchor_time=round(expanded_start + candidate.anchor_time, 2),
        score_hint=candidate.score_hint,
        tags=list(candidate.tags),
    )


def _boundary_perturbation_candidates(
    label: dict,
    transcript: list,
    reference_candidate,
    job_prefix: str,
):
    from services import boundary_candidates
    from services.timeline_types import Candidate

    label_class = str(label.get("label_class") or "")
    if label_class not in _POSITIVE_LABEL_CLASSES or not transcript:
        return []

    points = boundary_candidates.build_boundary_points(
        transcript,
        reference_candidate,
        context_lookback_sec=8.0,
        context_lookahead_sec=8.0,
    )
    times = sorted({round(point.time, 2) for point in points})
    if len(times) < 2:
        return []

    generated: list[Candidate] = []
    seen: set[tuple[float, float]] = set()
    for rank, (kind, start_offset, end_offset) in enumerate(_BOUNDARY_PERTURBATION_SPECS):
        start = _snap_perturbed_time(times, reference_candidate.start, start_offset, lower_bound=0.0)
        end = _snap_perturbed_time(times, reference_candidate.end, end_offset, lower_bound=reference_candidate.anchor_time)
        if start is None or end is None:
            continue
        if end - start < 6.0:
            continue
        if not (start < reference_candidate.anchor_time < end):
            continue
        key = (round(start, 2), round(end, 2))
        if key in seen:
            continue
        total_error = abs(start - reference_candidate.start) + abs(end - reference_candidate.end)
        if total_error < 0.75:
            continue
        seen.add(key)
        generated.append(
            Candidate(
                id=f"{job_prefix}:boundary:{rank}:{int(start * 100)}",
                start=round(start, 2),
                end=round(end, 2),
                source="boundary_perturbation",
                anchor_time=round(reference_candidate.anchor_time, 2),
                score_hint=max(float(reference_candidate.score_hint) * 0.35, 0.1),
                tags=["boundary_near_miss", f"boundary_perturbation:{kind}"],
            )
        )
        if len(generated) >= _BOUNDARY_NEAR_MISS_LIMIT:
            break
    return generated


def _snap_perturbed_time(
    boundaries: list[float],
    reference_time: float,
    offset: float,
    *,
    lower_bound: float,
) -> float | None:
    target = reference_time + offset
    if offset > 0:
        candidates = [boundary for boundary in boundaries if boundary >= max(target, lower_bound)]
        return candidates[0] if candidates else None
    if offset < 0:
        candidates = [boundary for boundary in boundaries if lower_bound <= boundary <= target]
        return candidates[-1] if candidates else None
    candidates = [boundary for boundary in boundaries if boundary >= lower_bound]
    if not candidates:
        return None
    return min(candidates, key=lambda value: abs(value - reference_time))


def _training_row_from_label(
    label: dict,
    candidate,
    features,
    *,
    sample_weight_scale: float,
    dataset_source: str,
    supervision_reason_suffix: str,
    window_origin: str,
    label_class_override: str | None = None,
    label_value_override: int | None = None,
    supervision_tier: str = "twitch_clip_proxy",
    rejected_reason_override: str | None = None,
    reference_window: tuple[float, float] | None = None,
    perturbation_kind: str | None = None,
) -> dict:
    label_class = str(label_class_override or label.get("label_class") or "")
    if label_value_override is None:
        is_positive = label_class in _POSITIVE_LABEL_CLASSES
        label_value = 1 if is_positive else 0
    else:
        label_value = int(label_value_override)
        is_positive = label_value == 1
    base_reason = str(label.get("label_source") or "twitch_clip_api")
    row = {
        "job_id": f"twitch_vod_{label['vod_id']}",
        "candidate_id": features.candidate_id,
        "label": label_value,
        "selected": is_positive,
        "exported": True,
        "final_start": candidate.start if is_positive else None,
        "final_end": candidate.end if is_positive else None,
        "rejected_reason": None if is_positive else str(rejected_reason_override or label.get("negative_source") or "twitch_vod_negative"),
        "rejection_source": None if is_positive else str(rejected_reason_override or label.get("negative_source") or "twitch_vod_negative"),
        "creator_id": str(label.get("broadcaster") or ""),
        "streamer": str(label.get("broadcaster") or ""),
        "platform": "twitch",
        "language": "",
        "tags": ["twitch", "vod"],
        "notes": f"twitch_label={label_class}",
        "quality_bucket": str(label.get("quality_bucket", "")),
        "label_reason": base_reason,
        "viral_score": float(label.get("label_score", 0.0) or 0.0),
        "view_count": int(label.get("view_count", 0) or 0),
        "like_count": 0,
        "comment_count": 0,
        "publish_timestamp": 0,
        "upload_date": "",
        "platform_source": str(label.get("platform_source", "twitch_clip")),
        "label_class": label_class,
        "label_score": float(label.get("label_score", 0.0) or 0.0),
        "sample_weight": round(
            sample_weight_for_label(label_class)
            * sample_weight_scale
            * _label_sample_weight_multiplier(label),
            3,
        ),
        "supervision_tier": supervision_tier,
        "supervision_reason": f"{base_reason}{supervision_reason_suffix}",
        "ordinal_label": collect_viral_clips._ordinal_label(label_class, label_value),
        "window_start": round(candidate.start, 2),
        "window_end": round(candidate.end, 2),
        "proxy_window_start": float(label.get("window_start") or 0.0),
        "proxy_window_end": float(label.get("window_end") or 0.0),
        "window_origin": window_origin,
        "candidate_source": candidate.source,
        "vod_id": str(label["vod_id"]),
        "clip_ids": list(label.get("clip_ids") or []),
        "source_url": str(label["vod_url"]),
        "dataset_source": dataset_source,
        "feature_provenance": {},
        "features": features.to_dict(),
    }
    if reference_window is not None:
        ref_start, ref_end = reference_window
        start_error = round(candidate.start - ref_start, 2)
        end_error = round(candidate.end - ref_end, 2)
        row["boundary_reference_start"] = round(ref_start, 2)
        row["boundary_reference_end"] = round(ref_end, 2)
        row["boundary_error_start_sec"] = start_error
        row["boundary_error_end_sec"] = end_error
        row["boundary_abs_start_error_sec"] = round(abs(start_error), 2)
        row["boundary_abs_end_error_sec"] = round(abs(end_error), 2)
        row["boundary_abs_total_error_sec"] = round(abs(start_error) + abs(end_error), 2)
    if perturbation_kind:
        row["boundary_perturbation_kind"] = perturbation_kind
    if bool(label.get("targeted_feature_backfill")):
        row["targeted_feature_backfill"] = True
        row["targeted_feature_backfill_source"] = str(
            label.get("targeted_feature_backfill_source") or "targeted_backfill"
        )
    return row


def _label_sample_weight_multiplier(label: dict) -> float:
    try:
        value = float(label.get("training_sample_weight_multiplier", 1.0) or 1.0)
    except (TypeError, ValueError):
        return 1.0
    return max(value, 0.0) if math.isfinite(value) else 1.0


def _dedupe_ranker_rows(rows: list[dict]) -> list[dict]:
    canonical_positive_rows: dict[tuple[str, tuple[str, ...]], dict] = {}
    deduped: list[dict] = []
    for row in rows:
        positive_key = _positive_cluster_key(row)
        if positive_key is None:
            deduped.append(row)
            continue
        existing = canonical_positive_rows.get(positive_key)
        if existing is None or _prefer_positive_row(row, existing):
            canonical_positive_rows[positive_key] = row

    if not canonical_positive_rows:
        return deduped

    seen_positive_keys: set[tuple[str, tuple[str, ...]]] = set()
    merged: list[dict] = []
    for row in rows:
        positive_key = _positive_cluster_key(row)
        if positive_key is None:
            merged.append(row)
            continue
        if positive_key in seen_positive_keys:
            continue
        seen_positive_keys.add(positive_key)
        merged.append(canonical_positive_rows[positive_key])
    return merged


def _prepare_ranker_rows(rows: list[dict]) -> list[dict]:
    deduped = _dedupe_ranker_rows(rows)
    balanced = _balance_vod_negative_rows(deduped)
    return _cap_candidate_source_mix(balanced)


def _balance_vod_negative_rows(
    rows: list[dict],
    *,
    max_negative_rows_per_positive: int = _MAX_VOD_NEGATIVE_ROWS_PER_POSITIVE,
) -> list[dict]:
    if not rows or max_negative_rows_per_positive <= 0:
        return list(rows)

    groups: dict[str, list[dict]] = {}
    passthrough: list[dict] = []
    for row in rows:
        key = _vod_balance_group_key(row)
        if key:
            groups.setdefault(key, []).append(row)
        else:
            passthrough.append(row)

    keep_ids = {id(row) for row in passthrough}
    capped_groups = 0
    dropped_rows = 0
    for group_rows in groups.values():
        positive_rows = [row for row in group_rows if int(row.get("label", 0)) == 1]
        negative_rows = [row for row in group_rows if int(row.get("label", 0)) == 0]
        neutral_rows = [row for row in group_rows if int(row.get("label", 0)) not in {0, 1}]
        limit = len(positive_rows) * int(max_negative_rows_per_positive)
        if not positive_rows or len(negative_rows) <= limit:
            keep_ids.update(id(row) for row in group_rows)
            continue

        selected_negative_rows = _select_vod_negative_rows(negative_rows, positive_rows, limit)
        keep_ids.update(id(row) for row in positive_rows)
        keep_ids.update(id(row) for row in neutral_rows)
        keep_ids.update(id(row) for row in selected_negative_rows)
        capped_groups += 1
        dropped_rows += len(negative_rows) - len(selected_negative_rows)

    if dropped_rows:
        logger.info(
            "training_vod_negative_balance_capped groups=%s rows_dropped=%s max_negative_rows_per_positive=%s",
            capped_groups,
            dropped_rows,
            max_negative_rows_per_positive,
        )
    return [row for row in rows if id(row) in keep_ids]


def _select_vod_negative_rows(
    negative_rows: list[dict],
    positive_rows: list[dict],
    limit: int,
) -> list[dict]:
    if limit >= len(negative_rows):
        return list(negative_rows)
    if limit <= 0:
        return []

    boundary_rows = [row for row in negative_rows if _is_boundary_near_miss_row(row)]
    primary_rows = [row for row in negative_rows if not _is_boundary_near_miss_row(row)]
    selected: list[dict] = []
    if boundary_rows and primary_rows and limit >= 2:
        boundary_quota = min(
            len(boundary_rows),
            max(1, int(math.floor(limit * _MAX_BOUNDARY_NEGATIVE_SHARE))),
        )
        primary_quota = max(limit - boundary_quota, 0)
        selected.extend(_rank_vod_negative_rows(primary_rows, positive_rows)[:primary_quota])
        selected.extend(_rank_vod_negative_rows(boundary_rows, positive_rows)[:boundary_quota])
    else:
        selected.extend(_rank_vod_negative_rows(negative_rows, positive_rows)[:limit])

    if len(selected) < limit:
        selected_ids = {id(row) for row in selected}
        remaining = [row for row in negative_rows if id(row) not in selected_ids]
        selected.extend(_rank_vod_negative_rows(remaining, positive_rows)[: limit - len(selected)])

    selected_ids = {id(row) for row in selected}
    return [row for row in negative_rows if id(row) in selected_ids]


def _rank_vod_negative_rows(negative_rows: list[dict], positive_rows: list[dict]) -> list[dict]:
    return sorted(
        negative_rows,
        key=lambda row: (
            -float(row.get("sample_weight") or 0.0),
            _distance_to_nearest_positive(row, positive_rows),
            float(row.get("window_start") or 0.0),
            str(row.get("candidate_id") or ""),
        ),
    )


def _distance_to_nearest_positive(row: dict, positive_rows: list[dict]) -> float:
    if not positive_rows:
        return 0.0
    center = _window_center(row)
    return min(abs(center - _window_center(positive_row)) for positive_row in positive_rows)


def _window_center(row: dict) -> float:
    start = float(row.get("window_start") or row.get("proxy_window_start") or 0.0)
    end = float(row.get("window_end") or row.get("proxy_window_end") or start)
    return (start + end) / 2.0


def _is_boundary_near_miss_row(row: dict) -> bool:
    return (
        str(row.get("dataset_source") or "") == "twitch_vod_boundary_near_miss"
        or str(row.get("supervision_tier") or "") == "boundary_near_miss"
        or str(row.get("label_class") or "") == "boundary_near_miss_negative"
    )


def _vod_balance_group_key(row: dict) -> str:
    for key in ("vod_id", "source_url", "job_id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _cap_candidate_source_mix(
    rows: list[dict],
    *,
    max_share: float = _MAX_CANDIDATE_SOURCE_SHARE,
) -> list[dict]:
    if not rows:
        return []

    buckets: dict[str, list[dict]] = {}
    for row in rows:
        source = str(row.get("candidate_source") or "").strip().lower() or "unknown"
        buckets.setdefault(source, []).append(row)

    if len(buckets) < 3:
        logger.warning(
            "candidate_source_mix_unenforceable distinct_sources=%s max_share=%.2f",
            len(buckets),
            max_share,
        )
        return rows

    total_rows = len(rows)
    current_peak = max(len(bucket) for bucket in buckets.values()) / max(total_rows, 1)
    if current_peak <= max_share + 1e-9:
        return rows

    protected_ids = _candidate_source_cap_protected_row_ids(rows)
    source_counts = {source: 0 for source in buckets}
    kept: list[dict] = []
    for row in rows:
        if id(row) not in protected_ids:
            continue
        source = str(row.get("candidate_source") or "").strip().lower() or "unknown"
        kept.append(row)
        source_counts[source] += 1

    for source, bucket in list(buckets.items()):
        buckets[source] = [row for row in bucket if id(row) not in protected_ids]

    ordered_sources = sorted(buckets, key=lambda source: (-len(buckets[source]), source))

    for source in ordered_sources:
        if not buckets[source]:
            continue
        next_source_count = source_counts[source] + 1
        next_total = len(kept) + 1
        if source_counts[source] > 0 and (next_source_count / next_total) > max_share + 1e-9:
            continue
        kept.append(buckets[source].pop(0))
        source_counts[source] += 1

    while True:
        progressed = False
        for source in ordered_sources:
            bucket = buckets[source]
            if not bucket:
                continue
            next_source_count = source_counts[source] + 1
            next_total = len(kept) + 1
            if (next_source_count / next_total) > max_share + 1e-9:
                continue
            kept.append(bucket.pop(0))
            source_counts[source] = next_source_count
            progressed = True
        if not progressed:
            break

    if len(kept) < len(rows):
        logger.info(
            "candidate_source_mix_capped rows_before=%s rows_after=%s max_share=%.2f peak_source=%s peak_share_before=%.3f peak_share_after=%.3f",
            len(rows),
            len(kept),
            max_share,
            max(buckets, key=lambda key: len(buckets[key]) + source_counts[key]),
            current_peak,
            max((count / len(kept) for count in source_counts.values()), default=0.0),
        )
    return kept


def _candidate_source_cap_protected_row_ids(rows: list[dict]) -> set[int]:
    protected: set[int] = {id(row) for row in rows if int(row.get("label", 0)) == 1}
    groups: dict[str, list[dict]] = {}
    for row in rows:
        key = _vod_balance_group_key(row)
        if key:
            groups.setdefault(key, []).append(row)

    for group_rows in groups.values():
        positive_rows = [row for row in group_rows if int(row.get("label", 0)) == 1]
        negative_rows = [row for row in group_rows if int(row.get("label", 0)) == 0]
        if not positive_rows or not negative_rows:
            continue
        best_negative = _rank_vod_negative_rows(negative_rows, positive_rows)[0]
        protected.add(id(best_negative))
    return protected


def _positive_cluster_key(row: dict) -> tuple[str, tuple[str, ...]] | None:
    if int(row.get("label", 0)) != 1:
        return None
    label_class = str(row.get("label_class") or "").strip().lower()
    if label_class not in _POSITIVE_LABEL_CLASSES:
        return None
    clip_ids = tuple(sorted(str(item) for item in (row.get("clip_ids") or []) if str(item).strip()))
    if not clip_ids:
        return None
    vod_id = str(row.get("vod_id") or "").strip()
    if not vod_id:
        return None
    return (vod_id, clip_ids)


def _prefer_positive_row(candidate: dict, existing: dict) -> bool:
    def score(row: dict) -> tuple[int, int, float, float]:
        dataset_source = str(row.get("dataset_source") or "")
        window_origin = str(row.get("window_origin") or "")
        clip_count = len(row.get("clip_ids") or [])
        return (
            1 if dataset_source == "twitch_vod_primary" else 0,
            1 if window_origin == "proxy_label_refined" else 0,
            float(row.get("sample_weight") or 0.0),
            float(clip_count),
        )

    return score(candidate) > score(existing)


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
