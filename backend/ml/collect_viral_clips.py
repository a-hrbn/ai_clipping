"""
Collect real viral clip training data from YouTube Shorts / TikTok URLs.

Each manifest row can be either:

- a positive example (`label=1`) for a genuinely viral/strong short, or
- a negative example (`label=0`) for a weak / non-viral / low-performing short
  from the same creator distribution.

The input file can be either:

- legacy plain-text URLs (one per line), or
- a CSV/TSV manifest with metadata such as streamer, platform, and tags.

Usage
-----
1. Add clip URLs to ml/viral_clips.txt:

   Plain text format:
       https://www.youtube.com/shorts/xxxx
       https://www.tiktok.com/@streamer/video/xxxx

   TSV/CSV manifest format:
       url,streamer,platform,language,tags,label,notes
       https://www.youtube.com/shorts/xxxx,shroud,youtube,de,"fps,clutch",1,"viral short"
       https://www.tiktok.com/@streamer/video/xxxx,tarik,tiktok,en,"reaction,chatting",0,"flat weak clip"

2. Run:
       python3 ml/collect_viral_clips.py

3. Then retrain:
       python3 ml/train_ranker.py \\
         --dataset ml/artifacts/viral_training_rows.jsonl \\
         --profile gaming

Options
-------
  --urls PATH       Path to URL list file (default: ml/viral_clips.txt)
  --output PATH     Output JSONL path (default: ml/artifacts/viral_training_rows.jsonl)
  --profile NAME    Clip profile for semantic scoring (default: gaming)
  --negatives PATH  Optional JSONL of fallback negative rows to merge in only
                    when the manifest does not already provide negatives
                    (default: ml/artifacts/synthetic_training_rows.jsonl if it exists)
  --work-dir PATH   Temp dir for downloads (default: backend/tmp/viral_clip_training)
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import importlib
import json
import logging
import pathlib
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import uuid

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# Suppress noisy library output during collection
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

_METADATA_KEYS = {
    "url",
    "streamer",
    "platform",
    "language",
    "tags",
    "label",
    "notes",
    "quality_bucket",
    "label_reason",
    "viral_score",
    "view_count",
    "like_count",
    "comment_count",
    "duration",
    "publish_timestamp",
    "upload_date",
}

_NETWORK_RETRY_BASE_DELAY_SEC = 30
_NETWORK_RETRY_MAX_DELAY_SEC = 300
_MAX_BATCH_CUT_WINDOWS = 24
_FFMPEG_BATCH_HEARTBEAT_SEC = 15.0
_SHORT_FORM_ROW_CACHE_VERSION = 1
_DEFAULT_SHORT_FORM_COLLECTION_WORKERS = 3
_DEFAULT_SHORT_FORM_CLIP_STEP_WORKERS = 3
_DEFAULT_SHORT_FORM_PREFETCH_WORKERS = 2
_SHORT_FORM_MEDIA_CACHE_NAMESPACE = "short_form"
_SHORT_FORM_MEDIA_CACHE_TTL_SEC = -1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--urls",      default=str(BACKEND_ROOT / "ml" / "viral_clips.txt"))
    parser.add_argument("--output",    default=str(BACKEND_ROOT / "ml" / "artifacts" / "viral_training_rows.jsonl"))
    parser.add_argument("--profile",   default="gaming")
    parser.add_argument("--negatives", default="")
    parser.add_argument("--work-dir",  default=str(BACKEND_ROOT / "tmp" / "viral_clip_training"))
    args = parser.parse_args()

    entries = _read_entries(pathlib.Path(args.urls))
    if not entries:
        logger.error("No URLs found in %s", args.urls)
        logger.info("Create the file and add one YouTube Shorts / TikTok URL per line, or use a CSV/TSV manifest.")
        return 1

    work_dir = pathlib.Path(args.work_dir)
    collected_rows = collect_rows(entries, args.profile, work_dir)
    negatives = sum(1 for row in collected_rows if row["label"] == 0)

    negative_rows: list[dict] = []
    if negatives == 0:
        negative_rows = _load_negatives(args.negatives)
        logger.info("Loaded %d fallback negative rows", len(negative_rows))

    all_rows = collected_rows + negative_rows
    if not all_rows:
        logger.error("No training rows produced.")
        return 1

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=True) for r in all_rows),
        encoding="utf-8",
    )
    logger.info("Wrote %d total rows to %s", len(all_rows), output_path)
    logger.info("Now run:  python3 ml/train_ranker.py --dataset %s --profile %s", output_path, args.profile)
    return 0


def collect_rows(
    entries: list[dict[str, str]],
    profile: str,
    work_dir: pathlib.Path,
    proxy: str | None = None,
    skip_platforms: set[str] | None = None,
) -> list[dict]:
    logger.info("Processing %d URLs…", len(entries))
    work_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = _short_form_row_cache_dir(work_dir)
    total_entries = len(entries)
    resolved_workers = _resolved_short_form_collection_workers(total_entries)
    resolved_prefetch_workers = _resolved_short_form_prefetch_workers(total_entries)
    cache_fingerprint = _short_form_cache_fingerprint(profile)
    logger.info(
        "short_form_collection_runtime urls=%s workers=%s prefetch_workers=%s cache_dir=%s",
        total_entries,
        resolved_workers,
        resolved_prefetch_workers,
        cache_dir,
    )

    collected_rows_by_index: dict[int, dict] = {}
    failed = 0
    skipped_unavailable = 0
    cache_hits = 0
    skipped_platforms = {item.strip().lower() for item in (skip_platforms or set()) if item.strip()}
    logger.info("%s", _render_progress_bar(0, total_entries, failed))
    prefetcher = _ShortFormMediaPrefetcher(proxy=proxy, max_workers=resolved_prefetch_workers)
    _schedule_short_form_prefetch(prefetcher, entries, start_index=1, width=_short_form_prefetch_window(resolved_workers))

    completed = 0
    try:
        if resolved_workers <= 1:
            _warm_clip_worker_runtime(profile)
            for index, entry in enumerate(entries, start=1):
                outcome = _collect_row_for_entry(
                    entry,
                    entries=entries,
                    index=index,
                    total_entries=total_entries,
                    profile=profile,
                    work_dir=work_dir,
                    cache_dir=cache_dir,
                    cache_fingerprint=cache_fingerprint,
                    proxy=proxy,
                    skipped_platforms=skipped_platforms,
                    prefetcher=prefetcher,
                    prefetch_width=_short_form_prefetch_window(resolved_workers),
                )
                completed, failed, skipped_unavailable, cache_hits = _record_collect_outcome(
                    outcome,
                    completed=completed,
                    total_entries=total_entries,
                    failed=failed,
                    skipped_unavailable=skipped_unavailable,
                    cache_hits=cache_hits,
                    collected_rows_by_index=collected_rows_by_index,
                )
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=resolved_workers,
                initializer=_warm_clip_worker_runtime,
                initargs=(profile,),
            ) as executor:
                futures = [
                    executor.submit(
                        _collect_row_for_entry,
                        entry,
                        entries=entries,
                        index=index,
                        total_entries=total_entries,
                        profile=profile,
                        work_dir=work_dir,
                        cache_dir=cache_dir,
                        cache_fingerprint=cache_fingerprint,
                        proxy=proxy,
                        skipped_platforms=skipped_platforms,
                        prefetcher=prefetcher,
                        prefetch_width=_short_form_prefetch_window(resolved_workers),
                    )
                    for index, entry in enumerate(entries, start=1)
                ]
                for future in concurrent.futures.as_completed(futures):
                    outcome = future.result()
                    completed, failed, skipped_unavailable, cache_hits = _record_collect_outcome(
                        outcome,
                        completed=completed,
                        total_entries=total_entries,
                        failed=failed,
                        skipped_unavailable=skipped_unavailable,
                        cache_hits=cache_hits,
                        collected_rows_by_index=collected_rows_by_index,
                    )
    finally:
        prefetcher.close()

    collected_rows = [collected_rows_by_index[index] for index in sorted(collected_rows_by_index)]
    positives = sum(1 for row in collected_rows if row["label"] == 1)
    negatives = sum(1 for row in collected_rows if row["label"] == 0)
    logger.info("%s", _render_progress_bar(total_entries, total_entries, failed))
    logger.info(
        "Collected %d labeled rows (%d positive, %d negative, %d failed, %d skipped_unavailable, %d cache_hits)",
        len(collected_rows),
        positives,
        negatives,
        failed,
        skipped_unavailable,
        cache_hits,
    )
    return collected_rows


def _collect_row_for_entry(
    entry: dict[str, str],
    *,
    entries: list[dict[str, str]],
    index: int,
    total_entries: int,
    profile: str,
    work_dir: pathlib.Path,
    cache_dir: pathlib.Path,
    cache_fingerprint: dict[str, object],
    proxy: str | None,
    skipped_platforms: set[str],
    prefetcher: "_ShortFormMediaPrefetcher",
    prefetch_width: int,
) -> dict[str, object]:
    from services import downloader

    logger.info("[%d/%d] %s", index, total_entries, entry["url"])
    started_at = time.perf_counter()
    platform = _infer_platform(entry)
    if platform in skipped_platforms:
        logger.warning("  ↷ skipped_unavailable: platform %s explicitly skipped", platform)
        return {
            "index": index,
            "row": None,
            "failed": 0,
            "skipped_unavailable": 1,
            "cache_hit": False,
        }

    cached_row = _load_cached_row(cache_dir, entry, cache_fingerprint)
    if cached_row is not None:
        elapsed = time.perf_counter() - started_at
        logger.info(
            "  ✓ cache hit  (label=%s hook=%.2f  hype=%.2f  arc=%.2f  %.1fs)",
            cached_row["label"],
            cached_row["features"]["hook_strength"],
            cached_row["features"]["max_hype"],
            cached_row["features"]["arc_score"],
            elapsed,
        )
        return {
            "index": index,
            "row": cached_row,
            "failed": 0,
            "skipped_unavailable": 0,
            "cache_hit": True,
        }

    _schedule_short_form_prefetch(prefetcher, entries, start_index=index + 1, width=prefetch_width)

    job_id = f"viral_{uuid.uuid4().hex[:8]}"
    clip_dir = work_dir / job_id
    network_attempt = 0
    while True:
        clip_dir.mkdir(parents=True, exist_ok=True)
        try:
            media_path = prefetcher.get(entry)
            row = _process_clip(entry, job_id, clip_dir, profile, proxy=proxy, media_path=media_path)
            if row:
                _write_cached_row(cache_dir, entry, cache_fingerprint, row)
                elapsed = time.perf_counter() - started_at
                logger.info(
                    "  ✓ features extracted  (label=%s hook=%.2f  hype=%.2f  arc=%.2f  %.1fs)",
                    row["label"],
                    row["features"]["hook_strength"],
                    row["features"]["max_hype"],
                    row["features"]["arc_score"],
                    elapsed,
                )
                return {
                    "index": index,
                    "row": row,
                    "failed": 0,
                    "skipped_unavailable": 0,
                    "cache_hit": False,
                }
            logger.warning("  ✗ skipped (no transcript or too short)")
            return {
                "index": index,
                "row": None,
                "failed": 1,
                "skipped_unavailable": 0,
                "cache_hit": False,
            }
        except downloader.DownloadPausedError as exc:
            network_attempt += 1
            delay_sec = _network_retry_delay_sec(network_attempt)
            logger.warning(
                "  ‖ paused_network_retry: %s (attempt %d, retry in %ds)",
                exc,
                network_attempt,
                delay_sec,
            )
            time.sleep(delay_sec)
            continue
        except downloader.DownloadUnavailableError as exc:
            logger.warning("  ↷ skipped_unavailable: %s", exc)
            return {
                "index": index,
                "row": None,
                "failed": 0,
                "skipped_unavailable": 1,
                "cache_hit": False,
            }
        except Exception as exc:
            logger.error("  ✗ failed: %s", exc)
            return {
                "index": index,
                "row": None,
                "failed": 1,
                "skipped_unavailable": 0,
                "cache_hit": False,
            }
        finally:
            shutil.rmtree(clip_dir, ignore_errors=True)


def _record_collect_outcome(
    outcome: dict[str, object],
    *,
    completed: int,
    total_entries: int,
    failed: int,
    skipped_unavailable: int,
    cache_hits: int,
    collected_rows_by_index: dict[int, dict],
) -> tuple[int, int, int, int]:
    completed += 1
    failed += int(outcome.get("failed", 0) or 0)
    skipped_unavailable += int(outcome.get("skipped_unavailable", 0) or 0)
    cache_hits += 1 if outcome.get("cache_hit") else 0
    row = outcome.get("row")
    if isinstance(row, dict):
        collected_rows_by_index[int(outcome["index"])] = row
    logger.info("%s", _render_progress_bar(completed, total_entries, failed))
    return completed, failed, skipped_unavailable, cache_hits


def _short_form_row_cache_dir(work_dir: pathlib.Path) -> pathlib.Path:
    cache_dir = work_dir / "_row_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _cacheable_entry(entry: dict[str, str]) -> dict[str, str]:
    return {key: str(value or "") for key, value in sorted(entry.items())}


def _import_service_module(module_name: str):
    try:
        return importlib.import_module(f"services.{module_name}")
    except Exception:
        return None


def _short_form_cache_fingerprint(profile: str) -> dict[str, object]:
    transcriber = _import_service_module("transcriber")
    video_features = _import_service_module("video_features")
    feature_builder = _import_service_module("feature_builder")

    transcriber_runtime = {}
    if transcriber is not None and hasattr(transcriber, "runtime_summary"):
        try:
            transcriber_runtime = transcriber.runtime_summary()
        except Exception:
            transcriber_runtime = {"status": "unavailable"}

    video_runtime = {}
    if video_features is not None and hasattr(video_features, "runtime_summary"):
        try:
            video_runtime = video_features.runtime_summary()
        except Exception:
            video_runtime = {"status": "unavailable"}

    return {
        "cache_version": _SHORT_FORM_ROW_CACHE_VERSION,
        "profile": profile,
        "feature_build_version": getattr(feature_builder, "FEATURE_BUILD_VERSION", 1),
        "whisper_model": str(settings.whisper_model or ""),
        "whisper_device": str(settings.whisper_device or ""),
        "semantic_device": str(settings.semantic_device or ""),
        "semantic_batch_size": int(settings.semantic_batch_size or 0),
        "video_expression_sample_sec": int(settings.video_expression_sample_sec or 0),
        "transcriber_runtime": transcriber_runtime,
        "video_runtime": video_runtime,
    }


def _short_form_row_cache_key(entry: dict[str, str], cache_fingerprint: dict[str, object]) -> str:
    payload = {
        "entry": _cacheable_entry(entry),
        "runtime": cache_fingerprint,
    }
    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _short_form_row_cache_path(
    cache_dir: pathlib.Path,
    entry: dict[str, str],
    cache_fingerprint: dict[str, object],
) -> pathlib.Path:
    return cache_dir / f"{_short_form_row_cache_key(entry, cache_fingerprint)}.json"


def _load_cached_row(
    cache_dir: pathlib.Path,
    entry: dict[str, str],
    cache_fingerprint: dict[str, object],
) -> dict | None:
    cache_path = _short_form_row_cache_path(cache_dir, entry, cache_fingerprint)
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _write_cached_row(
    cache_dir: pathlib.Path,
    entry: dict[str, str],
    cache_fingerprint: dict[str, object],
    row: dict,
) -> None:
    cache_path = _short_form_row_cache_path(cache_dir, entry, cache_fingerprint)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(row, ensure_ascii=True), encoding="utf-8")


def _resolved_short_form_collection_workers(total_entries: int) -> int:
    if total_entries <= 0:
        return 1
    configured = int(getattr(settings, "short_form_collection_workers", _DEFAULT_SHORT_FORM_COLLECTION_WORKERS) or 0)
    if configured <= 0:
        configured = _DEFAULT_SHORT_FORM_COLLECTION_WORKERS
    return max(1, min(configured, total_entries))


def _resolved_short_form_clip_step_workers() -> int:
    configured = int(getattr(settings, "short_form_clip_step_workers", _DEFAULT_SHORT_FORM_CLIP_STEP_WORKERS) or 0)
    if configured <= 0:
        configured = _DEFAULT_SHORT_FORM_CLIP_STEP_WORKERS
    return max(1, min(configured, 3))


def _resolved_short_form_prefetch_workers(total_entries: int) -> int:
    if total_entries <= 1:
        return 1
    configured = int(getattr(settings, "short_form_prefetch_workers", _DEFAULT_SHORT_FORM_PREFETCH_WORKERS) or 0)
    if configured <= 0:
        configured = _DEFAULT_SHORT_FORM_PREFETCH_WORKERS
    return max(1, min(configured, total_entries))


def _short_form_prefetch_window(collection_workers: int) -> int:
    return max(1, int(collection_workers) + _DEFAULT_SHORT_FORM_PREFETCH_WORKERS)


def _schedule_short_form_prefetch(
    prefetcher: "_ShortFormMediaPrefetcher",
    entries: list[dict[str, str]],
    *,
    start_index: int,
    width: int,
) -> None:
    if not entries:
        return
    start = max(int(start_index), 1)
    end = min(len(entries), start + max(int(width), 0) - 1)
    if end < start:
        return
    prefetcher.schedule_entries(entries[start - 1:end])


def _warm_clip_worker_runtime(profile: str) -> None:
    logger.info("short_form_collection_worker_warmup profile=%s", profile)
    transcriber = _import_service_module("transcriber")
    video_features = _import_service_module("video_features")
    semantic_candidates = _import_service_module("semantic_candidates")

    try:
        get_model = getattr(transcriber, "_get_model", None) if transcriber is not None else None
        if callable(get_model):
            get_model()
    except Exception as exc:
        logger.warning("short_form_collection_warmup_transcriber_failed profile=%s error=%s", profile, exc)
    try:
        ensure_deepface_ready = getattr(video_features, "_ensure_deepface_ready", None) if video_features is not None else None
        if callable(ensure_deepface_ready):
            ensure_deepface_ready()
        if video_features is not None and hasattr(video_features, "runtime_summary"):
            video_features.runtime_summary()
    except Exception as exc:
        logger.warning("short_form_collection_warmup_video_failed profile=%s error=%s", profile, exc)
    try:
        get_model = getattr(semantic_candidates, "_get_model", None) if semantic_candidates is not None else None
        get_pattern_embeddings = getattr(semantic_candidates, "_get_pattern_embeddings", None) if semantic_candidates is not None else None
        model = get_model() if callable(get_model) else None
        if model is not None and callable(get_pattern_embeddings):
            get_pattern_embeddings(model)
    except Exception as exc:
        logger.warning("short_form_collection_warmup_semantic_failed profile=%s error=%s", profile, exc)


def _network_retry_delay_sec(attempt: int) -> int:
    bounded_attempt = max(int(attempt), 1)
    return min(_NETWORK_RETRY_BASE_DELAY_SEC * bounded_attempt, _NETWORK_RETRY_MAX_DELAY_SEC)


# ---------------------------------------------------------------------------
# Per-clip processing
# ---------------------------------------------------------------------------

def _process_clip(
    entry: dict[str, str],
    job_id: str,
    clip_dir: pathlib.Path,
    profile: str,
    proxy: str | None = None,
    media_path: pathlib.Path | None = None,
) -> dict | None:
    from services import (
        downloader, audio_extractor, transcriber,
        audio_features, video_features, scorer, feature_builder,
    )
    from services.timeline_types import Candidate

    url = entry["url"]

    # 1. Download
    video_path = media_path or downloader.download(url, job_id, proxy=proxy)

    # 2. Extract audio
    audio_path = audio_extractor.extract(video_path, job_id, output_path=clip_dir / "audio.wav")

    # 3. Transcribe + per-modality features
    transcript, audio_frames, visual_frames = _collect_clip_modalities(
        audio_path,
        video_path,
        transcriber=transcriber,
        audio_features=audio_features,
        video_features=video_features,
    )
    if not transcript:
        return None

    # 4. Merge feature modalities into the transcript timeline
    transcript = audio_features.enrich_segments(transcript, audio_frames)
    transcript = video_features.enrich_segments(transcript, visual_frames)

    # 5. Score
    transcript = scorer.score_segments(transcript)

    # 6. Treat the entire clip as one candidate window
    duration = transcript[-1].end - transcript[0].start
    if duration < 5.0:
        return None

    # Find the peak segment as anchor
    anchor_seg = max(transcript, key=lambda s: s.hype_score)

    candidate = Candidate(
        id=f"training:{job_id}",
        start=transcript[0].start,
        end=transcript[-1].end,
        source="training_clip",
        anchor_time=(anchor_seg.start + anchor_seg.end) / 2.0,
        score_hint=anchor_seg.hype_score,
        tags=list(set(anchor_seg.emotions)),
    )

    # 7. Build features
    feats_list = feature_builder.build_features(
        [candidate],
        transcript,
        profile,
        llm_provider="ollama",
        video_path=clip_path,
    )
    if not feats_list:
        return None
    feats = feats_list[0]

    # chat_peak is always 0 for standalone clips (no VOD chat available)
    # context_cost is 0 (no wasted pre-context — this IS the clip)
    feats.chat_peak = 0.0
    feats.context_cost = 0.0

    return _build_training_row(entry, candidate, feats)


def process_local_video(
    entry: dict[str, str],
    job_id: str,
    clip_dir: pathlib.Path,
    video_path: pathlib.Path,
    profile: str,
) -> dict | None:
    from services import (
        audio_extractor, transcriber,
        audio_features, video_features, scorer, feature_builder,
    )
    from services.timeline_types import Candidate

    audio_path = audio_extractor.extract(video_path, job_id, output_path=clip_dir / "audio.wav")
    transcript, audio_frames, visual_frames = _collect_clip_modalities(
        audio_path,
        video_path,
        transcriber=transcriber,
        audio_features=audio_features,
        video_features=video_features,
    )
    if not transcript:
        return None

    transcript = audio_features.enrich_segments(transcript, audio_frames)
    transcript = video_features.enrich_segments(transcript, visual_frames)
    transcript = scorer.score_segments(transcript)
    duration = transcript[-1].end - transcript[0].start
    if duration < 5.0:
        return None

    anchor_seg = max(transcript, key=lambda s: s.hype_score)
    candidate = Candidate(
        id=f"training:{job_id}",
        start=transcript[0].start,
        end=transcript[-1].end,
        source="training_clip",
        anchor_time=(anchor_seg.start + anchor_seg.end) / 2.0,
        score_hint=anchor_seg.hype_score,
        tags=list(set(anchor_seg.emotions)),
    )
    feats_list = feature_builder.build_features(
        [candidate],
        transcript,
        profile,
        llm_provider="ollama",
        video_path=video_path,
    )
    if not feats_list:
        return None
    feats = feats_list[0]
    feats.chat_peak = 0.0
    feats.context_cost = 0.0
    return _build_training_row(entry, candidate, feats)


class _ShortFormMediaPrefetcher:
    def __init__(self, *, proxy: str | None, max_workers: int) -> None:
        self._proxy = proxy
        self._max_workers = max(int(max_workers), 1)
        self._lock = threading.Lock()
        self._futures: dict[str, concurrent.futures.Future[pathlib.Path]] = {}
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers)

    def schedule_entries(self, entries: list[dict[str, str]]) -> None:
        for entry in entries:
            self._schedule(entry)

    def get(self, entry: dict[str, str]) -> pathlib.Path:
        return self._schedule(entry).result()

    def close(self) -> None:
        self._executor.shutdown(wait=True)

    def _schedule(self, entry: dict[str, str]) -> concurrent.futures.Future[pathlib.Path]:
        key = _short_form_media_cache_key(entry["url"])
        with self._lock:
            future = self._futures.get(key)
            if future is None:
                future = self._executor.submit(_ensure_short_form_media_cached, entry["url"], self._proxy)
                self._futures[key] = future
            return future


def _short_form_media_cache_key(url: str) -> str:
    return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()


def _ensure_short_form_media_cached(url: str, proxy: str | None = None) -> pathlib.Path:
    from services import downloader

    return downloader.ensure_cached_download(
        url,
        cache_namespace=_SHORT_FORM_MEDIA_CACHE_NAMESPACE,
        cache_ttl_sec=_SHORT_FORM_MEDIA_CACHE_TTL_SEC,
        proxy=proxy,
    )


def _collect_clip_modalities(
    audio_path: pathlib.Path,
    video_path: pathlib.Path,
    *,
    transcriber,
    audio_features,
    video_features,
) -> tuple[list, list, list]:
    resolved_workers = _resolved_short_form_clip_step_workers()
    if resolved_workers <= 1:
        transcript = transcriber.transcribe(audio_path)
        audio_frames = audio_features.extract(audio_path)
        visual_frames = video_features.extract(video_path)
        return transcript, audio_frames, visual_frames

    with concurrent.futures.ThreadPoolExecutor(max_workers=resolved_workers) as executor:
        transcript_future = executor.submit(transcriber.transcribe, audio_path)
        audio_future = executor.submit(audio_features.extract, audio_path)
        visual_future = executor.submit(video_features.extract, video_path)
        transcript = transcript_future.result()
        audio_frames = audio_future.result()
        visual_frames = visual_future.result()
    return transcript, audio_frames, visual_frames


def cut_video_window(
    video_path: pathlib.Path,
    output_path: pathlib.Path,
    start: float,
    end: float,
) -> pathlib.Path:
    return cut_video_windows(video_path, [(output_path, start, end)])[0]


def cut_video_windows(
    video_path: pathlib.Path,
    windows: list[tuple[pathlib.Path, float, float]],
) -> list[pathlib.Path]:
    if not windows:
        return []

    normalized_windows: list[tuple[pathlib.Path, float, float]] = []
    for output_path, start, end in windows:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        start_sec = max(float(start), 0.0)
        duration = max(float(end) - float(start), 1.0)
        normalized_windows.append((output_path, start_sec, duration))

    batches = _window_batches(normalized_windows, _MAX_BATCH_CUT_WINDOWS)
    total_windows = len(normalized_windows)
    logger.info(
        "training_vod_batch_cut_plan video=%s windows=%s batches=%s batch_limit=%s",
        video_path,
        total_windows,
        len(batches),
        _MAX_BATCH_CUT_WINDOWS,
    )
    completed_windows = 0
    for batch_index, chunk in enumerate(batches, start=1):
        batch_start = completed_windows + 1
        batch_end = completed_windows + len(chunk)
        batch_media_seconds = sum(duration for _output_path, _start_sec, duration in chunk)
        started_at = time.perf_counter()
        logger.info(
            "training_vod_batch_cut_start video=%s batch=%s/%s windows=%s-%s total_windows=%s batch_media_seconds=%.1f",
            video_path,
            batch_index,
            len(batches),
            batch_start,
            batch_end,
            total_windows,
            batch_media_seconds,
        )
        _cut_video_windows_chunk(
            video_path,
            chunk,
            progress_label=(
                f"video={video_path} batch={batch_index}/{len(batches)} "
                f"windows={batch_start}-{batch_end}/{total_windows}"
            ),
        )
        completed_windows = batch_end
        logger.info(
            "training_vod_batch_cut_done video=%s batch=%s/%s completed_windows=%s/%s elapsed_sec=%.1f",
            video_path,
            batch_index,
            len(batches),
            completed_windows,
            total_windows,
            time.perf_counter() - started_at,
        )
    return [output_path for output_path, _start_sec, _duration in normalized_windows]


def _window_batches(
    windows: list[tuple[pathlib.Path, float, float]],
    batch_size: int,
) -> list[list[tuple[pathlib.Path, float, float]]]:
    resolved_batch_size = max(1, int(batch_size))
    return [
        windows[index:index + resolved_batch_size]
        for index in range(0, len(windows), resolved_batch_size)
    ]


def _cut_video_windows_batch(
    video_path: pathlib.Path,
    normalized_windows: list[tuple[pathlib.Path, float, float]],
    *,
    progress_label: str,
) -> None:
    command = ["ffmpeg", "-y", "-nostdin", "-loglevel", "error"]
    for _output_path, start_sec, duration in normalized_windows:
        command.extend([
            "-ss", f"{start_sec:.3f}",
            "-t", f"{duration:.3f}",
            "-i", str(video_path),
        ])

    video_codec = str(settings.training_ffmpeg_video_codec).strip()
    audio_codec = str(settings.training_ffmpeg_audio_codec).strip()
    preset = str(settings.training_ffmpeg_preset).strip()
    threads = int(settings.training_ffmpeg_threads)
    for input_index, (output_path, _start_sec, _duration) in enumerate(normalized_windows):
        output_command = [
            "-map", f"{input_index}:v:0",
            "-map", f"{input_index}:a:0?",
        ]
        if threads > 0:
            output_command.extend(["-threads", str(threads)])
        if video_codec:
            output_command.extend(["-c:v", video_codec])
        if audio_codec:
            output_command.extend(["-c:a", audio_codec])
        if preset:
            output_command.extend(["-preset", preset])
        output_command.append(str(output_path))
        command.extend(output_command)

    completed = _run_ffmpeg_command(command, progress_label=progress_label)
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        outputs = ", ".join(str(output_path) for output_path, _start_sec, _duration in normalized_windows)
        raise RuntimeError(
            "ffmpeg failed while cutting training windows "
            f"{video_path} -> [{outputs}]"
            + (f": {stderr}" if stderr else "")
        )


def _cut_video_windows_chunk(
    video_path: pathlib.Path,
    normalized_windows: list[tuple[pathlib.Path, float, float]],
    *,
    progress_label: str,
) -> None:
    try:
        _cut_video_windows_batch(
            video_path,
            normalized_windows,
            progress_label=progress_label,
        )
        return
    except RuntimeError:
        if len(normalized_windows) <= 1:
            raise

    midpoint = max(1, len(normalized_windows) // 2)
    left = normalized_windows[:midpoint]
    right = normalized_windows[midpoint:]
    logger.warning(
        "ffmpeg batch cut retry split video=%s windows=%s -> (%s,%s)",
        video_path,
        len(normalized_windows),
        len(left),
        len(right),
    )
    _cut_video_windows_chunk(video_path, left, progress_label=progress_label)
    _cut_video_windows_chunk(video_path, right, progress_label=progress_label)


def _run_ffmpeg_command(command: list[str], *, progress_label: str) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    started_at = time.perf_counter()
    next_heartbeat = _FFMPEG_BATCH_HEARTBEAT_SEC
    while process.poll() is None:
        elapsed = time.perf_counter() - started_at
        if elapsed >= next_heartbeat:
            logger.info(
                "training_vod_batch_cut_heartbeat %s elapsed_sec=%.1f",
                progress_label,
                elapsed,
            )
            next_heartbeat += _FFMPEG_BATCH_HEARTBEAT_SEC
        time.sleep(1.0)

    _stdout, stderr = process.communicate()
    return subprocess.CompletedProcess(
        args=command,
        returncode=process.returncode,
        stdout="",
        stderr=stderr or "",
    )


def _build_training_row(entry: dict[str, str], candidate, feats) -> dict:
    label = _parse_label(entry.get("label", "1"))
    selected = label == 1
    rejection_source = None if selected else "real_short_negative"
    rejected_reason = None if selected else "real_short_negative"
    label_class = str(entry.get("label_class", "")).strip() or (
        "short_form_positive" if label == 1 else "short_form_negative"
    )

    return {
        "job_id": "short_form_collection",
        "candidate_id": feats.candidate_id,
        "label": label,
        "selected": selected,
        "exported": True,
        "final_start": candidate.start,
        "final_end": candidate.end,
        "rejected_reason": rejected_reason,
        "rejection_source": rejection_source,
        "creator_id": entry["streamer"] or _infer_creator_id(entry["url"]),
        "streamer": entry["streamer"],
        "platform": entry["platform"] or _infer_platform(entry["url"]),
        "language": entry["language"],
        "tags": _parse_tags(entry["tags"]),
        "notes": entry["notes"],
        "quality_bucket": entry.get("quality_bucket", ""),
        "label_reason": entry.get("label_reason", ""),
        "viral_score": _parse_float(entry.get("viral_score", "")),
        "view_count": _parse_int(entry.get("view_count", "")),
        "like_count": _parse_int(entry.get("like_count", "")),
        "comment_count": _parse_int(entry.get("comment_count", "")),
        "publish_timestamp": _parse_int(entry.get("publish_timestamp", "")),
        "upload_date": str(entry.get("upload_date", "")).strip(),
        "platform_source": str(entry.get("platform_source", "")).strip(),
        "label_class": label_class,
        "label_score": _parse_float(entry.get("label_score", "")),
        "supervision_tier": "short_form_discovered",
        "supervision_reason": str(entry.get("label_reason", "")).strip() or "manifest_label",
        "ordinal_label": _ordinal_label(label_class, label),
        "window_start": _parse_float(entry.get("window_start", candidate.start)),
        "window_end": _parse_float(entry.get("window_end", candidate.end)),
        "vod_id": str(entry.get("vod_id", "")).strip(),
        "clip_ids": _parse_tags(str(entry.get("clip_ids", "")).replace("|", ",")),
        "source_url": entry["url"],
        "feature_provenance": _feature_provenance(feats, has_vod_context=False),
        "features": feats.to_dict(),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_entries(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    content_lines = [line for line in lines if line.strip() and not line.lstrip().startswith("#")]
    if not content_lines:
        return []

    first = content_lines[0].lower()
    if any(key in first for key in _METADATA_KEYS) and ("," in first or "\t" in first):
        delimiter = "\t" if "\t" in content_lines[0] else ","
        reader = csv.DictReader(content_lines, delimiter=delimiter)
        entries: list[dict[str, str]] = []
        for row in reader:
            normalized = {
                "url": (row.get("url") or "").strip(),
                "streamer": (row.get("streamer") or "").strip(),
                "platform": (row.get("platform") or "").strip().lower(),
                "language": (row.get("language") or "").strip().lower(),
                "tags": (row.get("tags") or "").strip(),
                "label": (row.get("label") or "1").strip(),
                "notes": (row.get("notes") or "").strip(),
                "quality_bucket": (row.get("quality_bucket") or "").strip(),
                "label_reason": (row.get("label_reason") or "").strip(),
                "viral_score": (row.get("viral_score") or "").strip(),
                "view_count": (row.get("view_count") or "").strip(),
                "like_count": (row.get("like_count") or "").strip(),
                "comment_count": (row.get("comment_count") or "").strip(),
                "duration": (row.get("duration") or "").strip(),
                "publish_timestamp": (row.get("publish_timestamp") or "").strip(),
                "upload_date": (row.get("upload_date") or "").strip(),
            }
            if normalized["url"]:
                entries.append(normalized)
        return _dedupe_entries(entries)

    return _dedupe_entries([
        {
            "url": line.strip(),
            "streamer": "",
            "platform": "",
            "language": "",
            "tags": "",
            "label": "1",
            "notes": "",
            "quality_bucket": "",
            "label_reason": "",
            "viral_score": "",
            "view_count": "",
            "like_count": "",
            "comment_count": "",
            "duration": "",
            "publish_timestamp": "",
            "upload_date": "",
        }
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ])


def _dedupe_entries(entries: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen: set[str] = set()
    for entry in entries:
        key = _canonical_entry_url(str(entry.get("url") or ""))
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    return deduped


def _canonical_entry_url(raw: str) -> str:
    url = str(raw or "").strip()
    if not url:
        return ""
    parsed = urllib.parse.urlsplit(url)
    path = re.sub(r"/+", "/", parsed.path or "").rstrip("/")
    if not path:
        path = "/"
    return urllib.parse.urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            path,
            "",
            "",
        )
    )


def _load_negatives(path_arg: str) -> list[dict]:
    """Load negative rows from a JSONL file. Defaults to synthetic data if available."""
    candidates = []
    if path_arg:
        candidates.append(pathlib.Path(path_arg))
    # Always try the synthetic fallback
    candidates.append(BACKEND_ROOT / "ml" / "artifacts" / "synthetic_training_rows.jsonl")

    for path in candidates:
        if not path.exists():
            continue
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("label") == 0:
                rows.append(row)
        if rows:
            return rows
    return []


def _parse_tags(raw: str) -> list[str]:
    if not raw:
        return []
    return [part.strip().lower() for part in re.split(r"[|,;]", raw) if part.strip()]


def _parse_label(raw: str) -> int:
    value = str(raw).strip().lower()
    if value in {"0", "neg", "negative", "weak", "bad"}:
        return 0
    return 1


def _parse_int(raw: str) -> int:
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return 0


def _parse_float(raw: str) -> float:
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return 0.0


def _ordinal_label(label_class: str, label: int) -> int:
    normalized = str(label_class or "").strip().lower()
    mapping = {
        "viral_positive": 4,
        "cross_platform_outcome_positive": 4,
        "positive": 3,
        "weak_positive": 2,
        "short_form_positive": 3,
        "outcome_positive": 3,
        "pipeline_positive": 2,
        "hard_negative": 1,
        "background_negative": 0,
        "short_form_negative": 0,
        "outcome_negative": 0,
        "pipeline_negative": 0,
    }
    if normalized in mapping:
        return mapping[normalized]
    return 3 if int(label) == 1 else 0


def _feature_provenance(feats, *, has_vod_context: bool) -> dict[str, object]:
    getter = feats.get if isinstance(feats, dict) else getattr

    def _read(name: str, default):
        if isinstance(feats, dict):
            return getter(name, default)
        return getter(feats, name, default)

    return {
        "has_llm_semantics": bool(_read("has_llm_semantics", False)),
        "has_chat_features": bool(_read("has_chat_features", False)),
        "has_vector_similarity": bool(_read("has_vector_similarity", False)),
        "has_vod_context": bool(has_vod_context),
        "has_vod_memory": bool(
            _read("retrieved_memory_span_count", 0.0)
            or _read("setup_depth_score", 0.0)
            or _read("callback_strength", 0.0)
        ),
        "feature_build_version": int(_read("feature_build_version", 1) or 1),
    }


def _render_progress_bar(completed: int, total: int, failed: int, width: int = 28) -> str:
    total = max(total, 1)
    completed = max(0, min(completed, total))
    ratio = completed / total
    filled = min(width, int(round(width * ratio)))
    bar = "#" * filled + "-" * (width - filled)
    percent = int(round(ratio * 100))
    return f"Progress [{bar}] {completed}/{total} ({percent}%) failed={failed}"


def _infer_platform(url: str | dict[str, str]) -> str:
    if isinstance(url, dict):
        explicit = str(url.get("platform") or "").strip().lower()
        if explicit:
            return explicit
        url = str(url.get("url") or "")
    lowered = url.lower()
    if "tiktok.com" in lowered:
        return "tiktok"
    if "youtube.com/shorts/" in lowered or "youtu.be/" in lowered or "youtube.com/watch" in lowered:
        return "youtube"
    return "unknown"


def _infer_creator_id(url: str) -> str:
    match = re.search(r"tiktok\.com/@([^/]+)/video/", url, re.IGNORECASE)
    if match:
        return match.group(1).strip().lower()
    match = re.search(r"[?&]ab_channel=([^&]+)", url, re.IGNORECASE)
    if match:
        return match.group(1).strip().lower()
    return _infer_platform(url)


if __name__ == "__main__":
    raise SystemExit(main())
