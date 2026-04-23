from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass
import hashlib
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
from ml import collect_viral_clips
from services import downloader

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_POSITIVE_LABEL_CLASSES = {"viral_positive", "positive", "weak_positive"}
_SUPPLEMENTAL_CANDIDATE_LIMIT = 2
_SUPPLEMENTAL_CANDIDATE_WEIGHT = 0.65
_BOUNDARY_NEAR_MISS_SAMPLE_WEIGHT = 0.45
_BOUNDARY_NEAR_MISS_LIMIT = 6
_MAX_VOD_NEGATIVE_ROWS_PER_POSITIVE = 8
_MAX_CANDIDATE_SOURCE_SHARE = 0.40
_DATASET_BUILD_LLM_PROVIDER = (os.environ.get("TRAINING_DATASET_LLM_PROVIDER") or "none").strip() or "none"
_VOD_CHECKPOINT_VERSION = 4
_VOD_FEATURE_CACHE_VERSION = 1
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


@dataclass(frozen=True)
class VodFeatureContext:
    transcript: list
    audio_signal: object
    audio_sample_rate: int
    cache_dir: pathlib.Path

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
        "supplemental_candidate_limit": _SUPPLEMENTAL_CANDIDATE_LIMIT,
        "supplemental_candidate_weight": _SUPPLEMENTAL_CANDIDATE_WEIGHT,
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
) -> VodFeatureContext:
    from services import audio_extractor, transcriber
    import librosa

    cache_dir = _vod_feature_cache_dir(vod_id)
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

    audio_signal, audio_sample_rate = librosa.load(str(audio_path), sr=None, mono=True)
    return VodFeatureContext(
        transcript=transcript,
        audio_signal=audio_signal,
        audio_sample_rate=int(audio_sample_rate),
        cache_dir=cache_dir,
    )


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
                if vod_checkpoint_rows:
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
        download_futures: dict[str, concurrent.futures.Future[pathlib.Path]] = {}
        for vod_id, vod_rows in vod_items:
            checkpoint_rows = _load_vod_checkpoint(work_dir, profile, vod_id, vod_rows)
            if checkpoint_rows is not None:
                logger.info("training_vod_checkpoint_hit %s", vod_rows[0]["vod_url"])
                scheduled_results.append(("checkpoint", checkpoint_rows))
                continue

            job_id = f"twitch_vod_{vod_id}"
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

        if vod_path_future is not None:
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

        if vod_checkpoint_rows:
            _write_vod_checkpoint(work_dir, profile, vod_id, vod_rows, vod_checkpoint_rows)
        return vod_checkpoint_rows
    finally:
        shutil.rmtree(vod_dir, ignore_errors=True)
        shutil.rmtree(download_dir, ignore_errors=True)


def _build_rows_for_vod_labels(
    vod_id: str,
    vod_rows: list[dict],
    *,
    vod_path: pathlib.Path,
    vod_dir: pathlib.Path,
    job_id: str,
    profile: str,
    context_index: dict,
) -> list[dict]:
    ordered_labels = sorted(vod_rows, key=lambda item: float(item["window_start"]))
    if not ordered_labels:
        return []
    vod_feature_context = _load_or_build_vod_feature_context(
        vod_id,
        vod_path=vod_path,
        job_id=job_id,
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
        return rows


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
        return _finalize_label_rows(label_rows, label, context_index)
    finally:
        shutil.rmtree(clip_dir, ignore_errors=True)


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
    if next_index >= len(vod_items):
        logger.info("training_vod_prefetch_skip end_of_queue")
        return None, None, None

    next_vod_id, next_rows = vod_items[next_index]
    if _load_vod_checkpoint(work_dir, profile, next_vod_id, next_rows) is not None:
        logger.info("training_vod_prefetch_skip checkpoint_hit %s", next_rows[0]["vod_url"])
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

    expanded_start, expanded_end = _expanded_training_window(label)
    transcript = _slice_transcript_for_window(
        vod_feature_context.transcript,
        window_start=expanded_start,
        window_end=expanded_end,
    )
    audio_frames, visual_frames = _load_or_build_window_modalities(
        str(label.get("vod_id") or ""),
        vod_path=vod_path,
        vod_context=vod_feature_context,
        window_start=expanded_start,
        window_end=expanded_end,
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
    )
    if not feature_rows:
        return []

    rows: list[dict] = []
    primary_feature_rows = feature_rows[: len(candidates)]
    boundary_feature_rows = feature_rows[len(candidates):]
    for idx, (candidate, features) in enumerate(zip(candidates, primary_feature_rows)):
        absolute_candidate = _absolute_candidate(candidate, expanded_start)
        rows.append(
            _training_row_from_label(
                label,
                absolute_candidate,
                features,
                sample_weight_scale=1.0 if idx == 0 else _SUPPLEMENTAL_CANDIDATE_WEIGHT,
                dataset_source="twitch_vod_primary" if idx == 0 else "twitch_vod_candidate_supplemental",
                supervision_reason_suffix="" if idx == 0 else ":candidate_generator",
                window_origin="proxy_label_refined" if idx == 0 else "candidate_generator_overlap",
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


def _vod_checkpoint_signature(vod_rows: list[dict]) -> str:
    normalized = json.dumps(
        sorted(vod_rows, key=lambda row: (float(row.get("window_start") or 0.0), float(row.get("window_end") or 0.0), str(row.get("label_class") or ""))),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


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
    if str(payload.get("label_signature") or "") != _vod_checkpoint_signature(vod_rows):
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
        "sample_weight": round(sample_weight_for_label(label_class) * sample_weight_scale, 3),
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
    return row


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
        boundary_quota = min(len(boundary_rows), max(1, min(limit // 4, limit // 2)))
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
