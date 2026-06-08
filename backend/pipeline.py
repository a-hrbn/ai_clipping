import asyncio
import json
import logging
import os
import pathlib
import re
import shutil
import subprocess
import uuid
from dataclasses import asdict, is_dataclass
from config import settings
from job_manager import is_job_cancelled, update_job
from services import (
    downloader,
    audio_extractor,
    transcriber,
    clipper,
    clip_strategy,
    moment_finder,
    chat_signal,
)

logger = logging.getLogger(__name__)

_TWITCH_VOD_RE = re.compile(r"twitch\.tv/videos/(\d+)")
_AUDIO_CACHE_VERSION = 1


class PipelineCancelledError(RuntimeError):
    pass


def _raise_if_cancelled(job_id: str) -> None:
    if is_job_cancelled(job_id):
        raise PipelineCancelledError(f"Job {job_id} cancelled by client")


def _frontend_cache_enabled_for_job(job_id: str) -> bool:
    if bool(getattr(settings, "frontend_job_cache_enabled", False)):
        return True
    job_id = str(job_id or "")
    return job_id.startswith(("twitch_vod", "judge_backfill_", "pipeline_debug_"))


def _debug_job(job_id: str) -> bool:
    return str(job_id or "").startswith("pipeline_debug_")


def _transcript_cache_enabled_for_job(job_id: str, url: str | None) -> bool:
    if _frontend_cache_enabled_for_job(job_id):
        return True
    if not bool(getattr(settings, "frontend_transcript_cache_enabled", True)):
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



def _source_twitch_vod_id(url: str | None) -> str:
    if not url:
        return ""
    match = _TWITCH_VOD_RE.search(str(url).strip().lower())
    return match.group(1) if match else ""



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



def _audio_cache_path(cache_source: str | None) -> pathlib.Path | None:
    digest = transcriber.stable_cache_key_for_source(cache_source)
    if not digest:
        return None
    return pathlib.Path(settings.storage_dir) / "_audio_cache" / digest / "audio.wav"


def _audio_cache_metadata_path(audio_cache_path: pathlib.Path) -> pathlib.Path:
    return audio_cache_path.with_name("audio_metadata.json")



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



def _write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def _moments_from_post_processed(rows: list[dict], platform: str) -> list[dict]:
    moments = []
    for i, row in enumerate(rows, start=1):
        start = float(row["start_time"])
        end = float(row["end_time"])
        moments.append({
            "start_time": start,
            "end_time": end,
            "viral_score": float(row.get("viral_score") or 1.0),
            "reason": str(row.get("reason") or ""),
            "candidate_id": f"llm_moment_{i:02d}",
            "ranking_mode": "llm_moment_finder",
            "platform": platform,
            "projected_hook_score": row.get("hook"),
            "projected_retention_score": row.get("flow"),
        })
    return moments



async def run_pipeline(
    job_id: str,
    url: str | None = None,
    video_path: pathlib.Path | None = None,
    profile: str | None = None,
    strategy: dict | None = None,
) -> None:
    effective_profile = profile or settings.clip_profile
    normalized_strategy = clip_strategy.normalize_strategy(strategy)
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
            pct = 30 + int((done_chunks / max(total_chunks, 1)) * 25)
            update_job(job_id, status="transcribing", progress_pct=min(max(pct, 30), 55))

        _raise_if_cancelled(job_id)
        transcript_cache_source = (
            _transcript_cache_source(url, video_path)
            if _transcript_cache_enabled_for_job(job_id, url)
            else None
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
        if transcript is not None:
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
                    cache_source=transcript_cache_source,
                ),
            )
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

        job_dir = pathlib.Path(settings.storage_dir) / job_id
        debug_dir = job_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        if settings.enable_chat_analysis and url:
            update_job(job_id, status="downloading_chat", progress_pct=60)
            enriched = transcriber.apply_chat_enrichment(transcript_cache_source, transcript)
            if enriched is not None:
                transcript = enriched
            else:
                chat_path = await loop.run_in_executor(
                    None,
                    lambda: chat_signal.download_chat(
                        url, job_id, debug_dir,
                        timeout_sec=settings.chat_download_timeout_sec,
                    ),
                )
                if chat_path:
                    transcript = chat_signal.enrich_segments(transcript, chat_path)
                    if transcript_cache_source:
                        await loop.run_in_executor(
                            None,
                            lambda: transcriber.save_chat_enrichment(transcript_cache_source, transcript),
                        )

        llm_stage: dict[str, object] = {
            "attempted": True,
            "status": "running",
            "provider": str(getattr(settings, "moment_finder_provider", "gemini") or "gemini"),
            "model": str(getattr(settings, "moment_finder_model", "") or ""),
            "max_moments": int(getattr(settings, "moment_finder_max_moments", 180) or 180),
        }
        _write_json(debug_dir / "llm_moment_finder_stage.json", llm_stage)
        update_job(job_id, status="analyzing", progress_pct=73)
        try:
            llm_result = await loop.run_in_executor(
                None,
                lambda: moment_finder.find_moments(
                    transcript,
                    profile=effective_profile,
                    vod_id=_source_twitch_vod_id(url) or job_id,
                    provider=str(getattr(settings, "moment_finder_provider", "gemini") or "gemini"),
                    model=str(getattr(settings, "moment_finder_model", "") or ""),
                    max_moments=int(getattr(settings, "moment_finder_max_moments", 180) or 180),
                    min_duration_sec=float(getattr(settings, "moment_finder_min_duration_sec", 8.0) or 8.0),
                    max_duration_sec=float(getattr(settings, "moment_finder_max_duration_sec", 90.0) or 90.0),
                    max_tokens=int(getattr(settings, "moment_finder_max_tokens", 16000) or 16000),
                    temperature=0.0,
                    timeout_sec=float(getattr(settings, "moment_finder_timeout_sec", 900.0) or 900.0),
                    api_retries=int(getattr(settings, "moment_finder_api_retries", 2) or 0),
                    retry_backoff_sec=float(getattr(settings, "moment_finder_retry_backoff_sec", 20.0) or 20.0),
                    filter_transcript_noise=bool(getattr(settings, "moment_finder_filter_transcript_noise", True)),
                ),
            )
            if not llm_result.ranked_candidates:
                raise moment_finder.MomentFinderError("LLM moment finder returned no ranked candidates.")
            llm_stage.update({"status": "ok", **llm_result.summary})
            (debug_dir / "llm_moment_finder_prompt.txt").write_text(llm_result.prompt, encoding="utf-8")
            (debug_dir / "llm_moment_finder_response.txt").write_text(llm_result.response_text, encoding="utf-8")
            _write_json(debug_dir / "llm_moment_finder_stage.json", llm_stage)
            _write_json(debug_dir / "llm_moment_finder_usage.json", llm_result.usage)
            _write_jsonl(debug_dir / "llm_moment_finder_moments.jsonl", llm_result.post_processed_moments)
        except moment_finder.MomentFinderError as exc:
            llm_stage.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
            _write_json(debug_dir / "llm_moment_finder_stage.json", llm_stage)
            logger.error(
                json.dumps(
                    {"event": "llm_moment_finder_failed", "job_id": job_id,
                     "error_type": type(exc).__name__, "error": str(exc)},
                    ensure_ascii=True, sort_keys=True,
                )
            )
            raise

        transcripts_dir = pathlib.Path(__file__).parents[1] / "transcripts"
        transcripts_dir.mkdir(exist_ok=True)
        (transcripts_dir / f"{job_id}.txt").write_text(transcriber.format_transcript(transcript), encoding="utf-8")
        (transcripts_dir / f"{job_id}.jsonl").write_text(transcriber.format_jsonl(transcript), encoding="utf-8")
        _write_jsonl(debug_dir / "segments.jsonl", _segment_records(transcript))

        platform = str(normalized_strategy.get("platform") or "general")
        moments = _moments_from_post_processed(llm_result.post_processed_moments, platform)
        _write_json(debug_dir / "final_moments.json", {"moments": moments, "count": len(moments)})

        logger.info(
            json.dumps(
                {"event": "pipeline_moments_ready", "job_id": job_id, "moment_count": len(moments)},
                ensure_ascii=True, sort_keys=True,
            )
        )

        update_job(job_id, status="clipping", progress_pct=88)
        _raise_if_cancelled(job_id)

        def _clip_progress(so_far: list, done: int, total: int) -> None:
            pct = 88 + int(done / total * 11)
            try:
                update_job(job_id, status="clipping", progress_pct=pct, clips=so_far)
            except Exception:
                pass

        clips = await loop.run_in_executor(
            None, lambda: clipper.cut_clips(video_path, moments, job_id, progress_cb=_clip_progress)
        )

        _cleanup_audio_artifacts(audio_path)
        audio_path = None

        update_job(job_id, status="done", progress_pct=100, clips=clips)

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
    ttl = 7 * 24 * 3600 if _debug_job(job_id) else 3600
    await asyncio.sleep(ttl)
    job_dir = pathlib.Path(settings.storage_dir) / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)


def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in rows),
        encoding="utf-8",
    )


def _segment_records(transcript) -> list[dict]:
    lines = transcriber.format_jsonl(transcript).splitlines()
    return [json.loads(line) for line in lines if line.strip()]
