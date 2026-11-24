import concurrent.futures
import logging
import pathlib
import threading
from typing import Callable
import ffmpeg
from config import settings
from models import ClipResult

logger = logging.getLogger(__name__)
Moment = dict

_CLIP_MAX_WORKERS = 4

# NVENC availability — probed once against the first real video file.
_nvenc_lock = threading.Lock()
_nvenc_available: bool | None = None


def _probe_nvenc(video_path: pathlib.Path) -> bool:
    try:
        (
            ffmpeg
            .input(str(video_path), ss=0, t=1)
            .output("/dev/null", vcodec="h264_nvenc", preset="fast", acodec="aac", f="null")
            .overwrite_output()
            .run(quiet=True)
        )
        return True
    except Exception:
        return False


def _get_nvenc_available(video_path: pathlib.Path) -> bool:
    global _nvenc_available
    if _nvenc_available is None:
        with _nvenc_lock:
            if _nvenc_available is None:
                _nvenc_available = _probe_nvenc(video_path)
                logger.info("NVENC probe result: %s", "available" if _nvenc_available else "unavailable — using libx264")
    return _nvenc_available


def _video_output_kwargs(video_path: pathlib.Path) -> dict:
    if _get_nvenc_available(video_path):
        return {"vcodec": "h264_nvenc", "preset": "fast", "acodec": "aac", "movflags": "+faststart"}
    return {"vcodec": "libx264", "preset": "ultrafast", "acodec": "aac", "movflags": "+faststart"}


class ClipCuttingError(RuntimeError):
    """Not a single clip of a job could be cut."""


def cut_raw_segment(
    video_path: pathlib.Path, start: float, end: float, output_path: pathlib.Path
) -> None:
    """Cut ``[start, end)`` of ``video_path`` into ``output_path`` as a raw 16:9 clip.

    The same ffmpeg ``ss``/``t`` invocation + :func:`_video_output_kwargs` ``cut_clips``
    uses, but to an EXPLICIT ``output_path`` instead of ``video_path.parent/clips`` — so the
    archetype-retest render adapter (#99) can point two arms at distinct
    ``storage_dir/<arm_job_id>/clips/`` dirs without colliding. No reframe, no packaging, no
    copy: it re-reads the cached video in place.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    (
        ffmpeg
        .input(str(video_path), ss=start, t=end - start)
        .output(str(output_path), **_video_output_kwargs(video_path))
        .overwrite_output()
        .run(quiet=True)
    )


def cut_clips(
    video_path: pathlib.Path,
    moments: list[Moment],
    job_id: str,
    *,
    progress_cb: Callable[[list[ClipResult], int, int], None] | None = None,
) -> list[ClipResult]:
    clips_dir = video_path.parent / "clips"
    clips_dir.mkdir(exist_ok=True)
    min_story_duration = 8.0
    total = len(moments)

    slot: list[ClipResult | None] = [None] * total
    failures: list[str] = []
    done_count = 0
    _lock = threading.Lock()

    def _cut_one(idx: int, moment: Moment) -> None:
        nonlocal done_count
        start = moment["start_time"]
        end = moment["end_time"]
        duration = end - start
        if duration > settings.max_clip_duration:
            end = start + settings.max_clip_duration
        elif 0 < duration < min_story_duration:
            end = start + min(min_story_duration, settings.max_clip_duration)
        elif duration <= 0:
            end = start + min(min_story_duration, settings.max_clip_duration)

        rank = idx + 1
        filename = f"clip_{rank:02d}.mp4"
        output_path = clips_dir / filename
        error: str | None = None
        result: ClipResult | None = None
        try:
            cut_raw_segment(video_path, start, end, output_path)
        except Exception as exc:
            # One broken cut (e.g. missing VOD fragments) must not sink the job:
            # log it and keep the other clips. Survivors keep their rank/filename.
            error = _decode_ffmpeg_error(exc)
            logger.warning(
                "clip_cut_failed job_id=%s index=%s rank=%s start=%s end=%s error=%s",
                job_id, idx, rank, start, end, error,
            )
        else:
            result = ClipResult(
                candidate_id=str(moment.get("candidate_id") or f"clip_{rank:02d}"),
                filename=filename,
                download_url=f"/static/{job_id}/clips/{filename}",
                start_time=start,
                end_time=end,
                viral_score=moment["viral_score"],
                reason=moment["reason"],
                selection_reasons=list(moment.get("selection_reasons") or []),
                ranking_mode=moment.get("ranking_mode", "heuristic"),
                platform=moment.get("platform", "general"),
                variant_key=str(moment.get("variant_key") or "balanced"),
                variant_label=moment.get("variant_label"),
                projected_hook_score=moment.get("projected_hook_score"),
                projected_retention_score=moment.get("projected_retention_score"),
                composite_score=moment.get("composite_score"),
                value_score=moment.get("value_score"),
                trend_score=moment.get("trend_score"),
            )
        with _lock:
            if result is None:
                failures.append(error)
            else:
                slot[idx] = result
            done_count += 1
            d = done_count
            so_far = [r for r in slot if r is not None]
        if progress_cb is not None:
            progress_cb(so_far, d, total)

    with concurrent.futures.ThreadPoolExecutor(max_workers=_CLIP_MAX_WORKERS) as executor:
        futures = {executor.submit(_cut_one, i, m): i for i, m in enumerate(moments)}
        for fut in concurrent.futures.as_completed(futures):
            fut.result()

    if total and len(failures) == total:
        raise ClipCuttingError(
            f"All {total} clips failed to cut (first error: {failures[0]})"
        )
    return [r for r in slot if r is not None]


def _decode_ffmpeg_error(exc: Exception) -> str:
    stderr = getattr(exc, "stderr", b"") or b""
    if isinstance(stderr, bytes):
        message = stderr.decode("utf-8", errors="replace")
    else:
        message = str(stderr)
    return message.strip().splitlines()[-1] if message.strip() else str(exc)
