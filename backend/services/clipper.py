import concurrent.futures
import hashlib
import pathlib
import re
import threading
from typing import Callable
import ffmpeg
from config import settings
from models import ClipResult
Moment = dict

_CLIP_MAX_WORKERS = 4


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
        (
            ffmpeg
            .input(str(video_path), ss=start, t=end - start)
            .output(
                str(output_path),
                vcodec="libx264",
                acodec="aac",
                preset="veryfast",
                movflags="+faststart",
            )
            .overwrite_output()
            .run(quiet=True)
        )
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
        )
        with _lock:
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

    return [r for r in slot if r is not None]


def cut_review_clips(
    video_path: pathlib.Path,
    candidates: list[dict],
    job_id: str,
    limit: int,
) -> list[dict]:
    review_dir = video_path.parent / "review_clips"
    review_dir.mkdir(exist_ok=True)
    results: list[dict] = []

    for index, row in enumerate(candidates[: max(int(limit or 0), 0)], start=1):
        candidate_id = str(row.get("candidate_id") or f"candidate_{index:02d}")
        start = _as_float(row.get("start_time"))
        end = _as_float(row.get("end_time"))
        if end <= start:
            continue
        duration = min(max(end - start, 1.0), float(settings.max_clip_duration))
        filename = f"review_{index:02d}_{_safe_candidate_slug(candidate_id)}.mp4"
        output_path = review_dir / filename

        try:
            (
                ffmpeg
                .input(str(video_path), ss=start, t=duration)
                .output(
                    str(output_path),
                    vcodec="libx264",
                    acodec="aac",
                    preset="veryfast",
                    movflags="+faststart",
                )
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error as exc:
            failed = dict(row)
            failed["preview_error"] = _decode_ffmpeg_error(exc)
            results.append(failed)
            continue

        enriched = dict(row)
        enriched["preview_filename"] = filename
        enriched["preview_url"] = f"/static/{job_id}/review_clips/{filename}"
        results.append(enriched)

    return results


def _safe_candidate_slug(candidate_id: str) -> str:
    compact = re.sub(r"[^0-9A-Za-z]+", "-", candidate_id).strip("-").lower()
    digest = hashlib.sha1(candidate_id.encode("utf-8")).hexdigest()[:10]
    if compact:
        return f"{compact[:24]}-{digest}"
    return digest


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _decode_ffmpeg_error(exc: ffmpeg.Error) -> str:
    stderr = getattr(exc, "stderr", b"") or b""
    if isinstance(stderr, bytes):
        message = stderr.decode("utf-8", errors="replace")
    else:
        message = str(stderr)
    return message.strip().splitlines()[-1] if message.strip() else str(exc)
