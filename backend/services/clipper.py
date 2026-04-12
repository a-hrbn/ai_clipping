import pathlib
import ffmpeg
from config import settings
from models import ClipResult
from services.analyzer import Moment


def cut_clips(video_path: pathlib.Path, moments: list[Moment], job_id: str) -> list[ClipResult]:
    clips_dir = video_path.parent / "clips"
    clips_dir.mkdir(exist_ok=True)
    min_story_duration = 8.0

    results: list[ClipResult] = []

    for i, moment in enumerate(moments, start=1):
        start = moment["start_time"]
        end = moment["end_time"]

        # Clamp duration to configured limits
        duration = end - start
        if duration > settings.max_clip_duration:
            end = start + settings.max_clip_duration
        elif 0 < duration < min_story_duration:
            end = start + min(min_story_duration, settings.max_clip_duration)
        elif duration <= 0:
            end = start + min(min_story_duration, settings.max_clip_duration)

        filename = f"clip_{i:02d}.mp4"
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

        results.append(ClipResult(
            candidate_id=str(moment.get("candidate_id") or f"clip_{i:02d}"),
            filename=filename,
            download_url=f"/static/{job_id}/clips/{filename}",
            start_time=start,
            end_time=end,
            viral_score=moment["viral_score"],
            reason=moment["reason"],
            ranking_mode=moment.get("ranking_mode", "heuristic"),
        ))

    return results
