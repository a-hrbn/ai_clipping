from __future__ import annotations

import pathlib
from dataclasses import asdict, dataclass

import ffmpeg

from config import settings
from services.timeline_types import RankedCandidate


@dataclass
class SampledKeyframe:
    candidate_id: str
    label: str
    time_sec: float
    path: str
    mime_type: str = "image/jpeg"

    def to_dict(self) -> dict:
        return asdict(self)


def sample_candidate_keyframes(
    video_path: pathlib.Path,
    ranked_candidates: list[RankedCandidate],
    debug_dir: pathlib.Path,
    per_candidate: int | None = None,
) -> dict[str, list[SampledKeyframe]]:
    frames_dir = debug_dir / "judge_keyframes"
    frames_dir.mkdir(parents=True, exist_ok=True)
    desired = max(1, int(per_candidate or settings.judge_keyframes_per_candidate))
    sampled: dict[str, list[SampledKeyframe]] = {}

    for ranked in ranked_candidates:
        candidate = ranked.candidate
        candidate_dir = frames_dir / _safe_id(candidate.id)
        candidate_dir.mkdir(parents=True, exist_ok=True)
        frames: list[SampledKeyframe] = []
        for label, time_sec in _frame_times_for_candidate(candidate.start, candidate.end, candidate.anchor_time, desired):
            output_path = candidate_dir / f"{label}.jpg"
            try:
                _extract_frame(video_path, time_sec, output_path)
            except Exception:
                continue
            frames.append(
                SampledKeyframe(
                    candidate_id=candidate.id,
                    label=label,
                    time_sec=round(time_sec, 3),
                    path=str(output_path),
                )
            )
        sampled[candidate.id] = frames

    return sampled


def _frame_times_for_candidate(
    start: float,
    end: float,
    anchor_time: float,
    count: int,
) -> list[tuple[str, float]]:
    if count <= 0:
        return []

    duration = max(end - start, 0.5)
    clamped_anchor = min(max(anchor_time, start), end)
    nominal = [
        ("start", start + min(0.4, duration * 0.08)),
        ("hook", start + min(1.8, duration * 0.18)),
        ("anchor", clamped_anchor),
        ("payoff", min(end - 0.15, max(clamped_anchor + max(duration * 0.18, 0.8), start))),
    ]
    trimmed = nominal[:count]
    deduped: list[tuple[str, float]] = []
    used: list[float] = []
    for label, time_sec in trimmed:
        safe_time = min(max(time_sec, start), max(start, end - 0.05))
        if any(abs(safe_time - existing) < 0.15 for existing in used):
            continue
        deduped.append((label, safe_time))
        used.append(safe_time)
    return deduped


def _extract_frame(video_path: pathlib.Path, time_sec: float, output_path: pathlib.Path) -> None:
    stream = ffmpeg.input(str(video_path), ss=max(time_sec, 0.0))
    width = max(int(settings.judge_keyframe_width), 0)
    if width > 0:
        stream = stream.filter("scale", width, -2)
    (
        stream
        .output(str(output_path), vframes=1, format="image2", vcodec="mjpeg", qscale=4)
        .overwrite_output()
        .run(quiet=True)
    )


def _safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
