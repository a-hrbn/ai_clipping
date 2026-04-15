from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
import uuid

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import collect_viral_clips

_HOOK_SECONDS = 6.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Enrich vector reference rows with real clip transcripts.")
    parser.add_argument("--dataset", default=str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_training_rows.jsonl"))
    parser.add_argument("--output", default=str(BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_vector_manifest_enriched.jsonl"))
    parser.add_argument("--profile", default="jynxzi")
    parser.add_argument("--work-dir", default=str(BACKEND_ROOT / "tmp" / "vector_refs"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    rows = _read_jsonl(pathlib.Path(args.dataset))
    if args.limit > 0:
        rows = rows[: args.limit]

    work_dir = pathlib.Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    enriched: list[dict] = []
    for row in rows:
        updated = enrich_reference_row(row, work_dir, args.profile, force=args.force)
        if updated is not None:
            enriched.append(updated)

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in enriched),
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(enriched), "output": str(output_path)}, ensure_ascii=True))
    return 0


def enrich_reference_row(
    row: dict,
    work_dir: pathlib.Path,
    profile: str,
    *,
    force: bool = False,
) -> dict | None:
    if not force and str(row.get("transcript") or "").strip():
        return row

    source_url = str(row.get("source_url") or row.get("url") or "").strip()
    if not source_url:
        return None

    job_id = f"vector_ref_{uuid.uuid4().hex[:10]}"
    clip_dir = work_dir / job_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    try:
        segments = _transcribe_reference_clip(row, job_id, clip_dir, profile, source_url)
        if not segments:
            return None
        transcript_fields = build_transcript_fields(segments)
        enriched = dict(row)
        enriched.update(transcript_fields)
        enriched["transcript_segment_count"] = len(segments)
        return enriched
    finally:
        shutil.rmtree(clip_dir, ignore_errors=True)


def _transcribe_reference_clip(
    row: dict,
    job_id: str,
    clip_dir: pathlib.Path,
    profile: str,
    source_url: str,
):
    from services import (
        audio_extractor,
        audio_features,
        downloader,
        scorer,
        transcriber,
        video_features,
    )

    video_path = downloader.download(source_url, job_id)
    clip_path = _materialize_reference_clip(row, clip_dir, video_path)
    audio_path = audio_extractor.extract(clip_path, job_id)
    transcript = transcriber.transcribe(audio_path)
    if not transcript:
        return []

    audio_frames = audio_features.extract(audio_path)
    visual_frames = video_features.extract(clip_path)
    transcript = audio_features.enrich_segments(transcript, audio_frames)
    transcript = video_features.enrich_segments(transcript, visual_frames)
    transcript = scorer.score_segments(transcript)
    return transcript


def _materialize_reference_clip(row: dict, clip_dir: pathlib.Path, video_path: pathlib.Path) -> pathlib.Path:
    start, end = _clip_window(row)
    if end > start and (start > 0.0 or row.get("vod_id")):
        output_path = clip_dir / "reference_window.mp4"
        return collect_viral_clips.cut_video_window(video_path, output_path, start, end)
    return video_path


def _clip_window(row: dict) -> tuple[float, float]:
    start = _as_float(row.get("final_start"))
    end = _as_float(row.get("final_end"))
    if end > start:
        return start, end
    start = _as_float(row.get("window_start"))
    end = _as_float(row.get("window_end"))
    if end > start:
        return start, end
    return 0.0, 0.0


def build_transcript_fields(segments) -> dict[str, str]:
    full_text = " ".join(seg.text.strip() for seg in segments if seg.text.strip()).strip()
    if not full_text:
        return {"transcript": "", "hook_text": "", "payoff_text": ""}

    start_time = segments[0].start
    hook_segments = [seg for seg in segments if seg.end <= start_time + _HOOK_SECONDS]
    if not hook_segments:
        hook_segments = segments[:2]
    anchor_idx = max(range(len(segments)), key=lambda idx: getattr(segments[idx], "hype_score", 0.0))
    payoff_segments = segments[max(0, anchor_idx - 1): min(len(segments), anchor_idx + 2)]
    if not payoff_segments:
        payoff_segments = segments[-2:]

    hook_text = " ".join(seg.text.strip() for seg in hook_segments if seg.text.strip()).strip()
    payoff_text = " ".join(seg.text.strip() for seg in payoff_segments if seg.text.strip()).strip()
    return {
        "transcript": full_text,
        "hook_text": hook_text or full_text,
        "payoff_text": payoff_text or full_text,
    }


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    raise SystemExit(main())
