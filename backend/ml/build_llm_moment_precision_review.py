from __future__ import annotations

import argparse
import csv
import datetime
import html
import json
import os
import pathlib
import shutil
import subprocess
import sys
from collections import Counter
from typing import Any, Mapping, Sequence

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import run_llm_moment_finder_d0 as d0
from ml import vod_download_preflight


DEFAULT_PROFILE = "jynxzi"
DEFAULT_ARTIFACT_DIR = BACKEND_ROOT / "ml" / "artifacts"
DEFAULT_OUTPUT_PREFIX = DEFAULT_ARTIFACT_DIR / "jynxzi_llm_moment_precision_review"
DEFAULT_MOMENT_PREFIX = DEFAULT_ARTIFACT_DIR / "jynxzi_llm_moment_finder_d0"
DEFAULT_POSITIVE_ROWS = DEFAULT_ARTIFACT_DIR / "jynxzi_cross_platform_fingerprint_rows.jsonl"
DEFAULT_VOD_CACHE_ROOT = BACKEND_ROOT / "tmp" / "vod_feature_cache"
DEFAULT_VIDEO_SOURCE_ROOT = vod_download_preflight.DEFAULT_VIDEO_SOURCE_ROOT
DEFAULT_CLIP_OUTPUT_DIR = DEFAULT_ARTIFACT_DIR / "jynxzi_llm_moment_precision_review_clips"
DEFAULT_TOP_N_PER_VOD = 12
DEFAULT_MAX_DURATION_SEC = 120.0
DEFAULT_EXCERPT_PAD_SEC = 4.0
DEFAULT_EXCERPT_MAX_CHARS = 900
DEFAULT_CLIP_PAD_SEC = 0.0
DEFAULT_MATCH_MODE = "center_or_overlap"
DEFAULT_OVERLAP_RATIO = 0.45


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a human precision review queue from LLM moment-finder outputs.")
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--vod-ids", default="", help="Comma-separated VOD IDs. Defaults to VODs with moment artifacts.")
    parser.add_argument("--moment-run-dir", type=pathlib.Path, default=None, help="LLM moment-finder run directory to use as input.")
    parser.add_argument("--moment-prefix", type=pathlib.Path, default=DEFAULT_MOMENT_PREFIX)
    parser.add_argument("--positive-rows", type=pathlib.Path, default=DEFAULT_POSITIVE_ROWS)
    parser.add_argument("--vod-cache-root", type=pathlib.Path, default=DEFAULT_VOD_CACHE_ROOT)
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--cut-clips", action="store_true", help="Cut raw MP4 review clips from local full-VOD files.")
    parser.add_argument("--download-missing-vods", action="store_true", help="Download missing Twitch VODs before cutting clips. Implied by --cut-clips.")
    parser.add_argument("--video-source-root", type=pathlib.Path, default=DEFAULT_VIDEO_SOURCE_ROOT)
    parser.add_argument(
        "--clip-output-dir",
        type=pathlib.Path,
        default=DEFAULT_CLIP_OUTPUT_DIR,
        help="Root directory for raw clips. Each --cut-clips run writes to a run-specific child directory.",
    )
    parser.add_argument("--clip-run-id", default="", help="Optional deterministic child directory name under --clip-output-dir.")
    parser.add_argument("--clip-pad-sec", type=float, default=DEFAULT_CLIP_PAD_SEC)
    parser.add_argument("--download-cache-ttl-sec", type=int, default=-1)
    parser.add_argument("--ffmpeg-bin", type=pathlib.Path, default=None)
    parser.add_argument("--require-clips", action="store_true", help="Exit non-zero when requested clips cannot all be created. Implied by --cut-clips.")
    parser.add_argument("--top-n-per-vod", type=int, default=DEFAULT_TOP_N_PER_VOD)
    parser.add_argument("--max-duration-sec", type=float, default=DEFAULT_MAX_DURATION_SEC)
    parser.add_argument("--excerpt-pad-sec", type=float, default=DEFAULT_EXCERPT_PAD_SEC)
    parser.add_argument("--excerpt-max-chars", type=int, default=DEFAULT_EXCERPT_MAX_CHARS)
    parser.add_argument("--overlap-ratio", type=float, default=DEFAULT_OVERLAP_RATIO)
    parser.add_argument("--match-mode", choices=sorted(d0.MATCH_MODES), default=DEFAULT_MATCH_MODE)
    args = parser.parse_args(argv)
    moment_prefix = resolve_moment_prefix(args.moment_prefix, args.moment_run_dir)

    try:
        summary = build_review_queue(
            profile=args.profile,
            vod_ids=d0.split_requested_vod_ids(args.vod_ids),
            moment_prefix=moment_prefix,
            moment_run_dir=args.moment_run_dir,
            positive_rows_path=args.positive_rows,
            vod_cache_root=args.vod_cache_root,
            out_base=args.out,
            top_n_per_vod=max(int(args.top_n_per_vod or 0), 1),
            max_duration_sec=max(float(args.max_duration_sec or 0.0), 1.0),
            excerpt_pad_sec=max(float(args.excerpt_pad_sec or 0.0), 0.0),
            excerpt_max_chars=max(int(args.excerpt_max_chars or 0), 120),
            overlap_ratio=float(args.overlap_ratio),
            match_mode=str(args.match_mode),
            cut_clips=bool(args.cut_clips),
            download_missing_vods=bool(args.download_missing_vods or args.cut_clips),
            video_source_root=args.video_source_root,
            clip_output_dir=args.clip_output_dir,
            clip_run_id=str(args.clip_run_id or ""),
            clip_pad_sec=max(float(args.clip_pad_sec or 0.0), 0.0),
            download_cache_ttl_sec=int(args.download_cache_ttl_sec),
            ffmpeg_bin=args.ffmpeg_bin,
            require_clips=bool(args.require_clips),
        )
    except vod_download_preflight.VodDownloadPreflightError as exc:
        summary = {
            "event": "llm_moment_precision_review_preflight_failed",
            "cut_clips": bool(args.cut_clips),
            "download_missing_vods": bool(args.download_missing_vods or args.cut_clips),
            "clip_summary": {
                "requested": bool(args.cut_clips),
                "all_clips_ready": False,
                "download_preflight": exc.summary,
            },
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
        return 2
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    if int(summary.get("queue_rows") or 0) <= 0:
        return 1
    if (args.require_clips or args.cut_clips) and summary.get("clip_summary", {}).get("all_clips_ready") is False:
        return 2
    return 0


def build_review_queue(
    *,
    profile: str,
    vod_ids: Sequence[str],
    moment_prefix: pathlib.Path,
    positive_rows_path: pathlib.Path,
    vod_cache_root: pathlib.Path,
    out_base: pathlib.Path,
    top_n_per_vod: int,
    max_duration_sec: float,
    excerpt_pad_sec: float,
    excerpt_max_chars: int,
    overlap_ratio: float,
    match_mode: str,
    moment_run_dir: pathlib.Path | None = None,
    cut_clips: bool = False,
    download_missing_vods: bool = False,
    video_source_root: pathlib.Path = DEFAULT_VIDEO_SOURCE_ROOT,
    clip_output_dir: pathlib.Path = DEFAULT_CLIP_OUTPUT_DIR,
    clip_run_id: str = "",
    clip_pad_sec: float = DEFAULT_CLIP_PAD_SEC,
    download_cache_ttl_sec: int = -1,
    ffmpeg_bin: pathlib.Path | None = None,
    require_clips: bool = False,
) -> dict[str, Any]:
    selected_vod_ids = list(vod_ids) if vod_ids else discover_vod_ids(moment_prefix, moment_run_dir=moment_run_dir)
    positives_by_vod = d0.positive_windows_by_vod(d0.read_jsonl(positive_rows_path))
    rows: list[dict[str, Any]] = []
    by_vod: dict[str, dict[str, Any]] = {}
    for vod_id in selected_vod_ids:
        moments_path = moment_path_for_vod(moment_prefix, vod_id)
        all_moments = d0.load_moment_rows(moments_path)
        kept_moments, dropped_long = d0.filter_rescore_moment_rows(
            all_moments,
            max_duration_sec=max_duration_sec,
        )
        kept_moments.sort(key=lambda row: int(d0.int_or_none(row.get("rank")) or 1_000_000))
        top_rows = kept_moments[:top_n_per_vod]
        transcript_segments = load_transcript_or_empty(vod_cache_root, vod_id)
        positive_windows = positives_by_vod.get(vod_id, [])
        vod_rows = [
            review_row(
                profile=profile,
                vod_id=vod_id,
                row=row,
                review_rank=index,
                transcript_segments=transcript_segments,
                positive_windows=positive_windows,
                excerpt_pad_sec=excerpt_pad_sec,
                excerpt_max_chars=excerpt_max_chars,
                overlap_ratio=overlap_ratio,
                match_mode=match_mode,
            )
            for index, row in enumerate(top_rows, start=1)
        ]
        rows.extend(vod_rows)
        by_vod[vod_id] = {
            "input_moment_count": len(all_moments),
            "kept_moment_count": len(kept_moments),
            "dropped_long_moment_count": len(dropped_long),
            "queue_rows": len(vod_rows),
            "known_repost_matches_in_queue": sum(1 for row in vod_rows if row["known_repost_match"]),
        }

    out_base.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_base.with_suffix(".csv")
    jsonl_path = out_base.with_suffix(".jsonl")
    html_path = out_base.with_suffix(".html")
    summary_path = out_base.with_suffix(".summary.json")
    clip_summary = initialize_clip_fields(rows)
    resolved_clip_run_id = ""
    resolved_clip_output_dir = clip_output_dir
    if cut_clips:
        resolved_clip_run_id = resolve_clip_run_id(clip_run_id)
        resolved_clip_output_dir = clip_output_dir / resolved_clip_run_id
        effective_download_missing_vods = bool(download_missing_vods or cut_clips)
        preflight_summary = vod_download_preflight.ensure_vod_sources(
            selected_vod_ids,
            video_source_root=video_source_root,
            download_missing=effective_download_missing_vods,
            require_all=bool(require_clips or effective_download_missing_vods),
            cache_ttl_sec=download_cache_ttl_sec,
        )
        clip_summary = cut_review_clips(
            rows,
            video_source_root=video_source_root,
            source_paths_by_vod=source_paths_from_preflight(preflight_summary),
            output_dir=resolved_clip_output_dir,
            html_dir=html_path.parent,
            pad_sec=clip_pad_sec,
            ffmpeg_bin=ffmpeg_bin,
        )
        clip_summary["download_preflight"] = preflight_summary
        clip_summary["clip_output_root"] = str(clip_output_dir)
        clip_summary["clip_run_id"] = resolved_clip_run_id
    write_csv(csv_path, rows)
    write_jsonl(jsonl_path, rows)
    write_html(html_path, rows, title=f"{profile} LLM Moment Precision Review")
    summary = {
        "event": "llm_moment_precision_review_built",
        "profile": profile,
        "vod_ids": selected_vod_ids,
        "vod_count": len(selected_vod_ids),
        "queue_rows": len(rows),
        "top_n_per_vod": int(top_n_per_vod),
        "max_duration_sec": float(max_duration_sec),
        "excerpt_pad_sec": float(excerpt_pad_sec),
        "excerpt_max_chars": int(excerpt_max_chars),
        "cut_clips": bool(cut_clips),
        "download_missing_vods": bool(download_missing_vods or cut_clips),
        "video_source_root": str(video_source_root),
        "clip_output_root": str(clip_output_dir),
        "clip_output_dir": str(resolved_clip_output_dir),
        "clip_run_id": resolved_clip_run_id,
        "clip_pad_sec": float(clip_pad_sec),
        "download_cache_ttl_sec": int(download_cache_ttl_sec),
        "clip_summary": clip_summary,
        "positive_rows_path": str(positive_rows_path),
        "moment_prefix": str(moment_prefix),
        "moment_run_dir": str(moment_run_dir) if moment_run_dir is not None else "",
        "vod_cache_root": str(vod_cache_root),
        "match_mode": d0.normalize_match_mode(match_mode),
        "overlap_ratio": d0.bounded_ratio(overlap_ratio),
        "known_repost_matches_in_queue": sum(1 for row in rows if row["known_repost_match"]),
        "review_required": True,
        "auto_publish_allowed": False,
        "human_label_fields": ["human_viral_label", "human_boundary_label", "human_notes"],
        "by_vod": by_vod,
        "by_known_repost_match": dict(Counter(str(bool(row["known_repost_match"])).lower() for row in rows)),
        "csv_path": str(csv_path),
        "jsonl_path": str(jsonl_path),
        "html_path": str(html_path),
        "summary_path": str(summary_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def discover_vod_ids(moment_prefix: pathlib.Path, *, moment_run_dir: pathlib.Path | None = None) -> list[str]:
    if moment_run_dir is not None:
        manifest_vod_ids = vod_ids_from_manifest(moment_run_dir)
        if manifest_vod_ids:
            return manifest_vod_ids
    pattern = f"{moment_prefix.name}_*.moments.json"
    vod_ids: list[str] = []
    for path in sorted(moment_prefix.parent.glob(pattern)):
        suffix = path.name.removeprefix(f"{moment_prefix.name}_").removesuffix(".moments.json")
        if suffix and suffix not in vod_ids:
            vod_ids.append(suffix)
    return vod_ids


def vod_ids_from_manifest(moment_run_dir: pathlib.Path) -> list[str]:
    manifest_path = moment_run_dir / "manifest.json"
    if not manifest_path.exists():
        return []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(manifest, Mapping):
        return []
    raw_vod_ids = manifest.get("vod_ids")
    if not isinstance(raw_vod_ids, Sequence) or isinstance(raw_vod_ids, str):
        return []
    vod_ids: list[str] = []
    for raw_vod_id in raw_vod_ids:
        vod_id = str(raw_vod_id or "").strip()
        if vod_id and vod_id not in vod_ids:
            vod_ids.append(vod_id)
    return vod_ids


def resolve_moment_prefix(moment_prefix: pathlib.Path, moment_run_dir: pathlib.Path | None) -> pathlib.Path:
    if moment_run_dir is None:
        return moment_prefix
    manifest_path = moment_run_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
        manifest_prefix = str(manifest.get("moment_prefix") or "").strip() if isinstance(manifest, Mapping) else ""
        if manifest_prefix:
            return pathlib.Path(manifest_prefix)
    return moment_run_dir / DEFAULT_MOMENT_PREFIX.name


def moment_path_for_vod(moment_prefix: pathlib.Path, vod_id: str) -> pathlib.Path:
    vod_path = moment_prefix.parent / f"{moment_prefix.name}_{vod_id}.moments.json"
    if vod_path.exists():
        return vod_path
    direct_path = moment_prefix.with_suffix(".moments.json")
    if direct_path.exists():
        return direct_path
    raise FileNotFoundError(f"No LLM moment artifact found for VOD {vod_id}: {vod_path}")


def load_transcript_or_empty(vod_cache_root: pathlib.Path, vod_id: str) -> list[d0.TranscriptSegment]:
    transcript_path = d0.latest_transcript_path(vod_cache_root, vod_id)
    if transcript_path is None:
        return []
    return d0.load_transcript_segments(transcript_path)


def review_row(
    *,
    profile: str,
    vod_id: str,
    row: Mapping[str, Any],
    review_rank: int,
    transcript_segments: Sequence[d0.TranscriptSegment],
    positive_windows: Sequence[d0.PositiveWindow],
    excerpt_pad_sec: float,
    excerpt_max_chars: int,
    overlap_ratio: float,
    match_mode: str,
) -> dict[str, Any]:
    start = float(d0.float_or_none(row.get("start")) or 0.0)
    end = float(d0.float_or_none(row.get("end")) or start)
    rank = int(d0.int_or_none(row.get("rank")) or review_rank)
    moment = d0.LlmMoment(
        start=start,
        end=end,
        hook=d0.score01(row.get("hook")),
        flow=d0.score01(row.get("flow")),
        value=d0.score01(row.get("value")),
        trend=d0.score01(row.get("trend")),
        virality=d0.score01(row.get("virality")),
        rationale=str(row.get("rationale") or ""),
    )
    best = best_positive_match(moment, positive_windows, overlap_ratio=overlap_ratio, match_mode=match_mode)
    return {
        "review_id": f"llm_precision:{vod_id}:rank{review_rank:02d}",
        "profile": profile,
        "vod_id": vod_id,
        "review_rank": review_rank,
        "source_rank": rank,
        "start": round(start, 3),
        "end": round(end, 3),
        "duration": round(max(end - start, 0.0), 3),
        "hook": d0.score01(row.get("hook")),
        "flow": d0.score01(row.get("flow")),
        "value": d0.score01(row.get("value")),
        "trend": d0.score01(row.get("trend")),
        "virality": d0.score01(row.get("virality")),
        "rationale": str(row.get("rationale") or ""),
        "transcript_excerpt": d0.transcript_excerpt(
            transcript_segments,
            start,
            end,
            pad_sec=excerpt_pad_sec,
            max_chars=excerpt_max_chars,
        ),
        "known_repost_match": bool(best),
        "known_repost_positive_id": str(best.get("positive_id") or "") if best else "",
        "known_repost_title": str(best.get("title") or "") if best else "",
        "known_repost_source_url": str(best.get("source_url") or "") if best else "",
        "known_repost_overlap_ratio": best.get("overlap_ratio") if best else "",
        "known_repost_center_inside": best.get("positive_center_inside") if best else "",
        "human_viral_label": "",
        "human_boundary_label": "",
        "human_notes": "",
        "review_required": True,
        "auto_publish_allowed": False,
    }


def best_positive_match(
    moment: d0.LlmMoment,
    positives: Sequence[d0.PositiveWindow],
    *,
    overlap_ratio: float,
    match_mode: str,
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_score = -1.0
    threshold = d0.bounded_ratio(overlap_ratio)
    normalized_match_mode = d0.normalize_match_mode(match_mode)
    for positive in positives:
        ratio = d0.window_overlap_ratio(moment.start, moment.end, positive.start, positive.end)
        center_delta = d0.positive_center_delta_sec(positive, moment)
        center_inside = center_delta <= 0.0
        candidate = {
            "positive_id": positive.positive_id,
            "title": positive.title,
            "source_url": positive.source_url,
            "overlap_ratio": round(ratio, 6),
            "positive_center_inside": bool(center_inside),
            "positive_center_delta_sec": round(center_delta, 3),
        }
        if not d0.moment_match_passes(candidate, threshold=threshold, match_mode=normalized_match_mode):
            continue
        score = max(float(candidate["overlap_ratio"]), 1.0 if center_inside else 0.0)
        if score > best_score:
            best = candidate
            best_score = score
    return best


def initialize_clip_fields(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        row.setdefault("clip_status", "not_requested")
        row.setdefault("clip_path", "")
        row.setdefault("clip_rel_path", "")
        row.setdefault("clip_error", "")
        row.setdefault("source_video_path", "")
    return {
        "requested": False,
        "all_clips_ready": False,
        "clip_count": 0,
        "ready_clip_count": 0,
        "missing_source_count": 0,
        "error_count": 0,
        "missing_source_vods": [],
        "output_dir": "",
    }


def cut_review_clips(
    rows: Sequence[dict[str, Any]],
    *,
    video_source_root: pathlib.Path,
    source_paths_by_vod: Mapping[str, pathlib.Path] | None = None,
    output_dir: pathlib.Path,
    html_dir: pathlib.Path,
    pad_sec: float,
    ffmpeg_bin: pathlib.Path | None = None,
    runner: Any = subprocess.run,
) -> dict[str, Any]:
    initialize_clip_fields(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_path = resolve_ffmpeg_binary(ffmpeg_bin)
    missing_vods: set[str] = set()
    ready_count = 0
    error_count = 0
    missing_count = 0
    for row in rows:
        vod_id = str(row.get("vod_id") or "")
        source_path = (source_paths_by_vod or {}).get(vod_id) or source_video_path(video_source_root, vod_id)
        if source_path is None:
            row["clip_status"] = "missing_source"
            row["clip_error"] = f"Missing full VOD source: {video_source_root / f'vod_{vod_id}.mp4'}"
            missing_vods.add(vod_id)
            missing_count += 1
            continue
        row["source_video_path"] = str(source_path)
        clip_path = output_dir / clip_filename(row)
        try:
            run_ffmpeg_cut(
                ffmpeg_path=ffmpeg_path,
                source_path=source_path,
                output_path=clip_path,
                start_sec=float(row.get("start") or 0.0),
                end_sec=float(row.get("end") or 0.0),
                pad_sec=pad_sec,
                runner=runner,
            )
        except Exception as exc:  # noqa: BLE001 - artifact summaries need the exact failure text.
            row["clip_status"] = "error"
            row["clip_error"] = str(exc)
            error_count += 1
            continue
        row["clip_status"] = "ready"
        row["clip_path"] = str(clip_path)
        row["clip_rel_path"] = relative_path_for_html(clip_path, html_dir)
        ready_count += 1
    return {
        "requested": True,
        "all_clips_ready": ready_count == len(rows),
        "clip_count": len(rows),
        "ready_clip_count": ready_count,
        "missing_source_count": missing_count,
        "error_count": error_count,
        "missing_source_vods": sorted(vod for vod in missing_vods if vod),
        "output_dir": str(output_dir),
        "ffmpeg_bin": str(ffmpeg_path),
    }


def source_video_path(video_source_root: pathlib.Path, vod_id: str) -> pathlib.Path | None:
    return vod_download_preflight.source_video_path(video_source_root, vod_id)


def source_paths_from_preflight(summary: Mapping[str, Any]) -> dict[str, pathlib.Path]:
    by_vod = summary.get("by_vod")
    if not isinstance(by_vod, Mapping):
        return {}
    paths: dict[str, pathlib.Path] = {}
    for vod_id, row in by_vod.items():
        if not isinstance(row, Mapping):
            continue
        path = pathlib.Path(str(row.get("source_video_path") or ""))
        if path.exists() and path.is_file():
            paths[str(vod_id)] = path
    return paths


def resolve_clip_run_id(value: str = "") -> str:
    raw_value = str(value or "").strip()
    explicit = safe_name(raw_value) if raw_value else ""
    if explicit:
        return explicit
    return datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S%fZ")


def resolve_ffmpeg_binary(explicit_path: pathlib.Path | None) -> pathlib.Path:
    if explicit_path is not None:
        return explicit_path
    path = shutil.which("ffmpeg")
    if path:
        return pathlib.Path(path)
    bundled = BACKEND_ROOT / ".venv" / "bin" / "ffmpeg"
    if bundled.exists():
        return bundled
    raise FileNotFoundError("ffmpeg not found on PATH and backend/.venv/bin/ffmpeg is missing")


def clip_filename(row: Mapping[str, Any]) -> str:
    vod_id = safe_name(str(row.get("vod_id") or "vod"))
    rank = int(d0.int_or_none(row.get("review_rank")) or 0)
    start = int(round(float(row.get("start") or 0.0)))
    end = int(round(float(row.get("end") or 0.0)))
    return f"{vod_id}_rank{rank:02d}_{start:06d}_{end:06d}.mp4"


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value).strip("_") or "item"


def run_ffmpeg_cut(
    *,
    ffmpeg_path: pathlib.Path,
    source_path: pathlib.Path,
    output_path: pathlib.Path,
    start_sec: float,
    end_sec: float,
    pad_sec: float,
    runner: Any = subprocess.run,
) -> None:
    clip_start = max(float(start_sec) - float(pad_sec), 0.0)
    clip_end = max(float(end_sec) + float(pad_sec), clip_start + 0.1)
    duration = max(clip_end - clip_start, 0.1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(ffmpeg_path),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{clip_start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(source_path),
        "-map",
        "0:v:0?",
        "-map",
        "0:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    completed = runner(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if completed.returncode != 0:
        error = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(f"ffmpeg failed for {output_path.name}: {error}")
    if not output_path.exists():
        raise RuntimeError(f"ffmpeg reported success but did not create {output_path}")


def relative_path_for_html(path: pathlib.Path, html_dir: pathlib.Path) -> str:
    try:
        return path.resolve().relative_to(html_dir.resolve()).as_posix()
    except ValueError:
        return pathlib.Path(os.path.relpath(path, html_dir)).as_posix()


def write_csv(path: pathlib.Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "review_id",
        "profile",
        "vod_id",
        "review_rank",
        "source_rank",
        "start",
        "end",
        "duration",
        "virality",
        "hook",
        "flow",
        "value",
        "trend",
        "rationale",
        "transcript_excerpt",
        "clip_status",
        "clip_path",
        "clip_rel_path",
        "clip_error",
        "source_video_path",
        "known_repost_match",
        "known_repost_positive_id",
        "known_repost_title",
        "known_repost_source_url",
        "known_repost_overlap_ratio",
        "known_repost_center_inside",
        "human_viral_label",
        "human_boundary_label",
        "human_notes",
        "review_required",
        "auto_publish_allowed",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_jsonl(path: pathlib.Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def write_html(path: pathlib.Path, rows: Sequence[Mapping[str, Any]], *, title: str) -> None:
    cards = []
    for row in rows:
        match_text = "known repost match" if row["known_repost_match"] else "new/non-matching moment"
        clip_block = html_clip_block(row)
        cards.append(
            f"""
<section class="card">
  <header>
    <div class="rank">#{html.escape(str(row['review_rank']))} · VOD {html.escape(str(row['vod_id']))}</div>
    <div class="time">{html.escape(str(row['start']))}s - {html.escape(str(row['end']))}s · {html.escape(str(row['duration']))}s</div>
  </header>
  <div class="scores">virality {html.escape(str(row['virality']))} · hook {html.escape(str(row['hook']))} · flow {html.escape(str(row['flow']))} · value {html.escape(str(row['value']))} · trend {html.escape(str(row['trend']))}</div>
  {clip_block}
  <p class="rationale">{html.escape(str(row['rationale']))}</p>
  <pre>{html.escape(str(row['transcript_excerpt']))}</pre>
  <div class="match">{html.escape(match_text)} {html.escape(str(row['known_repost_title']))}</div>
  <div class="labels">viral: _____ · boundary: _____ · notes: ______________________________</div>
</section>"""
        )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; background: #f7f7f5; color: #1f2933; }}
    h1 {{ font-size: 22px; }}
    .card {{ background: white; border: 1px solid #d8d8d2; border-radius: 8px; padding: 16px; margin: 14px 0; }}
    header {{ display: flex; justify-content: space-between; gap: 16px; font-weight: 700; }}
    .scores, .match, .labels {{ color: #52606d; margin-top: 8px; font-size: 13px; }}
    .rationale {{ margin: 10px 0; }}
    video {{ width: 100%; max-width: 720px; display: block; margin: 12px 0; background: black; }}
    .clip-missing {{ margin: 12px 0; padding: 10px; background: #fff7ed; border: 1px solid #fed7aa; color: #9a3412; border-radius: 6px; font-size: 13px; }}
    pre {{ white-space: pre-wrap; background: #f0f2f4; padding: 10px; border-radius: 6px; font-size: 13px; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p>Review each moment for human viral-worthiness. Known repost matches are recall-floor hints, not target labels.</p>
  {''.join(cards)}
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def html_clip_block(row: Mapping[str, Any]) -> str:
    status = str(row.get("clip_status") or "not_requested")
    if status == "ready":
        src = html.escape(str(row.get("clip_rel_path") or row.get("clip_path") or ""))
        return f'<video controls preload="metadata" src="{src}"></video>'
    if status == "missing_source":
        return (
            '<div class="clip-missing">'
            f"Raw clip missing: {html.escape(str(row.get('clip_error') or 'source video missing'))}"
            "</div>"
        )
    if status == "error":
        return (
            '<div class="clip-missing">'
            f"Raw clip error: {html.escape(str(row.get('clip_error') or 'ffmpeg failed'))}"
            "</div>"
        )
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
