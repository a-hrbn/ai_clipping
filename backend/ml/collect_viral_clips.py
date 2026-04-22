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
import csv
import json
import logging
import pathlib
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import uuid

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

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
    from services import downloader

    logger.info("Processing %d URLs…", len(entries))
    work_dir.mkdir(parents=True, exist_ok=True)

    collected_rows: list[dict] = []
    failed = 0
    skipped_unavailable = 0
    skipped_platforms = {item.strip().lower() for item in (skip_platforms or set()) if item.strip()}

    for i, entry in enumerate(entries, 1):
        progress = _render_progress_bar(i - 1, len(entries), failed)
        logger.info("%s", progress)
        logger.info("[%d/%d] %s", i, len(entries), entry["url"])
        job_id = f"viral_{uuid.uuid4().hex[:8]}"
        clip_dir = work_dir / job_id
        started_at = time.perf_counter()
        platform = _infer_platform(entry)
        if platform in skipped_platforms:
            logger.warning("  ↷ skipped_unavailable: platform %s explicitly skipped", platform)
            skipped_unavailable += 1
            continue
        network_attempt = 0
        while True:
            clip_dir.mkdir(parents=True, exist_ok=True)
            try:
                row = _process_clip(entry, job_id, clip_dir, profile, proxy=proxy)
                if row:
                    collected_rows.append(row)
                    elapsed = time.perf_counter() - started_at
                    logger.info(
                        "  ✓ features extracted  (label=%s hook=%.2f  hype=%.2f  arc=%.2f  %.1fs)",
                        row["label"],
                        row["features"]["hook_strength"],
                        row["features"]["max_hype"],
                        row["features"]["arc_score"],
                        elapsed,
                    )
                else:
                    logger.warning("  ✗ skipped (no transcript or too short)")
                    failed += 1
                break
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
                skipped_unavailable += 1
                break
            except Exception as exc:
                logger.error("  ✗ failed: %s", exc)
                failed += 1
                break
            finally:
                shutil.rmtree(clip_dir, ignore_errors=True)

    positives = sum(1 for row in collected_rows if row["label"] == 1)
    negatives = sum(1 for row in collected_rows if row["label"] == 0)
    logger.info("%s", _render_progress_bar(len(entries), len(entries), failed))
    logger.info(
        "Collected %d labeled rows (%d positive, %d negative, %d failed, %d skipped_unavailable)",
        len(collected_rows),
        positives,
        negatives,
        failed,
        skipped_unavailable,
    )
    return collected_rows


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
) -> dict | None:
    from services import (
        downloader, audio_extractor, transcriber,
        audio_features, video_features, scorer, feature_builder,
    )
    from services.timeline_types import Candidate

    url = entry["url"]

    # 1. Download
    video_path = downloader.download(url, job_id, proxy=proxy)

    # 2. Extract audio
    audio_path = audio_extractor.extract(video_path, job_id)

    # 3. Transcribe
    transcript = transcriber.transcribe(audio_path)
    if not transcript:
        return None

    # 4. Audio + video features (parallel via threads would be nice, but keep simple)
    audio_frames = audio_features.extract(audio_path)
    visual_frames = video_features.extract(video_path)

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

    audio_path = audio_extractor.extract(video_path, job_id)
    transcript = transcriber.transcribe(audio_path)
    if not transcript:
        return None

    audio_frames = audio_features.extract(audio_path)
    visual_frames = video_features.extract(video_path)
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
    )
    if not feats_list:
        return None
    feats = feats_list[0]
    feats.chat_peak = 0.0
    feats.context_cost = 0.0
    return _build_training_row(entry, candidate, feats)


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

    for chunk in _window_batches(normalized_windows, _MAX_BATCH_CUT_WINDOWS):
        _cut_video_windows_chunk(video_path, chunk)
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
) -> None:
    command = ["ffmpeg", "-y", "-nostdin", "-loglevel", "error"]
    for _output_path, start_sec, duration in normalized_windows:
        command.extend([
            "-ss", f"{start_sec:.3f}",
            "-t", f"{duration:.3f}",
            "-i", str(video_path),
        ])

    for input_index, (output_path, _start_sec, _duration) in enumerate(normalized_windows):
        command.extend([
            "-map", f"{input_index}:v:0",
            "-map", f"{input_index}:a:0?",
            "-threads", "1",
            "-c:v", "libx264",
            "-c:a", "aac",
            "-preset", "veryfast",
            "-movflags", "+faststart",
            str(output_path),
        ])

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
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
) -> None:
    try:
        _cut_video_windows_batch(video_path, normalized_windows)
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
    _cut_video_windows_chunk(video_path, left)
    _cut_video_windows_chunk(video_path, right)


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
