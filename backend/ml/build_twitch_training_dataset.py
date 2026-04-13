from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pathlib
import shutil
import time
import sys

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config import settings
from ml import collect_viral_clips
from services import downloader

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a Twitch VOD-window training dataset from labeled clip windows.")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", default="general")
    parser.add_argument("--work-dir", default="/tmp/ai_clipping_twitch_training")
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
    rows: list[dict] = []
    work_dir.mkdir(parents=True, exist_ok=True)
    labels_by_vod: dict[str, list[dict]] = {}
    for row in labels:
        if row.get("label_class") == "ambiguous":
            continue
        labels_by_vod.setdefault(str(row["vod_id"]), []).append(row)

    for vod_id, vod_rows in labels_by_vod.items():
        job_id = f"twitch_vod_{vod_id}"
        vod_dir = work_dir / job_id
        download_dir = pathlib.Path(settings.storage_dir) / job_id
        context_index = _build_vod_context_index(vod_rows)
        vod_dir.mkdir(parents=True, exist_ok=True)
        try:
            vod_path = _materialize_vod(vod_rows[0]["vod_url"], job_id)
            for index, label in enumerate(sorted(vod_rows, key=lambda item: float(item["window_start"])), start=1):
                clip_dir = vod_dir / f"window_{index:04d}"
                clip_dir.mkdir(parents=True, exist_ok=True)
                try:
                    try:
                        clip_path = collect_viral_clips.cut_video_window(
                            vod_path,
                            clip_dir / "window.mp4",
                            float(label["window_start"]),
                            float(label["window_end"]),
                        )
                        entry = _entry_from_label(label)
                        row = collect_viral_clips.process_local_video(entry, f"{job_id}_{index}", clip_dir, clip_path, profile)
                    except Exception as exc:
                        logger.warning(
                            "Skipping VOD window %s[%s] %.2f-%.2f: %s",
                            vod_id,
                            index,
                            float(label["window_start"]),
                            float(label["window_end"]),
                            exc,
                        )
                        continue
                    if row:
                        row["label_class"] = label["label_class"]
                        row["label_score"] = float(label.get("label_score", 0.0))
                        row["platform_source"] = label.get("platform_source", "twitch_clip")
                        row["window_start"] = float(label["window_start"])
                        row["window_end"] = float(label["window_end"])
                        row["vod_id"] = str(label["vod_id"])
                        row["clip_ids"] = list(label.get("clip_ids") or [])
                        row["label_source"] = label.get("label_source", "twitch_clip_api")
                        row["sample_weight"] = sample_weight_for_label(row["label_class"])
                        row["dataset_source"] = "twitch_vod_primary"
                        row["supervision_tier"] = "twitch_clip_proxy"
                        row["supervision_reason"] = row["label_source"]
                        row["ordinal_label"] = collect_viral_clips._ordinal_label(row["label_class"], int(row["label"]))
                        _apply_vod_context_features(row, label, context_index)
                        row["feature_provenance"] = collect_viral_clips._feature_provenance(
                            row["features"],
                            has_vod_context=True,
                        )
                        rows.append(row)
                finally:
                    shutil.rmtree(clip_dir, ignore_errors=True)
        finally:
            shutil.rmtree(vod_dir, ignore_errors=True)
            shutil.rmtree(download_dir, ignore_errors=True)
    return rows


def _materialize_vod(vod_url: str, job_id: str) -> pathlib.Path:
    cache_dir = _training_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    _evict_expired_training_cache(cache_dir)
    cache_path = _training_cache_path(vod_url)
    if cache_path.exists():
        logger.info("training_vod_cache_hit  %s", vod_url)
        return cache_path

    logger.info("training_vod_cache_miss %s", vod_url)
    downloaded = downloader.download(vod_url, job_id)
    try:
        shutil.copy2(downloaded, cache_path)
        logger.info("training_vod_cache_write %s", cache_path)
    except OSError:
        pass
    return cache_path if cache_path.exists() else downloaded


def _training_cache_dir() -> pathlib.Path:
    return pathlib.Path(settings.storage_dir) / "_download_cache" / "training_vods"


def _training_cache_path(vod_url: str) -> pathlib.Path:
    digest = hashlib.sha256(vod_url.strip().encode("utf-8")).hexdigest()
    return _training_cache_dir() / f"{digest}.mp4"


def _evict_expired_training_cache(cache_dir: pathlib.Path) -> None:
    ttl = max(int(settings.training_vod_cache_ttl_sec), 0)
    if ttl <= 0 or not cache_dir.exists():
        return

    cutoff = time.time() - ttl
    for path in cache_dir.glob("*.mp4"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
        except OSError:
            continue


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
    for row in rows:
        by_class[row.get("label_class", "")] = by_class.get(row.get("label_class", ""), 0) + 1
        if row.get("vod_id"):
            vods.add(str(row["vod_id"]))
    return {
        "rows": len(rows),
        "label_classes": by_class,
        "vod_count": len(vods),
    }


def _entry_from_label(label: dict) -> dict[str, str]:
    label_class = str(label.get("label_class", ""))
    is_positive = label_class in {"viral_positive", "positive", "weak_positive"}
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
    start = float(label.get("window_start") or 0.0)
    end = float(label.get("window_end") or 0.0)
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
