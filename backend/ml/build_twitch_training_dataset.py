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

_POSITIVE_LABEL_CLASSES = {"viral_positive", "positive", "weak_positive"}
_SUPPLEMENTAL_CANDIDATE_LIMIT = 2
_SUPPLEMENTAL_CANDIDATE_WEIGHT = 0.65


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
                        label_rows = _build_rows_for_label(
                            label,
                            vod_path=vod_path,
                            clip_dir=clip_dir,
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
                        continue
                    for row in label_rows:
                        row.setdefault("label_class", str(label.get("label_class", "")))
                        row.setdefault("label_score", float(label.get("label_score", 0.0)))
                        row.setdefault("platform_source", label.get("platform_source", "twitch_clip"))
                        row.setdefault("label_source", label.get("label_source", "twitch_clip_api"))
                        row.setdefault("window_start", float(label.get("window_start", 0.0)))
                        row.setdefault("window_end", float(label.get("window_end", 0.0)))
                        row.setdefault("vod_id", str(label.get("vod_id", "")))
                        row.setdefault("clip_ids", list(label.get("clip_ids") or []))
                        row.setdefault("sample_weight", sample_weight_for_label(str(row.get("label_class", ""))))
                        row.setdefault("dataset_source", "twitch_vod_primary")
                        row.setdefault("supervision_tier", "twitch_clip_proxy")
                        row.setdefault("supervision_reason", str(row.get("label_source", "twitch_clip_api")))
                        row.setdefault("ordinal_label", collect_viral_clips._ordinal_label(str(row.get("label_class", "")), int(row.get("label", 0))))
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


def _build_rows_for_label(
    label: dict,
    *,
    vod_path: pathlib.Path,
    clip_dir: pathlib.Path,
    job_prefix: str,
    profile: str,
) -> list[dict]:
    from services import (
        audio_extractor, audio_features, feature_builder, scorer, transcriber, video_features,
    )
    from services.timeline_types import Candidate

    expanded_start, expanded_end = _expanded_training_window(label)
    clip_path = collect_viral_clips.cut_video_window(
        vod_path,
        clip_dir / "window.mp4",
        expanded_start,
        expanded_end,
    )
    audio_path = audio_extractor.extract(clip_path, job_prefix)
    transcript = transcriber.transcribe(audio_path)
    if not transcript:
        return []

    audio_frames = audio_features.extract(audio_path)
    visual_frames = video_features.extract(clip_path)
    transcript = audio_features.enrich_segments(transcript, audio_frames)
    transcript = video_features.enrich_segments(transcript, visual_frames)
    transcript = scorer.score_segments(transcript)
    if transcript[-1].end - transcript[0].start < 5.0:
        return []

    proxy_candidate = _proxy_candidate(label, expanded_start, job_prefix)
    candidates = _refined_candidates_for_label(label, transcript, proxy_candidate, job_prefix)
    if not candidates:
        return []

    feature_rows = feature_builder.build_features(
        candidates,
        transcript,
        profile,
        llm_candidate_ids={candidates[0].id},
        llm_provider="ollama",
    )
    if not feature_rows:
        return []

    rows: list[dict] = []
    for idx, (candidate, features) in enumerate(zip(candidates, feature_rows)):
        absolute_candidate = Candidate(
            id=candidate.id,
            start=round(expanded_start + candidate.start, 2),
            end=round(expanded_start + candidate.end, 2),
            source=candidate.source,
            anchor_time=round(expanded_start + candidate.anchor_time, 2),
            score_hint=candidate.score_hint,
            tags=list(candidate.tags),
        )
        rows.append(
            _training_row_from_label(
                label,
                absolute_candidate,
                features,
                sample_weight_scale=1.0 if idx == 0 else _SUPPLEMENTAL_CANDIDATE_WEIGHT,
                dataset_source="twitch_vod_primary" if idx == 0 else "twitch_vod_candidate_supplemental",
                supervision_reason_suffix="" if idx == 0 else ":candidate_generator",
                window_origin="proxy_label_refined" if idx == 0 else "candidate_generator_overlap",
            )
        )
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


def _training_row_from_label(
    label: dict,
    candidate,
    features,
    *,
    sample_weight_scale: float,
    dataset_source: str,
    supervision_reason_suffix: str,
    window_origin: str,
) -> dict:
    label_class = str(label.get("label_class") or "")
    is_positive = label_class in _POSITIVE_LABEL_CLASSES
    label_value = 1 if is_positive else 0
    base_reason = str(label.get("label_source") or "twitch_clip_api")
    return {
        "job_id": f"twitch_vod_{label['vod_id']}",
        "candidate_id": features.candidate_id,
        "label": label_value,
        "selected": is_positive,
        "exported": True,
        "final_start": candidate.start if is_positive else None,
        "final_end": candidate.end if is_positive else None,
        "rejected_reason": None if is_positive else str(label.get("negative_source") or "twitch_vod_negative"),
        "rejection_source": None if is_positive else str(label.get("negative_source") or "twitch_vod_negative"),
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
        "supervision_tier": "twitch_clip_proxy",
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
