from __future__ import annotations

import argparse
import concurrent.futures
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
_BOUNDARY_NEAR_MISS_SAMPLE_WEIGHT = 0.45
_BOUNDARY_NEAR_MISS_LIMIT = 6
_DATASET_BUILD_LLM_PROVIDER = "ollama"
_VOD_CHECKPOINT_VERSION = 2
_BOUNDARY_PERTURBATION_SPECS = (
    ("start_late_small", 0.8, 0.0),
    ("start_late_large", 1.8, 0.0),
    ("start_early_small", -0.8, 0.0),
    ("end_early_small", 0.0, -0.8),
    ("end_early_large", 0.0, -1.8),
    ("end_late_small", 0.0, 1.2),
    ("both_tight", 0.8, -1.0),
    ("both_loose", -1.0, 1.2),
)


def _dataset_build_fingerprint(profile: str) -> str:
    # Fingerprint the code-path knobs that materially change generated rows.
    from services import feature_builder

    payload = {
        "profile": profile,
        "feature_build_version": int(getattr(feature_builder, "FEATURE_BUILD_VERSION", 1)),
        "llm_provider": _DATASET_BUILD_LLM_PROVIDER,
        "supplemental_candidate_limit": _SUPPLEMENTAL_CANDIDATE_LIMIT,
        "supplemental_candidate_weight": _SUPPLEMENTAL_CANDIDATE_WEIGHT,
        "boundary_near_miss_limit": _BOUNDARY_NEAR_MISS_LIMIT,
        "boundary_near_miss_sample_weight": _BOUNDARY_NEAR_MISS_SAMPLE_WEIGHT,
        "boundary_perturbation_specs": list(_BOUNDARY_PERTURBATION_SPECS),
    }
    normalized = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a Twitch VOD-window training dataset from labeled clip windows.")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", default="general")
    parser.add_argument("--work-dir", default=str(BACKEND_ROOT / "tmp" / "training_dataset_build"))
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

    vod_items = list(labels_by_vod.items())
    prefetch_future: concurrent.futures.Future[pathlib.Path] | None = None
    prefetched_vod_id: str | None = None
    prefetched_job_id: str | None = None

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        for position, (vod_id, vod_rows) in enumerate(vod_items):
            job_id = f"twitch_vod_{vod_id}"
            vod_dir = work_dir / job_id
            download_dir = pathlib.Path(settings.storage_dir) / job_id
            prefetched_download_dir: pathlib.Path | None = None
            context_index = _build_vod_context_index(vod_rows)
            vod_checkpoint_rows: list[dict] = []
            vod_dir.mkdir(parents=True, exist_ok=True)
            try:
                checkpoint_rows = _load_vod_checkpoint(work_dir, profile, vod_id, vod_rows)
                if checkpoint_rows is not None:
                    logger.info("training_vod_checkpoint_hit %s", vod_rows[0]["vod_url"])
                    rows.extend(checkpoint_rows)
                    prefetch_future, prefetched_vod_id, prefetched_job_id = _schedule_prefetch(
                        executor,
                        vod_items,
                        position + 1,
                    )
                    continue

                if prefetch_future is not None and prefetched_vod_id == vod_id:
                    try:
                        vod_path = prefetch_future.result()
                        if prefetched_job_id:
                            prefetched_download_dir = pathlib.Path(settings.storage_dir) / prefetched_job_id
                        logger.info("training_vod_prefetch_ready %s", vod_rows[0]["vod_url"])
                    except Exception as exc:
                        logger.warning("training_vod_prefetch_failed %s: %s", vod_rows[0]["vod_url"], exc)
                        logger.info("training_vod_prefetch_fallback %s", vod_rows[0]["vod_url"])
                        vod_path = _materialize_vod(vod_rows[0]["vod_url"], job_id)
                else:
                    vod_path = _materialize_vod(vod_rows[0]["vod_url"], job_id)

                prefetch_future, prefetched_vod_id, prefetched_job_id = _schedule_prefetch(
                    executor,
                    vod_items,
                    position + 1,
                )

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
                            row.setdefault("vod_duration_seconds", float(label.get("vod_duration_seconds", 0.0)))
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
                            vod_checkpoint_rows.append(row)
                    finally:
                        shutil.rmtree(clip_dir, ignore_errors=True)
                if vod_checkpoint_rows:
                    _write_vod_checkpoint(work_dir, profile, vod_id, vod_rows, vod_checkpoint_rows)
            finally:
                shutil.rmtree(vod_dir, ignore_errors=True)
                shutil.rmtree(download_dir, ignore_errors=True)
                if prefetched_download_dir is not None and prefetched_download_dir != download_dir:
                    shutil.rmtree(prefetched_download_dir, ignore_errors=True)
    return rows


def _schedule_prefetch(
    executor: concurrent.futures.ThreadPoolExecutor,
    vod_items: list[tuple[str, list[dict]]],
    next_index: int,
) -> tuple[concurrent.futures.Future[pathlib.Path] | None, str | None, str | None]:
    if next_index >= len(vod_items):
        logger.info("training_vod_prefetch_skip end_of_queue")
        return None, None, None

    next_vod_id, next_rows = vod_items[next_index]
    next_url = str(next_rows[0]["vod_url"])
    if _training_cache_enabled():
        cache_path = _training_cache_path(next_url)
        if cache_path.exists():
            logger.info("training_vod_prefetch_skip cache_hit %s", next_url)
            return None, None, None

    prefetch_job_id = f"twitch_vod_prefetch_{next_vod_id}"
    logger.info("training_vod_prefetch_start %s", next_url)
    return (
        executor.submit(_materialize_vod, next_url, prefetch_job_id),
        next_vod_id,
        prefetch_job_id,
    )


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

    boundary_candidates = _boundary_perturbation_candidates(
        label,
        transcript,
        candidates[0],
        job_prefix,
    )
    all_candidates = list(candidates) + boundary_candidates

    feature_rows = feature_builder.build_features(
        all_candidates,
        transcript,
        profile,
        llm_candidate_ids={candidates[0].id},
        llm_provider=_DATASET_BUILD_LLM_PROVIDER,
    )
    if not feature_rows:
        return []

    rows: list[dict] = []
    primary_feature_rows = feature_rows[: len(candidates)]
    boundary_feature_rows = feature_rows[len(candidates):]
    for idx, (candidate, features) in enumerate(zip(candidates, primary_feature_rows)):
        absolute_candidate = _absolute_candidate(candidate, expanded_start)
        rows.append(
            _training_row_from_label(
                label,
                absolute_candidate,
                features,
                sample_weight_scale=1.0 if idx == 0 else _SUPPLEMENTAL_CANDIDATE_WEIGHT,
                dataset_source="twitch_vod_primary" if idx == 0 else "twitch_vod_candidate_supplemental",
                supervision_reason_suffix="" if idx == 0 else ":candidate_generator",
                window_origin="proxy_label_refined" if idx == 0 else "candidate_generator_overlap",
                reference_window=(absolute_candidate.start, absolute_candidate.end) if idx == 0 else None,
            )
        )

    if boundary_candidates and boundary_feature_rows:
        reference_candidate = _absolute_candidate(candidates[0], expanded_start)
        for candidate, features in zip(boundary_candidates, boundary_feature_rows):
            absolute_candidate = _absolute_candidate(candidate, expanded_start)
            perturbation_kind = next(
                (tag.split(":", 1)[1] for tag in candidate.tags if tag.startswith("boundary_perturbation:")),
                "boundary_near_miss",
            )
            rows.append(
                _training_row_from_label(
                    label,
                    absolute_candidate,
                    features,
                    sample_weight_scale=_BOUNDARY_NEAR_MISS_SAMPLE_WEIGHT,
                    dataset_source="twitch_vod_boundary_near_miss",
                    supervision_reason_suffix=f":boundary_near_miss:{perturbation_kind}",
                    window_origin="boundary_perturbed",
                    label_class_override="boundary_near_miss_negative",
                    label_value_override=0,
                    supervision_tier="boundary_near_miss",
                    rejected_reason_override="boundary_near_miss",
                    reference_window=(reference_candidate.start, reference_candidate.end),
                    perturbation_kind=perturbation_kind,
                )
            )
    return rows


def _materialize_vod(vod_url: str, job_id: str) -> pathlib.Path:
    if not _training_cache_enabled():
        logger.info("training_vod_cache_disabled %s", vod_url)
        return downloader.download(vod_url, job_id)

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


def _checkpoint_dir(work_dir: pathlib.Path) -> pathlib.Path:
    return work_dir / "_vod_checkpoints"


def _vod_checkpoint_path(work_dir: pathlib.Path, vod_id: str) -> pathlib.Path:
    return _checkpoint_dir(work_dir) / f"{vod_id}.json"


def _vod_checkpoint_signature(vod_rows: list[dict]) -> str:
    normalized = json.dumps(
        sorted(vod_rows, key=lambda row: (float(row.get("window_start") or 0.0), float(row.get("window_end") or 0.0), str(row.get("label_class") or ""))),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _load_vod_checkpoint(
    work_dir: pathlib.Path,
    profile: str,
    vod_id: str,
    vod_rows: list[dict],
) -> list[dict] | None:
    path = _vod_checkpoint_path(work_dir, vod_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if int(payload.get("version") or 0) != _VOD_CHECKPOINT_VERSION:
        return None
    if str(payload.get("profile") or "") != profile:
        return None
    if str(payload.get("vod_id") or "") != vod_id:
        return None
    if str(payload.get("dataset_build_fingerprint") or "") != _dataset_build_fingerprint(profile):
        return None
    if str(payload.get("label_signature") or "") != _vod_checkpoint_signature(vod_rows):
        return None
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return None
    return rows


def _write_vod_checkpoint(
    work_dir: pathlib.Path,
    profile: str,
    vod_id: str,
    vod_rows: list[dict],
    rows: list[dict],
) -> None:
    path = _vod_checkpoint_path(work_dir, vod_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": _VOD_CHECKPOINT_VERSION,
        "profile": profile,
        "vod_id": vod_id,
        "dataset_build_fingerprint": _dataset_build_fingerprint(profile),
        "label_signature": _vod_checkpoint_signature(vod_rows),
        "rows": rows,
    }
    path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


def _training_cache_enabled() -> bool:
    return max(int(settings.training_vod_cache_ttl_sec), 0) > 0


def _training_cache_dir() -> pathlib.Path:
    configured = str(settings.training_vod_cache_dir or "").strip()
    if configured:
        return pathlib.Path(configured)
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


def _absolute_candidate(candidate, expanded_start: float):
    from services.timeline_types import Candidate

    return Candidate(
        id=candidate.id,
        start=round(expanded_start + candidate.start, 2),
        end=round(expanded_start + candidate.end, 2),
        source=candidate.source,
        anchor_time=round(expanded_start + candidate.anchor_time, 2),
        score_hint=candidate.score_hint,
        tags=list(candidate.tags),
    )


def _boundary_perturbation_candidates(
    label: dict,
    transcript: list,
    reference_candidate,
    job_prefix: str,
):
    from services import boundary_candidates
    from services.timeline_types import Candidate

    label_class = str(label.get("label_class") or "")
    if label_class not in _POSITIVE_LABEL_CLASSES or not transcript:
        return []

    points = boundary_candidates.build_boundary_points(
        transcript,
        reference_candidate,
        context_lookback_sec=8.0,
        context_lookahead_sec=8.0,
    )
    times = sorted({round(point.time, 2) for point in points})
    if len(times) < 2:
        return []

    generated: list[Candidate] = []
    seen: set[tuple[float, float]] = set()
    for rank, (kind, start_offset, end_offset) in enumerate(_BOUNDARY_PERTURBATION_SPECS):
        start = _snap_perturbed_time(times, reference_candidate.start, start_offset, lower_bound=0.0)
        end = _snap_perturbed_time(times, reference_candidate.end, end_offset, lower_bound=reference_candidate.anchor_time)
        if start is None or end is None:
            continue
        if end - start < 6.0:
            continue
        if not (start < reference_candidate.anchor_time < end):
            continue
        key = (round(start, 2), round(end, 2))
        if key in seen:
            continue
        total_error = abs(start - reference_candidate.start) + abs(end - reference_candidate.end)
        if total_error < 0.75:
            continue
        seen.add(key)
        generated.append(
            Candidate(
                id=f"{job_prefix}:boundary:{rank}:{int(start * 100)}",
                start=round(start, 2),
                end=round(end, 2),
                source="boundary_perturbation",
                anchor_time=round(reference_candidate.anchor_time, 2),
                score_hint=max(float(reference_candidate.score_hint) * 0.35, 0.1),
                tags=["boundary_near_miss", f"boundary_perturbation:{kind}"],
            )
        )
        if len(generated) >= _BOUNDARY_NEAR_MISS_LIMIT:
            break
    return generated


def _snap_perturbed_time(
    boundaries: list[float],
    reference_time: float,
    offset: float,
    *,
    lower_bound: float,
) -> float | None:
    target = reference_time + offset
    if offset > 0:
        candidates = [boundary for boundary in boundaries if boundary >= max(target, lower_bound)]
        return candidates[0] if candidates else None
    if offset < 0:
        candidates = [boundary for boundary in boundaries if lower_bound <= boundary <= target]
        return candidates[-1] if candidates else None
    candidates = [boundary for boundary in boundaries if boundary >= lower_bound]
    if not candidates:
        return None
    return min(candidates, key=lambda value: abs(value - reference_time))


def _training_row_from_label(
    label: dict,
    candidate,
    features,
    *,
    sample_weight_scale: float,
    dataset_source: str,
    supervision_reason_suffix: str,
    window_origin: str,
    label_class_override: str | None = None,
    label_value_override: int | None = None,
    supervision_tier: str = "twitch_clip_proxy",
    rejected_reason_override: str | None = None,
    reference_window: tuple[float, float] | None = None,
    perturbation_kind: str | None = None,
) -> dict:
    label_class = str(label_class_override or label.get("label_class") or "")
    if label_value_override is None:
        is_positive = label_class in _POSITIVE_LABEL_CLASSES
        label_value = 1 if is_positive else 0
    else:
        label_value = int(label_value_override)
        is_positive = label_value == 1
    base_reason = str(label.get("label_source") or "twitch_clip_api")
    row = {
        "job_id": f"twitch_vod_{label['vod_id']}",
        "candidate_id": features.candidate_id,
        "label": label_value,
        "selected": is_positive,
        "exported": True,
        "final_start": candidate.start if is_positive else None,
        "final_end": candidate.end if is_positive else None,
        "rejected_reason": None if is_positive else str(rejected_reason_override or label.get("negative_source") or "twitch_vod_negative"),
        "rejection_source": None if is_positive else str(rejected_reason_override or label.get("negative_source") or "twitch_vod_negative"),
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
        "supervision_tier": supervision_tier,
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
    if reference_window is not None:
        ref_start, ref_end = reference_window
        start_error = round(candidate.start - ref_start, 2)
        end_error = round(candidate.end - ref_end, 2)
        row["boundary_reference_start"] = round(ref_start, 2)
        row["boundary_reference_end"] = round(ref_end, 2)
        row["boundary_error_start_sec"] = start_error
        row["boundary_error_end_sec"] = end_error
        row["boundary_abs_start_error_sec"] = round(abs(start_error), 2)
        row["boundary_abs_end_error_sec"] = round(abs(end_error), 2)
        row["boundary_abs_total_error_sec"] = round(abs(start_error) + abs(end_error), 2)
    if perturbation_kind:
        row["boundary_perturbation_kind"] = perturbation_kind
    return row


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
