from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import Counter

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml.row_identity import read_jsonl, write_json, write_jsonl
from services import candidate_generators, clip_strategy, proposal_scorer
from services.transcriber import RichSegment

DEFAULT_ARTIFACT_DIR = BACKEND_ROOT / "ml" / "artifacts"
DEFAULT_CROSS_PLATFORM_ROWS = DEFAULT_ARTIFACT_DIR / "jynxzi_cross_platform_fingerprint_rows.jsonl"
DEFAULT_OUTPUT = DEFAULT_ARTIFACT_DIR / "jynxzi_proposal_training_rows.jsonl"
DEFAULT_SUMMARY = DEFAULT_ARTIFACT_DIR / "jynxzi_proposal_training_summary.json"
_DENSE_RECALL_POSITIVE_WEIGHT = 4.0
_CROSS_PLATFORM_POSITIVE_WEIGHT = 2.0
DEFAULT_STORAGE_DIR = BACKEND_ROOT / "tmp" / "storage"
_VOD_URL_RE = re.compile(r"twitch\.tv/videos/(\d+)")
_GOLD_SOURCE_TOKENS = ("gold_eval", "gold-eval", "frontend_gold")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build proposal-model training rows from cross-platform positives and VOD segment JSONL files."
    )
    parser.add_argument("--profile", default="jynxzi")
    parser.add_argument("--cross-platform-rows", type=pathlib.Path, default=DEFAULT_CROSS_PLATFORM_ROWS)
    parser.add_argument(
        "--holdout-policy-rows",
        type=pathlib.Path,
        default=None,
        help="Optional JSONL policy rows; VOD ids marked holdout are excluded from proposal training.",
    )
    parser.add_argument("--storage-dir", type=pathlib.Path, default=DEFAULT_STORAGE_DIR)
    parser.add_argument("--segments", action="append", type=pathlib.Path, default=[])
    parser.add_argument(
        "--existing-feature-rows",
        action="append",
        type=pathlib.Path,
        default=[],
        help=(
            "Optional legacy VOD-ranker feature rows to translate into proposal training rows. "
            "This is off by default because translated ranker features do not match serving-time proposal features."
        ),
    )
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=pathlib.Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--durations-sec", default="12,20,30,45,60")
    parser.add_argument("--stride-sec", type=float, default=5.0)
    parser.add_argument("--max-candidates-per-vod", type=int, default=5000)
    parser.add_argument("--positive-overlap-ratio", type=float, default=0.45)
    parser.add_argument("--near-miss-sec", type=float, default=180.0)
    parser.add_argument("--background-negatives-per-positive", type=int, default=4)
    parser.add_argument("--max-background-negatives-per-vod", type=int, default=120)
    args = parser.parse_args(argv)

    holdout_vod_ids = _holdout_vod_ids(args.holdout_policy_rows)
    positive_index = _positive_window_index(read_jsonl(args.cross_platform_rows), excluded_vod_ids=holdout_vod_ids)
    segment_sources = _discover_segment_sources(args.storage_dir, args.segments)
    rows: list[dict] = []
    summaries: list[dict] = []
    for source in segment_sources:
        vod_id = source["vod_id"]
        if vod_id in holdout_vod_ids:
            summaries.append({**source, "status": "skipped_holdout_vod"})
            continue
        positives = positive_index.get(vod_id, [])
        if not vod_id or not positives:
            summaries.append({**source, "status": "skipped_no_cross_platform_positives"})
            continue
        segments = _load_segments(pathlib.Path(source["segments_path"]))
        if not segments:
            summaries.append({**source, "status": "skipped_empty_segments"})
            continue
        uncapped_candidates = candidate_generators.generate_dense_candidate_inventory(
            segments,
            strategy=clip_strategy.normalize_strategy(None),
            durations=_parse_float_list(args.durations_sec),
            stride_sec=args.stride_sec,
            max_candidates=0,
        )
        candidates, recall_summary = _promote_positive_recall_candidates(
            uncapped_candidates,
            positives,
            max_candidates=max(args.max_candidates_per_vod, 0),
            positive_overlap_ratio=args.positive_overlap_ratio,
        )
        labelled, summary = _label_candidates(
            candidates,
            segments,
            positives,
            profile=args.profile,
            vod_id=vod_id,
            source=source,
            positive_overlap_ratio=args.positive_overlap_ratio,
            near_miss_sec=args.near_miss_sec,
            background_negatives_per_positive=args.background_negatives_per_positive,
            max_background_negatives_per_vod=args.max_background_negatives_per_vod,
        )
        rows.extend(labelled)
        summaries.append({**summary, **recall_summary})

    existing_rows, existing_summary = build_rows_from_existing_feature_rows(
        list(args.existing_feature_rows),
        profile=args.profile,
        excluded_vod_ids=holdout_vod_ids,
    )
    rows.extend(existing_rows)

    rows.sort(key=lambda row: (str(row.get("vod_id") or ""), float(row.get("window_start") or 0.0), row["candidate_id"]))
    write_jsonl(args.output, rows)
    summary_payload = _summary(rows, summaries, existing_summary, args, positive_index, holdout_vod_ids)
    write_json(args.summary_output, summary_payload)
    print(json.dumps({"event": "proposal_training_rows_built", **summary_payload}, ensure_ascii=True, sort_keys=True))
    return 0


def build_rows_from_existing_feature_rows(
    paths: list[pathlib.Path],
    *,
    profile: str,
    excluded_vod_ids: set[str] | None = None,
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    excluded_vod_ids = excluded_vod_ids or set()
    input_rows = 0
    skipped_gold_rows = 0
    skipped_holdout_rows = 0
    skipped_unusable_rows = 0
    source_counts: Counter[str] = Counter()
    for path in _dedupe_paths(paths):
        for row in read_jsonl(path):
            input_rows += 1
            if _is_gold_eval_row(row):
                skipped_gold_rows += 1
                continue
            if str(row.get("vod_id") or "").strip() in excluded_vod_ids:
                skipped_holdout_rows += 1
                continue
            converted = _existing_feature_training_row(row, profile=profile, source_path=path)
            if converted is None:
                skipped_unusable_rows += 1
                continue
            rows.append(converted)
            source_counts[str(converted.get("label_source") or "unknown")] += 1
    return rows, {
        "input_paths": [str(path) for path in _dedupe_paths(paths)],
        "input_rows": input_rows,
        "rows": len(rows),
        "positive_rows": sum(1 for row in rows if int(row.get("label", 0)) == 1),
        "negative_rows": sum(1 for row in rows if int(row.get("label", 0)) == 0),
        "skipped_gold_rows": skipped_gold_rows,
        "skipped_holdout_rows": skipped_holdout_rows,
        "skipped_unusable_rows": skipped_unusable_rows,
        "label_sources": dict(sorted(source_counts.items())),
    }


def _existing_feature_training_row(row: dict, *, profile: str, source_path: pathlib.Path) -> dict | None:
    source_features = row.get("features") or {}
    if not isinstance(source_features, dict):
        return None
    candidate_id = str(row.get("candidate_id") or row.get("queue_id") or "").strip()
    if not candidate_id:
        return None
    try:
        label = int(row.get("label"))
    except (TypeError, ValueError):
        return None
    label = 1 if label == 1 else 0
    features = _proposal_features_from_existing_features(source_features, row)
    label_source = str(row.get("dataset_source") or row.get("label_source") or source_path.stem or "existing_feature_rows")
    return {
        "queue_id": f"proposal_existing:{candidate_id}",
        "job_id": row.get("job_id") or "",
        "vod_id": str(row.get("vod_id") or "").strip(),
        "source_url": row.get("source_url") or row.get("vod_url") or "",
        "candidate_id": candidate_id,
        "window_start": _as_float(row.get("window_start") or row.get("start")) or 0.0,
        "window_end": _as_float(row.get("window_end") or row.get("end")) or 0.0,
        "label": label,
        "label_class": "proposal_positive" if label else "proposal_negative",
        "label_source": f"existing_feature_row:{label_source}",
        "dataset_source": "proposal_existing_feature_backfill",
        "supervision_tier": "proposal_train",
        "sample_weight": _as_float(row.get("sample_weight")) or (1.4 if label else 0.8),
        "candidate_source": row.get("candidate_source") or row.get("source") or source_features.get("source") or "",
        "features": features,
        "profile": profile,
    }


def _proposal_features_from_existing_features(source_features: dict, row: dict) -> dict:
    candidate_source = str(row.get("candidate_source") or row.get("source") or source_features.get("source") or "").strip()
    payoff = _scaled_feature(source_features, "payoff_score", divisor=8.0)
    hook = _feature(source_features, "hook_strength")
    return {
        "duration": _feature(source_features, "duration"),
        "segment_count": _feature(source_features, "segment_count"),
        "token_count": _feature(source_features, "token_count"),
        "max_hype": _feature(source_features, "max_hype"),
        "mean_hype": _feature(source_features, "mean_hype"),
        "arc_peak": _feature(source_features, "arc_score"),
        "chat_peak": _feature(source_features, "chat_peak"),
        "motion_peak": _feature(source_features, "motion_peak"),
        "face_peak": _feature(source_features, "face_reaction_peak"),
        "laughter_count": _feature(source_features, "laughter_count"),
        "exclamation_count": _feature(source_features, "exclamation_count"),
        "question_count": _feature(source_features, "question_count"),
        "speaker_turns": _feature(source_features, "speaker_turns"),
        "hook_strength": hook,
        "payoff_strength": payoff,
        "hype_delta": _feature(source_features, "hype_delta"),
        "early_hook_score": hook,
        "late_payoff_score": payoff,
        "standalone_text_score": _feature(source_features, "standalone_score"),
        "source_prior": proposal_scorer.source_prior_for_candidate(candidate_source),
        "heuristic_hit_count": 1.0 if candidate_source else 0.0,
        "strategy_relevance": _feature(source_features, "goal_relevance_score"),
        "dense_inventory": 0.0,
        "multimodal_proxy_score": _feature(source_features, "multimodal_proxy_score"),
        "rage_score": _feature(source_features, "rage_score"),
        "clutch_score": _feature(source_features, "clutch_score"),
        "funny_reaction_score": _feature(source_features, "funny_reaction_score"),
        "trash_talk_payoff_score": _feature(source_features, "trash_talk_payoff_score"),
        "chat_shock_score": _feature(source_features, "chat_shock_score"),
        "ranked_tension_score": _feature(source_features, "ranked_tension_score"),
        "failed_setup_score": _feature(source_features, "failed_setup_score"),
        "streamer_persona_beat_score": _feature(source_features, "streamer_persona_beat_score"),
        "context_pattern_score": _feature(source_features, "context_pattern_score"),
        "context_pattern_support": _feature(source_features, "context_pattern_support"),
        "context_pattern_views_prior": _feature(source_features, "context_pattern_views_prior"),
        "context_pattern_confidence": _feature(source_features, "context_pattern_confidence"),
        "context_pattern_similarity_margin": _feature(source_features, "context_pattern_similarity_margin"),
    }


def _label_candidates(
    candidates,
    segments: list[RichSegment],
    positives: list[dict],
    *,
    profile: str,
    vod_id: str,
    source: dict,
    positive_overlap_ratio: float,
    near_miss_sec: float,
    background_negatives_per_positive: int,
    max_background_negatives_per_vod: int,
) -> tuple[list[dict], dict]:
    positive_rows: list[dict] = []
    near_miss_rows: list[dict] = []
    background_candidates: list[tuple[float, object, dict | None]] = []
    for candidate in candidates:
        match = _best_positive_match(candidate.start, candidate.end, positives)
        if match["overlap_ratio"] >= positive_overlap_ratio:
            dense_recall_target = candidate.source == "dense_inventory"
            positive_rows.append(
                _training_row(
                    candidate,
                    segments,
                    profile=profile,
                    vod_id=vod_id,
                    source=source,
                    label=1,
                    sample_weight=(
                        _DENSE_RECALL_POSITIVE_WEIGHT
                        if dense_recall_target
                        else _CROSS_PLATFORM_POSITIVE_WEIGHT
                    ),
                    label_source=(
                        "cross_platform_dense_recall_positive"
                        if dense_recall_target
                        else "cross_platform_positive_overlap"
                    ),
                    match=match["row"],
                    positive_overlap_ratio=match["overlap_ratio"],
                )
            )
        elif match["distance_sec"] <= near_miss_sec:
            near_miss_rows.append(
                _training_row(
                    candidate,
                    segments,
                    profile=profile,
                    vod_id=vod_id,
                    source=source,
                    label=0,
                    sample_weight=0.9,
                    label_source="cross_platform_near_miss",
                    match=match["row"],
                    positive_overlap_ratio=match["overlap_ratio"],
                )
            )
        else:
            background_candidates.append((match["distance_sec"], candidate, match["row"]))

    positive_count = len(positive_rows)
    background_limit = min(
        max_background_negatives_per_vod,
        max(positive_count * max(background_negatives_per_positive, 0), 0),
    )
    background_rows = [
        _training_row(
            candidate,
            segments,
            profile=profile,
            vod_id=vod_id,
            source=source,
            label=0,
            sample_weight=0.35,
            label_source="cross_platform_background_negative",
            match=match,
            positive_overlap_ratio=0.0,
        )
        for _distance, candidate, match in _even_sample(background_candidates, background_limit)
    ]
    rows = [*positive_rows, *near_miss_rows, *background_rows]
    return rows, {
        **source,
        "status": "labelled",
        "candidate_windows": len(candidates),
        "cross_platform_positive_windows": len(positives),
        "positive_rows": len(positive_rows),
        "near_miss_negative_rows": len(near_miss_rows),
        "background_negative_rows": len(background_rows),
        "rows": len(rows),
    }


def _promote_positive_recall_candidates(
    candidates,
    positives: list[dict],
    *,
    max_candidates: int,
    positive_overlap_ratio: float,
) -> tuple[list, dict]:
    if max_candidates <= 0 or len(candidates) <= max_candidates:
        recall_targets = [
            candidate for candidate in candidates
            if _best_positive_match(candidate.start, candidate.end, positives)["overlap_ratio"] >= positive_overlap_ratio
        ]
        return list(candidates), {
            "uncapped_candidate_windows": len(candidates),
            "positive_recall_candidate_windows": len(recall_targets),
            "positive_recall_candidates_promoted": 0,
        }

    recall_targets = []
    recall_ids: set[str] = set()
    for candidate in candidates:
        if _best_positive_match(candidate.start, candidate.end, positives)["overlap_ratio"] < positive_overlap_ratio:
            continue
        recall_targets.append(candidate)
        recall_ids.add(candidate.id)

    remaining_slots = max(max_candidates - len(recall_targets), 0)
    non_targets = [candidate for candidate in candidates if candidate.id not in recall_ids]
    selected_non_targets = sorted(
        non_targets,
        key=lambda candidate: (-float(candidate.score_hint), float(candidate.start), float(candidate.end), candidate.id),
    )[:remaining_slots]
    selected = sorted([*recall_targets, *selected_non_targets], key=lambda candidate: (candidate.start, candidate.end, candidate.id))
    previous_cap_ids = {
        candidate.id
        for candidate in sorted(
            candidates,
            key=lambda candidate: (-float(candidate.score_hint), float(candidate.start), float(candidate.end), candidate.id),
        )[:max_candidates]
    }
    promoted = max(0, len(recall_targets) - sum(1 for candidate in recall_targets if candidate.id in previous_cap_ids))
    return selected, {
        "uncapped_candidate_windows": len(candidates),
        "positive_recall_candidate_windows": len(recall_targets),
        "positive_recall_candidates_promoted": promoted,
    }


def _training_row(
    candidate,
    segments: list[RichSegment],
    *,
    profile: str,
    vod_id: str,
    source: dict,
    label: int,
    sample_weight: float,
    label_source: str,
    match: dict | None,
    positive_overlap_ratio: float,
) -> dict:
    features = proposal_scorer.build_proposal_features(
        candidate,
        segments,
        clip_strategy.normalize_strategy(None),
        profile=profile,
    ).to_dict()
    match = match or {}
    return {
        "queue_id": f"proposal_backfill:{vod_id}:{candidate.id}",
        "job_id": source.get("job_id") or "",
        "vod_id": vod_id,
        "source_url": source.get("source_url") or "",
        "candidate_id": candidate.id,
        "window_start": candidate.start,
        "window_end": candidate.end,
        "label": int(label),
        "label_class": "proposal_positive" if label else "proposal_negative",
        "label_source": label_source,
        "dataset_source": "proposal_cross_platform_backfill",
        "supervision_tier": "proposal_train",
        "sample_weight": sample_weight,
        "candidate_source": candidate.source,
        "features": features,
        "positive_overlap_ratio": round(float(positive_overlap_ratio or 0.0), 6),
        "proposal_recall_target": bool(label and candidate.source == "dense_inventory"),
        "matched_cross_platform_candidate_id": match.get("candidate_id") or "",
        "matched_cross_platform_source_url": match.get("source_url") or "",
        "matched_cross_platform_title": _first_text(
            match,
            ("repost_title", "short_title", "video_title", "title"),
        ),
        "matched_cross_platform_similarity": match.get("transcript_similarity"),
        "transcript_repost_excerpt": match.get("transcript_repost_excerpt") or "",
        "transcript_vod_excerpt": match.get("transcript_vod_excerpt") or "",
        "profile": profile,
    }


def _first_text(row: dict, fields: tuple[str, ...]) -> str:
    for field in fields:
        value = str(row.get(field) or "").strip()
        if value:
            return value
    return ""


def _positive_window_index(rows: list[dict], *, excluded_vod_ids: set[str] | None = None) -> dict[str, list[dict]]:
    index: dict[str, list[dict]] = {}
    excluded_vod_ids = excluded_vod_ids or set()
    for row in rows:
        try:
            label = int(row.get("label", 1))
            start = float(row.get("window_start"))
            end = float(row.get("window_end"))
        except (TypeError, ValueError):
            continue
        vod_id = str(row.get("vod_id") or "").strip()
        if not vod_id or vod_id in excluded_vod_ids or label != 1 or end <= start:
            continue
        copied = dict(row)
        copied["window_start"] = start
        copied["window_end"] = end
        index.setdefault(vod_id, []).append(copied)
    return index


def _discover_segment_sources(storage_dir: pathlib.Path, explicit_segments: list[pathlib.Path]) -> list[dict]:
    sources: list[dict] = []
    seen_paths: set[pathlib.Path] = set()
    for path in explicit_segments:
        resolved = path.resolve()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        sources.append(
            {
                "segments_path": str(path),
                "job_id": "",
                "vod_id": _vod_id_from_path_or_rows(path),
                "source_url": "",
                "source_kind": "explicit_segments",
            }
        )
    for segments_path in sorted(storage_dir.glob("*/debug/segments.jsonl")):
        resolved = segments_path.resolve()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        job_dir = segments_path.parents[1]
        request = _read_json(job_dir / "request.json")
        source_url = str(request.get("url") or "").strip()
        sources.append(
            {
                "segments_path": str(segments_path),
                "job_id": job_dir.name,
                "vod_id": _vod_id_from_url(source_url),
                "source_url": source_url,
                "source_kind": "storage_job_debug",
            }
        )
    return sources


def _load_segments(path: pathlib.Path) -> list[RichSegment]:
    segments: list[RichSegment] = []
    for row in read_jsonl(path):
        data = dict(row)
        data["emotions"] = list(data.pop("tags", data.get("emotions") or []) or [])
        allowed = RichSegment.__dataclass_fields__.keys()
        kwargs = {key: data[key] for key in allowed if key in data}
        if "start" not in kwargs or "end" not in kwargs or "text" not in kwargs:
            continue
        try:
            segments.append(RichSegment(**kwargs))
        except TypeError:
            continue
    return sorted(segments, key=lambda segment: (segment.start, segment.end))


def _best_positive_match(start: float, end: float, positives: list[dict]) -> dict:
    best = {"overlap_ratio": 0.0, "distance_sec": float("inf"), "row": None}
    for row in positives:
        pos_start = float(row["window_start"])
        pos_end = float(row["window_end"])
        overlap = max(0.0, min(end, pos_end) - max(start, pos_start))
        pos_duration = max(pos_end - pos_start, 1e-6)
        candidate_duration = max(end - start, 1e-6)
        overlap_ratio = max(overlap / pos_duration, overlap / candidate_duration)
        if overlap > 0:
            distance = 0.0
        else:
            distance = min(abs(start - pos_end), abs(pos_start - end))
        if overlap_ratio > best["overlap_ratio"] or (
            overlap_ratio == best["overlap_ratio"] and distance < best["distance_sec"]
        ):
            best = {"overlap_ratio": overlap_ratio, "distance_sec": distance, "row": row}
    return best


def _even_sample(items: list[tuple[float, object, dict | None]], limit: int) -> list[tuple[float, object, dict | None]]:
    if limit <= 0 or not items:
        return []
    items = sorted(items, key=lambda item: (-item[0], item[1].start, item[1].id))
    if len(items) <= limit:
        return items
    step = len(items) / limit
    return [items[min(int(index * step), len(items) - 1)] for index in range(limit)]


def _summary(
    rows: list[dict],
    source_summaries: list[dict],
    existing_feature_summary: dict,
    args,
    positive_index: dict[str, list[dict]],
    holdout_vod_ids: set[str],
) -> dict:
    label_counts = Counter(int(row["label"]) for row in rows)
    label_sources = Counter(str(row.get("label_source") or "unknown") for row in rows)
    return {
        "profile": args.profile,
        "output": str(args.output),
        "rows": len(rows),
        "positive_rows": label_counts.get(1, 0),
        "negative_rows": label_counts.get(0, 0),
        "label_sources": dict(sorted(label_sources.items())),
        "segment_sources": len(source_summaries),
        "labelled_segment_sources": sum(1 for item in source_summaries if item.get("status") == "labelled"),
        "skipped_holdout_segment_sources": sum(1 for item in source_summaries if item.get("status") == "skipped_holdout_vod"),
        "excluded_holdout_vods": sorted(holdout_vod_ids),
        "cross_platform_vods": len(positive_index),
        "existing_feature_backfill": existing_feature_summary,
        "source_summaries": source_summaries,
    }


def _holdout_vod_ids(path: pathlib.Path | None) -> set[str]:
    if path is None or not pathlib.Path(path).is_file():
        return set()
    vod_ids: set[str] = set()
    for row in read_jsonl(path):
        split = str(row.get("gold_holdout_split") or row.get("outcome_split") or row.get("dataset_role") or "").lower()
        source = str(row.get("dataset_source") or row.get("supervision_tier") or "").lower()
        if "holdout" not in split and "holdout" not in source:
            continue
        vod_id = str(row.get("vod_id") or "").strip()
        if vod_id:
            vod_ids.add(vod_id)
    return vod_ids


def _parse_float_list(value: str) -> tuple[float, ...]:
    parsed = []
    for item in str(value or "").split(","):
        try:
            number = float(item.strip())
        except ValueError:
            continue
        if number > 0:
            parsed.append(number)
    return tuple(parsed)


def _read_json(path: pathlib.Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _vod_id_from_url(url: str) -> str:
    match = _VOD_URL_RE.search(url)
    return match.group(1) if match else ""


def _vod_id_from_path_or_rows(path: pathlib.Path) -> str:
    match = re.search(r"(\d{8,})", str(path))
    if match:
        return match.group(1)
    for row in read_jsonl(path):
        value = str(row.get("vod_id") or "").strip()
        if value:
            return value
    return ""


def _dedupe_paths(paths: list[pathlib.Path] | tuple[pathlib.Path, ...]) -> list[pathlib.Path]:
    deduped: list[pathlib.Path] = []
    seen: set[pathlib.Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    return deduped


def _is_gold_eval_row(row: dict) -> bool:
    haystack = " ".join(
        str(row.get(key) or "").lower()
        for key in (
            "dataset_source",
            "label_source",
            "supervision_tier",
            "label_class",
            "dataset_role",
            "original_gold_split",
        )
    )
    return any(token in haystack for token in _GOLD_SOURCE_TOKENS)


def _feature(features: dict, key: str, default: float = 0.0) -> float:
    return _as_float(features.get(key)) if _as_float(features.get(key)) is not None else default


def _scaled_feature(features: dict, key: str, *, divisor: float, default: float = 0.0) -> float:
    value = _feature(features, key, default)
    if divisor <= 0:
        return value
    return max(0.0, min(value / divisor, 1.0))


def _as_float(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
