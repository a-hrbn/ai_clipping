from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter
from dataclasses import asdict

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import build_twitch_training_dataset as vod_training
from ml.row_identity import read_jsonl, write_json, write_jsonl
from services import candidate_generators, chapter_segmenter, clip_strategy, proposal_scorer, scorer

DEFAULT_ARTIFACT_DIR = BACKEND_ROOT / "ml" / "artifacts"
DEFAULT_CROSS_PLATFORM_ROWS = DEFAULT_ARTIFACT_DIR / "jynxzi_cross_platform_fingerprint_rows.jsonl"
DEFAULT_STORAGE_DIR = BACKEND_ROOT / "tmp" / "storage"
DEFAULT_VOD_CACHE_ROOT = BACKEND_ROOT / "tmp" / "vod_feature_cache"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Materialize proposal-training debug rows from cached positive VOD transcripts."
    )
    parser.add_argument("--profile", default="jynxzi")
    parser.add_argument("--cross-platform-rows", type=pathlib.Path, default=DEFAULT_CROSS_PLATFORM_ROWS)
    parser.add_argument("--storage-dir", type=pathlib.Path, default=DEFAULT_STORAGE_DIR)
    parser.add_argument("--vod-cache-root", type=pathlib.Path, default=DEFAULT_VOD_CACHE_ROOT)
    parser.add_argument("--max-vods", type=int, default=20)
    parser.add_argument("--durations-sec", default="12,20,30,45,60")
    parser.add_argument("--stride-sec", type=float, default=5.0)
    parser.add_argument("--max-candidates-per-vod", type=int, default=5000)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--summary-output",
        type=pathlib.Path,
        default=DEFAULT_ARTIFACT_DIR / "jynxzi_proposal_materialize_summary.json",
    )
    args = parser.parse_args(argv)

    summary = materialize_proposal_debug(
        profile=args.profile,
        cross_platform_rows=args.cross_platform_rows,
        storage_dir=args.storage_dir,
        vod_cache_root=args.vod_cache_root,
        max_vods=max(int(args.max_vods or 0), 0),
        durations=_parse_float_list(args.durations_sec),
        stride_sec=args.stride_sec,
        max_candidates_per_vod=max(int(args.max_candidates_per_vod or 0), 0),
        refresh=bool(args.refresh),
    )
    write_json(args.summary_output, summary)
    print(json.dumps({"event": "proposal_training_debug_materialized", **summary}, ensure_ascii=True, sort_keys=True))
    return 0


def materialize_proposal_debug(
    *,
    profile: str,
    cross_platform_rows: pathlib.Path,
    storage_dir: pathlib.Path,
    vod_cache_root: pathlib.Path,
    max_vods: int,
    durations: tuple[float, ...],
    stride_sec: float,
    max_candidates_per_vod: int,
    refresh: bool = False,
) -> dict:
    positives_by_vod = _positive_rows_by_vod(read_jsonl(cross_platform_rows))
    selected_vods = [
        vod_id
        for vod_id, _count in Counter({vod_id: len(rows) for vod_id, rows in positives_by_vod.items()}).most_common()
    ]
    if max_vods > 0:
        selected_vods = selected_vods[:max_vods]

    results: list[dict] = []
    for vod_id in selected_vods:
        debug_dir = storage_dir / f"proposal_training_{vod_id}" / "debug"
        if (
            not refresh
            and (debug_dir / "segments.jsonl").exists()
            and (debug_dir / "candidate_proposals.jsonl").exists()
            and (debug_dir / "positive_recall_targets.jsonl").exists()
        ):
            results.append(
                {
                    "vod_id": vod_id,
                    "status": "cached_debug",
                    "positive_rows": len(positives_by_vod.get(vod_id, [])),
                    "debug_dir": str(debug_dir),
                }
            )
            continue
        loaded = _load_cached_vod_transcript(vod_cache_root / vod_id, vod_id)
        if loaded is None:
            results.append(
                {
                    "vod_id": vod_id,
                    "status": "missing_transcript_cache",
                    "positive_rows": len(positives_by_vod.get(vod_id, [])),
                }
            )
            continue
        transcript_path, transcript = loaded
        if not transcript:
            results.append(
                {
                    "vod_id": vod_id,
                    "status": "empty_transcript_cache",
                    "positive_rows": len(positives_by_vod.get(vod_id, [])),
                    "transcript_path": str(transcript_path),
                }
            )
            continue
        transcript = scorer.score_segments(transcript)
        chapters = chapter_segmenter.segment_chapters(transcript)
        strategy = clip_strategy.normalize_strategy(None)
        uncapped_candidates = candidate_generators.generate_dense_candidate_inventory(
            transcript,
            chapters,
            strategy,
            durations=durations,
            stride_sec=stride_sec,
            max_candidates=0,
        )
        candidates, recall_summary = _promote_positive_recall_candidates(
            uncapped_candidates,
            positives_by_vod.get(vod_id, []),
            max_candidates=max_candidates_per_vod,
            positive_overlap_ratio=0.45,
        )
        proposal = proposal_scorer.select_candidates(
            candidates,
            transcript,
            profile,
            strategy,
            top_k=max_candidates_per_vod,
            min_score=0.0,
        )
        positive_recall_targets = _positive_recall_target_rows(
            candidates,
            proposal.audit_rows,
            positives_by_vod.get(vod_id, []),
            positive_overlap_ratio=0.45,
        )
        job_dir = debug_dir.parent
        debug_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            job_dir / "request.json",
            {
                "url": f"https://www.twitch.tv/videos/{vod_id}",
                "vod_id": vod_id,
                "profile": profile,
                "source": "proposal_training_materializer",
            },
        )
        write_jsonl(debug_dir / "segments.jsonl", [asdict(segment) for segment in transcript])
        write_jsonl(debug_dir / "chapters.jsonl", [chapter.to_dict() for chapter in chapters])
        write_jsonl(debug_dir / "candidates_dense_inventory.jsonl", [candidate.to_dict() for candidate in candidates])
        write_jsonl(debug_dir / "candidate_proposals.jsonl", _annotate_proposal_recall_targets(proposal.audit_rows, positive_recall_targets))
        write_jsonl(debug_dir / "positive_recall_targets.jsonl", positive_recall_targets)
        write_json(
            debug_dir / "candidate_proposal_summary.json",
            {
                **proposal.summary,
                **recall_summary,
                "positive_recall_targets": len(positive_recall_targets),
                "vod_id": vod_id,
                "positive_rows": len(positives_by_vod.get(vod_id, [])),
                "transcript_path": str(transcript_path),
            },
        )
        results.append(
            {
                "vod_id": vod_id,
                "status": "materialized",
                "positive_rows": len(positives_by_vod.get(vod_id, [])),
                "segments": len(transcript),
                "candidate_windows": len(candidates),
                "proposal_rows": len(proposal.audit_rows),
                **recall_summary,
                "positive_recall_targets": len(positive_recall_targets),
                "debug_dir": str(debug_dir),
                "transcript_path": str(transcript_path),
            }
        )

    statuses = Counter(str(row.get("status") or "unknown") for row in results)
    return {
        "profile": profile,
        "cross_platform_rows": str(cross_platform_rows),
        "storage_dir": str(storage_dir),
        "vod_cache_root": str(vod_cache_root),
        "positive_vods": len(positives_by_vod),
        "selected_vods": len(selected_vods),
        "statuses": dict(sorted(statuses.items())),
        "materialized_vods": statuses.get("materialized", 0),
        "cached_debug_vods": statuses.get("cached_debug", 0),
        "results": results,
    }


def _positive_rows_by_vod(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        vod_id = str(row.get("vod_id") or "").strip()
        if not vod_id:
            continue
        try:
            label = int(row.get("label", 1))
        except (TypeError, ValueError):
            label = 1
        if label != 1:
            continue
        grouped.setdefault(vod_id, []).append(row)
    return grouped


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
    selected_non_targets = sorted(
        [candidate for candidate in candidates if candidate.id not in recall_ids],
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


def _positive_recall_target_rows(
    candidates,
    proposal_rows: list[dict],
    positives: list[dict],
    *,
    positive_overlap_ratio: float,
) -> list[dict]:
    proposal_by_id = {str(row.get("candidate_id") or ""): row for row in proposal_rows}
    rows: list[dict] = []
    for candidate in candidates:
        match = _best_positive_match(candidate.start, candidate.end, positives)
        if match["overlap_ratio"] < positive_overlap_ratio:
            continue
        proposal = proposal_by_id.get(candidate.id) or {}
        rows.append(
            {
                "candidate_id": candidate.id,
                "source": candidate.source,
                "start": round(float(candidate.start), 3),
                "end": round(float(candidate.end), 3),
                "score_hint": round(float(candidate.score_hint), 6),
                "proposal_score": proposal.get("score"),
                "proposal_accepted": bool(proposal.get("accepted") is True),
                "positive_overlap_ratio": round(float(match["overlap_ratio"] or 0.0), 6),
                "matched_cross_platform_candidate_id": (match["row"] or {}).get("candidate_id") or "",
                "matched_cross_platform_source_url": (match["row"] or {}).get("source_url") or "",
                "matched_cross_platform_views": (match["row"] or {}).get("repost_views"),
            }
        )
    return rows


def _annotate_proposal_recall_targets(proposal_rows: list[dict], recall_rows: list[dict]) -> list[dict]:
    recall_by_id = {str(row.get("candidate_id") or ""): row for row in recall_rows}
    annotated = []
    for row in proposal_rows:
        copied = dict(row)
        recall = recall_by_id.get(str(copied.get("candidate_id") or ""))
        if recall:
            copied["proposal_recall_target"] = True
            copied["positive_overlap_ratio"] = recall.get("positive_overlap_ratio")
            copied["matched_cross_platform_candidate_id"] = recall.get("matched_cross_platform_candidate_id")
            copied["matched_cross_platform_source_url"] = recall.get("matched_cross_platform_source_url")
        annotated.append(copied)
    return annotated


def _best_positive_match(start: float, end: float, positives: list[dict]) -> dict:
    best = {"overlap_ratio": 0.0, "row": None}
    for row in positives:
        try:
            pos_start = float(row["window_start"])
            pos_end = float(row["window_end"])
        except (TypeError, ValueError, KeyError):
            continue
        overlap = max(0.0, min(float(end), pos_end) - max(float(start), pos_start))
        pos_duration = max(pos_end - pos_start, 1e-6)
        candidate_duration = max(float(end) - float(start), 1e-6)
        overlap_ratio = max(overlap / pos_duration, overlap / candidate_duration)
        if overlap_ratio > float(best["overlap_ratio"]):
            best = {"overlap_ratio": overlap_ratio, "row": row}
    return best


def _load_cached_vod_transcript(cache_dir: pathlib.Path, vod_id: str) -> tuple[pathlib.Path, list] | None:
    compatible = vod_training._load_compatible_transcript_cache(cache_dir, vod_id)
    if compatible is not None:
        return compatible
    paths = sorted(
        cache_dir.glob("transcript_*.json"),
        key=lambda path: path.stat().st_mtime_ns if path.exists() else 0,
        reverse=True,
    )
    for path in paths:
        cached = vod_training._load_transcript_cache(path)
        if cached is None:
            continue
        metadata, transcript = cached
        if int(metadata.get("version") or 0) != vod_training._VOD_FEATURE_CACHE_VERSION:
            continue
        if str(metadata.get("kind") or "") != "transcript":
            continue
        if str(metadata.get("vod_id") or "") != str(vod_id):
            continue
        return path, transcript
    return None


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


if __name__ == "__main__":
    raise SystemExit(main())
