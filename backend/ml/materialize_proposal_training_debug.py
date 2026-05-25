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
        if not refresh and (debug_dir / "segments.jsonl").exists() and (debug_dir / "candidate_proposals.jsonl").exists():
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
        candidates = candidate_generators.generate_dense_candidate_inventory(
            transcript,
            chapters,
            strategy,
            durations=durations,
            stride_sec=stride_sec,
            max_candidates=max_candidates_per_vod,
        )
        proposal = proposal_scorer.select_candidates(
            candidates,
            transcript,
            profile,
            strategy,
            top_k=max_candidates_per_vod,
            min_score=0.0,
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
        write_jsonl(debug_dir / "candidate_proposals.jsonl", proposal.audit_rows)
        write_json(
            debug_dir / "candidate_proposal_summary.json",
            {
                **proposal.summary,
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
