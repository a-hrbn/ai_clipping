"""Convert cross-platform match rows into a fixed eval manifest.

Reads iboclipss_matches.jsonl (output of match_cross_platform_transcripts.py),
groups by VOD, deduplicates near-overlapping windows, and writes
jynxzi_eval_manifest.jsonl.

This manifest is IMMUTABLE once built — never use it for training.

Usage:
  python ml/build_eval_manifest.py
  python ml/build_eval_manifest.py --matches ml/artifacts/iboclipss_matches.jsonl
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml.row_identity import read_jsonl, window_overlap_ratio, write_jsonl  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MATCHES_IN = BACKEND_ROOT / "ml" / "artifacts" / "iboclipss_matches.jsonl"
DEFAULT_MANIFEST_OUT = BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_eval_manifest.jsonl"
OVERLAP_DEDUP_RATIO = 0.5


def main() -> int:
    parser = argparse.ArgumentParser(description="Build eval manifest from cross-platform match rows.")
    parser.add_argument("--matches", default=str(DEFAULT_MATCHES_IN))
    parser.add_argument("--output", default=str(DEFAULT_MANIFEST_OUT))
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.55,
        help="Minimum transcript_similarity to include (default: 0.55).",
    )
    args = parser.parse_args()

    matches_path = pathlib.Path(args.matches)
    if not matches_path.exists():
        logger.error("Matches file not found: %s", matches_path)
        return 1

    rows = read_jsonl(matches_path)
    logger.info("Loaded %d match rows from %s", len(rows), matches_path)

    rows = [r for r in rows if float(r.get("transcript_similarity") or 0) >= args.min_similarity]
    logger.info("%d rows pass similarity filter (>= %.2f)", len(rows), args.min_similarity)

    by_vod: dict[str, list[dict]] = {}
    for row in rows:
        vod_id = str(row.get("vod_id") or "").strip()
        if vod_id:
            by_vod.setdefault(vod_id, []).append(row)

    output_rows = []
    total_moments = 0
    for vod_id, vod_rows in sorted(by_vod.items()):
        gt_moments = dedup_moments(vod_rows)
        total_moments += len(gt_moments)
        output_rows.append({"vod_id": vod_id, "gt_moments": gt_moments})
        logger.info("VOD %s: %d match rows → %d gt_moments", vod_id, len(vod_rows), len(gt_moments))

    output_path = pathlib.Path(args.output)
    write_jsonl(output_path, output_rows)
    logger.info(
        "Eval manifest: %d VODs, %d gt_moments total → %s",
        len(output_rows), total_moments, output_path,
    )
    return 0


def dedup_moments(rows: list[dict]) -> list[dict]:
    """Deduplicate near-overlapping windows per VOD into eval-manifest moments.

    Projects each match row into the eval-manifest **moment** schema
    (``repost_title``/``views``/``source_url``/``match_kind``), then collapses
    near-overlapping windows: highest views wins a slot, a candidate is dropped
    when it covers >= ``OVERLAP_DEDUP_RATIO`` of an already-kept window. Output
    is sorted by ``start``. The pure geometry is delegated to
    :func:`ml.row_identity.window_overlap_ratio`; this function owns the
    eval-manifest schema knowledge, so it stays in the manifest builder.
    """
    moments = []
    for row in rows:
        start = float(row.get("window_start") or 0)
        end = float(row.get("window_end") or 0)
        if end <= start:
            continue
        moment = {
            "start": round(start, 2),
            "end": round(end, 2),
            "desc": str(row.get("repost_title") or ""),
            "views": float(row.get("views") or 0),
            "source_url": str(row.get("source_url") or ""),
            "match_similarity": round(float(row.get("transcript_similarity") or 0), 4),
            "match_kind": str(row.get("transcript_match_kind") or "single_window"),
        }
        if row.get("transcript_position_ambiguous"):
            moment["position_ambiguous"] = True
        moments.append(moment)

    moments.sort(key=lambda m: m["views"], reverse=True)
    return keep_non_overlapping(moments)


def keep_non_overlapping(ranked: list[dict]) -> list[dict]:
    """Walk moments in priority order; drop one that covers >= ``OVERLAP_DEDUP_RATIO``
    of an already-kept window. Returns the survivors sorted by ``start``."""
    kept: list[dict] = []
    for candidate in ranked:
        start, end = float(candidate.get("start") or 0), float(candidate.get("end") or 0)
        if not any(
            window_overlap_ratio(start, end, float(k.get("start") or 0), float(k.get("end") or 0))
            >= OVERLAP_DEDUP_RATIO
            for k in kept
        ):
            kept.append(candidate)
    return sorted(kept, key=lambda m: float(m.get("start") or 0))


if __name__ == "__main__":
    sys.exit(main())
