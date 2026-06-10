"""Expand the eval manifest with curated cross-platform positives.

The original iboclipss_matches.jsonl that fed build_eval_manifest.py is gone,
but its accepted matches survive as ``cross_platform_outcome_positive`` rows in
jynxzi_vod_training_rows.jsonl (with full repost metadata: views, title,
source_url, transcript similarity). This script promotes those windows into
jynxzi_eval_manifest.jsonl so the eval grows from 5 to ~37 VODs.

Curation rules:
- only rows with transcript match metadata (227 legacy rows without it are
  ~96% window-duplicates of the metadata rows and carry no confidence signal)
- transcript_similarity >= --min-similarity (default 0.55, same as
  build_eval_manifest.py)
- window duration sanity bounds, near-duplicate windows per VOD deduped
- VODs already in the manifest keep their existing (manually curated)
  gt_moments untouched; only NEW VODs are added
- every gt_moment gets a ``clip_group`` (derived from source_url) so
  multi-window entries of the same repost can be scored clip-level

Outputs:
- jynxzi_eval_manifest.jsonl        (expanded, sorted by vod_id)
- jynxzi_eval_manifest.v1.jsonl     (backup of the previous manifest)
- jynxzi_eval_manifest.meta.json    (version stamp + provenance counts)

The manifest stays IMMUTABLE once built — never use it for training. Note the
source rows are marked gold_holdout_split=train: fine for the LLM-only
pipeline, but if a trained ranker returns, re-split before tuning on this.

Usage:
  python ml/expand_eval_manifest.py [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import pathlib
import sys

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TRAINING_ROWS = BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_vod_training_rows.jsonl"
DEFAULT_MANIFEST = BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_eval_manifest.jsonl"
MANIFEST_VERSION = 2
_OVERLAP_DEDUP_RATIO = 0.5
_MIN_DURATION_SEC = 4.0
_MAX_DURATION_SEC = 120.0


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _clip_group(source_url: str, vod_id: str) -> str:
    basis = (source_url or "").strip() or f"vod:{vod_id}"
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:10]


def _overlap_ratio(a: dict, b: dict) -> float:
    overlap = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    shorter = max(min(a["end"] - a["start"], b["end"] - b["start"]), 1e-9)
    return overlap / shorter


def _dedupe_windows(moments: list[dict]) -> list[dict]:
    """Drop near-duplicate windows; the higher-similarity row wins."""
    kept: list[dict] = []
    for moment in sorted(moments, key=lambda m: m.get("match_similarity") or 0, reverse=True):
        if any(_overlap_ratio(moment, existing) >= _OVERLAP_DEDUP_RATIO for existing in kept):
            continue
        kept.append(moment)
    kept.sort(key=lambda m: m["start"])
    return kept


def main() -> int:
    parser = argparse.ArgumentParser(description="Expand eval manifest from training positives.")
    parser.add_argument("--training-rows", default=str(DEFAULT_TRAINING_ROWS))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--min-similarity", type=float, default=0.55)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = pathlib.Path(args.manifest)
    existing = _read_jsonl(manifest_path) if manifest_path.exists() else []
    existing_vods = {str(row.get("vod_id")) for row in existing}
    logger.info("Existing manifest: %d VODs, %d GT moments",
                len(existing), sum(len(r.get("gt_moments") or []) for r in existing))

    rows = _read_jsonl(pathlib.Path(args.training_rows))
    positives = [
        r for r in rows
        if r.get("label_class") == "cross_platform_outcome_positive"
        and r.get("transcript_match_kind")  # rows without metadata are unverifiable dupes
        and float(r.get("transcript_similarity") or 0) >= args.min_similarity
    ]
    logger.info("Curated positives (metadata + similarity >= %.2f): %d", args.min_similarity, len(positives))

    new_by_vod: dict[str, list[dict]] = {}
    skipped_duration = 0
    for r in positives:
        vod_id = str(r.get("vod_id") or "").strip()
        if not vod_id or vod_id in existing_vods:
            continue  # anchor VODs keep their manually curated entries
        start = float(r.get("window_start") or 0)
        end = float(r.get("window_end") or 0)
        if not (_MIN_DURATION_SEC <= end - start <= _MAX_DURATION_SEC):
            skipped_duration += 1
            continue
        source_url = str(r.get("source_url") or "")
        new_by_vod.setdefault(vod_id, []).append({
            "start": round(start, 2),
            "end": round(end, 2),
            "desc": str(r.get("repost_title") or ""),
            "views": float(r.get("views") or 0),
            "source_url": source_url,
            "match_similarity": round(float(r.get("transcript_similarity") or 0), 4),
            "match_kind": str(r.get("transcript_match_kind") or ""),
            "clip_group": _clip_group(source_url, vod_id),
            "origin": "training_positive_v2",
        })

    new_rows = []
    for vod_id in sorted(new_by_vod):
        moments = _dedupe_windows(new_by_vod[vod_id])
        if moments:
            new_rows.append({"vod_id": vod_id, "gt_moments": moments})

    # Stamp clip_group onto the existing curated rows too (idempotent).
    for row in existing:
        for moment in row.get("gt_moments") or []:
            moment.setdefault("clip_group", _clip_group(str(moment.get("source_url") or ""), str(row.get("vod_id"))))
            moment.setdefault("origin", "anchor_v1")

    merged = sorted(existing + new_rows, key=lambda r: str(r.get("vod_id")))
    gt_total = sum(len(r.get("gt_moments") or []) for r in merged)
    clip_groups = {m["clip_group"] for r in merged for m in r.get("gt_moments") or []}
    logger.info("Expanded manifest: %d VODs, %d GT moments, %d distinct clips (dropped %d by duration)",
                len(merged), gt_total, len(clip_groups), skipped_duration)

    if args.dry_run:
        for row in new_rows:
            logger.info("  + VOD %s: %d moments", row["vod_id"], len(row["gt_moments"]))
        return 0

    backup = manifest_path.with_name("jynxzi_eval_manifest.v1.jsonl")
    if manifest_path.exists() and not backup.exists():
        backup.write_text(manifest_path.read_text(encoding="utf-8"), encoding="utf-8")
        logger.info("Backup written: %s", backup.name)

    manifest_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in merged) + "\n",
        encoding="utf-8",
    )
    meta = {
        "version": MANIFEST_VERSION,
        "built_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "vod_count": len(merged),
        "gt_moment_count": gt_total,
        "distinct_clip_count": len(clip_groups),
        "sources": {
            "anchor_v1_vods": len(existing),
            "training_positive_v2_vods": len(new_rows),
            "min_similarity": args.min_similarity,
        },
    }
    manifest_path.with_name("jynxzi_eval_manifest.meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8",
    )
    logger.info("Manifest v%d written: %s", MANIFEST_VERSION, manifest_path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
