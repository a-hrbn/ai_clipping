"""
Evaluate LLM clip finder against the Ground-Truth eval manifest.

Overlap rule (bidirectional): a prediction is a "hit" for a GT moment when it
covers >= 50% of the GT window OR the GT window covers >= 50% of the prediction.
The second direction lets a short, precise clip fully inside a long, loose
Iboclipss-repost window count; the finder's 8s minimum clip length keeps a
degenerate mini-prediction from gaming it. The reported per-GT `coverage` column
stays GT-centric (overlap_seconds / gt_duration).

Usage (standalone CLI inside the Docker container):
  python ml/eval_clip_finder.py --json
  python ml/eval_clip_finder.py --workflow-id 20260608T120000000000Z
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import pathlib
import sys
from typing import Any

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

DEFAULT_MANIFEST = BACKEND_ROOT / "ml" / "artifacts" / "jynxzi_eval_manifest.jsonl"
DEFAULT_ANCHOR_CSV = BACKEND_ROOT / "ml" / "jynxzi_gt_vods.csv"
DEFAULT_STORAGE_DIR = BACKEND_ROOT / "tmp" / "storage"
DEFAULT_KS: tuple[int, ...] = (5, 10, 20)
COVERAGE_THRESHOLD = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Core arithmetic
# ─────────────────────────────────────────────────────────────────────────────

def coverage_ratio(gt: dict, pred: dict) -> float:
    """Fraction of the GT window covered by the prediction window.

    GT uses start/end keys; pred uses start_time/end_time keys.
    Returns 0.0 if the GT window is zero-length.
    """
    gt_start = float(gt.get("start") or 0)
    gt_end = float(gt.get("end") or 0)
    if gt_end <= gt_start:
        return 0.0
    pred_start = float(pred.get("start_time") or 0)
    pred_end = float(pred.get("end_time") or 0)
    overlap = max(0.0, min(gt_end, pred_end) - max(gt_start, pred_start))
    return overlap / (gt_end - gt_start)


def pred_coverage_ratio(gt: dict, pred: dict) -> float:
    """Fraction of the prediction window that lies within the GT window.

    The bidirectional counterpart to coverage_ratio: a short, precise clip
    fully inside a long, loose GT window covers a tiny fraction of the GT but
    (almost) all of itself. Returns 0.0 if the prediction window is zero-length.
    """
    pred_start = float(pred.get("start_time") or 0)
    pred_end = float(pred.get("end_time") or 0)
    if pred_end <= pred_start:
        return 0.0
    gt_start = float(gt.get("start") or 0)
    gt_end = float(gt.get("end") or 0)
    overlap = max(0.0, min(gt_end, pred_end) - max(gt_start, pred_start))
    return overlap / (pred_end - pred_start)


def is_hit(gt: dict, pred: dict, threshold: float = COVERAGE_THRESHOLD) -> bool:
    """Bidirectional hit: the prediction covers >= threshold of the GT window,
    OR the GT window covers >= threshold of the prediction.
    """
    return (
        coverage_ratio(gt, pred) >= threshold
        or pred_coverage_ratio(gt, pred) >= threshold
    )


def _pred_rank(pred: dict, fallback: int) -> int:
    try:
        return int(pred.get("rank") or fallback)
    except (TypeError, ValueError):
        return fallback


# ─────────────────────────────────────────────────────────────────────────────
# Per-VOD evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_vod(
    gt_moments: list[dict],
    pred_moments: list[dict],
    ks: tuple[int, ...] = DEFAULT_KS,
    *,
    vod_id: str = "",
) -> dict:
    """Evaluate one VOD.

    pred_moments should be rank-sorted ascending (rank 1 = best).
    Returns per-VOD metrics + per-GT-moment hit table.
    """
    has_gt = bool(gt_moments)
    gt_count = len(gt_moments)
    pred_count = len(pred_moments)
    max_k = max(ks) if ks else 20

    # Rank-sort predictions (defensive; pipeline guarantees this already)
    ranked_preds = sorted(
        enumerate(pred_moments),
        key=lambda ip: _pred_rank(ip[1], ip[0] + 1),
    )

    gt_hits: list[dict] = []
    for gi, gt in enumerate(gt_moments):
        best_cov = 0.0
        hit_rank: int | None = None
        hit_pred: dict | None = None

        for _idx, pred in ranked_preds:
            rank = _pred_rank(pred, _idx + 1)
            if rank > max_k:
                break  # sorted ascending — no further hits possible
            cov = coverage_ratio(gt, pred)
            if cov > best_cov:
                best_cov = cov  # reported coverage stays GT-centric
            if hit_rank is None and is_hit(gt, pred):
                hit_rank = rank
                hit_pred = pred
                break  # first hit (by rank) wins, no double-count

        gt_hits.append({
            "gt_index": gi,
            "gt_start": float(gt.get("start") or 0),
            "gt_end": float(gt.get("end") or 0),
            "desc": str(gt.get("desc") or ""),
            "views": float(gt.get("views") or 0),
            "source_url": str(gt.get("source_url") or ""),
            "hit": hit_rank is not None,
            "hit_rank": hit_rank,
            "coverage": round(best_cov, 4),
            "pred_start_time": float(hit_pred.get("start_time") or 0) if hit_pred else None,
            "pred_end_time": float(hit_pred.get("end_time") or 0) if hit_pred else None,
        })

    hit_count = sum(1 for h in gt_hits if h["hit"])
    miss_count = gt_count - hit_count

    recalls: dict[str, float | None] = {}
    for k in ks:
        if gt_count == 0:
            recalls[str(k)] = None
        else:
            hits_at_k = sum(
                1 for h in gt_hits
                if h["hit"] and h["hit_rank"] is not None and h["hit_rank"] <= k
            )
            recalls[str(k)] = round(hits_at_k / gt_count, 6)

    hit_ranks = [h["hit_rank"] for h in gt_hits if h["hit"] and h["hit_rank"] is not None]
    avg_rank = round(sum(hit_ranks) / len(hit_ranks), 4) if hit_ranks else None
    miss_rate = round(miss_count / gt_count, 6) if gt_count > 0 else None

    return {
        "vod_id": vod_id,
        "has_gt": has_gt,
        "gt_count": gt_count,
        "pred_count": pred_count,
        "recall": recalls,
        "avg_rank": avg_rank,
        "miss_rate": miss_rate,
        "hit_count": hit_count,
        "miss_count": miss_count,
        "gt_hits": gt_hits,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Multi-VOD aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_overall(per_vod: list[dict], ks: tuple[int, ...] = DEFAULT_KS) -> dict:
    """Micro-average across all has_gt VODs (VODs without GT excluded)."""
    active = [v for v in per_vod if v.get("has_gt")]
    null_recalls: dict[str, float | None] = {str(k): None for k in ks}

    if not active:
        return {
            "vods_with_gt": 0,
            "vods_total": len(per_vod),
            "gt_count": 0,
            "pred_count": sum(v.get("pred_count", 0) for v in per_vod),
            "recall": null_recalls,
            "avg_rank": None,
            "miss_rate": None,
            "hit_count": 0,
            "miss_count": 0,
        }

    total_gt = sum(v["gt_count"] for v in active)
    recalls: dict[str, float | None] = {}
    for k in ks:
        if total_gt == 0:
            recalls[str(k)] = None
        else:
            hits_at_k = sum(
                sum(
                    1 for h in v["gt_hits"]
                    if h["hit"] and h.get("hit_rank") is not None and h["hit_rank"] <= k
                )
                for v in active
            )
            recalls[str(k)] = round(hits_at_k / total_gt, 6)

    all_hit_ranks = [
        h["hit_rank"]
        for v in active
        for h in v["gt_hits"]
        if h["hit"] and h.get("hit_rank") is not None
    ]
    avg_rank = round(sum(all_hit_ranks) / len(all_hit_ranks), 4) if all_hit_ranks else None

    total_hits = sum(v["hit_count"] for v in active)
    total_misses = sum(v["miss_count"] for v in active)
    miss_rate = round(total_misses / total_gt, 6) if total_gt > 0 else None

    return {
        "vods_with_gt": len(active),
        "vods_total": len(per_vod),
        "gt_count": total_gt,
        "pred_count": sum(v.get("pred_count", 0) for v in per_vod),
        "recall": recalls,
        "avg_rank": avg_rank,
        "miss_rate": miss_rate,
        "hit_count": total_hits,
        "miss_count": total_misses,
    }


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_manifest(path: pathlib.Path = DEFAULT_MANIFEST) -> dict[str, list[dict]]:
    """Parse GT manifest JSONL → {vod_id: gt_moments}. Missing file → {}."""
    result: dict[str, list[dict]] = {}
    if not path.exists():
        return result
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("vod_id"):
                result[str(obj["vod_id"])] = list(obj.get("gt_moments") or [])
    except OSError:
        pass
    return result


def filter_manifest_tiers(manifest: dict[str, list[dict]], tiers: set[str]) -> dict[str, list[dict]]:
    """Keep only GT moments whose gt_tier is in `tiers`.

    Moments without a gt_tier (manifest not yet verified via
    ml.verify_gt_matches) are excluded when a filter is requested.
    """
    return {
        vod_id: [m for m in moments if str(m.get("gt_tier") or "") in tiers]
        for vod_id, moments in manifest.items()
    }


def read_anchor_vod_ids(path: pathlib.Path = DEFAULT_ANCHOR_CSV) -> list[str]:
    """Read anchor VOD IDs from CSV (lightweight, no heavy imports)."""
    if not path.exists():
        return []
    vod_ids: list[str] = []
    try:
        with path.open(encoding="utf-8") as f:
            non_comment = (line for line in f if not line.strip().startswith("#"))
            reader = csv.DictReader(non_comment)
            for row in reader:
                vod_id = (row.get("vod_id") or "").strip()
                if vod_id:
                    vod_ids.append(vod_id)
    except OSError:
        pass
    return vod_ids


def _read_moments_jsonl(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except OSError:
        pass
    return rows


def _read_final_moments_json(path: pathlib.Path) -> list[dict]:
    """Read the product's final_moments.json -> moment list (order = product rank)."""
    if not path.exists():
        return []
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    moments = obj.get("moments") if isinstance(obj, dict) else None
    return [m for m in (moments or []) if isinstance(m, dict)]


def _resolve_job_dir(
    storage: pathlib.Path,
    vod_id: str,
    workflow_id: str,
) -> pathlib.Path | None:
    """Resolve a VOD's pipeline_debug job dir.

    With a workflow_id, points at that exact run; otherwise the most-recently
    modified pipeline_debug job for the VOD. None if no job exists.
    """
    if workflow_id:
        return storage / f"pipeline_debug_{workflow_id}_{vod_id}"
    candidates = sorted(
        storage.glob(f"pipeline_debug_*_{vod_id}"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _finder_moments_path(job_dir: pathlib.Path) -> pathlib.Path:
    """Prefer the raw (pre-filter) finder moments, fall back to the filtered set."""
    raw_path = job_dir / "debug" / "llm_moment_finder_moments_raw.jsonl"
    filtered_path = job_dir / "debug" / "llm_moment_finder_moments.jsonl"
    return raw_path if raw_path.exists() else filtered_path


def load_predictions(
    anchor_ids: list[str],
    storage: pathlib.Path,
    workflow_id: str = "",
) -> dict[str, list[dict]]:
    """Load *raw* finder predictions per VOD from pipeline_debug job dirs.

    With a workflow_id, reads that exact run; otherwise picks the most-recently
    modified pipeline_debug job for each VOD. VODs with no job map to [].
    """
    predictions_by_vod: dict[str, list[dict]] = {}
    for vod_id in anchor_ids:
        job_dir = _resolve_job_dir(storage, vod_id, workflow_id)
        moments_path = _finder_moments_path(job_dir) if job_dir else pathlib.Path("/dev/null")
        predictions_by_vod[vod_id] = _read_moments_jsonl(moments_path)
    return predictions_by_vod


def load_final_predictions(
    anchor_ids: list[str],
    storage: pathlib.Path,
    workflow_id: str = "",
) -> dict[str, list[dict]]:
    """Load the product's *final* moments per VOD (post dedupe + enforce_diversity
    + top_k) from each pipeline_debug job's final_moments.json.

    The counterpart to load_predictions (which returns the raw finder moments):
    final_moments are what the product actually ships, and share the
    start_time/end_time keys, so the same hit logic applies. List order is the
    product rank. VODs with no job map to [].
    """
    predictions_by_vod: dict[str, list[dict]] = {}
    for vod_id in anchor_ids:
        job_dir = _resolve_job_dir(storage, vod_id, workflow_id)
        moments_path = job_dir / "debug" / "final_moments.json" if job_dir else pathlib.Path("/dev/null")
        predictions_by_vod[vod_id] = _read_final_moments_json(moments_path)
    return predictions_by_vod


# ─────────────────────────────────────────────────────────────────────────────
# Top-level orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all(
    predictions_by_vod: dict[str, list[dict]],
    manifest: dict[str, list[dict]] | None = None,
    anchor_vod_ids: list[str] | None = None,
    ks: tuple[int, ...] = DEFAULT_KS,
) -> dict:
    """Evaluate all anchor VODs and return complete result dict.

    Used by the pipeline-debug router (compute-at-hydration) and the CLI.
    """
    if manifest is None:
        manifest = load_manifest()
    if anchor_vod_ids is None:
        anchor_vod_ids = read_anchor_vod_ids()

    per_vod: list[dict] = []
    for vod_id in anchor_vod_ids:
        gt_moments = manifest.get(vod_id, [])
        preds = predictions_by_vod.get(vod_id, [])
        per_vod.append(evaluate_vod(gt_moments, preds, ks, vod_id=vod_id))

    return {
        "per_vod": per_vod,
        "overall": aggregate_overall(per_vod, ks),
        "ks": list(ks),
        "coverage_threshold": COVERAGE_THRESHOLD,
        "computed_at": datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--vods", default=str(DEFAULT_ANCHOR_CSV), help="Anchor VOD CSV")
    parser.add_argument("--predictions-dir", default=str(DEFAULT_STORAGE_DIR))
    parser.add_argument("--workflow-id", default="", help="Workflow ID to resolve job paths")
    parser.add_argument("--ks", default="5,10,20,50", help="Recall@K comma-separated")
    parser.add_argument("--json", dest="output_json", action="store_true")
    parser.add_argument("--tiers", default="",
                        help="Only score GT moments with these gt_tier values, e.g. gold,silver (default: all)")
    args = parser.parse_args(argv)

    ks = tuple(int(k.strip()) for k in args.ks.split(",") if k.strip())
    manifest = load_manifest(pathlib.Path(args.manifest))
    if args.tiers.strip():
        manifest = filter_manifest_tiers(manifest, {t.strip() for t in args.tiers.split(",") if t.strip()})
    anchor_ids = read_anchor_vod_ids(pathlib.Path(args.vods))
    storage = pathlib.Path(args.predictions_dir)

    predictions_by_vod = load_predictions(anchor_ids, storage, args.workflow_id)

    result = evaluate_all(predictions_by_vod, manifest=manifest, anchor_vod_ids=anchor_ids, ks=ks)

    if args.output_json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    def _fmt(v: Any) -> str:
        if v is None:
            return "  —  "
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    overall = result["overall"]
    print(f"\n{'='*64}")
    print(f"LLM Clip Finder — Baseline Eval  (threshold={result['coverage_threshold']})")
    print(f"{'='*64}")
    print(f"VODs with GT: {overall['vods_with_gt']}/{overall['vods_total']}   GT: {overall['gt_count']}   Preds: {overall['pred_count']}")
    recall = overall.get("recall", {})
    print(f"\nOVERALL  recall@5={_fmt(recall.get('5'))} r@10={_fmt(recall.get('10'))} r@20={_fmt(recall.get('20'))}  avg_rank={_fmt(overall.get('avg_rank'))}  miss={_fmt(overall.get('miss_rate'))}  hits={overall['hit_count']}/{overall['gt_count']}")

    print("\nPER VOD:")
    for v in result["per_vod"]:
        rec = v.get("recall", {})
        if not v["has_gt"]:
            print(f"  VOD {v['vod_id']}: keine GT  preds={v['pred_count']}")
            continue
        print(f"  VOD {v['vod_id']}: r@5={_fmt(rec.get('5'))} r@10={_fmt(rec.get('10'))} r@20={_fmt(rec.get('20'))} avg_rank={_fmt(v.get('avg_rank'))} miss={_fmt(v.get('miss_rate'))} hits={v['hit_count']}/{v['gt_count']}")
        for h in v["gt_hits"]:
            mark = "✓" if h["hit"] else "✗"
            rank_s = str(h["hit_rank"]) if h["hit_rank"] is not None else "-"
            print(f"    {mark} rank={rank_s:>3}  cov={h['coverage']:.0%}  {int(h['views']):>7,}v  {h['desc'][:55]}")

    print(f"\ncomputed_at: {result['computed_at']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
