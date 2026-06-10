"""Directional finder eval — localize the LLM moment finder's gap.

Scores the current LLM moment finder against the *verified gold* GT to learn
whether its weakness is **generation** (it never surfaces the viral moment) or
**ranking** (it surfaces it, just not in the top-K) — before changing anything
about the finder.

Two-tier recall:
  floor     every gold moment (the long tail)
  headline  gold moments with views >= --headline-min-views (interim 100k)

Each tier reports raw recall@50 alongside recall@10 (and @5/@20). A wide gap
between recall@50 and recall@10 is a *ranking* gap; a low recall@50 is a
*generation* gap.

"Gold" here means the `gold` verification tier only — i.e. audio-confirmed at
the claimed position. That exactly drops the `not_confirmed` matches (silver/
review tiers) and the `vod_unavailable` ones (unverifiable tier) the PRD calls
out, so phantom/mispositioned targets never poison the metric.

A gold VOD with no finder predictions is **surfaced before scoring** — closing
that gap means running the LLM finder over it (paid), so the eval refuses to
silently bury it as a miss.

Usage (inside the Docker container):
  python -m ml.directional_eval
  python -m ml.directional_eval --headline-min-views 500000 --ks 5,10,20,50
  python -m ml.directional_eval --json
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys
from typing import Any

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import eval_clip_finder
from ml.eval_clip_finder import COVERAGE_THRESHOLD, evaluate_vod, load_manifest, load_predictions

DEFAULT_KS: tuple[int, ...] = (5, 10, 20, 50)
DEFAULT_HEADLINE_MIN_VIEWS = 100_000
GOLD_TIER = "gold"


# ─────────────────────────────────────────────────────────────────────────────
# Pure decision functions
# ─────────────────────────────────────────────────────────────────────────────

def gold_manifest(
    manifest: dict[str, list[dict]],
    tiers: set[str] | tuple[str, ...] = (GOLD_TIER,),
) -> dict[str, list[dict]]:
    """Gold-filtered manifest: keep only gt_moments in `tiers`, drop empty VODs.

    With the default {gold} this excludes the `not_confirmed` (silver/review)
    and `vod_unavailable` (unverifiable) tiers, leaving only audio-confirmed
    positions.
    """
    keep = set(tiers)
    filtered = eval_clip_finder.filter_manifest_tiers(manifest, keep)
    return {vod_id: moments for vod_id, moments in filtered.items() if moments}


def missing_prediction_vods(
    gold: dict[str, list[dict]],
    predictions_by_vod: dict[str, list[dict]],
) -> list[str]:
    """Gold VODs (>=1 gold moment) with no finder predictions, sorted.

    These are coverage gaps: scoring them as misses would understate recall, so
    they are surfaced — generating their predictions is a paid finder run.
    """
    return sorted(
        vod_id
        for vod_id, moments in gold.items()
        if moments and not predictions_by_vod.get(vod_id)
    )


def aggregate_tier(
    per_vod: list[dict],
    ks: tuple[int, ...] = DEFAULT_KS,
    *,
    min_views: float = 0.0,
) -> dict:
    """Micro-averaged recall over gold gt_hits with views >= min_views.

    Operates on the per-VOD `gt_hits` already produced by evaluate_vod, so both
    tiers come from a single evaluation pass.
    """
    hits = [
        h
        for vod in per_vod
        if vod.get("has_gt")
        for h in vod.get("gt_hits", [])
        if float(h.get("views") or 0.0) >= min_views
    ]
    gt_count = len(hits)
    vods_in_tier = len({
        vod.get("vod_id")
        for vod in per_vod
        if vod.get("has_gt")
        and any(float(h.get("views") or 0.0) >= min_views for h in vod.get("gt_hits", []))
    })

    recalls: dict[str, float | None] = {}
    for k in ks:
        if gt_count == 0:
            recalls[str(k)] = None
        else:
            hits_at_k = sum(
                1 for h in hits
                if h["hit"] and h.get("hit_rank") is not None and h["hit_rank"] <= k
            )
            recalls[str(k)] = round(hits_at_k / gt_count, 6)

    hit_ranks = [h["hit_rank"] for h in hits if h["hit"] and h.get("hit_rank") is not None]
    hit_count = sum(1 for h in hits if h["hit"])
    return {
        "min_views": min_views,
        "vods": vods_in_tier,
        "gt_count": gt_count,
        "hit_count": hit_count,
        "miss_count": gt_count - hit_count,
        "recall": recalls,
        "avg_rank": round(sum(hit_ranks) / len(hit_ranks), 4) if hit_ranks else None,
        "miss_rate": round((gt_count - hit_count) / gt_count, 6) if gt_count else None,
    }


def two_tier_recall(
    per_vod: list[dict],
    ks: tuple[int, ...] = DEFAULT_KS,
    *,
    headline_min_views: float = DEFAULT_HEADLINE_MIN_VIEWS,
) -> dict:
    """Floor (all gold) + headline (gold with views >= headline_min_views)."""
    return {
        "headline_min_views": headline_min_views,
        "floor": aggregate_tier(per_vod, ks, min_views=0.0),
        "headline": aggregate_tier(per_vod, ks, min_views=headline_min_views),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run(
    manifest: dict[str, list[dict]],
    predictions_by_vod: dict[str, list[dict]],
    *,
    ks: tuple[int, ...] = DEFAULT_KS,
    headline_min_views: float = DEFAULT_HEADLINE_MIN_VIEWS,
    gold_tiers: set[str] | tuple[str, ...] = (GOLD_TIER,),
) -> dict:
    """Run the directional eval over the gold-filtered manifest.

    Returns a structured result: two-tier recall plus the surfaced
    missing-prediction coverage gap.
    """
    gold = gold_manifest(manifest, gold_tiers)
    missing = missing_prediction_vods(gold, predictions_by_vod)

    per_vod = [
        evaluate_vod(gold[vod_id], predictions_by_vod.get(vod_id, []), ks, vod_id=vod_id)
        for vod_id in sorted(gold)
    ]
    tiers = two_tier_recall(per_vod, ks, headline_min_views=headline_min_views)

    return {
        "ks": list(ks),
        "gold_tiers": sorted(gold_tiers),
        "gold_vods": len(gold),
        "missing_prediction_vods": missing,
        "headline_min_views": tiers["headline_min_views"],
        "floor": tiers["floor"],
        "headline": tiers["headline"],
        "coverage_threshold": COVERAGE_THRESHOLD,
        "per_vod": per_vod,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(v: Any) -> str:
    if v is None:
        return "  —  "
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _tier_line(label: str, tier: dict, ks: list[int]) -> str:
    recall = tier.get("recall", {})
    cells = "  ".join(f"r@{k}={_fmt(recall.get(str(k)))}" for k in ks)
    return (
        f"  {label:<9} n={tier['gt_count']:>3} ({tier['vods']} VODs)  {cells}"
        f"  avg_rank={_fmt(tier.get('avg_rank'))}  hits={tier['hit_count']}/{tier['gt_count']}"
    )


def format_report(result: dict) -> str:
    ks = result["ks"]
    lines = [
        "=" * 72,
        f"Directional Finder Eval — generation vs ranking  (coverage>={result['coverage_threshold']})",
        "=" * 72,
        f"Gold tiers: {','.join(result['gold_tiers'])}   Gold VODs: {result['gold_vods']}"
        f"   Headline: views>={int(result['headline_min_views']):,}",
    ]

    missing = result["missing_prediction_vods"]
    if missing:
        lines.append(
            f"⚠  {len(missing)}/{result['gold_vods']} gold VODs have NO finder predictions "
            f"(counted as misses): {', '.join(missing)}"
        )
        lines.append(
            "   Close this gap by running the LLM moment finder over them "
            "(paid LLM calls), then re-run."
        )
    else:
        lines.append("✓  Every gold VOD has finder predictions.")

    lines.append("")
    lines.append(_tier_line("FLOOR", result["floor"], ks))
    lines.append(_tier_line("HEADLINE", result["headline"], ks))
    lines.append("")
    lines.append(
        "Read: recall@50 ≫ recall@10 → ranking gap (surfaced, not top-K); "
        "low recall@50 → generation gap (never surfaced)."
    )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifest", default=str(eval_clip_finder.DEFAULT_MANIFEST))
    parser.add_argument("--predictions-dir", default=str(eval_clip_finder.DEFAULT_STORAGE_DIR))
    parser.add_argument("--workflow-id", default="", help="Score one finder run instead of latest-per-VOD")
    parser.add_argument("--ks", default="5,10,20,50", help="Recall@K comma-separated")
    parser.add_argument("--headline-min-views", type=float, default=DEFAULT_HEADLINE_MIN_VIEWS,
                        help="Min views for the headline tier (interim 100000)")
    parser.add_argument("--gold-tiers", default=GOLD_TIER,
                        help="Verification tiers treated as gold (default: gold)")
    parser.add_argument("--json", dest="output_json", action="store_true")
    args = parser.parse_args(argv)

    ks = tuple(int(k.strip()) for k in args.ks.split(",") if k.strip())
    gold_tiers = tuple(t.strip() for t in args.gold_tiers.split(",") if t.strip())

    manifest = load_manifest(pathlib.Path(args.manifest))
    gold = gold_manifest(manifest, gold_tiers)
    predictions_by_vod = load_predictions(sorted(gold), pathlib.Path(args.predictions_dir), args.workflow_id)

    # Surface the paid-finder gap before scoring (also embedded in the result).
    missing = missing_prediction_vods(gold, predictions_by_vod)
    if missing:
        print(
            f"WARNING: {len(missing)}/{len(gold)} gold VODs lack finder predictions; "
            f"generating them is a paid LLM finder run: {', '.join(missing)}",
            file=sys.stderr,
        )

    result = run(manifest, predictions_by_vod, ks=ks,
                 headline_min_views=args.headline_min_views, gold_tiers=gold_tiers)
    result["computed_at"] = datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z")

    if args.output_json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(format_report(result))
        print(f"\ncomputed_at: {result['computed_at']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
