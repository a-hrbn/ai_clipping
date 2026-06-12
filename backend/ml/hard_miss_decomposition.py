"""hard_miss Root-Cause-Decomposition — skeleton + denominator (PRD #56, Phase 1).

The thinnest end-to-end tracer of the Generation-Track. It does NOT add a new
measurement layer; it **composes** the existing seams:

  * `directional_eval` (#48) — the gold filtering, the anchor restriction and the
    coverage-gap rule (raw@any is the generation measurement seam),
  * `gap_decomposition` (#49) — the `hard_miss` set itself, via `classify_gold`;
    the decomposition *sub-classifies* that already-verified set, it never
    re-derives it (the reuse guard), and
  * the `replay_harness` (#47) loaders — raw moments + `segments.jsonl` from the
    `pipeline_debug_*` job dirs (no reimplemented I/O), reached through
    `gap_decomposition.build_inputs`.

In this slice the module classifies **nothing**: every `hard_miss` gold carries
`bucket = "unclassified"`. The purpose is to prove the seam composition and the
honest denominator before any detector exists — floor vs headline hard_miss split
over the scored anchor VODs, with the missing-prediction VODs surfaced as a
coverage gap (measured, not counted as a miss). No paid LLM call.

Usage (inside the Docker container):
  python -m ml.hard_miss_decomposition
  python -m ml.hard_miss_decomposition --json
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import pathlib
import sys

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import directional_eval, eval_clip_finder, gap_decomposition
from ml.gap_decomposition import DEFAULT_THRESHOLDS, RealnessThresholds

# This slice sub-classifies nothing: every hard_miss gold gets this placeholder
# bucket. The real root-cause buckets (no-transcript / truncation / llm-skipped)
# land in later slices.
UNCLASSIFIED = "unclassified"


def hard_miss_golds(
    gold_by_vod: dict[str, list[dict]],
    raw_by_vod: dict[str, list[dict]],
    pool_by_vod: dict[str, list[dict]],
    *,
    min_views: float = 0.0,
    thresholds: RealnessThresholds = DEFAULT_THRESHOLDS,
) -> list[dict]:
    """The gold moments (views >= min_views) #49 classifies as hard_miss.

    The reuse guard lives here: each gold is routed through
    ``gap_decomposition.classify_gold`` and kept only when its bucket is
    ``HARD_MISS``, so the set is bit-identical to #49's hard_miss — the
    decomposition sub-classifies that set, it never re-derives the membership.
    Each survivor is returned as a flat record carrying its window, views, desc
    and ``bucket = "unclassified"`` (this slice classifies nothing).
    """
    out: list[dict] = []
    for vod_id, moments in gold_by_vod.items():
        raw_pool = raw_by_vod.get(vod_id, [])
        ranked_pool = pool_by_vod.get(vod_id, [])
        for gold in moments:
            if float(gold.get("views") or 0.0) < min_views:
                continue
            verdict = gap_decomposition.classify_gold(
                gold, raw_pool, ranked_pool=ranked_pool, thresholds=thresholds)
            if verdict["bucket"] != gap_decomposition.HARD_MISS:
                continue
            out.append({
                "vod_id": vod_id,
                "start": float(gold.get("start") or 0.0),
                "end": float(gold.get("end") or 0.0),
                "views": float(gold.get("views") or 0.0),
                "desc": str(gold.get("desc") or ""),
                "bucket": UNCLASSIFIED,
            })
    return out


def _tier(golds: list[dict]) -> dict:
    """A reported hard_miss tier: its named count next to the golds themselves."""
    return {"count": len(golds), "golds": golds}


def run(
    manifest: dict[str, list[dict]],
    raw_by_vod: dict[str, list[dict]],
    pool_by_vod: dict[str, list[dict]],
    *,
    anchor_vod_ids: list[str] | None = None,
    headline_min_views: float = directional_eval.DEFAULT_HEADLINE_MIN_VIEWS,
    gold_tiers: tuple[str, ...] = (directional_eval.GOLD_TIER,),
    thresholds: RealnessThresholds = DEFAULT_THRESHOLDS,
) -> dict:
    """Report the hard_miss denominator, split floor vs headline (no classifier).

    Reuses ``directional_eval`` for the gold filtering and the anchor restriction
    (the same gold-tier + anchor set as the frozen baseline), so the denominator
    never drifts from the recall numbers. The hard_miss membership comes from
    ``hard_miss_golds`` (i.e. #49's ``classify_gold``) — this never re-classifies.
    Floor counts every hard_miss gold; headline keeps only the high-view ones
    (views >= ``headline_min_views``), so the routing weight reads honestly off
    the North-Star tier.
    """
    gold = directional_eval.gold_manifest(manifest, gold_tiers)
    if anchor_vod_ids is not None:
        anchor_set = set(anchor_vod_ids)
        gold = {vod_id: moments for vod_id, moments in gold.items() if vod_id in anchor_set}

    # The gold VODs the finder never scored are a coverage gap: surfaced, never
    # folded into the denominator. The decomposition runs over the scored set only.
    missing = directional_eval.missing_prediction_vods(gold, raw_by_vod)
    scored_gold = {vod_id: moments for vod_id, moments in gold.items() if raw_by_vod.get(vod_id)}

    floor = hard_miss_golds(scored_gold, raw_by_vod, pool_by_vod,
                            min_views=0.0, thresholds=thresholds)
    headline = hard_miss_golds(scored_gold, raw_by_vod, pool_by_vod,
                               min_views=headline_min_views, thresholds=thresholds)

    return {
        "gold_tiers": sorted(gold_tiers),
        "anchor_restricted": anchor_vod_ids is not None,
        "gold_vods": len(gold),
        "scored_vods": len(scored_gold),
        "missing_prediction_vods": missing,
        "headline_min_views": headline_min_views,
        "thresholds": dataclasses.asdict(thresholds),
        "hard_miss": {"floor": _tier(floor), "headline": _tier(headline)},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Artifact bridge — reuse the #49 bridge (which reuses the #47 replay loaders)
# ─────────────────────────────────────────────────────────────────────────────

def build_inputs(
    anchor_ids: list[str],
    storage: pathlib.Path,
) -> tuple[dict[str, list[dict]], dict[str, list[dict]], list[str]]:
    """Replay each anchor VOD into (raw, deduped-pool) dicts + the coverage-gap list.

    Delegates straight to ``gap_decomposition.build_inputs``, which composes the
    ``replay_harness`` (#47) loaders — no reimplemented I/O. Reusing the #49 bridge
    guarantees the raw rows here are the exact rows #49 classifies, so the hard_miss
    set reconciles. The product final set #49 also builds is unused in this skeleton
    (there is no recall metric yet) and dropped; a VOD with no replayable artifacts
    comes back in the gap list, never silently dropped.
    """
    raw_by_vod, pool_by_vod, _final_by_vod, gaps = gap_decomposition.build_inputs(
        anchor_ids, storage)
    return raw_by_vod, pool_by_vod, gaps


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _hard_miss_line(label: str, tier: dict) -> str:
    return f"  {label:<9} hard_miss={tier['count']:>2}  (all bucket=\"{UNCLASSIFIED}\")"


def format_report(result: dict) -> str:
    hm = result["hard_miss"]
    lines = [
        "=" * 78,
        "hard_miss Root-Cause-Decomposition (skeleton) — the denominator, unclassified",
        "=" * 78,
        f"Gold tiers: {','.join(result['gold_tiers'])}   Gold VODs: {result['gold_vods']}"
        f"   Scored: {result['scored_vods']}"
        f"{'  (anchor-restricted)' if result.get('anchor_restricted') else ''}"
        f"   Headline: views>={int(result['headline_min_views']):,}",
    ]
    missing = result["missing_prediction_vods"]
    if missing:
        lines.append(
            f"⚠  Coverage gap: {len(missing)}/{result['gold_vods']} gold VODs have NO finder "
            f"predictions — excluded from the hard_miss denominator, NOT counted as misses: "
            f"{', '.join(missing)}")
    else:
        lines.append("✓  Every gold VOD has finder predictions.")

    lines.append("")
    lines.append("  HARD_MISS DENOMINATOR (generation gap — no raw candidate covers the gold):")
    lines.append(_hard_miss_line("FLOOR", hm["floor"]))
    lines.append(_hard_miss_line("HEADLINE", hm["headline"]))
    lines.append("")
    lines.append(
        "Read: this slice classifies nothing — it pins the seam composition and the "
        "honest denominator. The no-transcript / truncation / llm-skipped buckets land next.")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=str(eval_clip_finder.DEFAULT_MANIFEST))
    parser.add_argument("--predictions-dir", default=str(eval_clip_finder.DEFAULT_STORAGE_DIR))
    parser.add_argument("--anchor-csv", default=str(eval_clip_finder.DEFAULT_ANCHOR_CSV))
    parser.add_argument("--headline-min-views", type=float,
                        default=directional_eval.DEFAULT_HEADLINE_MIN_VIEWS)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    parser.add_argument("--json", dest="output_json", action="store_true")
    args = parser.parse_args(argv)

    manifest = eval_clip_finder.load_manifest(pathlib.Path(args.manifest))
    anchors = eval_clip_finder.read_anchor_vod_ids(pathlib.Path(args.anchor_csv))

    raw_by_vod, pool_by_vod, gaps = build_inputs(anchors, pathlib.Path(args.predictions_dir))
    if gaps:
        # No silent truncation of the analysis: the artifact-less VODs are logged
        # here too (they also surface in the result as missing_prediction_vods).
        print(
            f"hard_miss_decomposition: {len(gaps)}/{len(anchors)} anchor VODs have no "
            f"replayable artifacts — coverage gap: {', '.join(gaps)}",
            file=sys.stderr)

    result = run(
        manifest, raw_by_vod, pool_by_vod,
        anchor_vod_ids=anchors, headline_min_views=args.headline_min_views)
    result["computed_at"] = datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z")

    payload = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    if args.output_json:
        print(payload)
    else:
        print(format_report(result))
        print(f"\ncomputed_at: {result['computed_at']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
