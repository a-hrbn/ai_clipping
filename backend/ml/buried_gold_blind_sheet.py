"""Buried-Gold Blind-Sheet — Selektion (PRD #69, Slice 1).

The read-only selection core of the blind-sheet gate: it mechanically determines
the population the (later) blind-discrimination sheet will test — the
``clean_but_buried`` viral golds whose real covering candidate sits *outside* the
delivered top-20 — by **composing** the existing seams, never re-classifying:

  * ``replay_harness.load_replay_inputs`` (#47) + ``replay_ranked_pool`` (production
    rank, identity ``rank_score_fn``) — the raw rows and the deduped, rank-ordered
    pool the buried gold live in,
  * ``gap_decomposition.classify_gold`` (#46/#49) — the frozen bucket + buried_rank,
  * the gold-eval-manifest loader + ``gt_tier`` filter (``eval_clip_finder`` /
    ``directional_eval``) — only audio-confirmed positions (gold/silver), and
  * ``outcome_view_weighting.view_count`` — the repost-view ranking the top-10 cut
    is read off.

Selection: ``bucket == CLEAN_BUT_BURIED`` **and** ``buried_rank > 20``, gt_tier in
{gold, silver}, top-10 by views. It builds NO excerpts/decoys/sheet and adds no
new measurement layer — the slice is verifiable as "the gate picks the right
population". The VODs with no replayable job dir stay a visible coverage gap
(measured-not-counted); a selection < 10 is logged, never silently capped.

Usage (inside the Docker container):
  python -m ml.buried_gold_blind_sheet
  python -m ml.buried_gold_blind_sheet --json
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import pathlib
import sys

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import (directional_eval, eval_clip_finder, gap_decomposition as gd,
                outcome_view_weighting, replay_harness)

logger = logging.getLogger(__name__)

# Audio-confirmed positions only — the gold window is trustworthy. review /
# unverifiable are excluded so a (later) excerpt never sits on a shaky boundary.
GOLD_SILVER_TIERS: tuple[str, ...] = ("gold", "silver")

# Outside the delivered top-20: a clean candidate buried deeper than this is what
# costs the shipped recall@20. The selection floor (strictly greater than).
BURIED_RANK_FLOOR = 20

# The blind-sheet population size — the top-N clean_but_buried golds by views.
DEFAULT_TOP_N = 10


def select_buried_golds(
    gold_by_vod: dict[str, list[dict]],
    raw_by_vod: dict[str, list[dict]],
    pool_by_vod: dict[str, list[dict]],
    *,
    buried_rank_floor: int = BURIED_RANK_FLOOR,
    top_n: int = DEFAULT_TOP_N,
    thresholds: gd.RealnessThresholds = gd.DEFAULT_THRESHOLDS,
) -> dict:
    """Pick the clean_but_buried golds with ``buried_rank > floor``.

    Per gold, reuses ``gap_decomposition.classify_gold`` (raw pool for hard-miss
    coverage, ranked pool for the buried rank) and keeps only the
    ``CLEAN_BUT_BURIED`` ones ranked deeper than ``buried_rank_floor``. The
    eligible golds are ranked by repost views (``outcome_view_weighting.view_count``)
    descending and the top ``top_n`` kept. The caller passes already
    gt_tier-filtered, scored gold.
    """
    eligible: list[dict] = []
    for vod_id, moments in gold_by_vod.items():
        raw_pool = raw_by_vod.get(vod_id, [])
        ranked_pool = pool_by_vod.get(vod_id, [])
        for gold in moments:
            verdict = gd.classify_gold(
                gold, raw_pool, ranked_pool=ranked_pool, thresholds=thresholds)
            if verdict["bucket"] != gd.CLEAN_BUT_BURIED:
                continue
            buried_rank = verdict["buried_rank"]
            if buried_rank is None or buried_rank <= buried_rank_floor:
                continue
            eligible.append({
                "vod_id": vod_id,
                "start": float(gold.get("start") or 0.0),
                "end": float(gold.get("end") or 0.0),
                "buried_rank": buried_rank,
                "views": outcome_view_weighting.view_count(gold),
                "gt_tier": str(gold.get("gt_tier") or ""),
                "desc": str(gold.get("desc") or ""),
            })
    # Highest repost views first (stable tie-break on vod_id + window) — the
    # burial that costs the most reach is tested first.
    eligible.sort(key=lambda g: (-g["views"], g["vod_id"], g["start"]))
    selected = eligible[:top_n] if top_n > 0 else eligible
    shortfall = max(0, top_n - len(eligible)) if top_n > 0 else 0
    if shortfall:
        # No silent cap: a thin selection is logged so it is not misread as a
        # full top-N population.
        logger.info(json.dumps(
            {"event": "blind_sheet_selection_shortfall", "eligible": len(eligible),
             "top_n": top_n, "shortfall": shortfall}, sort_keys=True))
    return {
        "selected": selected,
        "eligible_count": len(eligible),
        "top_n": top_n,
        "shortfall": shortfall,
    }


def run(
    manifest: dict[str, list[dict]],
    raw_by_vod: dict[str, list[dict]],
    pool_by_vod: dict[str, list[dict]],
    *,
    anchor_vod_ids: list[str] | None = None,
    tiers: tuple[str, ...] = GOLD_SILVER_TIERS,
    buried_rank_floor: int = BURIED_RANK_FLOOR,
    top_n: int = DEFAULT_TOP_N,
    thresholds: gd.RealnessThresholds = gd.DEFAULT_THRESHOLDS,
) -> dict:
    """Select the blind-sheet population from the manifest + replayed pools.

    Restricts the manifest to ``tiers`` (gold/silver — audio-confirmed) via
    ``directional_eval.gold_manifest`` and to the anchor VODs, surfaces the gold
    VODs with no finder predictions as ``missing_prediction_vods`` (a coverage gap,
    measured-not-counted, never a miss), and runs ``select_buried_golds`` over the
    scored gold. Composes only — no re-classification, no new measurement layer.
    """
    gold = directional_eval.gold_manifest(manifest, tuple(tiers))
    if anchor_vod_ids is not None:
        anchor_set = set(anchor_vod_ids)
        gold = {vod_id: ms for vod_id, ms in gold.items() if vod_id in anchor_set}

    missing = directional_eval.missing_prediction_vods(gold, raw_by_vod)
    scored_gold = {vod_id: ms for vod_id, ms in gold.items() if raw_by_vod.get(vod_id)}
    if missing:
        logger.info(json.dumps(
            {"event": "blind_sheet_coverage_gap", "missing_prediction_vods": missing},
            sort_keys=True))

    selection = select_buried_golds(
        scored_gold, raw_by_vod, pool_by_vod,
        buried_rank_floor=buried_rank_floor, top_n=top_n, thresholds=thresholds)

    return {
        "tiers": sorted(tiers),
        "buried_rank_floor": buried_rank_floor,
        "gold_vods": len(gold),
        "scored_vods": len(scored_gold),
        "missing_prediction_vods": missing,
        **selection,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Artifact bridge — replay each anchor VOD into (raw, ranked-pool) dicts
# ─────────────────────────────────────────────────────────────────────────────

def build_inputs(
    anchor_ids: list[str],
    storage: pathlib.Path,
    *,
    rank_score_fn: replay_harness.RankScoreFn | None = None,
) -> tuple[dict[str, list[dict]], dict[str, list[dict]], list[str]]:
    """Replay each anchor VOD into (raw rows, ranked pool) dicts + the gap list.

    For every anchor VOD it loads the newest replayable ``pipeline_debug_*`` job
    dir (``replay_harness.load_replay_inputs``) and builds the production-rank
    ranked pool (``replay_ranked_pool`` under the identity ``rank_score_fn`` — the
    shipped composite order). A VOD with no replayable artifacts is returned in the
    gap list, never silently dropped. No paid call, no network.
    """
    raw_by_vod: dict[str, list[dict]] = {}
    pool_by_vod: dict[str, list[dict]] = {}
    gaps: list[str] = []
    for vod_id in anchor_ids:
        loaded = replay_harness.load_replay_inputs(vod_id, storage)
        if loaded is None:
            gaps.append(vod_id)
            continue
        raw_rows, segments = loaded
        raw_by_vod[vod_id] = raw_rows
        pool_by_vod[vod_id] = replay_harness.replay_ranked_pool(
            raw_rows, segments, rank_score_fn=rank_score_fn)
    return raw_by_vod, pool_by_vod, gaps


# ─────────────────────────────────────────────────────────────────────────────
# Reporting + CLI
# ─────────────────────────────────────────────────────────────────────────────

def format_report(result: dict) -> str:
    """The selected blind-sheet population — one line per gold, caps surfaced."""
    lines = [
        "=" * 78,
        "Buried-Gold Blind-Sheet — SELECTION (clean_but_buried, buried_rank>20)",
        "=" * 78,
        f"Tiers: {','.join(result['tiers'])}   "
        f"Gold VODs: {result['gold_vods']}   Scored: {result['scored_vods']}   "
        f"buried_rank>{result['buried_rank_floor']}   top-{result['top_n']}",
    ]
    missing = result["missing_prediction_vods"]
    if missing:
        lines.append(
            f"⚠  Coverage gap: {len(missing)} gold VOD(s) have NO replayable job dir "
            f"— measured-not-counted, NOT a miss: {', '.join(missing)}")
    if result["shortfall"]:
        lines.append(
            f"⚠  Selection short of top-{result['top_n']}: only {result['eligible_count']} "
            f"eligible buried gold(s) ({result['shortfall']} short) — surfaced, not silently capped.")
    lines.append("")
    lines.append(f"  SELECTED ({len(result['selected'])} of {result['eligible_count']} eligible):")
    if result["selected"]:
        for g in result["selected"]:
            lines.append(
                f"    {g['vod_id']}  [{g['gt_tier']:^6}] "
                f"win {g['start']:.1f}-{g['end']:.1f}s  rank {g['buried_rank']:>3}  "
                f"views {int(g['views']):>9,}  {str(g['desc'])[:46]}")
    else:
        lines.append("    (no clean_but_buried golds beyond the rank floor)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=str(eval_clip_finder.DEFAULT_MANIFEST))
    parser.add_argument("--predictions-dir", default=str(eval_clip_finder.DEFAULT_STORAGE_DIR),
                        help="pipeline_debug_* dirs to replay the ranked pool from")
    parser.add_argument("--anchor-csv", default=str(eval_clip_finder.DEFAULT_ANCHOR_CSV))
    parser.add_argument("--tiers", default=",".join(GOLD_SILVER_TIERS),
                        help="Allowed gt_tier values (default: gold,silver)")
    parser.add_argument("--buried-rank-floor", type=int, default=BURIED_RANK_FLOOR,
                        help="Select clean candidates ranked strictly deeper than this")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N,
                        help="Keep the top-N selected golds by views (the cut is surfaced)")
    parser.add_argument("--output", type=pathlib.Path, default=None)
    parser.add_argument("--json", dest="output_json", action="store_true")
    args = parser.parse_args(argv)

    tiers = tuple(t.strip() for t in args.tiers.split(",") if t.strip())
    manifest = eval_clip_finder.load_manifest(pathlib.Path(args.manifest))
    anchors = eval_clip_finder.read_anchor_vod_ids(pathlib.Path(args.anchor_csv))

    raw_by_vod, pool_by_vod, gaps = build_inputs(anchors, pathlib.Path(args.predictions_dir))
    if gaps:
        # No silent truncation: the artifact-less VODs are logged here too (they
        # also surface in the result as missing_prediction_vods).
        print(
            f"buried_gold_blind_sheet: {len(gaps)}/{len(anchors)} anchor VODs have no "
            f"replayable artifacts — coverage gap: {', '.join(gaps)}",
            file=sys.stderr)

    result = run(
        manifest, raw_by_vod, pool_by_vod, anchor_vod_ids=anchors,
        tiers=tiers, buried_rank_floor=args.buried_rank_floor, top_n=args.top_n)
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
