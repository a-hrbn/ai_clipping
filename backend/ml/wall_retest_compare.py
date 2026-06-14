"""WALL-Re-Test verdict/compare harness — base-vs-neu recall@20 → pre-fixed verdict.

The deterministic heart of the WALL re-test (PRD #90): compare two prediction
dirs (``base`` vs ``neu``) and read the *pre-fixed* verdict off the recall@20
movement. It composes the existing eval seams — ``gap_decomposition.build_inputs``
(replay) → ``directional_eval.run`` (recall@20 + raw@any) → ``gap_decomposition.
decompose_tier`` (clean_but_buried) — over each dir, diffs the headline recall@20
and maps the movement to the verdict, never a new measurement layer.

The verdict is fixed in code, calibrated against the headline tier (shipped
recall@20 = 0.125, ceiling raw@any = 0.625):

  movement >= 0.25  → WALL_FALLS      (the cheap lever recovered the signal)
  movement <= 0.15  → ESCALATE_PASS2  (best-shot didn't move it — WALL hält)
  in between        → INCONCLUSIVE

Always reported alongside, both dirs and both tiers: raw@any (the generation
lift) and the clean_but_buried share (the sorting lift). base stays untouched —
neu lives in its own ``--predictions-dir``.
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys
from typing import Sequence

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import directional_eval, eval_clip_finder, gap_decomposition, replay_harness

# recall@20 is read at the shipped headline column (K = clip_top_k = 20).
RECALL_K = directional_eval.PRODUCT_HEADLINE_K

# The pre-fixed verdict — its three outcomes and the two movement bands live in
# exactly one place, calibrated against the headline tier before the paid re-run.
WALL_FALLS = "WALL_FALLS"
ESCALATE_PASS2 = "ESCALATE_PASS2"
INCONCLUSIVE = "INCONCLUSIVE"

WALL_FALLS_THRESHOLD = 0.25
ESCALATE_THRESHOLD = 0.15

# The tiers every metric is reported at; the verdict keys on the headline one
# (the tier the bands were calibrated against).
TIERS: tuple[str, ...] = ("floor", "headline")
DEFAULT_VERDICT_TIER = "headline"


def verdict_for_movement(movement: float | None) -> str:
    """Map a recall@20 movement (neu − base) to the pre-fixed verdict.

    A ``None`` movement (a tier with no gold to score on either side) can't be
    concluded → ``INCONCLUSIVE``.
    """
    if movement is None:
        return INCONCLUSIVE
    if movement >= WALL_FALLS_THRESHOLD:
        return WALL_FALLS
    if movement <= ESCALATE_THRESHOLD:
        return ESCALATE_PASS2
    return INCONCLUSIVE


def _lift(base_val: float | None, neu_val: float | None) -> dict:
    """A {base, neu, lift} cell; lift is None unless both sides are numbers."""
    lift = (
        round(neu_val - base_val, 6)
        if (base_val is not None and neu_val is not None)
        else None
    )
    return {"base": base_val, "neu": neu_val, "lift": lift}


def compare(base: dict, neu: dict, *, verdict_tier: str = DEFAULT_VERDICT_TIER) -> dict:
    """Diff two per-dir measures and read the pre-fixed verdict off the movement.

    Builds a base/neu/lift cell for recall@20, raw@any (generation lift) and the
    clean_but_buried share (sorting lift) at both tiers, then reads the verdict off
    the ``verdict_tier`` recall@20 lift (headline by default). base is never
    mutated — the lift is always neu − base.
    """
    recall = {t: _lift(base["recall_at_20"][t], neu["recall_at_20"][t]) for t in TIERS}
    raw_any = {t: _lift(base["raw_any"][t], neu["raw_any"][t]) for t in TIERS}
    clean_but_buried = {
        t: _lift(base["clean_but_buried"][t], neu["clean_but_buried"][t]) for t in TIERS
    }
    movement = recall[verdict_tier]["lift"]
    return {
        "verdict": verdict_for_movement(movement),
        "verdict_tier": verdict_tier,
        "movement": movement,
        "recall_at_20": recall,
        "raw_any": raw_any,
        "clean_but_buried": clean_but_buried,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Measurement — replay one prediction dir through the existing eval seams
# ─────────────────────────────────────────────────────────────────────────────

def measure_dir(
    anchor_ids: Sequence[str],
    storage: pathlib.Path,
    manifest: dict[str, list[dict]],
    *,
    gold_tiers: Sequence[str] = (directional_eval.GOLD_TIER,),
    ks: tuple[int, ...] = directional_eval.DEFAULT_KS,
    headline_min_views: float = directional_eval.DEFAULT_HEADLINE_MIN_VIEWS,
    top_k: int = replay_harness.DEFAULT_TOP_K,
) -> dict:
    """Replay one prediction dir and extract recall@20 + raw@any + clean_but_buried.

    Composes the existing seams, no new measurement layer:
      * ``gap_decomposition.build_inputs`` replays each anchor VOD's newest job dir
        into (raw, deduped-pool, final) via the real serving postprocess,
      * ``directional_eval.run`` scores the product recall@K and the generation
        ceiling (raw@any) over the gold-filtered manifest, and
      * ``gap_decomposition.decompose_tier`` partitions the same scored gold so the
        clean_but_buried share (the sorting lift) comes off the buried-real bucket.

    Both tiers (floor = all gold, headline = views ≥ ``headline_min_views``) are
    reported. Replay coverage gaps are surfaced, never folded in as misses.
    """
    anchor_list = list(anchor_ids)
    raw_by_vod, pool_by_vod, final_by_vod, gaps = gap_decomposition.build_inputs(
        anchor_list, storage, top_k=top_k)

    metrics = directional_eval.run(
        manifest, final_by_vod, raw_by_vod,
        anchor_vod_ids=anchor_list, ks=ks,
        headline_min_views=headline_min_views, gold_tiers=tuple(gold_tiers))

    # The same scored-gold restriction directional_eval applies (anchor set ∩ gold
    # VODs with raw predictions), so the decomposition denominator never drifts.
    gold = directional_eval.gold_manifest(manifest, gold_tiers)
    anchor_set = set(anchor_list)
    scored_gold = {
        vod_id: moments
        for vod_id, moments in gold.items()
        if vod_id in anchor_set and raw_by_vod.get(vod_id)
    }
    decomposition = {
        "floor": gap_decomposition.decompose_tier(
            scored_gold, raw_by_vod, pool_by_vod, min_views=0.0),
        "headline": gap_decomposition.decompose_tier(
            scored_gold, raw_by_vod, pool_by_vod, min_views=headline_min_views),
    }

    k = str(RECALL_K)
    return {
        "recall_at_20": {
            "floor": metrics["floor"]["recall"].get(k),
            "headline": metrics["headline"]["recall"].get(k),
        },
        "raw_any": {
            "floor": metrics["generation"]["floor"]["recall_any"],
            "headline": metrics["generation"]["headline"]["recall_any"],
        },
        "clean_but_buried": {
            "floor": decomposition["floor"]["shares"][gap_decomposition.CLEAN_BUT_BURIED],
            "headline": decomposition["headline"]["shares"][gap_decomposition.CLEAN_BUT_BURIED],
        },
        "scored_vods": metrics["scored_vods"],
        "missing_prediction_vods": metrics["missing_prediction_vods"],
        "replay_gaps": gaps,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(v) -> str:
    if v is None:
        return "  —  "
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _signed(v) -> str:
    if v is None:
        return "  —  "
    return f"{v:+.3f}" if isinstance(v, float) else f"{v:+}"


def _metric_line(label: str, cell: dict) -> str:
    return (f"  {label:<18} base={_fmt(cell['base'])}  neu={_fmt(cell['neu'])}  "
            f"lift={_signed(cell['lift'])}")


def format_report(result: dict) -> str:
    vt = result["verdict_tier"]
    lines = [
        "=" * 72,
        "WALL-Re-Test — base vs neu recall@20 → pre-fixed verdict",
        "=" * 72,
        f"Gold tiers: {','.join(result['gold_tiers'])}   Anchor VODs: {result['anchor_count']}"
        f"   Verdict tier: {vt}",
        "",
        f"  VERDICT: {result['verdict']}   "
        f"(recall@20 {vt} movement {_signed(result['movement'])}; "
        f"bands: ≥{WALL_FALLS_THRESHOLD} WALL_FALLS / ≤{ESCALATE_THRESHOLD} ESCALATE_PASS2)",
        "",
        "  RECALL@20 (the verdict driver):",
        _metric_line("floor", result["recall_at_20"]["floor"]),
        _metric_line("headline", result["recall_at_20"]["headline"]),
        "",
        "  raw@any (generation lift — did neu surface more gold at any rank):",
        _metric_line("floor", result["raw_any"]["floor"]),
        _metric_line("headline", result["raw_any"]["headline"]),
        "",
        "  clean_but_buried (sorting lift — more real-but-buried gold):",
        _metric_line("floor", result["clean_but_buried"]["floor"]),
        _metric_line("headline", result["clean_but_buried"]["headline"]),
    ]
    for name in ("base", "neu"):
        gaps = result[name].get("replay_gaps") or []
        missing = result[name].get("missing_prediction_vods") or []
        if gaps:
            lines.append(f"⚠  {name}: {len(gaps)} VOD(s) without replayable artifacts: "
                         f"{', '.join(gaps)}")
        if missing:
            lines.append(f"⚠  {name}: {len(missing)} gold VOD(s) with no predictions "
                         f"(coverage gap, not misses): {', '.join(missing)}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI — measure both dirs through the shared seams, diff, read the verdict.
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-dir", required=True,
                        help="The base prediction dir (untouched) — the WALL status quo")
    parser.add_argument("--predictions-dir", required=True,
                        help="The neu prediction dir (the re-run with the cheap lever applied)")
    parser.add_argument("--manifest", default=str(eval_clip_finder.DEFAULT_MANIFEST))
    parser.add_argument("--anchor-csv", default=str(eval_clip_finder.DEFAULT_ANCHOR_CSV),
                        help="CSV of the re-test subset VOD IDs (same subset, both dirs)")
    parser.add_argument("--ks", default="5,10,20,50", help="Recall@K comma-separated")
    parser.add_argument("--tiers", default=directional_eval.GOLD_TIER,
                        help="Verification tiers scored as gold (default: gold)")
    parser.add_argument("--headline-min-views", type=float,
                        default=directional_eval.DEFAULT_HEADLINE_MIN_VIEWS,
                        help="Min views for the headline tier (interim 100000)")
    parser.add_argument("--verdict-tier", default=DEFAULT_VERDICT_TIER, choices=TIERS,
                        help="Which tier's recall@20 movement drives the verdict")
    parser.add_argument("--top-k", type=int, default=replay_harness.DEFAULT_TOP_K)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    parser.add_argument("--json", dest="output_json", action="store_true")
    args = parser.parse_args(argv)

    ks = tuple(int(k.strip()) for k in args.ks.split(",") if k.strip())
    gold_tiers = tuple(t.strip() for t in args.tiers.split(",") if t.strip())
    manifest = eval_clip_finder.load_manifest(pathlib.Path(args.manifest))
    anchors = eval_clip_finder.read_anchor_vod_ids(pathlib.Path(args.anchor_csv))

    def _measure(storage_dir: str) -> dict:
        return measure_dir(
            anchors, pathlib.Path(storage_dir), manifest,
            gold_tiers=gold_tiers, ks=ks,
            headline_min_views=args.headline_min_views, top_k=args.top_k)

    base = _measure(args.base_dir)
    neu = _measure(args.predictions_dir)

    result = compare(base, neu, verdict_tier=args.verdict_tier)
    result["base"] = base
    result["neu"] = neu
    result["gold_tiers"] = sorted(gold_tiers)
    result["anchor_count"] = len(anchors)
    result["headline_min_views"] = args.headline_min_views
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
