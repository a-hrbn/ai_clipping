"""Gap-decomposition + baseline-delta report — localize WHY product recall is low.

Institutionalizes the diagnosis behind PRD #46: it takes the same offline
artifacts the replay harness reads and, for every verified-gold moment, decides
whether the finder's failure is

  * hard-miss        — no raw candidate covers the gold window at all
                       (= the generation_ceiling raw@any miss),
  * accidental-only  — covered, but only by an oversized/random window; no
                       *real* candidate (clip-sized, gold-proportioned) covers
                       it in the deduped pool, or
  * clean-but-buried — a real candidate DOES cover it and survives dedupe, it is
                       just ranked deep (the diagnosis' median rank ~141).

The three buckets partition every gold moment; the buried-rank stats on the
clean bucket are the diagnostic that says the bottleneck is *discrimination*,
not weighting. Alongside the decomposition the report carries the canonical
metrics (raw@any, floor/headline recall@K, avg_rank) for any ``rank_score_fn``
as a **delta against the frozen composite baseline** (``finder_recall_baseline``)
so every ranking proposal is judged against the status quo.

Every capping bound (the judge shortlist depth) is surfaced explicitly, and the
gold VODs with no finder predictions stay a visible coverage gap — never folded
in as silent misses. No paid LLM call: it replays existing artifacts.

Usage (inside the Docker container):
  python -m ml.gap_decomposition
  python -m ml.gap_decomposition --json
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import pathlib
import statistics
import sys
from typing import Any

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import directional_eval, eval_clip_finder, replay_harness
from ml.eval_clip_finder import is_hit

# The frozen composite baseline every proposal is measured against.
DEFAULT_BASELINE = BACKEND_ROOT / "ml" / "artifacts" / "finder_recall_baseline.json"

# The classification heuristic lives in exactly one place. "real" = a clip-sized
# candidate (8-90s) whose length is 0.5-3x the gold window — anything else that
# covers the gold does so by an oversized/random overlap (accidental).
@dataclasses.dataclass(frozen=True)
class RealnessThresholds:
    min_duration_s: float = 8.0
    max_duration_s: float = 90.0
    min_gold_ratio: float = 0.5
    max_gold_ratio: float = 3.0


DEFAULT_THRESHOLDS = RealnessThresholds()

# The judge shortlist depth — the report surfaces this cut so a clean candidate
# ranked beyond it reads as truncated-away, not "covered". Configurable here.
DEFAULT_SHORTLIST_DEPTH = 200

HARD_MISS = "hard_miss"
ACCIDENTAL_ONLY = "accidental_only"
CLEAN_BUT_BURIED = "clean_but_buried"
BUCKETS: tuple[str, ...] = (HARD_MISS, ACCIDENTAL_ONLY, CLEAN_BUT_BURIED)


def _duration(cand: dict) -> float:
    return float(cand.get("end_time") or 0.0) - float(cand.get("start_time") or 0.0)


def _gold_duration(gold: dict) -> float:
    return float(gold.get("end") or 0.0) - float(gold.get("start") or 0.0)


def is_real_candidate(
    gold: dict, cand: dict, thresholds: RealnessThresholds = DEFAULT_THRESHOLDS
) -> bool:
    """A clip-sized, gold-proportioned candidate — not an oversized accidental window.

    True iff the candidate window is within [min,max] seconds AND its length is
    within [min,max] x the gold window. A zero-length gold window is never real.
    """
    dur = _duration(cand)
    if not (thresholds.min_duration_s <= dur <= thresholds.max_duration_s):
        return False
    gold_dur = _gold_duration(gold)
    if gold_dur <= 0:
        return False
    return thresholds.min_gold_ratio * gold_dur <= dur <= thresholds.max_gold_ratio * gold_dur


def classify_gold(
    gold: dict,
    raw_pool: list[dict],
    *,
    ranked_pool: list[dict],
    thresholds: RealnessThresholds = DEFAULT_THRESHOLDS,
) -> dict:
    """Partition one gold moment into hard-miss / accidental-only / clean-but-buried.

    Coverage for hard-miss is decided on ``raw_pool`` (the unfiltered finder
    output) so the hard-miss set is exactly the generation_ceiling raw@any miss —
    a gold the finder never surfaced. Everything else is classified against the
    ``ranked_pool`` (the deduped, rank-ordered candidates the product actually
    ranks): a *real* covering candidate there → clean-but-buried with its best
    (shallowest, 1-based) rank as ``buried_rank``; only oversized/random windows
    cover it → accidental-only.
    """
    if not any(is_hit(gold, cand) for cand in raw_pool):
        return {"bucket": HARD_MISS, "buried_rank": None}

    real_ranks = [
        rank
        for rank, cand in enumerate(ranked_pool, start=1)
        if is_hit(gold, cand) and is_real_candidate(gold, cand, thresholds)
    ]
    if not real_ranks:
        return {"bucket": ACCIDENTAL_ONLY, "buried_rank": None}
    return {"bucket": CLEAN_BUT_BURIED, "buried_rank": min(real_ranks)}


def _buried_rank_stats(ranks: list[int]) -> dict:
    """median/min/max of the clean bucket's buried ranks (None-filled when empty)."""
    if not ranks:
        return {"median": None, "min": None, "max": None, "n": 0}
    return {
        "median": statistics.median(ranks),
        "min": min(ranks),
        "max": max(ranks),
        "n": len(ranks),
    }


def decompose_tier(
    gold_by_vod: dict[str, list[dict]],
    raw_by_vod: dict[str, list[dict]],
    pool_by_vod: dict[str, list[dict]],
    *,
    min_views: float = 0.0,
    thresholds: RealnessThresholds = DEFAULT_THRESHOLDS,
    shortlist_depth: int = DEFAULT_SHORTLIST_DEPTH,
) -> dict:
    """Classify every gold moment (views >= min_views) into the three buckets.

    Iterates the scored VODs the caller passes (coverage-gap VODs are excluded
    upstream, never folded in here as misses), classifies each gold against its
    VOD's raw + deduped pools, and aggregates counts, shares and the clean
    bucket's buried-rank stats. ``shortlist_depth`` is the judge shortlist cut:
    it is echoed verbatim and the clean bucket is split into reachable
    (``buried_rank <= depth``) vs truncated-away (``> depth``) so the cap is
    explicit, not silent.
    """
    counts = {bucket: 0 for bucket in BUCKETS}
    buried_ranks: list[int] = []
    within = beyond = 0
    gold_count = 0

    for vod_id, moments in gold_by_vod.items():
        raw_pool = raw_by_vod.get(vod_id, [])
        ranked_pool = pool_by_vod.get(vod_id, [])
        for gold in moments:
            if float(gold.get("views") or 0.0) < min_views:
                continue
            gold_count += 1
            verdict = classify_gold(gold, raw_pool, ranked_pool=ranked_pool, thresholds=thresholds)
            counts[verdict["bucket"]] += 1
            rank = verdict["buried_rank"]
            if rank is not None:
                buried_ranks.append(rank)
                if rank <= shortlist_depth:
                    within += 1
                else:
                    beyond += 1

    shares = {
        bucket: round(counts[bucket] / gold_count, 6) if gold_count else None
        for bucket in BUCKETS
    }
    return {
        "min_views": min_views,
        "shortlist_depth": shortlist_depth,
        "gold_count": gold_count,
        "buckets": counts,
        "shares": shares,
        "buried_rank": _buried_rank_stats(buried_ranks),
        "clean_within_shortlist": within,
        "clean_beyond_shortlist": beyond,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Baseline delta — every metric as a move against the frozen composite baseline
# ─────────────────────────────────────────────────────────────────────────────

def _pair(current, baseline) -> dict:
    """A {current, baseline, delta} cell; delta is None unless both are numbers."""
    delta = round(current - baseline, 6) if (current is not None and baseline is not None) else None
    return {"current": current, "baseline": baseline, "delta": delta}


def baseline_delta(current: dict, baseline: dict) -> dict:
    """raw@any, recall@K and avg_rank for both tiers, as a delta vs the baseline.

    ``current`` and ``baseline`` are both ``directional_eval.run`` results (the
    frozen one is ``finder_recall_baseline.json``). For each tier the function
    reports the generation ceiling (raw@any), every recall@K the two share, and
    avg_rank — each as a current/baseline/delta cell so a ``rank_score_fn``
    proposal is read as a move against the status quo, never an absolute number.
    """
    def tier_recall(tier: str) -> dict:
        cur = current.get(tier, {}).get("recall", {})
        base = baseline.get(tier, {}).get("recall", {})
        ks = sorted(set(cur) & set(base), key=lambda k: int(k))
        return {k: _pair(cur.get(k), base.get(k)) for k in ks}

    def tier_raw_any(tier: str):
        return _pair(
            current.get("generation", {}).get(tier, {}).get("recall_any"),
            baseline.get("generation", {}).get(tier, {}).get("recall_any"),
        )

    def tier_avg_rank(tier: str):
        return _pair(current.get(tier, {}).get("avg_rank"), baseline.get(tier, {}).get("avg_rank"))

    return {
        "raw_any": {"floor": tier_raw_any("floor"), "headline": tier_raw_any("headline")},
        "recall": {"floor": tier_recall("floor"), "headline": tier_recall("headline")},
        "avg_rank": {"floor": tier_avg_rank("floor"), "headline": tier_avg_rank("headline")},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator — decomposition + canonical metrics + baseline delta in one pass
# ─────────────────────────────────────────────────────────────────────────────

def run(
    manifest: dict[str, list[dict]],
    raw_by_vod: dict[str, list[dict]],
    pool_by_vod: dict[str, list[dict]],
    final_by_vod: dict[str, list[dict]],
    baseline: dict,
    *,
    anchor_vod_ids: list[str] | None = None,
    ks: tuple[int, ...] = directional_eval.DEFAULT_KS,
    headline_min_views: float = directional_eval.DEFAULT_HEADLINE_MIN_VIEWS,
    gold_tiers: tuple[str, ...] = (directional_eval.GOLD_TIER,),
    thresholds: RealnessThresholds = DEFAULT_THRESHOLDS,
    shortlist_depth: int = DEFAULT_SHORTLIST_DEPTH,
) -> dict:
    """Decompose the gap and carry the canonical metrics as a baseline delta.

    Reuses ``directional_eval`` for the gold filtering, the coverage-gap rule and
    the metric pass (same gold-tier + anchor restriction as the frozen baseline),
    so the decomposition and the recall numbers never drift apart. The gold VODs
    with no finder predictions are surfaced as ``missing_prediction_vods`` and
    dropped from the decomposition denominator — a visible coverage gap, not a
    silent miss. Both ``raw_by_vod`` (hard-miss coverage) and ``pool_by_vod``
    (the deduped, ranked candidates) feed the per-tier bucket counts.
    """
    gold = directional_eval.gold_manifest(manifest, gold_tiers)
    if anchor_vod_ids is not None:
        anchor_set = set(anchor_vod_ids)
        gold = {vod_id: moments for vod_id, moments in gold.items() if vod_id in anchor_set}

    missing = directional_eval.missing_prediction_vods(gold, raw_by_vod)
    scored_gold = {vod_id: moments for vod_id, moments in gold.items() if raw_by_vod.get(vod_id)}

    def tier(min_views: float) -> dict:
        return decompose_tier(
            scored_gold, raw_by_vod, pool_by_vod,
            min_views=min_views, thresholds=thresholds, shortlist_depth=shortlist_depth,
        )

    metrics = directional_eval.run(
        manifest, final_by_vod, raw_by_vod,
        anchor_vod_ids=anchor_vod_ids, ks=ks,
        headline_min_views=headline_min_views, gold_tiers=gold_tiers,
    )

    return {
        "gold_tiers": sorted(gold_tiers),
        "anchor_restricted": anchor_vod_ids is not None,
        "gold_vods": len(gold),
        "scored_vods": len(scored_gold),
        "missing_prediction_vods": missing,
        "headline_min_views": headline_min_views,
        "shortlist_depth": shortlist_depth,
        "thresholds": dataclasses.asdict(thresholds),
        "decomposition": {
            "floor": tier(0.0),
            "headline": tier(headline_min_views),
        },
        "metrics": metrics,
        "baseline_delta": baseline_delta(metrics, baseline),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Artifact bridge — replay the real artifacts into the three input dicts
# ─────────────────────────────────────────────────────────────────────────────

def build_inputs(
    anchor_ids: list[str],
    storage: pathlib.Path,
    *,
    rank_score_fn: replay_harness.RankScoreFn | None = None,
    top_k: int = replay_harness.DEFAULT_TOP_K,
) -> tuple[dict[str, list[dict]], dict[str, list[dict]], dict[str, list[dict]], list[str]]:
    """Replay each anchor VOD into (raw, deduped-pool, final) dicts + the gap list.

    For every anchor VOD it loads the newest replayable job dir and runs the real
    serving functions under ``rank_score_fn``: the raw rows feed hard-miss
    coverage, ``replay_ranked_pool`` gives the deduped, rank-ordered candidates
    the buried gold live in, and ``replay_postprocess`` gives the product final
    set for recall. A VOD with no replayable artifacts is returned in the gap
    list — never silently dropped.
    """
    raw_by_vod: dict[str, list[dict]] = {}
    pool_by_vod: dict[str, list[dict]] = {}
    final_by_vod: dict[str, list[dict]] = {}
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
        final_by_vod[vod_id] = replay_harness.replay_postprocess(
            raw_rows, segments, rank_score_fn=rank_score_fn, top_k=top_k)
    return raw_by_vod, pool_by_vod, final_by_vod, gaps


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(v: Any) -> str:
    if v is None:
        return "  —  "
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _signed(v: Any) -> str:
    if v is None:
        return "  —  "
    return f"{v:+.3f}" if isinstance(v, float) else f"{v:+}"


def _bucket_line(label: str, tier: dict) -> str:
    b, s = tier["buckets"], tier["shares"]
    cells = "  ".join(
        f"{name}={b[name]:>2} ({_fmt(s[name])})" for name in BUCKETS
    )
    return f"  {label:<9} n={tier['gold_count']:>3}  {cells}"


def _buried_line(label: str, tier: dict) -> str:
    br = tier["buried_rank"]
    return (
        f"  {label:<9} buried-rank median={_fmt(br['median'])} "
        f"(min {_fmt(br['min'])}, max {_fmt(br['max'])}, n {br['n']})  "
        f"shortlist≤{tier['shortlist_depth']}: {tier['clean_within_shortlist']} reachable, "
        f"{tier['clean_beyond_shortlist']} truncated-away"
    )


def format_report(result: dict) -> str:
    delta = result["baseline_delta"]
    k = str(directional_eval.PRODUCT_HEADLINE_K)
    lines = [
        "=" * 78,
        "Gap-Decomposition + Baseline-Delta — WHERE the recall gap lives",
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
            f"predictions — excluded from the decomposition, NOT counted as misses: "
            f"{', '.join(missing)}")
    else:
        lines.append("✓  Every gold VOD has finder predictions.")

    lines.append("")
    lines.append("  DECOMPOSITION (hard-miss = generation, accidental/buried = discrimination):")
    lines.append(_bucket_line("FLOOR", result["decomposition"]["floor"]))
    lines.append(_bucket_line("HEADLINE", result["decomposition"]["headline"]))
    lines.append("")
    lines.append(f"  CLEAN-BUT-BURIED depth (shortlist_depth={result['shortlist_depth']}):")
    lines.append(_buried_line("FLOOR", result["decomposition"]["floor"]))
    lines.append(_buried_line("HEADLINE", result["decomposition"]["headline"]))

    lines.append("")
    lines.append("  BASELINE DELTA (current − frozen composite baseline):")
    for tier in ("floor", "headline"):
        raw_any = delta["raw_any"][tier]
        recall = delta["recall"][tier].get(k, {})
        avg = delta["avg_rank"][tier]
        lines.append(
            f"  {tier.upper():<9} raw@any {_signed(raw_any['delta'])}   "
            f"product r@{k} {_signed(recall.get('delta'))}   "
            f"avg_rank {_signed(avg['delta'])}")
    lines.append("")
    lines.append(
        "Read: hard-miss is a generation gap (no candidate); accidental-only + "
        "clean-but-buried are discrimination gaps (the candidate exists, ranking buries it).")
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
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE),
                        help="Frozen composite baseline JSON to delta against")
    parser.add_argument("--ks", default="5,10,20,50", help="Recall@K comma-separated")
    parser.add_argument("--headline-min-views", type=float,
                        default=directional_eval.DEFAULT_HEADLINE_MIN_VIEWS)
    parser.add_argument("--shortlist-depth", type=int, default=DEFAULT_SHORTLIST_DEPTH,
                        help="Judge shortlist depth; the clean-bucket cut is surfaced, not silent")
    parser.add_argument("--top-k", type=int, default=replay_harness.DEFAULT_TOP_K)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    parser.add_argument("--json", dest="output_json", action="store_true")
    args = parser.parse_args(argv)

    ks = tuple(int(k.strip()) for k in args.ks.split(",") if k.strip())
    manifest = eval_clip_finder.load_manifest(pathlib.Path(args.manifest))
    anchors = eval_clip_finder.read_anchor_vod_ids(pathlib.Path(args.anchor_csv))
    baseline = json.loads(pathlib.Path(args.baseline).read_text(encoding="utf-8"))

    raw_by_vod, pool_by_vod, final_by_vod, gaps = build_inputs(
        anchors, pathlib.Path(args.predictions_dir), top_k=args.top_k)
    if gaps:
        # No silent truncation: log the artifact-less VODs here too (they also
        # surface in the result as missing_prediction_vods).
        print(
            f"gap_decomposition: {len(gaps)}/{len(anchors)} anchor VODs have no "
            f"replayable artifacts — coverage gap: {', '.join(gaps)}",
            file=sys.stderr)

    result = run(
        manifest, raw_by_vod, pool_by_vod, final_by_vod, baseline,
        anchor_vod_ids=anchors, ks=ks,
        headline_min_views=args.headline_min_views, shortlist_depth=args.shortlist_depth)
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
