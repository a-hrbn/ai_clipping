"""Judge-Artefakt Separability-Probe — does the discrimination signal exist at all?

The free, read-only #54 follow-up to the paid judge spike. PRD #51 asks the gating
question *before* anyone builds a cheap feature-producer (Slice B): is repost
virality even separable by a content ranker, or is it partly exogenous
(clipper/title/timing) and catchable by *no* judge? The paid spike (#50) read the
final *blended* rank and went NO-GO — but it lost recall@20 to over-aggressive
top-concentration, not necessarily to missing separation. This probe isolates the
separation itself.

It re-reads the already-paid per-candidate judge scores
(``judge_spike/*.scores.json``) — no new model, no paid call — and, for every
``clean_but_buried`` gold (the #49 bucket: a real candidate covers it and survives
dedupe, just ranked deep), measures the **within-pool score-separation**: where the
judge ranked that gold's covering candidate *against its non-gold siblings in the
same pool*, as a probability-of-superiority percentile (0.5 = random, 1.0 = beats
every sibling). Independent of the final concentrated rank.

The verdict is pre-fixed, read explicitly, never rationalized after the fact:
median within-pool percentile **≥ 0.66 (upper third) → getrennt → B** (the signal
is there and semantic; a cheap feature can approximate it or distill the judge);
**below → nicht getrennt → Generation** (even a strong model does not lift the
buried gold over its siblings → the ranking track is exhausted). Reported per tier
(floor and headline).
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import statistics
import sys

from ml import directional_eval, eval_clip_finder, gap_decomposition
from ml.eval_clip_finder import is_hit

# The judged pool is the top-`depth` shortlist the paid spike scored (the buried-real
# gold median rank ~141, so 200 reaches them). Echoed so the cap is never silent.
DEFAULT_SHORTLIST_DEPTH = gap_decomposition.DEFAULT_SHORTLIST_DEPTH

# Where the paid spike persisted its parsed per-candidate scores — the artifacts
# this probe re-reads (read-only, no paid call). Mirrors judge_spike's default.
DEFAULT_ARTIFACTS_DIR = gap_decomposition.BACKEND_ROOT / "ml" / "artifacts" / "judge_spike"

# The pre-fixed separation verdict cut: the buried gold must land, in the median,
# in the upper third of their pool (clearly above the 0.5 random null) for the
# judge to count as separating them. Surfaced in the report, not rationalized.
SEPARATION_THRESHOLD = 0.66

# The two verdicts, read explicitly off the median within-pool percentile.
VERDICT_SEPARATED = "getrennt → B"
VERDICT_NOT_SEPARATED = "nicht getrennt → Generation"


def within_pool_separation(
    target_gold: dict,
    all_gold: list[dict],
    judged_pool: list[dict],
    *,
    thresholds: gap_decomposition.RealnessThresholds = gap_decomposition.DEFAULT_THRESHOLDS,
) -> dict | None:
    """Where the judge ranked one gold's covering candidate vs its non-gold siblings.

    The gold candidate is the best-scoring *real* (clip-sized, gold-proportioned)
    candidate in ``judged_pool`` that covers ``target_gold``; its judge score is
    compared against the non-gold siblings — pool candidates that hit *no* gold in
    ``all_gold``. Returns the probability-of-superiority percentile (fraction of
    siblings the gold candidate strictly outscores: 0.5 = random, 1.0 = beats all).

    Returns ``None`` when no real covering candidate for ``target_gold`` is present
    in the judged pool — the gold was not measurable here (e.g. its candidate sits
    beyond the judged shortlist depth); a surfaced sampling limit, never folded in.
    """
    gold_scores = [
        float(cand["composite_score"])
        for cand in judged_pool
        if is_hit(target_gold, cand)
        and gap_decomposition.is_real_candidate(target_gold, cand, thresholds)
    ]
    if not gold_scores:
        return None
    gold_score = max(gold_scores)

    siblings = [
        float(cand["composite_score"])
        for cand in judged_pool
        if not any(is_hit(g, cand) for g in all_gold)
    ]
    beaten = sum(1 for s in siblings if s < gold_score)
    percentile = beaten / len(siblings) if siblings else None
    return {
        "gold_score": gold_score,
        "sibling_n": len(siblings),
        "beaten": beaten,
        "percentile": percentile,
    }


def probe_tier(
    gold_by_vod: dict[str, list[dict]],
    judged_by_vod: dict[str, list[dict]],
    *,
    min_views: float = 0.0,
    thresholds: gap_decomposition.RealnessThresholds = gap_decomposition.DEFAULT_THRESHOLDS,
) -> dict:
    """Aggregate the within-pool separation over every gold (views >= min_views).

    For each gold the probe measures its covering candidate's within-pool percentile
    against the VOD's non-gold siblings. A gold with a real covering candidate in the
    judged pool is **measurable** (its percentile joins the distribution); one without
    is **unmeasurable** — surfaced, never counted as a 0.0 that would understate
    separation. The tier verdict reads the **median** of the measurable percentiles
    (the central tendency, robust to the long tail). ``min_views`` slices floor (0)
    from headline (the product view floor).
    """
    percentiles: list[float] = []
    measurable = unmeasurable = gold_count = 0

    for vod_id, gold_moments in gold_by_vod.items():
        judged_pool = judged_by_vod.get(vod_id, [])
        for gold in gold_moments:
            if float(gold.get("views") or 0.0) < min_views:
                continue
            gold_count += 1
            sep = within_pool_separation(gold, gold_moments, judged_pool, thresholds=thresholds)
            if sep is None or sep["percentile"] is None:
                unmeasurable += 1
                continue
            measurable += 1
            percentiles.append(sep["percentile"])

    return {
        "min_views": min_views,
        "gold_count": gold_count,
        "measurable": measurable,
        "unmeasurable": unmeasurable,
        "median_percentile": statistics.median(percentiles) if percentiles else None,
        "percentiles": sorted(percentiles),
    }


def classify_separation(
    median_percentile: float | None,
    *,
    threshold: float = SEPARATION_THRESHOLD,
) -> dict:
    """The pre-fixed verdict: median within-pool percentile ≥ threshold → getrennt → B.

    Read explicitly off the median, never rationalized after the fact. A ``None``
    median (no measurable clean_but_buried gold) fails closed → nicht getrennt: an
    unmeasurable signal is never read as separation. Returns the verdict, the cut,
    and a one-line reason so the read stays explicit.
    """
    separated = median_percentile is not None and median_percentile >= threshold
    verdict = VERDICT_SEPARATED if separated else VERDICT_NOT_SEPARATED
    if median_percentile is None:
        reason = (
            f"median within-pool percentile — (no measurable clean_but_buried gold) "
            f"< {threshold} → {verdict}"
        )
    else:
        reason = (
            f"median within-pool percentile {median_percentile:.3f} "
            f"{'≥' if separated else '<'} {threshold} → {verdict}"
        )
    return {
        "separated": separated,
        "median_percentile": median_percentile,
        "threshold": threshold,
        "verdict": verdict,
        "reason": reason,
    }


def top_view_buried_misses(
    gold_by_vod: dict[str, list[dict]],
    judged_by_vod: dict[str, list[dict]],
    *,
    limit: int | None = None,
    threshold: float = SEPARATION_THRESHOLD,
    thresholds: gap_decomposition.RealnessThresholds = gap_decomposition.DEFAULT_THRESHOLDS,
) -> list[dict]:
    """The highest-view buried gold, each followed individually (separated or not).

    Analogous to the #50 misses table, but per-gold within-pool separation instead
    of blended rank: every gold with a real covering candidate in the judged pool
    carries its candidate's judge score, its within-pool percentile, and whether
    that clears the cut. Gold with no measurable covering candidate are not listed
    here (they are surfaced as the unmeasurable count). Ordered by views desc so the
    biggest buried reposts are inspected one by one, not lost in the median.
    """
    misses: list[dict] = []
    for vod_id, gold_moments in gold_by_vod.items():
        judged_pool = judged_by_vod.get(vod_id, [])
        for gold in gold_moments:
            sep = within_pool_separation(gold, gold_moments, judged_pool, thresholds=thresholds)
            if sep is None or sep["percentile"] is None:
                continue
            misses.append({
                "vod_id": vod_id,
                "views": float(gold.get("views") or 0.0),
                "desc": str(gold.get("desc") or ""),
                "gt_start": float(gold.get("start") or 0.0),
                "gt_end": float(gold.get("end") or 0.0),
                "gold_score": sep["gold_score"],
                "sibling_n": sep["sibling_n"],
                "percentile": sep["percentile"],
                "separated": sep["percentile"] >= threshold,
            })
    misses.sort(key=lambda m: m["views"], reverse=True)
    return misses[:limit] if limit is not None else misses


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator — probe both tiers, read the verdict, surface the cap + gaps
# ─────────────────────────────────────────────────────────────────────────────

def run(
    manifest: dict[str, list[dict]],
    judged_by_vod: dict[str, list[dict]],
    *,
    anchor_vod_ids: list[str] | None = None,
    headline_min_views: float = directional_eval.DEFAULT_HEADLINE_MIN_VIEWS,
    gold_tiers: tuple[str, ...] = (directional_eval.GOLD_TIER,),
    shortlist_depth: int = DEFAULT_SHORTLIST_DEPTH,
    threshold: float = SEPARATION_THRESHOLD,
    miss_limit: int | None = None,
    thresholds: gap_decomposition.RealnessThresholds = gap_decomposition.DEFAULT_THRESHOLDS,
) -> dict:
    """The read-only probe end-to-end: separation per tier + the pre-fixed verdict.

    Gold-filters the manifest (same gold-tier + anchor restriction as the baseline),
    measures within-pool separation over both tiers, reads the verdict off each
    tier's median, and follows the highest-view buried gold individually. Gold VODs
    with no judged artifact are a surfaced coverage gap — never folded in as misses.
    The judged pool is the top-``shortlist_depth`` shortlist; that cap is echoed and
    the per-tier ``unmeasurable`` count (no real covering candidate inside it) makes
    its effect explicit, never a silent truncation. No paid call: it re-reads the
    persisted judge scores only.
    """
    gold = directional_eval.gold_manifest(manifest, gold_tiers)
    if anchor_vod_ids is not None:
        anchor_set = set(anchor_vod_ids)
        gold = {vod_id: moments for vod_id, moments in gold.items() if vod_id in anchor_set}

    coverage_gap_vods = sorted(vod_id for vod_id in gold if not judged_by_vod.get(vod_id))
    scored_gold = {vod_id: moments for vod_id, moments in gold.items() if judged_by_vod.get(vod_id)}

    def tier(min_views: float) -> dict:
        stats = probe_tier(scored_gold, judged_by_vod, min_views=min_views, thresholds=thresholds)
        stats["verdict"] = classify_separation(stats["median_percentile"], threshold=threshold)
        return stats

    misses = top_view_buried_misses(
        scored_gold, judged_by_vod, limit=miss_limit, threshold=threshold, thresholds=thresholds)

    return {
        "gold_tiers": sorted(gold_tiers),
        "anchor_restricted": anchor_vod_ids is not None,
        "gold_vods": len(gold),
        "scored_vods": sorted(scored_gold),
        "coverage_gap_vods": coverage_gap_vods,
        "shortlist_depth": shortlist_depth,
        "headline_min_views": headline_min_views,
        "threshold": threshold,
        "separation": {"floor": tier(0.0), "headline": tier(headline_min_views)},
        "top_view_buried_misses": misses,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Artifact bridge — re-read the persisted per-candidate judge scores (read-only)
# ─────────────────────────────────────────────────────────────────────────────

def _scores_path(artifacts_dir: pathlib.Path, vod_id: str) -> pathlib.Path:
    return artifacts_dir / f"{vod_id}.scores.json"


def load_judged_scores(artifacts_dir: pathlib.Path, vod_id: str) -> list[dict] | None:
    """The persisted judged pool for one VOD (``{vod_id}.scores.json``), or None.

    None means the spike never judged this VOD — a surfaced coverage gap, never a
    re-priced call. The file is the parsed per-candidate artifact the paid run wrote
    (window + judge composite_score), so the probe re-scores for free.
    """
    path = _scores_path(artifacts_dir, vod_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_judged_by_vod(
    artifacts_dir: pathlib.Path, vod_ids: list[str]
) -> dict[str, list[dict]]:
    """Load every VOD's persisted judged pool; VODs with no artifact are omitted
    (they surface as the run's coverage gap)."""
    out: dict[str, list[dict]] = {}
    for vod_id in vod_ids:
        scores = load_judged_scores(artifacts_dir, vod_id)
        if scores is not None:
            out[vod_id] = scores
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Reporting + CLI
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(value) -> str:
    if value is None:
        return "  —  "
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _tier_line(label: str, tier: dict) -> str:
    v = tier["verdict"]
    mark = "✅" if v["separated"] else "⛔"
    return (
        f"  {label:<9} median percentile {_fmt(tier['median_percentile'])}  "
        f"(measurable {tier['measurable']}, unmeasurable {tier['unmeasurable']} "
        f"of {tier['gold_count']} gold)  {mark} {v['verdict']}"
    )


def format_report(result: dict) -> str:
    """The within-pool separation per tier, the verdict read explicitly, then the
    highest-view buried gold individually (separated yes/no)."""
    floor = result["separation"]["floor"]
    headline = result["separation"]["headline"]
    lines = [
        "=" * 78,
        "Judge-Artefakt Separability-Probe — within-pool separation of buried gold",
        "=" * 78,
        f"Gold tiers: {','.join(result['gold_tiers'])}   Gold VODs: {result['gold_vods']}"
        f"   Scored: {len(result['scored_vods'])}"
        f"{'  (anchor-restricted)' if result.get('anchor_restricted') else ''}"
        f"   Headline: views>={int(result['headline_min_views']):,}",
    ]
    gaps = result["coverage_gap_vods"]
    if gaps:
        lines.append(
            f"⚠  Coverage gap: {len(gaps)} gold VOD(s) have NO judged artifact — "
            f"excluded, NOT counted as misses: {', '.join(gaps)}")
    else:
        lines.append("✓  Every gold VOD has a judged artifact.")

    lines += [
        "",
        f"  WITHIN-POOL SEPARATION (gold's covering candidate vs non-gold siblings; "
        f"cut ≥ {result['threshold']}):",
        _tier_line("FLOOR", floor),
        _tier_line("HEADLINE", headline),
        "",
        f"  ⚠  unmeasurable = no real covering candidate within the judged top-"
        f"{result['shortlist_depth']} shortlist (truncated beyond depth, or no real "
        f"candidate at all) — the cap is explicit, not silent.",
        "",
        f"  FLOOR    {floor['verdict']['reason']}",
        f"  HEADLINE {headline['verdict']['reason']}",
        "",
        "  HIGHEST-VIEW BURIED GOLD (each followed individually — sep=separated):",
    ]
    misses = result["top_view_buried_misses"]
    if not misses:
        lines.append("    (none measurable)")
    for miss in misses[:20]:
        sep = "sep" if miss["separated"] else "buried"
        lines.append(
            f"    {int(miss['views']):>10,}  {sep:<7} pct {_fmt(miss['percentile'])}  "
            f"score {_fmt(miss['gold_score'])}/10  {miss['vod_id']}  {miss['desc'][:44]}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=str(eval_clip_finder.DEFAULT_MANIFEST))
    parser.add_argument("--anchor-csv", default=str(eval_clip_finder.DEFAULT_ANCHOR_CSV))
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR),
                        help="Where the paid spike persisted *.scores.json (read-only)")
    parser.add_argument("--headline-min-views", type=float,
                        default=directional_eval.DEFAULT_HEADLINE_MIN_VIEWS)
    parser.add_argument("--shortlist-depth", type=int, default=DEFAULT_SHORTLIST_DEPTH,
                        help="Judged shortlist depth; the unmeasurable cap is surfaced, not silent")
    parser.add_argument("--threshold", type=float, default=SEPARATION_THRESHOLD,
                        help="Median within-pool percentile cut for getrennt → B")
    parser.add_argument("--output", type=pathlib.Path, default=None)
    parser.add_argument("--json", dest="output_json", action="store_true")
    args = parser.parse_args(argv)

    manifest = eval_clip_finder.load_manifest(pathlib.Path(args.manifest))
    anchors = eval_clip_finder.read_anchor_vod_ids(pathlib.Path(args.anchor_csv))
    artifacts_dir = pathlib.Path(args.artifacts_dir)
    judged_by_vod = load_judged_by_vod(artifacts_dir, anchors)
    print(f"separability_probe: read-only (no paid call)  judged-VODs={len(judged_by_vod)}  "
          f"depth={args.shortlist_depth}  artifacts={artifacts_dir}", file=sys.stderr)

    result = run(
        manifest, judged_by_vod, anchor_vod_ids=anchors,
        headline_min_views=args.headline_min_views, shortlist_depth=args.shortlist_depth,
        threshold=args.threshold)
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
