"""Grouped-by-VOD K-fold Out-of-Fold (OOF) eval-runner — the Phase-2b foundation.

The walking skeleton of the Phase-2b ranker eval: it splits the eval VODs into K
folds with the **VOD as the group** (every row of a VOD — positives AND the
negatives derived from them — lands in the same fold), "trains" a *pluggable*
model per fold on the other folds, and ranks each eval VOD with the fold that did
**not** see it in training (out-of-fold). The OOF rankings go straight through the
offline replay harness (#47) -> ``gap_decomposition`` (#49) -> ``directional_eval``
so the number is unleaked even though 83% of the outcome positives sit on eval
VODs.

This slice builds **only the plumbing**: the model seam is driven by stub/fake
adapters (identity + canned reorder). It *calls* the serving/harness/decomposition
functions, it does not reimplement them; the real ranker arrives in a later issue.

The report carries a **two-metric delta** against the frozen composite baseline:
the **full-denominator headline recall@20** (Ship-Gate) and the
``clean_but_buried`` recall@20 (Engineering-Gate, via the #49 buckets), plus floor
recall@20 (Guardrail) and avg_rank. The five VODs with no replayable artifacts stay
a visible coverage gap — never silently counted as a miss — and every fold/sampling
bound is logged. No paid LLM call.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import pathlib
import sys
from typing import Any, Callable, Sequence

from ml import directional_eval, eval_clip_finder, gap_decomposition, replay_harness

logger = logging.getLogger(__name__)

# The two gates are read at the product headline column (K = clip_top_k = 20).
GATE_K = directional_eval.PRODUCT_HEADLINE_K

# Default fold count for the grouped-by-VOD split — 5 over the ~18 scored anchors.
DEFAULT_K_FOLDS = 5

# Train a (pluggable) model on the given training VOD ids and return the harness's
# pluggable ranking seam. This slice drives it with stub/fake adapters (identity +
# canned reorder); the real within-VOD pairwise ranker plugs in here in a later
# issue. == Callable[[train_vod_ids], replay_harness.RankScoreFn].
TrainFn = Callable[[Sequence[str]], replay_harness.RankScoreFn]

# A fold boundary record: fold index, training-set size, the held-out eval VODs.
FoldRecord = dict
OnFoldFn = Callable[[FoldRecord], None]


def assign_folds(vod_ids: Sequence[str], k: int) -> dict[str, int]:
    """Deterministic group=VOD fold assignment: each VOD -> exactly one fold index.

    Round-robins the *sorted* unique VOD ids across ``k`` folds, so the split is
    reproducible regardless of the anchor list's order and balanced to within one.
    The VOD is the group: assigning at VOD granularity is what keeps every row of a
    VOD (positive and derived-negative alike) in the same fold.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    ordered = sorted(set(vod_ids))
    return {vod_id: index % k for index, vod_id in enumerate(ordered)}


def _identity_rank(rows: list[dict]) -> list[dict]:
    """Pass the deduped rows through untouched — the composite order is kept."""
    return rows


def identity_train_fn(_train_vods: Sequence[str]) -> replay_harness.RankScoreFn:
    """The stub model: ignore training, rank by the existing composite order.

    Plugged into the OOF path it reproduces the frozen composite baseline (the
    skew/regression guard) — the model seam is exercised end-to-end without a
    single learned weight.
    """
    return _identity_rank


def _log_fold(record: FoldRecord) -> None:
    """Default fold sink: a structured log line so each fold bound is visible."""
    logger.info(json.dumps({"event": "oof_fold", **record}, sort_keys=True))


def build_oof_inputs(
    anchor_ids: Sequence[str],
    storage: pathlib.Path,
    folds: dict[str, int],
    train_fn: TrainFn,
    *,
    top_k: int = replay_harness.DEFAULT_TOP_K,
    on_fold: OnFoldFn | None = None,
) -> tuple[
    dict[str, list[dict]],
    dict[str, list[dict]],
    dict[str, list[dict]],
    list[str],
    dict[int, list[str]],
]:
    """Replay each eval VOD under its out-of-fold model into (raw, pool, final).

    For each distinct fold it trains a model on the anchors the fold did *not* hold
    out (``train_fn`` over those VOD ids) and logs the fold bound. Then every eval
    VOD is replayed through the real serving functions under the model of its *own*
    fold — the one whose training set excludes it — so no eval VOD is ranked by a
    fold that saw it. ``replay_ranked_pool`` gives the deduped, rank-ordered pool
    the buried gold live in and ``replay_postprocess`` the served final set. A VOD
    with no replayable artifacts is returned in the gap list, never silently
    dropped. Returns (raw, pool, final, gaps, fold_train_vods).
    """
    fold_train_vods: dict[int, list[str]] = {}
    models: dict[int, replay_harness.RankScoreFn] = {}
    for fold in sorted(set(folds.values())):
        train_vods = [vod_id for vod_id in anchor_ids if folds.get(vod_id) != fold]
        fold_train_vods[fold] = sorted(train_vods)
        models[fold] = train_fn(train_vods)
        (on_fold or _log_fold)({
            "fold": fold,
            "train_vods": len(train_vods),
            "eval_vods": sorted(v for v in anchor_ids if folds.get(v) == fold),
        })

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
        rank_score_fn = models[folds[vod_id]]
        raw_by_vod[vod_id] = raw_rows
        pool_by_vod[vod_id] = replay_harness.replay_ranked_pool(
            raw_rows, segments, rank_score_fn=rank_score_fn)
        final_by_vod[vod_id] = replay_harness.replay_postprocess(
            raw_rows, segments, rank_score_fn=rank_score_fn, top_k=top_k)
    return raw_by_vod, pool_by_vod, final_by_vod, gaps, fold_train_vods


# ─────────────────────────────────────────────────────────────────────────────
# The Engineering metric — clean_but_buried recall@K, un-confounded by the ceiling
# ─────────────────────────────────────────────────────────────────────────────

def clean_but_buried_recall(
    gold_by_vod: dict[str, list[dict]],
    raw_by_vod: dict[str, list[dict]],
    pool_by_vod: dict[str, list[dict]],
    final_by_vod: dict[str, list[dict]],
    *,
    ks: tuple[int, ...] = directional_eval.DEFAULT_KS,
    min_views: float = 0.0,
    thresholds: gap_decomposition.RealnessThresholds = gap_decomposition.DEFAULT_THRESHOLDS,
) -> dict:
    """Recall@K over only the ``clean_but_buried`` gold — the re-rankable set.

    The denominator is every gold moment (views >= ``min_views``) the #49
    classification puts in the ``clean_but_buried`` bucket: a real, clip-sized
    candidate covers it AND survives dedupe, so a ranker *could* lift it into the
    served top-K. That excludes hard-miss (no candidate — a generation gap) and
    accidental-only (only an oversized window covers), which is what makes this
    Engineering-Gate un-confounded by the generation ceiling. The numerator at K is
    those whose served final set hits them at rank <= K — measured through the same
    ``evaluate_vod`` ranks the product recall uses, so it is a true subset of floor
    recall. Membership is read off ``pool_by_vod``; since dedupe runs before the
    rank_score_fn, that set is invariant to the model, so the OOF and baseline
    denominators match and the delta is a clean recall move.
    """
    denominator = 0
    hits = {k: 0 for k in ks}
    for vod_id, gold_moments in gold_by_vod.items():
        raw_pool = raw_by_vod.get(vod_id, [])
        ranked_pool = pool_by_vod.get(vod_id, [])
        final_pool = final_by_vod.get(vod_id, [])
        gt_hits = eval_clip_finder.evaluate_vod(gold_moments, final_pool, ks, vod_id=vod_id)["gt_hits"]
        for gi, gold in enumerate(gold_moments):
            if float(gold.get("views") or 0.0) < min_views:
                continue
            verdict = gap_decomposition.classify_gold(
                gold, raw_pool, ranked_pool=ranked_pool, thresholds=thresholds)
            if verdict["bucket"] != gap_decomposition.CLEAN_BUT_BURIED:
                continue
            denominator += 1
            hit = gt_hits[gi]
            if hit["hit"] and hit["hit_rank"] is not None:
                for k in ks:
                    if hit["hit_rank"] <= k:
                        hits[k] += 1
    return {
        "min_views": min_views,
        "denominator": denominator,
        "hits": {str(k): hits[k] for k in ks},
        "recall": {
            str(k): round(hits[k] / denominator, 6) if denominator else None for k in ks
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator — OOF decomposition + the two-metric delta against the baseline
# ─────────────────────────────────────────────────────────────────────────────

def _cell(current, baseline) -> dict:
    """A {current, baseline, delta} cell; delta is None unless both are numbers."""
    delta = round(current - baseline, 6) if (current is not None and baseline is not None) else None
    return {"current": current, "baseline": baseline, "delta": delta}


def run_oof(
    manifest: dict[str, list[dict]],
    raw_by_vod: dict[str, list[dict]],
    oof_pool_by_vod: dict[str, list[dict]],
    oof_final_by_vod: dict[str, list[dict]],
    baseline_pool_by_vod: dict[str, list[dict]],
    baseline_final_by_vod: dict[str, list[dict]],
    frozen_baseline: dict,
    *,
    anchor_vod_ids: Sequence[str],
    folds: dict[str, int],
    fold_train_vods: dict[int, list[str]],
    coverage_gaps: Sequence[str] = (),
    ks: tuple[int, ...] = directional_eval.DEFAULT_KS,
    headline_min_views: float = directional_eval.DEFAULT_HEADLINE_MIN_VIEWS,
    gold_tiers: tuple[str, ...] = (directional_eval.GOLD_TIER,),
    thresholds: gap_decomposition.RealnessThresholds = gap_decomposition.DEFAULT_THRESHOLDS,
    shortlist_depth: int = gap_decomposition.DEFAULT_SHORTLIST_DEPTH,
) -> dict:
    """Score the OOF rankings and report a two-metric delta against the baseline.

    The OOF pools feed ``gap_decomposition.run`` (which reuses ``directional_eval``)
    for the canonical metrics + #49 buckets + the delta vs the frozen composite
    baseline — that yields the **Ship-Gate** (full-denominator headline recall@20),
    the **Guardrail** (floor recall@20) and avg_rank as moves against
    ``frozen_baseline``. The **Engineering-Gate** (``clean_but_buried`` recall@20,
    un-confounded by the generation ceiling) is not in the frozen artifact, so it is
    deltaed against an inline identity (composite) replay: ``baseline_pool`` /
    ``baseline_final``. Coverage-gap VODs are surfaced, never folded in as misses,
    and the fold layout is echoed so the leak-free split is auditable.
    """
    decomposition = gap_decomposition.run(
        manifest, raw_by_vod, oof_pool_by_vod, oof_final_by_vod, frozen_baseline,
        anchor_vod_ids=list(anchor_vod_ids), ks=ks,
        headline_min_views=headline_min_views, gold_tiers=gold_tiers,
        thresholds=thresholds, shortlist_depth=shortlist_depth,
    )

    # The re-rankable gold live on the gold VODs the finder actually scored.
    gold = directional_eval.gold_manifest(manifest, gold_tiers)
    anchor_set = set(anchor_vod_ids)
    scored_gold = {
        vod_id: moments
        for vod_id, moments in gold.items()
        if vod_id in anchor_set and raw_by_vod.get(vod_id)
    }

    def cbb(pool, final, min_views):
        return clean_but_buried_recall(
            scored_gold, raw_by_vod, pool, final,
            ks=ks, min_views=min_views, thresholds=thresholds)

    oof_cbb = {"floor": cbb(oof_pool_by_vod, oof_final_by_vod, 0.0),
               "headline": cbb(oof_pool_by_vod, oof_final_by_vod, headline_min_views)}
    baseline_cbb = {"floor": cbb(baseline_pool_by_vod, baseline_final_by_vod, 0.0),
                    "headline": cbb(baseline_pool_by_vod, baseline_final_by_vod, headline_min_views)}

    delta = decomposition["baseline_delta"]
    k = str(GATE_K)

    def cbb_cell(tier: str) -> dict:
        return _cell(oof_cbb[tier]["recall"].get(k), baseline_cbb[tier]["recall"].get(k))

    two_metric_delta = {
        "k": GATE_K,
        # Ship-Gate + Guardrail + avg_rank ride on the frozen-baseline delta.
        "ship_headline_recall": delta["recall"]["headline"].get(k),
        "guardrail_floor_recall": delta["recall"]["floor"].get(k),
        "avg_rank": {"floor": delta["avg_rank"]["floor"], "headline": delta["avg_rank"]["headline"]},
        # Engineering-Gate deltas vs the inline identity (composite) replay.
        "engineering_clean_but_buried_recall": {
            "floor": cbb_cell("floor"), "headline": cbb_cell("headline"),
        },
    }

    return {
        "k": GATE_K,
        "anchor_vods": list(anchor_vod_ids),
        "scored_vods": sorted(raw_by_vod),
        "coverage_gap_vods": list(coverage_gaps),
        "missing_prediction_vods": decomposition["missing_prediction_vods"],
        "folds": folds,
        "fold_train_vods": {str(fold): vods for fold, vods in fold_train_vods.items()},
        "headline_min_views": headline_min_views,
        "decomposition": decomposition,
        "clean_but_buried": {"oof": oof_cbb, "baseline": baseline_cbb},
        "two_metric_delta": two_metric_delta,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Artifact bridge — replay the real artifacts into the OOF + baseline inputs
# ─────────────────────────────────────────────────────────────────────────────

def build_report(
    anchor_ids: Sequence[str],
    storage: pathlib.Path,
    manifest: dict[str, list[dict]],
    frozen_baseline: dict,
    *,
    train_fn: TrainFn = identity_train_fn,
    k: int = DEFAULT_K_FOLDS,
    top_k: int = replay_harness.DEFAULT_TOP_K,
    ks: tuple[int, ...] = directional_eval.DEFAULT_KS,
    headline_min_views: float = directional_eval.DEFAULT_HEADLINE_MIN_VIEWS,
    gold_tiers: tuple[str, ...] = (directional_eval.GOLD_TIER,),
    thresholds: gap_decomposition.RealnessThresholds = gap_decomposition.DEFAULT_THRESHOLDS,
    shortlist_depth: int = gap_decomposition.DEFAULT_SHORTLIST_DEPTH,
    on_fold: OnFoldFn | None = None,
) -> dict:
    """Replay the real artifacts into the OOF + baseline inputs and score them.

    Splits the anchors into ``k`` folds (group=VOD), replays each VOD's identity
    (composite) pools via the #49 bridge — the baseline for the Engineering-Gate
    delta — and replays the OOF pools under ``train_fn`` (default: the identity
    stub, which reproduces the frozen baseline). Both feed ``run_oof``. The same
    18 scored VODs as the frozen baseline flow through ``directional_eval``; VODs
    with no replayable artifacts are a surfaced coverage gap.
    """
    folds = assign_folds(anchor_ids, k)
    _braw, baseline_pool, baseline_final, _bgaps = gap_decomposition.build_inputs(
        anchor_ids, storage, top_k=top_k)
    raw, oof_pool, oof_final, gaps, fold_train = build_oof_inputs(
        anchor_ids, storage, folds, train_fn, top_k=top_k, on_fold=on_fold)
    return run_oof(
        manifest, raw, oof_pool, oof_final, baseline_pool, baseline_final, frozen_baseline,
        anchor_vod_ids=list(anchor_ids), folds=folds, fold_train_vods=fold_train,
        coverage_gaps=gaps, ks=ks, headline_min_views=headline_min_views,
        gold_tiers=gold_tiers, thresholds=thresholds, shortlist_depth=shortlist_depth)


# ─────────────────────────────────────────────────────────────────────────────
# Reporting + CLI
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(value: Any) -> str:
    if value is None:
        return "  —  "
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _signed(value: Any) -> str:
    if value is None:
        return "  —  "
    return f"{value:+.3f}" if isinstance(value, float) else f"{value:+}"


def _cell_str(cell: dict | None) -> str:
    cell = cell or {}
    return f"{_fmt(cell.get('current'))} vs {_fmt(cell.get('baseline'))}   Δ {_signed(cell.get('delta'))}"


def format_report(report: dict) -> str:
    """The #49 decomposition, then the OOF two-metric delta read explicitly."""
    tm = report["two_metric_delta"]
    k = tm["k"]
    eng = tm["engineering_clean_but_buried_recall"]
    lines = [gap_decomposition.format_report(report["decomposition"]), ""]
    lines += [
        "=" * 78,
        f"OOF TWO-METRIC DELTA (grouped-by-VOD K-fold, current OOF − baseline)  K={k}",
        "=" * 78,
        f"  SHIP        headline recall@{k}            : {_cell_str(tm['ship_headline_recall'])}",
        f"  ENGINEERING clean_but_buried recall@{k} (fl): {_cell_str(eng['floor'])}",
        f"              clean_but_buried recall@{k} (hd): {_cell_str(eng['headline'])}",
        f"  GUARDRAIL   floor recall@{k}               : {_cell_str(tm['guardrail_floor_recall'])}",
        f"  avg_rank    floor {_signed(tm['avg_rank']['floor'].get('delta'))}"
        f"   headline {_signed(tm['avg_rank']['headline'].get('delta'))}",
        "",
        f"  Folds (k={len(report['fold_train_vods'])}, group=VOD): "
        f"{len(report['scored_vods'])} scored VODs",
    ]
    gaps = report["coverage_gap_vods"]
    if gaps:
        lines.append(f"  ⚠  Coverage gap (no replayable artifacts): {', '.join(gaps)}")
    lines += [
        "",
        "Read: SHIP is the product gate (confounded by the generation ceiling); "
        "ENGINEERING is the un-confounded ranker signal over the re-rankable gold.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=str(eval_clip_finder.DEFAULT_MANIFEST))
    parser.add_argument("--predictions-dir", default=str(eval_clip_finder.DEFAULT_STORAGE_DIR))
    parser.add_argument("--anchor-csv", default=str(eval_clip_finder.DEFAULT_ANCHOR_CSV))
    parser.add_argument("--baseline", default=str(gap_decomposition.DEFAULT_BASELINE),
                        help="Frozen composite baseline JSON to delta the Ship/Guardrail gates against")
    parser.add_argument("--k", type=int, default=DEFAULT_K_FOLDS, help="Number of grouped-by-VOD folds")
    parser.add_argument("--top-k", type=int, default=replay_harness.DEFAULT_TOP_K)
    parser.add_argument("--ks", default="5,10,20,50", help="Recall@K comma-separated")
    parser.add_argument("--headline-min-views", type=float,
                        default=directional_eval.DEFAULT_HEADLINE_MIN_VIEWS)
    parser.add_argument("--shortlist-depth", type=int, default=gap_decomposition.DEFAULT_SHORTLIST_DEPTH)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    parser.add_argument("--json", dest="output_json", action="store_true")
    args = parser.parse_args(argv)

    ks = tuple(int(x.strip()) for x in args.ks.split(",") if x.strip())
    manifest = eval_clip_finder.load_manifest(pathlib.Path(args.manifest))
    anchors = eval_clip_finder.read_anchor_vod_ids(pathlib.Path(args.anchor_csv))
    baseline = json.loads(pathlib.Path(args.baseline).read_text(encoding="utf-8"))

    # The stub model — this slice ships only the plumbing; no paid call, no learned
    # weight. A later issue swaps in the real ranker as train_fn.
    report = build_report(
        anchors, pathlib.Path(args.predictions_dir), manifest, baseline,
        train_fn=identity_train_fn, k=args.k, top_k=args.top_k, ks=ks,
        headline_min_views=args.headline_min_views, shortlist_depth=args.shortlist_depth)
    if report["coverage_gap_vods"]:
        print(
            f"oof_eval_runner: {len(report['coverage_gap_vods'])}/{len(anchors)} anchor VODs have "
            f"no replayable artifacts — coverage gap: {', '.join(report['coverage_gap_vods'])}",
            file=sys.stderr)
    report["computed_at"] = datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z")

    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    if args.output_json:
        print(payload)
    else:
        print(format_report(report))
        print(f"\ncomputed_at: {report['computed_at']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
