"""Judge-Blend Survival-Probe — does the within-pool judge signal survive the cut?

The free, read-only #55 follow-up to the #54 separability probe. #54 proved the
already-paid judge separates the ``clean_but_buried`` gold from their non-gold
siblings *within the pool* (headline median 0.737). But the paid spike (#50) went
NO-GO by reading the final *blended* rank — it lost recall@20 to over-aggressive
top-concentration. PRD #51 designed, but never ran, the one experiment that gates
whether Slice B must be built at all: a **bounded-α blend** of the same persisted
judge, asking whether that within-pool signal **survives** ``snap → dedupe →
diversity → top_k`` into the global top-20.

It re-reads the persisted per-candidate judge scores (``judge_spike/*.scores.json``
— no new model, no paid call) and, per VOD pool, **rank-normalizes** the finder
composite and the judge virality to [0,1] and convex-blends them::

    finder' = rank01(finder_composite, pool)   # ties → avg-rank
    judge'  = rank01(judge_virality,   pool)
    score   = (1 - α) * finder' + α * judge'

Run through the real harness (``replay_harness`` → ``gap_decomposition`` #49 →
``directional_eval``) over a bounded α-sweep {0, .1, .2, .3, .4, .5}, the curve
drives a pre-fixed 3-way verdict (uniform PASS → B distill · oracle PASS → B
gating-feature · else → Generation). α=0 is rank-normalized finder alone and must
reproduce the frozen ``finder_recall_baseline.json`` — a built-in sanity check.
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys
from typing import Callable

from ml import directional_eval, eval_clip_finder, gap_decomposition, replay_harness, separability_probe

# The gate column: recall is read at the product headline K (= clip_top_k = 20).
GATE_K = directional_eval.PRODUCT_HEADLINE_K

# rows -> rows; the replay harness's pluggable ranking seam (#47). The blend
# overrides each row's composite_score; the downstream diversity + ranking honour
# it. == replay_harness.RankScoreFn (re-aliased here to avoid the heavy import).
RankScoreFn = Callable[[list[dict]], list[dict]]

# The bounded α-sweep: a diagnostic *curve*, not a tuned ship value. α_cap = 0.5
# keeps the blend judge-bounded; α=0 is the rank-normalized finder-only baseline.
ALPHA_SWEEP: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
ALPHA_CAP = 0.5

# The pre-fixed 3-way gate, read mechanically off the α-curve — never rationalized
# after the fact. head recall@20 ≥ 0.208 = ≥2 of the ~7 measurable headline-gold
# lifted onto the frozen 3/24 baseline (≥2, not 1, survives a single-fold fluke);
# floor recall@20 ≥ 0.118 = strictly no fall below the baseline floor; avg_rank Δ ≤
# +5.0 = the #53 cut (the blend may not bury existing hits deeper than that).
HEAD_RECALL_GATE = 0.208
FLOOR_RECALL_GATE = 0.118
AVG_RANK_DELTA_CAP = 5.0

# The three tracks, read explicitly off the curve (uniform first, oracle as the
# leakage upper bound, else the ranking track is exhausted).
VERDICT_B_DISTILL = "uniform PASS → B (distill judge into serving-cheap feature)"
VERDICT_B_GATING = "oracle PASS → B (serve-time gating feature)"
VERDICT_GENERATION = "no PASS → Generation-Track"


def rank01(values: list[float]) -> list[float]:
    """Within-pool rank-normalize ``values`` onto [0,1]; ties take the average rank.

    Each value maps to its ascending rank percentile: the lowest → 0.0, the
    highest → 1.0, evenly spaced between. Tied values share the average of their
    ranks, so the transform is **monotonic non-decreasing** in the value — which is
    what lets the α=0 blend (finder' alone) keep the exact composite order and
    reproduce the frozen baseline. A single value normalizes to 0.0 (a one-element
    pool has no spread); an empty pool returns an empty list.
    """
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [0.0]
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # 1-based average of the tied positions i..j
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return [(r - 1) / (n - 1) for r in ranks]


def _judge_virality_for(row: dict, judged_pool: list[dict]) -> float | None:
    """The judge's composite_score for the judged candidate whose window best
    matches ``row``, or ``None`` if no judged candidate covers it.

    Joins the deduped candidate to the persisted judged pool by **window** — the
    same bidirectional overlap (``is_hit``) #54 used — because the paid spike scored
    the post-snap deduped windows themselves; identical windows match trivially, and
    the best-overlap pick stays deterministic under any float drift. ``None`` means
    the candidate sat beyond the judged shortlist (no judge entry) — a surfaced
    sampling gap the caller falls back to pure finder for, never a spurious 0.
    """
    gt = {"start": float(row.get("start_time") or 0.0), "end": float(row.get("end_time") or 0.0)}
    best: dict | None = None
    best_overlap = -1.0
    for judged in judged_pool:
        if not eval_clip_finder.is_hit(gt, judged):
            continue
        j_start = float(judged.get("start_time") or 0.0)
        j_end = float(judged.get("end_time") or 0.0)
        overlap = max(0.0, min(gt["end"], j_end) - max(gt["start"], j_start))
        if overlap > best_overlap:
            best, best_overlap = judged, overlap
    return float(best["composite_score"]) if best is not None else None


def _covers_gold(row: dict, gold: list[dict]) -> bool:
    """Whether the candidate ``row`` covers any labeled gold window (``is_hit``)."""
    return any(eval_clip_finder.is_hit(g, row) for g in gold)


def make_blend_rank_score_fn(
    judged_pool: list[dict],
    alpha: float,
    *,
    gold: list[dict] | None = None,
) -> RankScoreFn:
    """A per-VOD ``rank_score_fn`` that convex-blends finder ⊕ judge within the pool.

    Rank-normalizes the finder composite over the whole pool and the judge virality
    over the judge-matched candidates, then writes ``composite_score = (1-α)·finder'
    + α·judge'`` on every matched candidate. A candidate with no judge match keeps
    its pure finder' (effectively α=0 — never pushed down or spuriously boosted). At
    α=0 the score is finder' alone, a monotonic image of the input composite, so the
    served order is unchanged and the frozen baseline reproduces.

    ``gold`` makes it the **oracle** pass: the judge boost is restricted to
    candidates covering a labeled gold window (the eval-view labels), everyone else
    falling back to finder'. That is deliberate, un-shippable leakage — it measures
    the ceiling a perfect serve-time gating feature could buy, never a gate number.
    With ``gold=None`` (the default) it is the **uniform** pass: the serving-honest
    blend that touches every deduped candidate.
    """
    def _blend(rows: list[dict]) -> list[dict]:
        finder01 = rank01([float(r.get("composite_score") or 0.0) for r in rows])

        per_row_judge: list[float | None] = []
        for row in rows:
            jv = _judge_virality_for(row, judged_pool)
            if jv is not None and gold is not None and not _covers_gold(row, gold):
                jv = None  # oracle: a non-gold candidate keeps pure finder'
            per_row_judge.append(jv)
        matched = [i for i, jv in enumerate(per_row_judge) if jv is not None]
        judge01_matched = rank01([per_row_judge[i] for i in matched])
        judge01: list[float | None] = [None] * len(rows)
        for pos, i in enumerate(matched):
            judge01[i] = judge01_matched[pos]

        out: list[dict] = []
        for i, row in enumerate(rows):
            jv = judge01[i]
            score = finder01[i] if jv is None else (1 - alpha) * finder01[i] + alpha * jv
            out.append({**row, "composite_score": score})
        return out

    return _blend


# ─────────────────────────────────────────────────────────────────────────────
# Scoring one α-column — head/floor recall@20 + avg_rank Δ through the #49 harness
# ─────────────────────────────────────────────────────────────────────────────

def score_curve_cell(
    manifest: dict[str, list[dict]],
    raw_by_vod: dict[str, list[dict]],
    pool_by_vod: dict[str, list[dict]],
    final_by_vod: dict[str, list[dict]],
    baseline: dict,
    *,
    alpha: float,
    anchor_vod_ids: list[str] | None = None,
    ks: tuple[int, ...] = directional_eval.DEFAULT_KS,
    headline_min_views: float = directional_eval.DEFAULT_HEADLINE_MIN_VIEWS,
    gold_tiers: tuple[str, ...] = (directional_eval.GOLD_TIER,),
    thresholds: gap_decomposition.RealnessThresholds = gap_decomposition.DEFAULT_THRESHOLDS,
    shortlist_depth: int = gap_decomposition.DEFAULT_SHORTLIST_DEPTH,
) -> dict:
    """One α-column of the curve, read off ``gap_decomposition.run`` — no re-impl.

    Feeds the blended (raw, pool, final) for this α straight through the #49
    decomposition (which reuses ``directional_eval``) and extracts the gate cell:
    head recall@20, floor recall@20, and the avg_rank Δ vs the frozen baseline for
    both tiers. ``decomposition`` is carried through verbatim so the report can show
    the buckets + coverage gap the metric rides on.
    """
    decomposition = gap_decomposition.run(
        manifest, raw_by_vod, pool_by_vod, final_by_vod, baseline,
        anchor_vod_ids=anchor_vod_ids, ks=ks, headline_min_views=headline_min_views,
        gold_tiers=gold_tiers, thresholds=thresholds, shortlist_depth=shortlist_depth,
    )
    k = str(GATE_K)
    metrics = decomposition["metrics"]
    avg_rank_delta = decomposition["baseline_delta"]["avg_rank"]
    return {
        "alpha": alpha,
        "head_recall": metrics["headline"]["recall"].get(k),
        "floor_recall": metrics["floor"]["recall"].get(k),
        "avg_rank_delta": {
            "floor": avg_rank_delta["floor"]["delta"],
            "headline": avg_rank_delta["headline"]["delta"],
        },
        "decomposition": decomposition,
    }


# ─────────────────────────────────────────────────────────────────────────────
# The pre-fixed 3-way verdict — read mechanically off the α-curve
# ─────────────────────────────────────────────────────────────────────────────

def _in_bounded_sweep(alpha: float) -> bool:
    """α ∈ (0, α_cap]: the baseline column (α=0) is never a *judge* win, and the
    blend stays bounded at α_cap — both ends are excluded from the PASS search."""
    return 0.0 < alpha <= ALPHA_CAP


def _uniform_passes(cell: dict) -> bool:
    """A uniform α column clears the gate: head ≥ 0.208 AND floor ≥ 0.118 AND the
    worse of the two avg_rank Δ ≤ +5.0 (a None Δ = no hits to bury → no constraint)."""
    if not _in_bounded_sweep(cell["alpha"]):
        return False
    head, floor = cell["head_recall"], cell["floor_recall"]
    if head is None or head < HEAD_RECALL_GATE:
        return False
    if floor is None or floor < FLOOR_RECALL_GATE:
        return False
    deltas = [d for d in cell["avg_rank_delta"].values() if d is not None]
    if deltas and max(deltas) > AVG_RANK_DELTA_CAP:
        return False
    return True


def _oracle_passes(cell: dict) -> bool:
    """The oracle upper bound clears on head recall alone — it is the labeled ceiling
    a perfect gating feature could buy, so floor/avg_rank guardrails do not apply."""
    if not _in_bounded_sweep(cell["alpha"]):
        return False
    head = cell["head_recall"]
    return head is not None and head >= HEAD_RECALL_GATE


def classify_verdict(uniform_curve: list[dict], oracle_curve: list[dict]) -> dict:
    """The pre-fixed 3-way gate, read mechanically off the two α-curves.

    **Uniform PASS** (∃ α∈(0, 0.5] clearing head/floor/avg_rank) → B (distill the
    judge into a serving-cheap feature; the bounded-α paid judge is the interim).
    Else **Oracle PASS** (∃ α∈(0, 0.5] whose *leakage* head recall clears 0.208) →
    B (the value is exactly the serve-time gating feature). Else → Generation-Track
    (even the labeled upper bound cannot lift the buried gold; the ranking track is
    exhausted, matching #54's floor and #50's NO-GO). Each gate is surfaced.
    """
    u_alpha = next((c["alpha"] for c in uniform_curve if _uniform_passes(c)), None)
    o_alpha = next((c["alpha"] for c in oracle_curve if _oracle_passes(c)), None)

    if u_alpha is not None:
        verdict = VERDICT_B_DISTILL
    elif o_alpha is not None:
        verdict = VERDICT_B_GATING
    else:
        verdict = VERDICT_GENERATION

    def _gate(alpha: float | None, *, leakage: bool) -> dict:
        if alpha is None:
            reason = (
                f"no α∈(0,{ALPHA_CAP}] cleared head recall@20 ≥ {HEAD_RECALL_GATE}"
                + ("" if leakage else f" + floor ≥ {FLOOR_RECALL_GATE} + avg_rank Δ ≤ +{AVG_RANK_DELTA_CAP}")
            )
            return {"passed": False, "alpha": None, "reason": reason}
        kind = "oracle (leakage upper-bound)" if leakage else "uniform"
        return {"passed": True, "alpha": alpha,
                "reason": f"{kind} α={alpha} cleared the gate"}

    return {
        "verdict": verdict,
        "thresholds": {
            "head_recall_gate": HEAD_RECALL_GATE,
            "floor_recall_gate": FLOOR_RECALL_GATE,
            "avg_rank_delta_cap": AVG_RANK_DELTA_CAP,
            "alpha_cap": ALPHA_CAP,
        },
        "uniform": _gate(u_alpha, leakage=False),
        "oracle": _gate(o_alpha, leakage=True),
        "reason": verdict,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator — sweep α over both passes through the real harness, read the gate
# ─────────────────────────────────────────────────────────────────────────────

def _blend_inputs(
    loaded: dict[str, tuple[list[dict], list]],
    judged_by_vod: dict[str, list[dict]],
    alpha: float,
    *,
    gold_by_vod: dict[str, list[dict]] | None,
    top_k: int,
) -> tuple[dict[str, list[dict]], dict[str, list[dict]], dict[str, list[dict]]]:
    """Replay every pre-loaded VOD under its per-VOD blend fn into (raw, pool, final).

    The blend fn is closed over the VOD's own judged pool (and, for the oracle pass,
    its labeled gold), so the within-pool rank-normalize is automatically per-VOD —
    the replay calls the fn on one VOD's rows at a time. An un-judged VOD blends to
    pure finder' (identity), contributing its baseline ranking, never dropped.
    """
    raw_by_vod: dict[str, list[dict]] = {}
    pool_by_vod: dict[str, list[dict]] = {}
    final_by_vod: dict[str, list[dict]] = {}
    for vod_id, (raw_rows, segments) in loaded.items():
        # Oracle pass: a VOD with no labeled gold scopes to gold=[] (finder-only), not
        # to None (which would silently fall back to the uniform blend).
        gold = None if gold_by_vod is None else gold_by_vod.get(vod_id, [])
        fn = make_blend_rank_score_fn(judged_by_vod.get(vod_id, []), alpha, gold=gold)
        raw_by_vod[vod_id] = raw_rows
        pool_by_vod[vod_id] = replay_harness.replay_ranked_pool(raw_rows, segments, rank_score_fn=fn)
        final_by_vod[vod_id] = replay_harness.replay_postprocess(
            raw_rows, segments, rank_score_fn=fn, top_k=top_k)
    return raw_by_vod, pool_by_vod, final_by_vod


def _public_cell(cell: dict) -> dict:
    """The reportable α-column: drop the heavy decomposition, keep the gate numbers."""
    return {k: v for k, v in cell.items() if k != "decomposition"}


def _alpha0_sanity(uniform_curve: list[dict], baseline: dict) -> dict:
    """Whether the α=0 column reproduces the frozen baseline head/floor recall@20.

    α=0 is rank-normalized finder alone — a monotonic image of the composite — so it
    must reproduce ``finder_recall_baseline.json`` exactly. Surfacing the check makes
    any drift loud instead of silent.
    """
    k = str(GATE_K)
    cell0 = next((c for c in uniform_curve if c["alpha"] == 0.0), None)
    base_head = baseline.get("headline", {}).get("recall", {}).get(k)
    base_floor = baseline.get("floor", {}).get("recall", {}).get(k)
    cur_head = cell0["head_recall"] if cell0 else None
    cur_floor = cell0["floor_recall"] if cell0 else None
    return {
        "alpha": 0.0,
        "reproduces_baseline": cur_head == base_head and cur_floor == base_floor,
        "head_recall": {"current": cur_head, "baseline": base_head},
        "floor_recall": {"current": cur_floor, "baseline": base_floor},
    }


def run(
    manifest: dict[str, list[dict]],
    anchor_ids: list[str],
    storage: pathlib.Path,
    judged_by_vod: dict[str, list[dict]],
    baseline: dict,
    *,
    alphas: tuple[float, ...] = ALPHA_SWEEP,
    top_k: int = replay_harness.DEFAULT_TOP_K,
    ks: tuple[int, ...] = directional_eval.DEFAULT_KS,
    headline_min_views: float = directional_eval.DEFAULT_HEADLINE_MIN_VIEWS,
    gold_tiers: tuple[str, ...] = (directional_eval.GOLD_TIER,),
    thresholds: gap_decomposition.RealnessThresholds = gap_decomposition.DEFAULT_THRESHOLDS,
    shortlist_depth: int = gap_decomposition.DEFAULT_SHORTLIST_DEPTH,
) -> dict:
    """The read-only survival probe end-to-end: the bounded-α curve + the 3-way gate.

    Loads each anchor VOD's replay inputs once, then for every α sweeps **two**
    passes through the real harness (replay → #49 → directional_eval): the
    serving-honest **uniform** blend (touches every deduped candidate — the gate)
    and the leakage **oracle** blend (only labeled headline-gold candidates — the
    upper bound, never a gate number). The curve drives ``classify_verdict``; the
    α=0 column is asserted against the frozen baseline; gold VODs with no judged
    artifact are a surfaced coverage gap. No paid call — only persisted scores.
    """
    gold = directional_eval.gold_manifest(manifest, gold_tiers)
    anchor_set = set(anchor_ids)
    gold = {vod_id: moments for vod_id, moments in gold.items() if vod_id in anchor_set}

    loaded: dict[str, tuple[list[dict], list]] = {}
    for vod_id in anchor_ids:
        inputs = replay_harness.load_replay_inputs(vod_id, storage)
        if inputs is not None:
            loaded[vod_id] = inputs

    coverage_gap_vods = sorted(vod_id for vod_id in gold if vod_id not in loaded)
    judge_coverage_gap_vods = sorted(vod_id for vod_id in loaded if not judged_by_vod.get(vod_id))

    # The oracle scopes the blend to the labeled headline-gold windows per VOD.
    headline_gold_by_vod = {
        vod_id: [m for m in moments if float(m.get("views") or 0.0) >= headline_min_views]
        for vod_id, moments in gold.items()
    }

    def _score(gold_by_vod: dict[str, list[dict]] | None) -> list[dict]:
        curve: list[dict] = []
        for alpha in alphas:
            raw, pool, final = _blend_inputs(
                loaded, judged_by_vod, alpha, gold_by_vod=gold_by_vod, top_k=top_k)
            curve.append(score_curve_cell(
                manifest, raw, pool, final, baseline, alpha=alpha, anchor_vod_ids=anchor_ids,
                ks=ks, headline_min_views=headline_min_views, gold_tiers=gold_tiers,
                thresholds=thresholds, shortlist_depth=shortlist_depth))
        return curve

    uniform_curve = _score(None)
    oracle_curve = _score(headline_gold_by_vod)
    verdict = classify_verdict(uniform_curve, oracle_curve)

    return {
        "gold_tiers": sorted(gold_tiers),
        "anchor_restricted": True,
        "gold_vods": len(gold),
        "scored_vods": sorted(loaded),
        "coverage_gap_vods": coverage_gap_vods,
        "judge_coverage_gap_vods": judge_coverage_gap_vods,
        "shortlist_depth": shortlist_depth,
        "headline_min_views": headline_min_views,
        "alpha_sweep": list(alphas),
        "alpha_cap": ALPHA_CAP,
        "uniform_curve": [_public_cell(c) for c in uniform_curve],
        "oracle_curve": [_public_cell(c) for c in oracle_curve],
        "alpha0_sanity": _alpha0_sanity(uniform_curve, baseline),
        "verdict": verdict,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reporting + CLI
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(value) -> str:
    if value is None:
        return "  —  "
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _signed(value) -> str:
    if value is None:
        return "  —  "
    return f"{value:+.3f}" if isinstance(value, float) else f"{value:+}"


def _curve_lines(label: str, curve: list[dict]) -> list[str]:
    lines = [f"  {label}:", "    α     head@20   floor@20   avg_rank Δ (fl / hd)"]
    for c in curve:
        ar = c["avg_rank_delta"]
        lines.append(
            f"    {c['alpha']:<5} {_fmt(c['head_recall']):>7}   {_fmt(c['floor_recall']):>7}   "
            f"{_signed(ar['floor'])} / {_signed(ar['headline'])}")
    return lines


def format_report(result: dict) -> str:
    """The two α-curves (oracle marked leakage), the α=0 sanity, the 3-way verdict."""
    v = result["verdict"]
    lines = [
        "=" * 78,
        "Judge-Blend Survival-Probe — bounded-α blend, does the signal survive top-20?",
        "=" * 78,
        f"Gold VODs: {result['gold_vods']}   Scored: {len(result['scored_vods'])}"
        f"   Headline: views>={int(result['headline_min_views']):,}"
        f"   α∈{result['alpha_sweep']} (cap {result['alpha_cap']})",
    ]
    gaps = result["coverage_gap_vods"]
    if gaps:
        lines.append(
            f"⚠  Coverage gap: {len(gaps)} gold VOD(s) with NO judged/replay artifact — "
            f"surfaced, NOT counted as misses: {', '.join(gaps)}")
    else:
        lines.append("✓  Every gold VOD has a judged/replay artifact.")
    if result["judge_coverage_gap_vods"]:
        lines.append(
            f"⚠  {len(result['judge_coverage_gap_vods'])} replayable VOD(s) un-judged → "
            f"pure finder' (measured, not counted): {', '.join(result['judge_coverage_gap_vods'])}")

    lines += [""]
    lines += _curve_lines("UNIFORM (serving-honest — THE GATE)", result["uniform_curve"])
    lines += [""]
    lines += _curve_lines("ORACLE (leakage UPPER BOUND — never a gate number)", result["oracle_curve"])

    s = result["alpha0_sanity"]
    mark = "✅" if s["reproduces_baseline"] else "❌"
    lines += [
        "",
        f"  {mark} α=0 sanity: head {_fmt(s['head_recall']['current'])} vs baseline "
        f"{_fmt(s['head_recall']['baseline'])}   floor {_fmt(s['floor_recall']['current'])} vs "
        f"{_fmt(s['floor_recall']['baseline'])}  (must reproduce finder_recall_baseline.json)",
        "",
        f"  THRESHOLDS  head ≥ {v['thresholds']['head_recall_gate']}  "
        f"floor ≥ {v['thresholds']['floor_recall_gate']}  "
        f"avg_rank Δ ≤ +{v['thresholds']['avg_rank_delta_cap']}  α_cap {v['thresholds']['alpha_cap']}",
        f"  UNIFORM  {'✅' if v['uniform']['passed'] else '⛔'} {v['uniform']['reason']}",
        f"  ORACLE   {'✅' if v['oracle']['passed'] else '⛔'} {v['oracle']['reason']}",
        "",
        f"  ➤ VERDICT: {v['verdict']}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=str(eval_clip_finder.DEFAULT_MANIFEST))
    parser.add_argument("--anchor-csv", default=str(eval_clip_finder.DEFAULT_ANCHOR_CSV))
    parser.add_argument("--predictions-dir", default=str(eval_clip_finder.DEFAULT_STORAGE_DIR))
    parser.add_argument("--baseline", default=str(gap_decomposition.DEFAULT_BASELINE),
                        help="Frozen composite baseline JSON; the α=0 column must reproduce it")
    parser.add_argument("--artifacts-dir", default=str(separability_probe.DEFAULT_ARTIFACTS_DIR),
                        help="Where the paid spike persisted *.scores.json (read-only, no paid call)")
    parser.add_argument("--alphas", default=",".join(str(a) for a in ALPHA_SWEEP),
                        help="Bounded α-sweep, comma-separated (α_cap = 0.5)")
    parser.add_argument("--top-k", type=int, default=replay_harness.DEFAULT_TOP_K)
    parser.add_argument("--ks", default="5,10,20,50", help="Recall@K comma-separated")
    parser.add_argument("--headline-min-views", type=float,
                        default=directional_eval.DEFAULT_HEADLINE_MIN_VIEWS)
    parser.add_argument("--shortlist-depth", type=int, default=gap_decomposition.DEFAULT_SHORTLIST_DEPTH)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    parser.add_argument("--json", dest="output_json", action="store_true")
    args = parser.parse_args(argv)

    alphas = tuple(float(x.strip()) for x in args.alphas.split(",") if x.strip())
    ks = tuple(int(x.strip()) for x in args.ks.split(",") if x.strip())
    manifest = eval_clip_finder.load_manifest(pathlib.Path(args.manifest))
    anchors = eval_clip_finder.read_anchor_vod_ids(pathlib.Path(args.anchor_csv))
    baseline = json.loads(pathlib.Path(args.baseline).read_text(encoding="utf-8"))
    artifacts_dir = pathlib.Path(args.artifacts_dir)
    judged_by_vod = separability_probe.load_judged_by_vod(artifacts_dir, anchors)
    print(f"judge_blend_probe: read-only (no paid call)  judged-VODs={len(judged_by_vod)}  "
          f"α={list(alphas)}  artifacts={artifacts_dir}", file=sys.stderr)

    result = run(
        manifest, anchors, pathlib.Path(args.predictions_dir), judged_by_vod, baseline,
        alphas=alphas, top_k=args.top_k, ks=ks, headline_min_views=args.headline_min_views,
        shortlist_depth=args.shortlist_depth)
    if result["coverage_gap_vods"]:
        print(
            f"judge_blend_probe: {len(result['coverage_gap_vods'])} gold VOD(s) have no "
            f"judged/replay artifact — coverage gap: {', '.join(result['coverage_gap_vods'])}",
            file=sys.stderr)
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
