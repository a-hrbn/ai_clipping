"""Paid Judge-Spike orchestrator — the one paid decision-spike behind PRD #46.

Wires the three offline building blocks into the single end-to-end run that tests
the only hypothesis offline replay cannot: does a strong-model *repost*-virality
judge (#48) lift the clean-but-buried gold into the served top-20? It runs the
judge over the deduped top-~200 of each anchor VOD (the shortlist the buried-real
gold actually sit in — median rank ~141), plugs the per-candidate scores into the
replay harness (#47) as its ``rank_score_fn``, scores the result through
``directional_eval``, decomposes the gap as a delta vs the frozen composite
baseline (#49), classifies the highest-view misses, and reads a pre-fixed gate.

Two-Adapter-Disziplin carries through from the judge: the real run binds the
provider to paid Sonnet and persists every raw response; a re-eval binds the
provider to those persisted responses, so the spike is re-scorable without paying
again. Every shortlist cut is surfaced, never silent (no candidate beyond the
depth is back-filled with a default).

The gate and the Go/No-Go are read by a human — this module *prepares* the run and
*computes* the gate verdict; it does not decide Phase 2a vs 2b.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import pathlib
import sys
from typing import Callable

from ml import directional_eval, eval_clip_finder, gap_decomposition, replay_harness
from ml.eval_clip_finder import is_hit
from services.moment_finder import _rank_post_processed_rows
from ml import repost_virality_judge
from ml.repost_virality_judge import CallProviderFn, RerankFn, Row

logger = logging.getLogger(__name__)

# Where the paid run persists each VOD's raw response + parsed per-candidate scores,
# so a later --replay re-scores without paying again.
DEFAULT_ARTIFACTS_DIR = gap_decomposition.BACKEND_ROOT / "ml" / "artifacts" / "judge_spike"

# The gate is read at the product headline column (K = clip_top_k = 20).
GATE_K = directional_eval.PRODUCT_HEADLINE_K

# rows -> rows: the harness's pluggable ranking seam (== replay_harness.RankScoreFn).
RankScoreFn = Callable[[list[Row]], list[Row]]

# A shortlist cut record: pool_size, depth, shortlist_size, cut, scored, judge_dropped.
ShortlistCut = dict
OnCutFn = Callable[[ShortlistCut], None]

# The judge shortlist depth. ~200 because the buried-real gold median at rank ~141
# — a top-80 shortlist would not see them. Mirrors gap_decomposition's default.
DEFAULT_SHORTLIST_DEPTH = 200


def memoized_provider(provider: CallProviderFn) -> CallProviderFn:
    """Wrap a provider so each distinct prompt is paid for exactly once.

    The spike ranks each VOD's rescored pool twice — once for the ranked pool, once
    for the post-processed final set — and both build the *identical* judge prompt
    from the same shortlist. Memoizing on the prompt collapses that into a single
    paid call per VOD, so the double read never doubles the bill.
    """
    cache: dict[str, str] = {}

    def _call(prompt: str) -> str:
        if prompt not in cache:
            cache[prompt] = provider(prompt)
        return cache[prompt]

    return _call


def persisted_response_provider(raw: str) -> CallProviderFn:
    """A network-free provider that replays one persisted raw judge response.

    Re-evaluation binds the real judge adapter's provider seam to this instead of
    paid Sonnet: the prompt is ignored and the saved raw text is returned, so the
    adapter's own parser decodes it on the *same* path the paid run used — the spike
    is re-scorable without paying again (PRD #46).
    """

    def _call(_prompt: str) -> str:
        return raw

    return _call


def _log_cut(cut: ShortlistCut) -> None:
    """Default cut sink: structured log, mirroring the judge's own event line."""
    logger.info(json.dumps({"event": "judge_shortlist_cut", **cut}, sort_keys=True))


def judge_shortlist_rerank(
    judge: RerankFn,
    *,
    depth: int = DEFAULT_SHORTLIST_DEPTH,
    on_cut: OnCutFn | None = None,
) -> RankScoreFn:
    """Scope a per-candidate judge to the deduped top-``depth``, cutting the tail.

    Returns a harness ``rank_score_fn``: it ranks the deduped pool by the current
    composite order (the real serving rank), keeps the top ``depth``, and runs the
    judge over exactly those — so the judge's 1-based ids line up with the
    shortlist positions. Candidates ranked beyond ``depth`` are cut: never scored,
    never back-filled with a default that would distort top-k (PRD #46).

    Both reductions are surfaced through ``on_cut`` (default: a structured log):
    the depth ``cut`` AND the ``judge_dropped`` count of shortlisted candidates the
    fail-closed judge did not score — so a truncated response never reads as
    "scored everything".
    """

    def _rank_score_fn(rows: list[Row]) -> list[Row]:
        ranked = _rank_post_processed_rows(rows)
        shortlist = ranked[:depth]
        scored = judge(shortlist)
        (on_cut or _log_cut)({
            "pool_size": len(ranked),
            "depth": depth,
            "shortlist_size": len(shortlist),
            "cut": max(0, len(ranked) - depth),
            "scored": len(scored),
            "judge_dropped": len(shortlist) - len(scored),
        })
        return scored

    return _rank_score_fn


# ─────────────────────────────────────────────────────────────────────────────
# The pre-fixed gate — read explicitly, never rationalized after the fact
# ─────────────────────────────────────────────────────────────────────────────

def _recall_at_gate_k(eval_result: dict, tier: str) -> float | None:
    """The tier's product recall@GATE_K from a directional_eval.run result."""
    value = eval_result.get(tier, {}).get("recall", {}).get(str(GATE_K))
    return float(value) if value is not None else None


def _cell(current: float | None, baseline: float | None) -> dict:
    """A {current, baseline, delta} cell; delta is None unless both are numbers."""
    delta = round(current - baseline, 6) if (current is not None and baseline is not None) else None
    return {"current": current, "baseline": baseline, "delta": delta}


def evaluate_gate(current: dict, baseline: dict) -> dict:
    """The pre-fixed PRD #46 gate: headline recall@20 ↑ AND floor recall@20 not ↓.

    Both args are ``directional_eval.run`` results (the frozen one is
    ``finder_recall_baseline.json``). The judge run passes iff the headline tier's
    product recall@20 rises measurably (delta > 0) AND the floor tier's does not
    regress (delta >= 0). A missing recall on either side fails closed — an
    unmeasurable move is never read as an improvement. Returns the verdict, both
    delta cells, and a one-line reason so the read is explicit, not rationalized.
    """
    headline = _cell(_recall_at_gate_k(current, "headline"), _recall_at_gate_k(baseline, "headline"))
    floor = _cell(_recall_at_gate_k(current, "floor"), _recall_at_gate_k(baseline, "floor"))

    headline_up = headline["delta"] is not None and headline["delta"] > 0
    floor_held = floor["delta"] is not None and floor["delta"] >= 0
    passed = headline_up and floor_held

    reason = (
        f"headline r@{GATE_K} {_signed(headline['delta'])} ({'↑' if headline_up else 'not ↑'}) "
        f"AND floor r@{GATE_K} {_signed(floor['delta'])} ({'held' if floor_held else 'dropped'}) "
        f"→ {'PASS → Phase 2a' if passed else 'NO-GO → Phase 2b'}"
    )
    return {
        "passed": passed,
        "k": GATE_K,
        "headline_recall_at_20": headline,
        "floor_recall_at_20": floor,
        "reason": reason,
    }


def _signed(delta: float | None) -> str:
    return "—" if delta is None else f"{delta:+.6g}"


# ─────────────────────────────────────────────────────────────────────────────
# Highest-view misses — generation (no candidate) vs ranking (buried candidate)
# ─────────────────────────────────────────────────────────────────────────────

# The gap_decomposition bucket that means "no candidate ever covered the gold" —
# a generation miss; every other bucket is a ranking miss (the candidate exists).
_GENERATION = "generation"
_RANKING = "ranking"


def top_view_misses(
    gold_by_vod: dict[str, list[dict]],
    final_by_vod: dict[str, list[dict]],
    raw_by_vod: dict[str, list[dict]],
    pool_by_vod: dict[str, list[dict]],
    *,
    limit: int | None = None,
    thresholds: gap_decomposition.RealnessThresholds = gap_decomposition.DEFAULT_THRESHOLDS,
) -> list[dict]:
    """Gold the served top-k missed, ranked by views, each classified gen vs ranking.

    A gold moment is a miss when no candidate in its VOD's served ``final`` set
    hits it. Each miss is classified through ``gap_decomposition.classify_gold``:
    the hard-miss bucket (no raw candidate covers the window at all) is a
    *generation* miss; accidental-only / clean-but-buried (a candidate exists,
    ranking buries it) is a *ranking* miss, carrying the candidate's best rank as
    ``buried_rank``. Ordered by views desc so the highest-view misses (e.g. the
    1.4M ChainsFR reveal) are followed individually, not lost in an average.
    """
    misses: list[dict] = []
    for vod_id, gold_moments in gold_by_vod.items():
        raw_pool = raw_by_vod.get(vod_id)
        if not raw_pool:
            # No finder predictions for this VOD — a coverage gap, surfaced
            # elsewhere, never folded in as a generation miss (mirrors
            # gap_decomposition's scored_gold rule).
            continue
        final_pool = final_by_vod.get(vod_id, [])
        ranked_pool = pool_by_vod.get(vod_id, [])
        for gold in gold_moments:
            if any(is_hit(gold, cand) for cand in final_pool):
                continue  # served — not a miss
            verdict = gap_decomposition.classify_gold(
                gold, raw_pool, ranked_pool=ranked_pool, thresholds=thresholds)
            bucket = verdict["bucket"]
            misses.append({
                "vod_id": vod_id,
                "views": float(gold.get("views") or 0.0),
                "desc": str(gold.get("desc") or ""),
                "gt_start": float(gold.get("start") or 0.0),
                "gt_end": float(gold.get("end") or 0.0),
                "miss_type": _GENERATION if bucket == gap_decomposition.HARD_MISS else _RANKING,
                "bucket": bucket,
                "buried_rank": verdict["buried_rank"],
            })
    misses.sort(key=lambda m: m["views"], reverse=True)
    return misses[:limit] if limit is not None else misses


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator — judge each VOD's top-`depth`, score, decompose, read the gate
# ─────────────────────────────────────────────────────────────────────────────

# vod_id -> the per-VOD judge (paid Sonnet+persist, or replay-from-disk).
MakeJudgeFn = Callable[[str], RerankFn]
# vod_id -> (raw_rows, segments) or None when the VOD has no replayable artifacts.
LoadInputsFn = Callable[[str], "tuple[list[dict], list] | None"]


def run_judge_spike(
    anchor_ids: list[str],
    manifest: dict[str, list[dict]],
    baseline: dict,
    *,
    make_judge: MakeJudgeFn,
    load_inputs: LoadInputsFn,
    depth: int = DEFAULT_SHORTLIST_DEPTH,
    top_k: int = replay_harness.DEFAULT_TOP_K,
    ks: tuple[int, ...] = directional_eval.DEFAULT_KS,
    headline_min_views: float = directional_eval.DEFAULT_HEADLINE_MIN_VIEWS,
    gold_tiers: tuple[str, ...] = (directional_eval.GOLD_TIER,),
    miss_limit: int | None = None,
) -> dict:
    """The end-to-end spike: judge the top-``depth`` of each VOD, then score+decompose.

    For every anchor VOD it builds the per-VOD judge (``make_judge``), scopes it to
    the deduped top-``depth`` via ``judge_shortlist_rerank``, and runs the real
    serving replay (#47) under it to get the deduped pool and the served final set.
    The judge is invoked once per VOD for both reads — a memoized provider in the
    paid wiring keeps that a single paid call. The aggregated pools feed
    ``gap_decomposition.run`` for the decomposition + baseline delta (#49), the
    pre-fixed ``evaluate_gate``, and ``top_view_misses``. VODs with no replayable
    artifacts are a surfaced coverage gap, never silent misses; the per-VOD
    shortlist cut is recorded.

    Identical inputs + identical persisted responses ⇒ identical report, so a free
    replay reproduces the paid run bit-for-bit.
    """
    raw_by_vod: dict[str, list[dict]] = {}
    pool_by_vod: dict[str, list[dict]] = {}
    final_by_vod: dict[str, list[dict]] = {}
    shortlist_cuts: dict[str, ShortlistCut] = {}
    gaps: list[str] = []

    for vod_id in anchor_ids:
        loaded = load_inputs(vod_id)
        if loaded is None:
            gaps.append(vod_id)
            continue
        raw_rows, segments = loaded
        judge = make_judge(vod_id)
        rank_score_fn = judge_shortlist_rerank(
            judge, depth=depth,
            on_cut=lambda cut, v=vod_id: shortlist_cuts.__setitem__(v, cut),
        )
        raw_by_vod[vod_id] = raw_rows
        pool_by_vod[vod_id] = replay_harness.replay_ranked_pool(
            raw_rows, segments, rank_score_fn=rank_score_fn)
        final_by_vod[vod_id] = replay_harness.replay_postprocess(
            raw_rows, segments, rank_score_fn=rank_score_fn, top_k=top_k)

    decomposition = gap_decomposition.run(
        manifest, raw_by_vod, pool_by_vod, final_by_vod, baseline,
        anchor_vod_ids=list(anchor_ids), ks=ks,
        headline_min_views=headline_min_views, gold_tiers=tuple(gold_tiers),
        shortlist_depth=depth,
    )
    metrics = decomposition["metrics"]
    gate = evaluate_gate(metrics, baseline)

    gold = directional_eval.gold_manifest(manifest, tuple(gold_tiers))
    anchor_set = set(anchor_ids)
    gold = {vod_id: moments for vod_id, moments in gold.items() if vod_id in anchor_set}
    misses = top_view_misses(gold, final_by_vod, raw_by_vod, pool_by_vod, limit=miss_limit)

    return {
        "anchor_vods": list(anchor_ids),
        "scored_vods": sorted(raw_by_vod),
        "coverage_gap_vods": gaps,
        "shortlist_depth": depth,
        "top_k": top_k,
        "shortlist_cuts": shortlist_cuts,
        "decomposition": decomposition,
        "gate": gate,
        "top_view_misses": misses,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Paid / replay wiring — the two make_judge factories + artifact persistence
# ─────────────────────────────────────────────────────────────────────────────

def _raw_path(artifacts_dir: pathlib.Path, vod_id: str) -> pathlib.Path:
    return artifacts_dir / f"{vod_id}.raw.txt"


def _scores_path(artifacts_dir: pathlib.Path, vod_id: str) -> pathlib.Path:
    return artifacts_dir / f"{vod_id}.scores.json"


def _write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _persist_scores(path: pathlib.Path, scored_rows: list[Row]) -> None:
    """Persist the parsed per-candidate judge scores (window + score) for the run.

    The raw response already replays everything; this is the explicit, human-readable
    per-candidate score artifact (PRD #46 AC) — the judged window and its 1-10 score.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "start_time": row.get("start_time"),
            "end_time": row.get("end_time"),
            "composite_score": row.get("composite_score"),
            "reason": row.get("reason"),
        }
        for row in scored_rows
    ]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def paid_make_judge(artifacts_dir: pathlib.Path) -> MakeJudgeFn:
    """Per-VOD judge bound to paid Sonnet, persisting raw response + parsed scores.

    The provider is memoized so the pool + final reads share one paid call per VOD.
    The raw response is persisted before parsing (the re-evaluable artifact); the
    parsed per-candidate scores are written after.
    """

    def _make(vod_id: str) -> RerankFn:
        raw_path = _raw_path(artifacts_dir, vod_id)
        scores_path = _scores_path(artifacts_dir, vod_id)
        base = repost_virality_judge.repost_virality_judge(
            call_provider=memoized_provider(repost_virality_judge.anthropic_sonnet_provider()),
            persist_raw=lambda text: _write_text(raw_path, text),
        )

        def _judge(rows: list[Row]) -> list[Row]:
            scored = base(rows)
            _persist_scores(scores_path, scored)
            return scored

        return _judge

    return _make


def replay_make_judge(artifacts_dir: pathlib.Path) -> MakeJudgeFn:
    """Per-VOD judge bound to the persisted raw response — re-scores without paying."""

    def _make(vod_id: str) -> RerankFn:
        raw = _raw_path(artifacts_dir, vod_id).read_text(encoding="utf-8")
        return repost_virality_judge.repost_virality_judge(
            call_provider=persisted_response_provider(raw),
            persist_raw=lambda _t: None,
        )

    return _make


def _replay_load_inputs(storage: pathlib.Path, artifacts_dir: pathlib.Path) -> LoadInputsFn:
    """Replay loader: a VOD with no persisted raw was not part of the paid run, so
    it is a surfaced coverage gap (None) rather than a re-priced call."""

    def _load(vod_id: str):
        if not _raw_path(artifacts_dir, vod_id).exists():
            return None
        return replay_harness.load_replay_inputs(vod_id, storage)

    return _load


# ─────────────────────────────────────────────────────────────────────────────
# Reporting + CLI
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(value) -> str:
    if value is None:
        return "  —  "
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def format_spike_report(report: dict) -> str:
    """The decomposition + baseline-delta, then the GATE read explicitly, then the
    highest-view misses individually (generation vs ranking)."""
    lines = [gap_decomposition.format_report(report["decomposition"]), ""]
    gate = report["gate"]
    head, floor = gate["headline_recall_at_20"], gate["floor_recall_at_20"]
    lines += [
        "=" * 78,
        f"GATE (pre-fixed): headline r@{gate['k']} ↑ measurably AND floor r@{gate['k']} not ↓",
        "=" * 78,
        f"  headline r@{gate['k']}: {_fmt(head['current'])} vs baseline {_fmt(head['baseline'])}"
        f"   Δ {_signed(head['delta'])}",
        f"  floor    r@{gate['k']}: {_fmt(floor['current'])} vs baseline {_fmt(floor['baseline'])}"
        f"   Δ {_signed(floor['delta'])}",
        f"  VERDICT: {'✅ PASS → Phase 2a (judge blinded into serving)' if gate['passed'] else '⛔ NO-GO → Phase 2b (learned ranker on GT/outcome labels)'}",
        f"  {gate['reason']}",
        "",
        f"  HIGHEST-VIEW MISSES (served top-{report['top_k']} missed these — gen=generation, rank=ranking):",
    ]
    misses = report["top_view_misses"]
    if not misses:
        lines.append("    (none — every gold moment is in the served top-k)")
    for miss in misses[:20]:
        rank = f"rank {miss['buried_rank']}" if miss["buried_rank"] is not None else "—"
        lines.append(
            f"    {int(miss['views']):>10,}  {miss['miss_type']:<10} {miss['bucket']:<16} "
            f"{rank:<9} {miss['vod_id']}  {miss['desc'][:48]}")
    gaps = report["coverage_gap_vods"]
    if gaps:
        lines += ["", f"  ⚠  Coverage gap (no replayable/persisted artifacts): {', '.join(gaps)}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=str(eval_clip_finder.DEFAULT_MANIFEST))
    parser.add_argument("--predictions-dir", default=str(eval_clip_finder.DEFAULT_STORAGE_DIR))
    parser.add_argument("--anchor-csv", default=str(eval_clip_finder.DEFAULT_ANCHOR_CSV))
    parser.add_argument("--baseline", default=str(gap_decomposition.DEFAULT_BASELINE),
                        help="Frozen composite baseline JSON to delta + gate against")
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR),
                        help="Where raw responses + parsed scores persist (re-evaluable)")
    parser.add_argument("--depth", type=int, default=DEFAULT_SHORTLIST_DEPTH,
                        help="Judge shortlist depth (~200; buried-real gold median rank ~141)")
    parser.add_argument("--top-k", type=int, default=replay_harness.DEFAULT_TOP_K)
    parser.add_argument("--ks", default="5,10,20,50")
    parser.add_argument("--headline-min-views", type=float,
                        default=directional_eval.DEFAULT_HEADLINE_MIN_VIEWS)
    parser.add_argument("--replay", action="store_true",
                        help="Re-evaluate from persisted raw responses — NO paid call")
    parser.add_argument("--limit-vods", type=int, default=0,
                        help="Only judge the first N gold anchors (paid smoke test)")
    parser.add_argument("--output", type=pathlib.Path, default=None)
    parser.add_argument("--json", dest="output_json", action="store_true")
    args = parser.parse_args(argv)

    ks = tuple(int(k.strip()) for k in args.ks.split(",") if k.strip())
    manifest = eval_clip_finder.load_manifest(pathlib.Path(args.manifest))
    baseline = json.loads(pathlib.Path(args.baseline).read_text(encoding="utf-8"))
    storage = pathlib.Path(args.predictions_dir)
    artifacts_dir = pathlib.Path(args.artifacts_dir)

    # Only judge GOLD anchors — paying Sonnet for a non-gold VOD buys no gold recall.
    gold = directional_eval.gold_manifest(manifest, (directional_eval.GOLD_TIER,))
    anchors = [a for a in eval_clip_finder.read_anchor_vod_ids(pathlib.Path(args.anchor_csv)) if a in gold]
    if args.limit_vods:
        anchors = anchors[:args.limit_vods]

    if args.replay:
        make_judge = replay_make_judge(artifacts_dir)
        load_inputs = _replay_load_inputs(storage, artifacts_dir)
        mode = "replay (free, persisted responses)"
    else:
        make_judge = paid_make_judge(artifacts_dir)
        load_inputs = lambda vod_id: replay_harness.load_replay_inputs(vod_id, storage)
        mode = "PAID (anthropic Sonnet)"
    print(f"judge_spike: mode={mode}  depth={args.depth}  gold-anchors={len(anchors)}  "
          f"artifacts={artifacts_dir}", file=sys.stderr)

    report = run_judge_spike(
        anchors, manifest, baseline,
        make_judge=make_judge, load_inputs=load_inputs,
        depth=args.depth, top_k=args.top_k, ks=ks,
        headline_min_views=args.headline_min_views,
    )
    report["mode"] = "replay" if args.replay else "paid"
    report["computed_at"] = datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z")

    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    if args.output_json:
        print(payload)
    else:
        print(format_spike_report(report))
        print(f"\nmode: {report['mode']}   computed_at: {report['computed_at']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
