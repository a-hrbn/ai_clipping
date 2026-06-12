"""Strong-model Window-Probe — data assembly (PRD #60, Phase 1).

The free, offline tracer that de-risks the probe's data assembly *before* any paid
Sonnet call (#62) or separation analysis (#63). It does NOT add a new measurement
layer; it **composes** the existing seams:

  * `hard_miss_decomposition` (#56) — the `llm_skipped` bucket *is* the gold window
    set (consumed directly, never re-classified),
  * `eval_clip_finder.is_hit` (#49) — the gold/non-gold test + tier filtering, and
  * the `replay_harness` (#47) loaders — `segments.jsonl` and the delivered
    `final_moments.json` (the flash-lite top-20) from the `pipeline_debug_*` dirs.

For every window — gold AND control, treated identically — it builds a modest
symmetric transcript excerpt (~60-120 s each side) from `segments.jsonl`. The
control set is the delivered top-20 filtered to non-gold; coverage-gaps (windows
with no loadable artifacts) stay surfaced, never silently counted. No paid call:
the Sonnet rating (#62) and the separation/gate (#63) are later slices.

Usage (inside the Docker container):
  python -m ml.strong_model_window_probe
  python -m ml.strong_model_window_probe --json
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import directional_eval, eval_clip_finder, hard_miss_decomposition as hmd, replay_harness
from ml.eval_clip_finder import COVERAGE_THRESHOLD, is_hit

# The #56 bucket whose golds this probe rates: the model saw the transcript and
# skipped it. Consumed directly from hard_miss_decomposition, never re-classified.
GOLD_BUCKET = hmd.LLM_SKIPPED

# The symmetric transcript pad on each side of a window (~60-120 s; the mid of
# that range). Modest by design: not isolated (overstates recoverability), not the
# whole chunk (re-introduces the in-context competition the full re-gen tests).
DEFAULT_PAD_SEC = 90.0


def _seg_start(seg) -> float:
    return float((seg.get("start") if isinstance(seg, dict) else getattr(seg, "start", 0.0)) or 0.0)


def _seg_end(seg) -> float:
    return float((seg.get("end") if isinstance(seg, dict) else getattr(seg, "end", 0.0)) or 0.0)


def _seg_text(seg) -> str:
    return str((seg.get("text") if isinstance(seg, dict) else getattr(seg, "text", "")) or "")


def build_excerpt(
    start: float, end: float, segments, *, pad_sec: float = DEFAULT_PAD_SEC
) -> dict:
    """A modest symmetric transcript excerpt around the window ``[start, end]``.

    Pads the window by ``pad_sec`` on each side, clamped to the VOD bounds (the
    first segment start / last segment end), and concatenates the text of the
    segments overlapping that excerpt span (overlap = ``seg.start < excerpt_end and
    seg.end > excerpt_start``), in timeline order. The same pure function builds the
    excerpt for gold and control windows, so neither set gets special treatment.
    Accepts dict rows or RichSegment objects (composes with the replay loaders).
    """
    starts = [_seg_start(s) for s in segments]
    ends = [_seg_end(s) for s in segments]
    t0 = min(starts) if starts else float(start) - pad_sec
    t_end = max(ends) if ends else float(end) + pad_sec
    excerpt_start = max(t0, float(start) - pad_sec)
    excerpt_end = min(t_end, float(end) + pad_sec)
    kept = [s for s in segments
            if _seg_start(s) < excerpt_end and _seg_end(s) > excerpt_start]
    kept.sort(key=_seg_start)
    text = " ".join(_seg_text(s).strip() for s in kept if _seg_text(s).strip())
    return {
        "excerpt_start": excerpt_start,
        "excerpt_end": excerpt_end,
        "pad_sec": pad_sec,
        "text": text,
        "segment_count": len(kept),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Control set — the delivered top-20 (final_moments.json), filtered to non-gold
# ─────────────────────────────────────────────────────────────────────────────

def control_windows(
    final_moments: list[dict],
    gold_moments: list[dict],
    *,
    threshold: float = COVERAGE_THRESHOLD,
) -> dict:
    """The delivered flash-lite top-20, filtered to the non-gold moments.

    The control is the product question stated literally: "would Sonnet rank the
    skipped gold above what we actually showed in the slots that count?" A
    delivered moment that ``is_hit`` (#49, bidirectional) against *any* gold is a
    gold slot, not a control, and is dropped — the dropped count is surfaced so the
    sampling bound is logged, never silent. The kept controls preserve their
    1-based product rank (their position in the delivered order).
    """
    controls: list[dict] = []
    dropped = 0
    for rank, moment in enumerate(final_moments, start=1):
        if any(is_hit(gold, moment, threshold) for gold in gold_moments):
            dropped += 1
            continue
        controls.append({
            "start": float(moment.get("start_time") or 0.0),
            "end": float(moment.get("end_time") or 0.0),
            "rank": rank,
            "composite_score": float(moment.get("composite_score") or 0.0),
        })
    return {"controls": controls, "dropped_gold_overlap": dropped}


# ─────────────────────────────────────────────────────────────────────────────
# Gold windows — the #56 llm_skipped bucket, consumed directly per tier
# ─────────────────────────────────────────────────────────────────────────────

def llm_skipped_golds(hard_miss: dict, tier: str) -> list[dict]:
    """The ``llm_skipped`` golds of a tier, straight out of the #56 decomposition.

    ``hard_miss`` is the ``hard_miss_decomposition.run`` result's ``hard_miss``
    sub-dict (``{floor, headline}``, each a tier with classified ``golds``). The
    probe sub-selects the ``GOLD_BUCKET`` golds — it never re-classifies, so the
    set is bit-identical to #56's llm_skipped membership for that tier.
    """
    return [gold for gold in hard_miss[tier]["golds"] if gold["bucket"] == GOLD_BUCKET]


# ─────────────────────────────────────────────────────────────────────────────
# Assembly — per-VOD gold + control windows, each carrying its excerpt
# ─────────────────────────────────────────────────────────────────────────────

def run(
    manifest: dict[str, list[dict]],
    raw_by_vod: dict[str, list[dict]],
    pool_by_vod: dict[str, list[dict]],
    final_by_vod: dict[str, list[dict]],
    *,
    segments_by_vod: dict[str, list] | None = None,
    response_by_vod: dict[str, str] | None = None,
    chunk_duration_by_vod: dict[str, float] | None = None,
    anchor_vod_ids: list[str] | None = None,
    headline_min_views: float = directional_eval.DEFAULT_HEADLINE_MIN_VIEWS,
    hit_threshold: float = COVERAGE_THRESHOLD,
    pad_sec: float = DEFAULT_PAD_SEC,
    gold_tiers: tuple[str, ...] = (directional_eval.GOLD_TIER,),
) -> dict:
    """Assemble, per VOD, the gold + control windows with their transcript excerpts.

    Composes the seams end to end: ``hard_miss_decomposition.run`` (#56) gives the
    classified hard_miss, the ``llm_skipped`` golds become the gold windows (per
    tier, never re-classified), the delivered ``final_moments.json`` filtered to
    non-gold via #49 ``is_hit`` becomes the control windows, and ``build_excerpt``
    builds the *identical* symmetric excerpt for both. Coverage-gaps (VODs with no
    loadable ``final_moments``) stay surfaced, never silently counted; the dropped-
    as-gold control count is logged. No paid call — the Sonnet rating (#62) and the
    separation/gate (#63) consume this assembly later.
    """
    segments_by_vod = segments_by_vod or {}
    final_by_vod = final_by_vod or {}

    decomposition = hmd.run(
        manifest, raw_by_vod, pool_by_vod,
        segments_by_vod=segments_by_vod, response_by_vod=response_by_vod,
        chunk_duration_by_vod=chunk_duration_by_vod, anchor_vod_ids=anchor_vod_ids,
        headline_min_views=headline_min_views, gold_tiers=gold_tiers)
    hard_miss = decomposition["hard_miss"]

    floor_golds = llm_skipped_golds(hard_miss, "floor")
    headline_keys = {(g["vod_id"], g["start"], g["end"])
                     for g in llm_skipped_golds(hard_miss, "headline")}

    # The full gold set per VOD (anchor-restricted, same filter as #56) — what a
    # delivered control is filtered against, so a control is genuinely non-gold.
    gold_by_vod = directional_eval.gold_manifest(manifest, gold_tiers)
    if anchor_vod_ids is not None:
        anchor_set = set(anchor_vod_ids)
        gold_by_vod = {v: m for v, m in gold_by_vod.items() if v in anchor_set}

    vods: dict[str, dict] = {}
    for gold in floor_golds:
        vod_id = gold["vod_id"]
        entry = vods.setdefault(
            vod_id, {"gold_windows": [], "control_windows": [], "control_dropped_gold_overlap": 0})
        entry["gold_windows"].append({
            "vod_id": vod_id, "start": gold["start"], "end": gold["end"],
            "views": gold["views"], "desc": gold["desc"],
            "headline": (vod_id, gold["start"], gold["end"]) in headline_keys,
            "excerpt": build_excerpt(
                gold["start"], gold["end"], segments_by_vod.get(vod_id, []), pad_sec=pad_sec),
        })

    no_final_moments: list[str] = []
    for vod_id, entry in vods.items():
        delivered = final_by_vod.get(vod_id)
        if not delivered:
            no_final_moments.append(vod_id)
            continue
        segments = segments_by_vod.get(vod_id, [])
        filtered = control_windows(delivered, gold_by_vod.get(vod_id, []), threshold=hit_threshold)
        entry["control_dropped_gold_overlap"] = filtered["dropped_gold_overlap"]
        for control in filtered["controls"]:
            entry["control_windows"].append({
                "vod_id": vod_id, "start": control["start"], "end": control["end"],
                "rank": control["rank"], "composite_score": control["composite_score"],
                "excerpt": build_excerpt(
                    control["start"], control["end"], segments, pad_sec=pad_sec),
            })

    counts = {
        "gold_windows": {"floor": len(floor_golds), "headline": len(headline_keys)},
        "control_windows": sum(len(e["control_windows"]) for e in vods.values()),
        "control_dropped_gold_overlap": sum(e["control_dropped_gold_overlap"] for e in vods.values()),
        "vods_with_gold": len(vods),
    }
    return {
        "anchor_restricted": anchor_vod_ids is not None,
        "pad_sec": pad_sec,
        "hit_threshold": hit_threshold,
        "headline_min_views": headline_min_views,
        "gold_bucket": GOLD_BUCKET,
        "generator_model": decomposition["generator_model"],
        "ranking_model": decomposition["ranking_model"],
        "tiers": ["floor", "headline"],
        "counts": counts,
        "vods": vods,
        "gaps": {
            "missing_prediction_vods": decomposition["missing_prediction_vods"],
            "no_final_moments": no_final_moments,
        },
        "hard_miss": hard_miss,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Artifact bridge — compose the #56 inputs + the delivered final_moments loader
# ─────────────────────────────────────────────────────────────────────────────

def build_inputs(
    anchor_ids: list[str],
    storage: pathlib.Path,
) -> tuple[
    dict[str, list[dict]], dict[str, list[dict]], dict[str, list[dict]],
    dict[str, list], dict[str, str], dict[str, float], list[str],
]:
    """Replay each anchor VOD into the probe's inputs + the coverage-gap list.

    Composes ``hard_miss_decomposition.build_inputs`` (#56 → raw/pool/segments/
    response/chunk-duration, the exact rows the llm_skipped set is built from) and
    ``replay_harness.load_final_moments`` (#47 → the delivered top-20 control). No
    reimplemented I/O. A VOD with no replayable artifacts is in the gap list; one
    with no ``final_moments`` is simply absent from ``final_by_vod`` (surfaced by
    ``run`` as a ``no_final_moments`` gap).
    """
    raw_by_vod, pool_by_vod, segments_by_vod, response_by_vod, chunk_duration_by_vod, gaps = (
        hmd.build_inputs(anchor_ids, storage))
    final_by_vod: dict[str, list[dict]] = {}
    for vod_id in anchor_ids:
        moments = replay_harness.load_final_moments(vod_id, storage)
        if moments:
            final_by_vod[vod_id] = moments
    return (raw_by_vod, pool_by_vod, final_by_vod,
            segments_by_vod, response_by_vod, chunk_duration_by_vod, gaps)


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _excerpt_line(window: dict, label: str) -> str:
    ex = window["excerpt"]
    span = f"[{window['start']:.1f}-{window['end']:.1f}]"
    excerpt_span = f"excerpt [{ex['excerpt_start']:.1f}-{ex['excerpt_end']:.1f}], {ex['segment_count']} seg"
    return f"      {label} {span}  {excerpt_span}"


def format_report(result: dict) -> str:
    counts = result["counts"]
    lines = [
        "=" * 78,
        "Strong-model Window-Probe — assembled gold + control windows (no paid call)",
        "=" * 78,
        f"Source: #56 hard_miss_decomposition '{result['gold_bucket']}' bucket   "
        f"Headline: views>={int(result['headline_min_views']):,}   pad={result['pad_sec']:.0f}s",
        f"Generator: {result['generator_model']} (delivered top-20) — the probe rates the "
        f"skipped gold vs what {result['ranking_model']} will judge later (#62/#63).",
        f"Gold windows: floor={counts['gold_windows']['floor']}  "
        f"headline={counts['gold_windows']['headline']}   "
        f"Control windows: {counts['control_windows']} "
        f"(dropped as gold-overlap: {counts['control_dropped_gold_overlap']})   "
        f"VODs with gold: {counts['vods_with_gold']}",
    ]
    gaps = result["gaps"]
    if gaps["missing_prediction_vods"]:
        lines.append(
            f"⚠  Coverage gap: {len(gaps['missing_prediction_vods'])} gold VODs have NO finder "
            f"predictions (not counted): {', '.join(gaps['missing_prediction_vods'])}")
    if gaps["no_final_moments"]:
        lines.append(
            f"⚠  Coverage gap: {len(gaps['no_final_moments'])} VODs with gold have NO "
            f"final_moments (no controls assembled): {', '.join(gaps['no_final_moments'])}")

    for vod_id, entry in result["vods"].items():
        lines.append("")
        lines.append(f"  VOD {vod_id}:")
        lines.append(f"    GOLD (llm_skipped, {len(entry['gold_windows'])}):")
        for g in entry["gold_windows"]:
            tier = "headline" if g["headline"] else "floor"
            lines.append(_excerpt_line(g, f"[{tier:^8}] {int(g['views']):>9,} views"))
            lines.append(f"        {g['desc']}")
        lines.append(
            f"    CONTROL (delivered non-gold top-20, {len(entry['control_windows'])}; "
            f"dropped {entry['control_dropped_gold_overlap']} gold-overlap):")
        for c in entry["control_windows"]:
            lines.append(_excerpt_line(c, f"rank {c['rank']:>2}  comp {c['composite_score']:.2f}"))

    lines.append("")
    lines.append(
        f"Caveat (thin n): gold windows floor={counts['gold_windows']['floor']} / "
        f"headline={counts['gold_windows']['headline']} — the separation gate (#63) reads off "
        f"the North-Star (headline). This slice only assembles the data; no Sonnet call yet (#62).")
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
    parser.add_argument("--pad-sec", type=float, default=DEFAULT_PAD_SEC,
                        help="Symmetric transcript pad on each side of a window (~60-120s)")
    parser.add_argument("--hit-threshold", type=float, default=COVERAGE_THRESHOLD,
                        help="is_hit coverage threshold for filtering controls to non-gold")
    parser.add_argument("--output", type=pathlib.Path, default=None)
    parser.add_argument("--json", dest="output_json", action="store_true")
    args = parser.parse_args(argv)

    manifest = eval_clip_finder.load_manifest(pathlib.Path(args.manifest))
    anchors = eval_clip_finder.read_anchor_vod_ids(pathlib.Path(args.anchor_csv))

    (raw_by_vod, pool_by_vod, final_by_vod, segments_by_vod, response_by_vod,
     chunk_duration_by_vod, gaps) = build_inputs(anchors, pathlib.Path(args.predictions_dir))
    if gaps:
        # No silent truncation of the analysis: the artifact-less VODs are logged
        # here too (they also surface in the result as missing_prediction_vods).
        print(
            f"strong_model_window_probe: {len(gaps)}/{len(anchors)} anchor VODs have no "
            f"replayable artifacts — coverage gap: {', '.join(gaps)}",
            file=sys.stderr)

    result = run(
        manifest, raw_by_vod, pool_by_vod, final_by_vod,
        segments_by_vod=segments_by_vod, response_by_vod=response_by_vod,
        chunk_duration_by_vod=chunk_duration_by_vod, anchor_vod_ids=anchors,
        headline_min_views=args.headline_min_views, hit_threshold=args.hit_threshold,
        pad_sec=args.pad_sec)
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
