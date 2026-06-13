"""Buried-Gold Blind-Sheet — Selektion + Sheet (PRD #69, Slices 1 & 2).

The read-only blind-sheet gate. It mechanically determines the population to test
and assembles the blind discrimination task, by **composing** the existing seams,
never re-classifying and never adding a measurement layer:

  * ``replay_harness.load_replay_inputs`` (#47) + ``replay_ranked_pool`` (production
    rank, identity ``rank_score_fn``) — the raw rows, transcript segments, and the
    deduped, rank-ordered pool the buried gold live in,
  * ``gap_decomposition.classify_gold`` (#46/#49) — the frozen bucket + buried_rank,
  * the gold-eval-manifest loader + ``gt_tier`` filter (``eval_clip_finder`` /
    ``directional_eval``) — only audio-confirmed positions (gold/silver),
  * ``outcome_view_weighting.view_count`` — the repost-view ranking the top-10 cut,
  * ``eval_clip_finder.is_hit`` — the covering/decoy split (covering = is_hit, every
    decoy = not is_hit), and
  * ``strong_model_window_probe.build_excerpt`` (#61) — the one pure excerpt builder,
    same ``pad_sec`` for gold and decoys so length/format never leaks the answer.

Slice 1 — **selection**: ``bucket == CLEAN_BUT_BURIED`` **and** ``buried_rank > 20``,
gt_tier in {gold, silver}, top-10 by views. Slice 2 — **sheet**: per selected gold a
blind 8-way choice (the covering candidate at ``buried_rank`` + 7 over-ranking,
non-covering decoys), all rendered through the same ``build_excerpt`` and
deterministically shuffled (seed = vod_id + gold window), the answer hidden. Output:
a readable sheet (A–H + answer/recognized field, caveat header) and a **sealed key**
in a separate file (``sheet_id → {gold_letter, views, buried_rank, vod_id}``). VODs
with no replayable job dir stay a visible coverage gap (measured-not-counted);
selections < 10 golds and decoy shortfalls < 7 are logged, never silently capped.

Usage (inside the Docker container):
  python -m ml.buried_gold_blind_sheet                 # print the selection
  python -m ml.buried_gold_blind_sheet --json          # the selection as JSON
  python -m ml.buried_gold_blind_sheet --sheet-out sheet.txt --key-out key.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import pathlib
import random
import sys

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import (directional_eval, eval_clip_finder, gap_decomposition as gd,
                outcome_view_weighting, replay_harness,
                strong_model_window_probe as smwp)

logger = logging.getLogger(__name__)

# Audio-confirmed positions only — the gold window is trustworthy. review /
# unverifiable are excluded so a (later) excerpt never sits on a shaky boundary.
GOLD_SILVER_TIERS: tuple[str, ...] = ("gold", "silver")

# Outside the delivered top-20: a clean candidate buried deeper than this is what
# costs the shipped recall@20. The selection floor (strictly greater than).
BURIED_RANK_FLOOR = 20

# The blind-sheet population size — the top-N clean_but_buried golds by views.
DEFAULT_TOP_N = 10

# The 8-way blind choice: 1 covering candidate (the answer) + 7 over-ranking,
# non-covering decoys. Letters A-H label the shuffled excerpts on the sheet.
N_DECOYS = 7
N_CHOICES = N_DECOYS + 1
CHOICE_LETTERS = "ABCDEFGH"

# All 8 excerpts use this identical pad (±12 s) through the same pure build_excerpt,
# so length/format never leaks which window is the gold. (smwp's own default is the
# much wider 90 s window-probe pad — the sheet deliberately overrides it.)
EXCERPT_PAD_SEC = 12.0

# Header caveat — a PASS must not be over-read. It guards the upper-bound reading,
# the recognized exclusion, and every logged cap, so the gate decision stays honest.
SHEET_CAVEAT = (
    "CAVEAT — read before scoring:\n"
    "  * UPPER BOUND: this measures what a Jynxzi-literate human can pick from\n"
    "    pure transcript text — an upper bound on what a text/embedding ranker\n"
    "    could learn. A PASS is necessary, not sufficient.\n"
    "  * recognized picks (you remember the actual clip) are EXCLUDED from the\n"
    "    score — that is memory, not text signal. Mark them honestly.\n"
    "  * Any logged cap (< 10 eligible golds, < 7 available decoys) is surfaced\n"
    "    below — a short sheet is NOT full coverage."
)


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
# Sheet layer (#71) — the blind 8-way choice (covering + 7 decoys) + sealed key
# ─────────────────────────────────────────────────────────────────────────────

def _covers(gold: dict, cand: dict) -> bool:
    """True iff the candidate window covers the gold (``eval_clip_finder.is_hit``).

    The gold carries start/end and a candidate carries start_time/end_time —
    exactly the shapes ``is_hit`` reads — so the sheet's covering/decoy split never
    drifts from the frozen coverage seam.
    """
    return eval_clip_finder.is_hit(gold, cand)


def select_decoys(
    gold: dict,
    ranked_pool: list[dict],
    buried_rank: int,
    *,
    n_decoys: int = N_DECOYS,
) -> dict:
    """The non-covering decoys for one gold, drawn first from its over-rankers.

    The honest task is "beat the candidates that buried you": the decoys are the
    candidates ranked *above* the covering one (1-based ranks 1..buried_rank-1)
    that do **not** cover the gold. If fewer than ``n_decoys`` such over-rankers
    exist, the gap is filled from the deeper pool (ranks > buried_rank, still
    non-covering); any remaining shortfall is surfaced, never silently capped.
    """
    over_rankers = [c for c in ranked_pool[: buried_rank - 1] if not _covers(gold, c)]
    decoys = list(over_rankers[:n_decoys])
    overrank_used = len(decoys)
    if len(decoys) < n_decoys:
        for cand in ranked_pool[buried_rank:]:  # deeper non-coverers fill the gap
            if len(decoys) >= n_decoys:
                break
            if not _covers(gold, cand):
                decoys.append(cand)
    return {
        "decoys": decoys,
        "overrank_used": overrank_used,
        "overrank_available": len(over_rankers),
        "shortfall": max(0, n_decoys - len(decoys)),
    }


def _shuffle_seed(vod_id: str, gold: dict) -> str:
    """A stable seed string from the VOD id + the gold window (its row identity).

    Sheet and key stay reproducible: the same gold always shuffles to the same
    arrangement, independent of process hash randomisation.
    """
    start = float(gold.get("start") or 0.0)
    end = float(gold.get("end") or 0.0)
    return f"{vod_id}:{start:.2f}:{end:.2f}"


def _excerpt(cand: dict, segments, *, pad_sec: float) -> dict:
    """One candidate window as a padded transcript excerpt (the shared seam)."""
    return smwp.build_excerpt(
        float(cand.get("start_time") or 0.0),
        float(cand.get("end_time") or 0.0),
        segments,
        pad_sec=pad_sec,
    )


def build_choice_entry(
    gold: dict,
    ranked_pool: list[dict],
    segments,
    *,
    sheet_id: str,
    pad_sec: float = EXCERPT_PAD_SEC,
    n_decoys: int = N_DECOYS,
) -> dict:
    """One gold's blind 8-way choice: covering candidate + decoys, shuffled.

    The covering candidate sits at the gold's ``buried_rank`` (1-based) in the
    ranked pool — the *real* candidate the human competes against, not the raw GT
    boundary. It and the decoys are rendered through the same ``build_excerpt``
    with identical ``pad_sec``, then deterministically shuffled (seed = vod_id +
    gold window); the covering candidate's landing letter is recorded as
    ``gold_letter`` — the sealed answer, never rendered on the sheet itself.
    """
    buried_rank = int(gold["buried_rank"])
    cover = ranked_pool[buried_rank - 1]
    decoy_pick = select_decoys(gold, ranked_pool, buried_rank, n_decoys=n_decoys)
    # Covering candidate first, decoys after — then a deterministic shuffle.
    windows = [(cover, True)] + [(d, False) for d in decoy_pick["decoys"]]
    order = list(range(len(windows)))
    random.Random(_shuffle_seed(gold["vod_id"], gold)).shuffle(order)

    choices: list[dict] = []
    gold_letter = ""
    for letter, idx in zip(CHOICE_LETTERS, order):
        cand, is_gold = windows[idx]
        if is_gold:
            gold_letter = letter
        choices.append({
            "letter": letter,
            "candidate": cand,
            "excerpt": _excerpt(cand, segments, pad_sec=pad_sec),
        })
    return {
        "sheet_id": sheet_id,
        "vod_id": gold["vod_id"],
        "buried_rank": buried_rank,
        "views": float(gold.get("views") or 0.0),
        "gold_letter": gold_letter,
        "shortfall": decoy_pick["shortfall"],
        "overrank_used": decoy_pick["overrank_used"],
        "choices": choices,
    }


def build_sheet(
    selected: list[dict],
    pool_by_vod: dict[str, list[dict]],
    segments_by_vod: dict[str, list],
    *,
    pad_sec: float = EXCERPT_PAD_SEC,
    n_decoys: int = N_DECOYS,
) -> dict:
    """Assemble the blind sheet + sealed key over the selected buried golds.

    Per gold (in the selection's view-ranked order) a blind 8-way choice block
    with a stable ``sheet_id`` (``sheet_01``…); the sealed ``key`` maps each
    ``sheet_id`` to its answer + provenance (``gold_letter``, ``views``,
    ``buried_rank``, ``vod_id``) so the human works the sheet without spoilers.
    Decoy shortfalls are collected and logged — never silently capped.
    """
    entries: list[dict] = []
    key: dict[str, dict] = {}
    shortfalls: list[dict] = []
    for i, gold in enumerate(selected):
        sheet_id = f"sheet_{i + 1:02d}"
        vod_id = gold["vod_id"]
        entry = build_choice_entry(
            gold, pool_by_vod.get(vod_id, []), segments_by_vod.get(vod_id, []),
            sheet_id=sheet_id, pad_sec=pad_sec, n_decoys=n_decoys)
        entries.append(entry)
        key[sheet_id] = {
            "gold_letter": entry["gold_letter"],
            "views": entry["views"],
            "buried_rank": entry["buried_rank"],
            "vod_id": vod_id,
        }
        if entry["shortfall"]:
            shortfalls.append({"sheet_id": sheet_id, "vod_id": vod_id,
                               "shortfall": entry["shortfall"]})
    if shortfalls:
        # No silent cap: a block with fewer than 8 options is logged, not hidden.
        logger.info(json.dumps(
            {"event": "blind_sheet_decoy_shortfall", "blocks": shortfalls},
            sort_keys=True))
    return {
        "entries": entries,
        "key": key,
        "caveat": SHEET_CAVEAT,
        "shortfalls": shortfalls,
        "n_golds": len(entries),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Artifact bridge — replay each anchor VOD into (raw, ranked-pool, segments) dicts
# ─────────────────────────────────────────────────────────────────────────────

def build_replay_inputs(
    anchor_ids: list[str],
    storage: pathlib.Path,
    *,
    rank_score_fn: replay_harness.RankScoreFn | None = None,
) -> tuple[dict[str, list[dict]], dict[str, list[dict]], dict[str, list], list[str]]:
    """Replay each anchor VOD into (raw rows, ranked pool, segments) + the gap list.

    For every anchor VOD it loads the newest replayable ``pipeline_debug_*`` job
    dir (``replay_harness.load_replay_inputs``) and builds the production-rank
    ranked pool (``replay_ranked_pool`` under the identity ``rank_score_fn`` — the
    shipped composite order). The sheet layer additionally needs the ``segments``
    (``build_excerpt``); the selection layer reads only raw rows + pool. A VOD with
    no replayable artifacts lands in the gap list, never silently dropped. No paid
    call, no network.
    """
    raw_by_vod: dict[str, list[dict]] = {}
    pool_by_vod: dict[str, list[dict]] = {}
    segments_by_vod: dict[str, list] = {}
    gaps: list[str] = []
    for vod_id in anchor_ids:
        loaded = replay_harness.load_replay_inputs(vod_id, storage)
        if loaded is None:
            gaps.append(vod_id)
            continue
        raw_rows, segments = loaded
        raw_by_vod[vod_id] = raw_rows
        segments_by_vod[vod_id] = segments
        pool_by_vod[vod_id] = replay_harness.replay_ranked_pool(
            raw_rows, segments, rank_score_fn=rank_score_fn)
    return raw_by_vod, pool_by_vod, segments_by_vod, gaps


def build_inputs(
    anchor_ids: list[str],
    storage: pathlib.Path,
    *,
    rank_score_fn: replay_harness.RankScoreFn | None = None,
) -> tuple[dict[str, list[dict]], dict[str, list[dict]], list[str]]:
    """The selection layer's (raw rows, ranked pool, gap list) — segments dropped.

    A thin projection of ``build_replay_inputs`` kept stable for the Slice 1
    selection callers that never need the transcript segments.
    """
    raw_by_vod, pool_by_vod, _segments, gaps = build_replay_inputs(
        anchor_ids, storage, rank_score_fn=rank_score_fn)
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


def format_sheet(sheet: dict) -> str:
    """The human-facing blind sheet — caveat header + one A–H block per gold.

    Shows only the lettered excerpts and an answer/recognized field. None of the
    sealed-key fields (gold_letter, views, buried_rank, vod_id) appear, so the
    sheet carries no spoiler — the answer lives only in the separate key.
    """
    lines = [
        "=" * 78,
        "Buried-Gold Blind-Sheet — BLIND 8-WAY CHOICE  (PRD #69)",
        "=" * 78,
        sheet["caveat"],
        "",
    ]
    if sheet["shortfalls"]:
        lines.append("⚠  Decoy shortfalls (surfaced, not silently capped):")
        for s in sheet["shortfalls"]:
            lines.append(
                f"     {s['sheet_id']}: {s['shortfall']} decoy(s) short "
                f"— {N_CHOICES - s['shortfall']} options instead of {N_CHOICES}")
        lines.append("")
    for entry in sheet["entries"]:
        lines.append("-" * 78)
        lines.append(entry["sheet_id"])
        for choice in entry["choices"]:
            text = choice["excerpt"]["text"] or "(no transcript text in this window)"
            lines.append(f"  [{choice['letter']}] {text}")
        lines.append("")
        lines.append("  Your pick (A–H): ____    recognized (you remember this clip)? [ ] yes")
        lines.append("")
    return "\n".join(lines)


def format_key(sheet: dict) -> str:
    """The sealed answer key as JSON — sheet_id → answer + provenance."""
    return json.dumps(sheet["key"], indent=2, ensure_ascii=False, sort_keys=True)


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
    parser.add_argument("--sheet-out", type=pathlib.Path, default=None,
                        help="Write the blind 8-way sheet here (enables the sheet layer)")
    parser.add_argument("--key-out", type=pathlib.Path, default=None,
                        help="Write the sealed answer key here (default: <sheet-out>.key.json)")
    args = parser.parse_args(argv)

    tiers = tuple(t.strip() for t in args.tiers.split(",") if t.strip())
    manifest = eval_clip_finder.load_manifest(pathlib.Path(args.manifest))
    anchors = eval_clip_finder.read_anchor_vod_ids(pathlib.Path(args.anchor_csv))

    raw_by_vod, pool_by_vod, segments_by_vod, gaps = build_replay_inputs(
        anchors, pathlib.Path(args.predictions_dir))
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

    # Sheet layer (#71) — additive: only when --sheet-out is given. The selection
    # output above is unchanged. Sheet + sealed key go to separate files.
    if args.sheet_out:
        sheet = build_sheet(result["selected"], pool_by_vod, segments_by_vod)
        key_out = args.key_out or args.sheet_out.with_name(args.sheet_out.name + ".key.json")
        args.sheet_out.write_text(format_sheet(sheet) + "\n", encoding="utf-8")
        key_out.write_text(format_key(sheet) + "\n", encoding="utf-8")
        print(
            f"\nblind sheet: {sheet['n_golds']} block(s) → {args.sheet_out}  "
            f"(sealed key → {key_out})",
            file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
