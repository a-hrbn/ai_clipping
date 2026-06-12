"""Strong-model Window-Probe — assembly (#61) + paid Sonnet rating (#62), PRD #60.

The probe that de-risks, then runs, the single paid step of Generation Phase 2. It
does NOT add a new measurement layer; it **composes** the existing seams:

  * `hard_miss_decomposition` (#56) — the `llm_skipped` bucket *is* the gold window
    set (consumed directly, never re-classified),
  * `eval_clip_finder.is_hit` (#49) — the gold/non-gold test + tier filtering,
  * the `replay_harness` (#47) loaders — `segments.jsonl` and the delivered
    `final_moments.json` (the flash-lite top-20) from the `pipeline_debug_*` dirs, and
  * the `repost_virality_judge` (#48) judge — its compact score-only prompt, parser
    (with truncation salvage) and Sonnet provider seam, reused not rebuilt.

Assembly (#61): for every window — gold AND control, treated identically — it builds
a modest symmetric transcript excerpt (~60-120 s each side) from `segments.jsonl`;
the control set is the delivered top-20 filtered to non-gold; coverage-gaps stay
surfaced, never silently counted.

Rating (#62): the strong model rates each window's excerpt on repost-virality, gold
AND control in ONE scored set (identical form). The raw response is persisted BEFORE
parsing, so `--replay` re-scores net-free without paying again; a window the model
did not score is dropped fail-closed (a truncated response never reads as "scored
everything"). The within-set separation + 0.66 gate is #63.

Usage (inside the Docker container):
  python -m ml.strong_model_window_probe                    # free assembly only (#61)
  python -m ml.strong_model_window_probe --rate             # PAID: one Sonnet call/VOD
  python -m ml.strong_model_window_probe --replay [--json]  # free re-score from disk
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import pathlib
import sys
from typing import Callable

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import (directional_eval, eval_clip_finder, hard_miss_decomposition as hmd,
                repost_virality_judge, replay_harness)
from ml.eval_clip_finder import COVERAGE_THRESHOLD, is_hit

logger = logging.getLogger(__name__)

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
# Paid Sonnet rating — score gold + control windows (one set), persist-before-parse
# ─────────────────────────────────────────────────────────────────────────────
# The single paid step (PRD #60, #62): the strong model (Sonnet via
# `anthropic_model_topk`, identical to the #50 judge) rates each window excerpt —
# gold AND control, in ONE scored set so both get the identical prompt/score form —
# on repost-virality. Composed, never rebuilt: the compact score-only prompt
# (`build_judge_prompt`, no rationales → no JSON truncation), the parser
# (`parse_judge_scores`, with truncation salvage) and the provider seam
# (`anthropic_sonnet_provider`) are the #48 judge's. The raw response is persisted
# BEFORE parsing, so a `--replay` re-scores net-free, without paying again.

GOLD = "gold"
CONTROL = "control"

# Where the paid run persists each VOD's raw response + parsed per-window scores,
# so a later --replay re-scores without paying again (its own artifacts subdir).
DEFAULT_ARTIFACTS_DIR = BACKEND_ROOT / "ml" / "artifacts" / "strong_model_window_probe"


def vod_windows(entry: dict) -> list[dict]:
    """One VOD's gold + control windows as a single list, each tagged with its kind.

    The probe rates gold and control together so they share one model invocation
    and one identical prompt — the within-set separation the #63 gate reads is only
    meaningful when both sets were scored by the same call. Original window fields
    are preserved; a ``kind`` of ``GOLD``/``CONTROL`` is added.
    """
    return ([{**g, "kind": GOLD} for g in entry.get("gold_windows", [])]
            + [{**c, "kind": CONTROL} for c in entry.get("control_windows", [])])


def _window_row(window: dict) -> dict:
    """A window as the compact-prompt Row shape (`build_judge_prompt` reads `text`)."""
    return {
        "start_time": window["start"],
        "end_time": window["end"],
        "text": str(window.get("excerpt", {}).get("text") or ""),
    }


def rate_windows(
    windows: list[dict],
    *,
    call_provider: repost_virality_judge.CallProviderFn,
    persist_raw,
) -> dict:
    """Score a single set of windows (gold + control) through the reused #48 judge.

    Builds the compact score-only prompt over the windows' excerpts, calls the
    injected provider, persists the raw response BEFORE parsing, then parses the
    compact scores (truncation-salvaged) and attaches the raw 0-1 ``repost_virality``
    to each window by its 1-based id. A window the model did not score is DROPPED,
    never back-filled — fail-closed, so a truncated response never reads as "scored
    everything"; the drop count is logged. Returns ``{scored, dropped, raw}``.
    """
    rows = [_window_row(w) for w in windows]
    prompt = repost_virality_judge.build_judge_prompt(rows)
    raw = call_provider(prompt)
    persist_raw(raw)
    scores = repost_virality_judge.parse_judge_scores(raw)
    scored: list[dict] = []
    dropped: list[dict] = []
    for index, window in enumerate(windows, start=1):
        if index in scores:
            scored.append({**window, "score": scores[index]})
        else:
            dropped.append(window)
    logger.info(json.dumps(
        {"event": "window_probe_rated", "windows": len(windows),
         "scored": len(scored), "dropped": len(dropped)}, sort_keys=True))
    return {"scored": scored, "dropped": dropped, "raw": raw}


# vod_id, windows -> the rate_windows result (paid Sonnet+persist, or replay-from-disk).
MakeRaterFn = Callable[[str, list[dict]], dict]


def _raw_path(artifacts_dir: pathlib.Path, vod_id: str) -> pathlib.Path:
    return artifacts_dir / f"{vod_id}.raw.txt"


def _scores_path(artifacts_dir: pathlib.Path, vod_id: str) -> pathlib.Path:
    return artifacts_dir / f"{vod_id}.scores.json"


def _write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _persist_scores(path: pathlib.Path, scored: list[dict]) -> None:
    """Persist the parsed per-window scores — the human-readable score artifact.

    The raw response already replays everything; this is the explicit per-window
    record the #63 separation reads: the window's identity (vod/tier/kind/desc) and
    its raw 0-1 ``repost_virality`` ``score``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "vod_id": w.get("vod_id"),
            "start": w.get("start"),
            "end": w.get("end"),
            "kind": w.get("kind"),
            "headline": w.get("headline"),
            "rank": w.get("rank"),
            "desc": w.get("desc"),
            "score": w.get("score"),
        }
        for w in scored
    ]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def persisted_response_provider(raw: str) -> repost_virality_judge.CallProviderFn:
    """A network-free provider that replays one persisted raw response.

    Re-evaluation binds the rating's provider seam to this instead of paid Sonnet:
    the prompt is ignored and the saved raw text is returned, so the *same* parser
    decodes it on the path the paid run used — re-scorable without paying again.
    """

    def _call(_prompt: str) -> str:
        return raw

    return _call


def paid_make_rater(artifacts_dir: pathlib.Path, *, provider=None) -> MakeRaterFn:
    """Per-VOD rater bound to paid Sonnet, persisting raw response + parsed scores.

    The provider defaults to the configured strong model (Sonnet via
    ``anthropic_model_topk``, identical to the #50 judge); tests inject a fake. One
    paid call per VOD (gold + control in one prompt). The raw response is persisted
    BEFORE parsing (the re-evaluable artifact); the parsed per-window scores after.
    """
    call_provider = provider or repost_virality_judge.anthropic_sonnet_provider()

    def _make(vod_id: str, windows: list[dict]) -> dict:
        raw_path = _raw_path(artifacts_dir, vod_id)
        result = rate_windows(
            windows, call_provider=call_provider,
            persist_raw=lambda text: _write_text(raw_path, text))
        _persist_scores(_scores_path(artifacts_dir, vod_id), result["scored"])
        return result

    return _make


def replay_make_rater(artifacts_dir: pathlib.Path) -> MakeRaterFn:
    """Per-VOD rater bound to the persisted raw response — re-scores without paying."""

    def _make(vod_id: str, windows: list[dict]) -> dict:
        raw = _raw_path(artifacts_dir, vod_id).read_text(encoding="utf-8")
        return rate_windows(
            windows, call_provider=persisted_response_provider(raw),
            persist_raw=lambda _text: None)

    return _make


def run_rating(assembled: dict, *, make_rater: MakeRaterFn) -> dict:
    """Rate every VOD's gold + control window set through the injected rater.

    Consumes the ``run`` assembly (#61) and, per VOD, scores its combined gold +
    control windows in one set (``vod_windows``) via ``make_rater`` (paid Sonnet or
    replay). Every cap is surfaced, never silent: a VOD whose assembly yielded no
    windows is named, and the per-VOD count of windows the model did not score
    (fail-closed drops) is recorded — so a truncated response never reads as "scored
    everything". Returns the per-VOD scored/dropped windows + the aggregate counts.
    """
    vods: dict[str, dict] = {}
    no_windows: list[str] = []
    dropped_unscored_by_vod: dict[str, int] = {}
    total_scored = 0
    total_dropped = 0
    for vod_id, entry in assembled["vods"].items():
        windows = vod_windows(entry)
        if not windows:
            no_windows.append(vod_id)
            continue
        result = make_rater(vod_id, windows)
        vods[vod_id] = {"scored": result["scored"], "dropped": result["dropped"]}
        total_scored += len(result["scored"])
        total_dropped += len(result["dropped"])
        if result["dropped"]:
            dropped_unscored_by_vod[vod_id] = len(result["dropped"])
    return {
        "vods": vods,
        "counts": {"scored": total_scored, "dropped": total_dropped},
        "caps": {
            "vods_without_windows": no_windows,
            "dropped_unscored_by_vod": dropped_unscored_by_vod,
        },
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

    if result.get("rating"):
        lines.append("")
        lines.append(_format_rating(result))

    lines.append("")
    lines.append(
        f"Caveat (thin n): gold windows floor={counts['gold_windows']['floor']} / "
        f"headline={counts['gold_windows']['headline']} — the separation gate (#63) reads off "
        f"the North-Star (headline). This slice rates the windows; the 0.66 gate is #63.")
    return "\n".join(lines)


def _format_rating(result: dict) -> str:
    """The per-window Sonnet scores (gold + control), by VOD, + the surfaced caps.

    #62 produces the scores; the within-set separation + 0.66 gate is #63. The raw
    0-1 ``repost_virality`` is shown so the gold-vs-control split is already legible.
    """
    rating = result["rating"]
    counts = rating["counts"]
    caps = rating["caps"]
    lines = [
        "=" * 78,
        f"WINDOW RATING ({result.get('rating_mode', '?')}): {result['ranking_model']} "
        f"repost-virality per window (gold + control, one scored set)",
        "=" * 78,
        f"Scored: {counts['scored']}   Dropped (unscored, fail-closed): {counts['dropped']}",
    ]
    if caps["vods_without_windows"]:
        lines.append(f"⚠  VODs with no windows to rate: {', '.join(caps['vods_without_windows'])}")
    if caps["dropped_unscored_by_vod"]:
        lines.append(f"⚠  Unscored drops by VOD (truncation/omission): {caps['dropped_unscored_by_vod']}")
    for vod_id, entry in rating["vods"].items():
        lines.append(f"  VOD {vod_id}:")
        for w in sorted(entry["scored"], key=lambda x: -x.get("score", 0.0)):
            tag = w["kind"] if w["kind"] == GOLD else f"control r{w.get('rank', '?')}"
            lines.append(f"    {w['score']:.3f}  [{tag:^12}] [{w['start']:.1f}-{w['end']:.1f}]"
                         f"  {str(w.get('desc') or '')[:48]}")
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
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR),
                        help="Where raw responses + parsed scores persist (re-evaluable)")
    parser.add_argument("--rate", action="store_true",
                        help="PAID: rate the assembled windows with Sonnet, persist raw+scores")
    parser.add_argument("--replay", action="store_true",
                        help="Re-score the windows from persisted raw responses — NO paid call")
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

    # Optional second step (#62): rate the assembled windows. Default (neither flag)
    # is the free assembly only (#61). --replay re-scores net-free from persisted
    # responses; --rate is the single paid Sonnet call (persists raw+scores).
    if args.rate or args.replay:
        artifacts_dir = pathlib.Path(args.artifacts_dir)
        if args.replay:
            make_rater = replay_make_rater(artifacts_dir)
            mode = "replay (free, persisted responses)"
        else:
            make_rater = paid_make_rater(artifacts_dir)
            mode = "PAID (anthropic Sonnet)"
        print(f"strong_model_window_probe: rating mode={mode}  artifacts={artifacts_dir}",
              file=sys.stderr)
        result["rating"] = run_rating(result, make_rater=make_rater)
        result["rating_mode"] = "replay" if args.replay else "paid"

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
