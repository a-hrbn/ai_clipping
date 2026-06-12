"""hard_miss Root-Cause-Decomposition (PRD #56, Phase 1).

A read-only offline tracer of the Generation-Track. It does NOT add a new
measurement layer; it **composes** the existing seams:

  * `directional_eval` (#48) — the gold filtering, the anchor restriction and the
    coverage-gap rule (raw@any is the generation measurement seam),
  * `gap_decomposition` (#49) — the `hard_miss` set itself, via `classify_gold`;
    the decomposition *sub-classifies* that already-verified set, it never
    re-derives it (the reuse guard), and
  * the `replay_harness` (#47) loaders — raw moments, `segments.jsonl` and the
    chunked `response.txt` from the `pipeline_debug_*` job dirs (no reimplemented
    I/O), reached through `gap_decomposition.build_inputs` and
    `replay_harness.load_generation_artifacts`.

Each `hard_miss` gold is routed into exactly one priority-resolved root-cause
bucket, first match wins:

  1. `visual`      — the gold window holds ~no transcript text (model-agnostic,
                     checked first); no transcript lever can recover it,
  2. `truncation`  — the chunk covering the gold aborted mid-JSON (a mechanical,
                     discrimination-independent product-recall win), else
  3. `llm_skipped` — transcript present, covering chunk clean: the model saw it
                     and skipped it (the residuum a stronger model could address).

The report splits floor vs headline hard_miss with per-bucket counts (summing to
the hard_miss count per tier), names every gold under its bucket (so the 1.43M
ChainsFR Face Reveal is visible in the bucket it routed to), holds the
generator-model fact (`gemini-2.5-flash-lite` generation vs Sonnet ranking) and
carries the truncation caveat. The missing-prediction VODs stay a measured,
never-counted coverage gap. No paid LLM call.

Usage (inside the Docker container):
  python -m ml.hard_miss_decomposition
  python -m ml.hard_miss_decomposition --json
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import pathlib
import re
import sys

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml import directional_eval, eval_clip_finder, gap_decomposition, replay_harness
from ml import llm_moment_finder as d0
from ml.gap_decomposition import DEFAULT_THRESHOLDS, RealnessThresholds

# The three priority-resolved root-cause buckets every hard_miss gold lands in,
# each routing to a different Phase-2 lever (PRD #56): visual → video-signal /
# product pivot (no transcript lever helps); truncation → mechanical chunk/
# max_tokens fix (a discrimination-independent product win); llm_skipped → a
# gated paid strong-model re-gen probe.
VISUAL = "visual"
TRUNCATION = "truncation"
LLM_SKIPPED = "llm_skipped"
BUCKETS: tuple[str, ...] = (VISUAL, TRUNCATION, LLM_SKIPPED)

# Generation runs on a cheap, weak model while the ranking/judge experiments ran
# on Sonnet — so only the llm_skipped bucket can plausibly profit from a stronger
# model. Held in the report so that routing reads honestly (PRD #56).
GENERATOR_MODEL = "gemini-2.5-flash-lite"
RANKING_MODEL = "Sonnet"

# ─────────────────────────────────────────────────────────────────────────────
# Detector 1 — no-transcript / visual (checked first, model-agnostic)
# ─────────────────────────────────────────────────────────────────────────────
# A gold window whose overlapping segments carry essentially no transcript text
# is visual: no transcript hook — not even a stronger text model — can recover it.
# Measured on the absolute non-whitespace char count inside the window, so it is
# decided before (and independently of) anything the generator did.
DEFAULT_MIN_TRANSCRIPT_CHARS = 40


def _seg_start(seg) -> float:
    return float((seg.get("start") if isinstance(seg, dict) else getattr(seg, "start", 0.0)) or 0.0)


def _seg_end(seg) -> float:
    return float((seg.get("end") if isinstance(seg, dict) else getattr(seg, "end", 0.0)) or 0.0)


def _seg_text(seg) -> str:
    return str((seg.get("text") if isinstance(seg, dict) else getattr(seg, "text", "")) or "")


def transcript_chars_in_window(gold: dict, segments) -> int:
    """Non-whitespace transcript chars from the segments overlapping the gold window.

    A segment overlaps when ``seg.start < gold.end and seg.end > gold.start``; its
    full text counts (whitespace stripped out). Accepts dict rows or RichSegment
    objects so the detector composes with the replay loaders without a conversion
    layer.
    """
    gold_start = float(gold.get("start") or 0.0)
    gold_end = float(gold.get("end") or 0.0)
    chars = 0
    for seg in segments:
        if _seg_start(seg) < gold_end and _seg_end(seg) > gold_start:
            chars += len("".join(_seg_text(seg).split()))
    return chars


def is_no_transcript(
    gold: dict, segments, *, min_chars: float = DEFAULT_MIN_TRANSCRIPT_CHARS
) -> bool:
    """True iff the gold window holds fewer than ``min_chars`` transcript chars."""
    return transcript_chars_in_window(gold, segments) < min_chars


# ─────────────────────────────────────────────────────────────────────────────
# Detector 2 — truncation (strict: the covering chunk block aborts mid-JSON)
# ─────────────────────────────────────────────────────────────────────────────
# The generator splits long VODs into ``===== CHUNK i/N =====`` blocks, each its
# own LLM call with its own ``max_tokens`` budget. A chunk whose JSON output ran
# out of budget aborts mid-moment — every gold in that chunk's span lost its
# candidate to a mechanical cut, not to model judgement. The proof is the JSON
# abort itself (a cleanly-closed block is never truncation, whatever the token
# count says); ``usage.json`` is only a cheap pre-filter flag, surfaced as
# evidence, never the verdict.

_CHUNK_DELIMITER = re.compile(r"^===== CHUNK (\d+)/(\d+) =====$", re.MULTILINE)


def split_response_chunks(response_text: str) -> list[str]:
    """Split ``response.txt`` into its per-chunk blocks, in chunk order.

    Splits on the ``===== CHUNK i/N =====`` delimiter. A single-call VOD has no
    delimiter, so the whole text is returned as one block (chunk 1/1).
    """
    if not _CHUNK_DELIMITER.search(response_text):
        return [response_text]
    parts = _CHUNK_DELIMITER.split(response_text)
    # re.split yields [pre, i, N, body, i, N, body, ...]; the bodies are every
    # 3rd element from index 3, already in chunk order.
    return [parts[i] for i in range(3, len(parts), 3)]


def _block_is_complete(block: str) -> bool:
    """True iff the block parses to complete JSON on a strict, no-repair load.

    Strips a ``` ```json ``` fence if present, then attempts a plain
    ``json.loads`` — deliberately NOT the serving ``parse_json_payload`` repair
    path, which salvages a partial array and would mask the very truncation this
    detector exists to catch. A mid-moment abort raises and reads as incomplete.
    """
    stripped = block.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return False
    return True


def covering_chunk_index(
    gold: dict, *, t0: float, chunk_duration_sec: float, n_chunks: int
) -> int:
    """The 1-based chunk whose primary VOD-time span covers the gold window start.

    The generator tiles the transcript into ``chunk_duration_sec`` windows from
    ``t0`` (the first segment start), so chunk ``i`` owns ``[t0+(i-1)·dur,
    t0+i·dur)``. The gold is attributed to the chunk owning its start; the index
    is clamped to ``[1, n_chunks]`` so a gold past the last window still maps to a
    real block.
    """
    if chunk_duration_sec <= 0 or n_chunks <= 1:
        return 1
    gold_start = float(gold.get("start") or 0.0)
    idx = int((gold_start - t0) // chunk_duration_sec) + 1
    return max(1, min(idx, n_chunks))


def is_truncated(
    gold: dict,
    segments,
    response_text: str,
    *,
    chunk_duration_sec: float = d0.DEFAULT_CHUNK_DURATION_SEC,
) -> bool:
    """True iff the chunk covering the gold window aborted mid-JSON.

    Splits ``response.txt`` into chunk blocks, maps the gold to its covering chunk
    via ``covering_chunk_index`` (over ``chunk_duration_sec`` from the segment
    timeline's ``t0``), and returns whether *that* block is incomplete. A gold
    sitting in a cleanly-closed chunk is NOT truncation even when a different
    chunk in the same VOD ran out of budget.
    """
    blocks = split_response_chunks(response_text)
    starts = [_seg_start(s) for s in segments]
    t0 = min(starts) if starts else 0.0
    idx = covering_chunk_index(
        gold, t0=t0, chunk_duration_sec=chunk_duration_sec, n_chunks=len(blocks))
    return not _block_is_complete(blocks[idx - 1])


# ─────────────────────────────────────────────────────────────────────────────
# Priority resolution — the one bucket each hard_miss gold lands in
# ─────────────────────────────────────────────────────────────────────────────

def classify_root_cause(
    gold: dict,
    *,
    segments,
    response_text: str,
    chunk_duration_sec: float = d0.DEFAULT_CHUNK_DURATION_SEC,
    min_transcript_chars: float = DEFAULT_MIN_TRANSCRIPT_CHARS,
) -> str:
    """Resolve one hard_miss gold into exactly one root-cause bucket.

    Precondition: ``gold`` is already a #49 hard_miss (no raw candidate covers
    it), so "no candidate" holds by construction and ``llm_skipped`` is the
    residuum — it needs no separate coverage check here. Priority order, first
    match wins:

      1. ``visual`` — the gold window holds ~no transcript text (model-agnostic,
         checked first; no transcript lever, not even a stronger model, helps).
      2. ``truncation`` — the chunk covering the gold aborted mid-JSON.
      3. ``llm_skipped`` — transcript present, covering chunk clean: the model
         saw it and skipped it.

    Degrades honestly when an artifact is unavailable: ``segments is None`` skips
    the visual check (transcript unknown, not assumed empty); an empty/absent
    ``response_text`` skips the truncation check (truncation can't be proven from
    a missing response). A real hard_miss VOD is replayable and carries both, so
    the residuum logic only triggers on synthetic inputs.
    """
    if segments is not None and is_no_transcript(gold, segments, min_chars=min_transcript_chars):
        return VISUAL
    if response_text and is_truncated(
        gold, segments or [], response_text, chunk_duration_sec=chunk_duration_sec
    ):
        return TRUNCATION
    return LLM_SKIPPED


def hard_miss_golds(
    gold_by_vod: dict[str, list[dict]],
    raw_by_vod: dict[str, list[dict]],
    pool_by_vod: dict[str, list[dict]],
    *,
    segments_by_vod: dict[str, list] | None = None,
    response_by_vod: dict[str, str] | None = None,
    chunk_duration_by_vod: dict[str, float] | None = None,
    min_views: float = 0.0,
    thresholds: RealnessThresholds = DEFAULT_THRESHOLDS,
    min_transcript_chars: float = DEFAULT_MIN_TRANSCRIPT_CHARS,
) -> list[dict]:
    """The gold moments (views >= min_views) #49 classifies as hard_miss, bucketed.

    The reuse guard lives here: each gold is routed through
    ``gap_decomposition.classify_gold`` and kept only when its bucket is
    ``HARD_MISS``, so the set is bit-identical to #49's hard_miss — the
    decomposition sub-classifies that set, it never re-derives the membership.
    Each survivor is then routed through ``classify_root_cause`` (no-transcript →
    truncation → llm-skipped) using its VOD's ``segments.jsonl`` text, chunked
    ``response.txt`` and ``chunk_duration_sec``, and returned as a flat record
    carrying its window, views, desc and resolved ``bucket``.
    """
    segments_by_vod = segments_by_vod or {}
    response_by_vod = response_by_vod or {}
    chunk_duration_by_vod = chunk_duration_by_vod or {}
    out: list[dict] = []
    for vod_id, moments in gold_by_vod.items():
        raw_pool = raw_by_vod.get(vod_id, [])
        ranked_pool = pool_by_vod.get(vod_id, [])
        segments = segments_by_vod.get(vod_id)
        response_text = response_by_vod.get(vod_id)
        chunk_duration_sec = chunk_duration_by_vod.get(vod_id, d0.DEFAULT_CHUNK_DURATION_SEC)
        for gold in moments:
            if float(gold.get("views") or 0.0) < min_views:
                continue
            verdict = gap_decomposition.classify_gold(
                gold, raw_pool, ranked_pool=ranked_pool, thresholds=thresholds)
            if verdict["bucket"] != gap_decomposition.HARD_MISS:
                continue
            bucket = classify_root_cause(
                gold, segments=segments, response_text=response_text,
                chunk_duration_sec=chunk_duration_sec,
                min_transcript_chars=min_transcript_chars)
            out.append({
                "vod_id": vod_id,
                "start": float(gold.get("start") or 0.0),
                "end": float(gold.get("end") or 0.0),
                "views": float(gold.get("views") or 0.0),
                "desc": str(gold.get("desc") or ""),
                "bucket": bucket,
            })
    return out


def _tier(golds: list[dict]) -> dict:
    """A reported hard_miss tier: its count, the per-bucket breakdown, the golds.

    The per-bucket counts partition the tier exactly (mutually exclusive +
    exhaustive), so ``sum(buckets.values()) == count`` always holds.
    """
    buckets = {bucket: 0 for bucket in BUCKETS}
    for gold in golds:
        buckets[gold["bucket"]] = buckets.get(gold["bucket"], 0) + 1
    return {"count": len(golds), "buckets": buckets, "golds": golds}


def run(
    manifest: dict[str, list[dict]],
    raw_by_vod: dict[str, list[dict]],
    pool_by_vod: dict[str, list[dict]],
    *,
    segments_by_vod: dict[str, list] | None = None,
    response_by_vod: dict[str, str] | None = None,
    chunk_duration_by_vod: dict[str, float] | None = None,
    anchor_vod_ids: list[str] | None = None,
    headline_min_views: float = directional_eval.DEFAULT_HEADLINE_MIN_VIEWS,
    gold_tiers: tuple[str, ...] = (directional_eval.GOLD_TIER,),
    thresholds: RealnessThresholds = DEFAULT_THRESHOLDS,
    min_transcript_chars: float = DEFAULT_MIN_TRANSCRIPT_CHARS,
) -> dict:
    """Decompose the hard_miss denominator into root-cause buckets, floor vs headline.

    Reuses ``directional_eval`` for the gold filtering and the anchor restriction
    (the same gold-tier + anchor set as the frozen baseline), so the denominator
    never drifts from the recall numbers. The hard_miss membership comes from
    ``hard_miss_golds`` (i.e. #49's ``classify_gold``) — never re-classified — and
    each member is routed into one of ``visual`` / ``truncation`` / ``llm_skipped``
    from its VOD's segments + chunked response. Floor counts every hard_miss gold;
    headline keeps only the high-view ones (views >= ``headline_min_views``), so
    the routing weight reads honestly off the North-Star tier. Each tier's
    per-bucket counts sum to its hard_miss count.
    """
    gold = directional_eval.gold_manifest(manifest, gold_tiers)
    if anchor_vod_ids is not None:
        anchor_set = set(anchor_vod_ids)
        gold = {vod_id: moments for vod_id, moments in gold.items() if vod_id in anchor_set}

    # The gold VODs the finder never scored are a coverage gap: surfaced, never
    # folded into the denominator. The decomposition runs over the scored set only.
    missing = directional_eval.missing_prediction_vods(gold, raw_by_vod)
    scored_gold = {vod_id: moments for vod_id, moments in gold.items() if raw_by_vod.get(vod_id)}

    def tier(min_views: float) -> dict:
        return _tier(hard_miss_golds(
            scored_gold, raw_by_vod, pool_by_vod,
            segments_by_vod=segments_by_vod, response_by_vod=response_by_vod,
            chunk_duration_by_vod=chunk_duration_by_vod,
            min_views=min_views, thresholds=thresholds,
            min_transcript_chars=min_transcript_chars))

    return {
        "gold_tiers": sorted(gold_tiers),
        "anchor_restricted": anchor_vod_ids is not None,
        "gold_vods": len(gold),
        "scored_vods": len(scored_gold),
        "missing_prediction_vods": missing,
        "headline_min_views": headline_min_views,
        "min_transcript_chars": min_transcript_chars,
        "thresholds": dataclasses.asdict(thresholds),
        "generator_model": GENERATOR_MODEL,
        "ranking_model": RANKING_MODEL,
        "buckets": list(BUCKETS),
        "hard_miss": {"floor": tier(0.0), "headline": tier(headline_min_views)},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Artifact bridge — reuse the #49 bridge (which reuses the #47 replay loaders)
# ─────────────────────────────────────────────────────────────────────────────

def build_inputs(
    anchor_ids: list[str],
    storage: pathlib.Path,
) -> tuple[
    dict[str, list[dict]], dict[str, list[dict]], dict[str, list],
    dict[str, str], dict[str, float], list[str],
]:
    """Replay each anchor VOD into the classifier's inputs + the coverage-gap list.

    Composes two existing seams, no reimplemented I/O:

      * ``gap_decomposition.build_inputs`` for (raw, deduped-pool) — reusing the
        #49 bridge guarantees the raw rows are the exact rows #49 classifies, so
        the hard_miss set reconciles, and
      * ``replay_harness.load_generation_artifacts`` for the per-VOD segments,
        chunked ``response.txt`` and ``chunk_duration_sec`` (from ``stage.json``)
        the root-cause detectors read.

    A VOD with no replayable artifacts comes back in the gap list (and is absent
    from the per-VOD dicts), never silently dropped. The #49 product final set is
    unused here (there is no recall metric) and dropped.
    """
    raw_by_vod, pool_by_vod, _final_by_vod, gaps = gap_decomposition.build_inputs(
        anchor_ids, storage)
    segments_by_vod: dict[str, list] = {}
    response_by_vod: dict[str, str] = {}
    chunk_duration_by_vod: dict[str, float] = {}
    for vod_id in anchor_ids:
        loaded = replay_harness.load_generation_artifacts(vod_id, storage)
        if loaded is None:
            continue
        segments, response_text, _usage, stage = loaded
        segments_by_vod[vod_id] = segments
        response_by_vod[vod_id] = response_text
        chunk_duration = stage.get("chunk_duration_sec")
        if chunk_duration:
            chunk_duration_by_vod[vod_id] = float(chunk_duration)
    return (raw_by_vod, pool_by_vod, segments_by_vod, response_by_vod,
            chunk_duration_by_vod, gaps)


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _bucket_counts_line(label: str, tier: dict) -> str:
    b = tier["buckets"]
    cells = "  ".join(f"{name}={b.get(name, 0):>2}" for name in BUCKETS)
    return f"  {label:<9} n={tier['count']:>2}  {cells}"


def _golds_by_bucket(tier: dict) -> list[str]:
    """List every hard_miss gold under its bucket, by name — so a named moment
    (e.g. the 1.43M ChainsFR Face Reveal) is visible in the bucket it routed to."""
    lines: list[str] = []
    for bucket in BUCKETS:
        members = [g for g in tier["golds"] if g["bucket"] == bucket]
        lines.append(f"    {bucket} ({len(members)}):")
        if not members:
            lines.append("      (none)")
        for g in sorted(members, key=lambda x: -x["views"]):
            lines.append(
                f"      - {g['vod_id']} [{g['start']:.1f}-{g['end']:.1f}] "
                f"{int(g['views']):,} views — {g['desc']}")
    return lines


def format_report(result: dict) -> str:
    hm = result["hard_miss"]
    lines = [
        "=" * 78,
        "hard_miss Root-Cause-Decomposition — WHY each gold has no candidate",
        "=" * 78,
        f"Gold tiers: {','.join(result['gold_tiers'])}   Gold VODs: {result['gold_vods']}"
        f"   Scored: {result['scored_vods']}"
        f"{'  (anchor-restricted)' if result.get('anchor_restricted') else ''}"
        f"   Headline: views>={int(result['headline_min_views']):,}",
        f"Generator: {result.get('generator_model', GENERATOR_MODEL)} (generation) "
        f"vs {result.get('ranking_model', RANKING_MODEL)} (ranking) — only llm_skipped "
        f"can profit from a stronger model.",
    ]
    missing = result["missing_prediction_vods"]
    if missing:
        lines.append(
            f"⚠  Coverage gap: {len(missing)}/{result['gold_vods']} gold VODs have NO finder "
            f"predictions — excluded from the hard_miss denominator, NOT counted as misses: "
            f"{', '.join(missing)}")
    else:
        lines.append("✓  Every gold VOD has finder predictions.")

    lines.append("")
    lines.append("  ROOT-CAUSE BUCKETS (no raw candidate covers the gold; first match wins):")
    lines.append(_bucket_counts_line("FLOOR", hm["floor"]))
    lines.append(_bucket_counts_line("HEADLINE", hm["headline"]))
    lines.append("")
    lines.append("  FLOOR golds by bucket:")
    lines.extend(_golds_by_bucket(hm["floor"]))

    lines.append("")
    lines.append(
        "Caveat: only the truncation subset is a discrimination-independent product "
        "recall@20 win (those candidates would have competed on merit). An llm_skipped "
        "or visual raw@any lift lands buried until discrimination is solved.")
    lines.append(
        "Routing (cheap-before-expensive): truncation non-trivial → mechanical "
        "chunk/max_tokens fix; llm_skipped dominant → gated paid strong-model re-gen; "
        "visual plurality → video-signal track / product pivot.")
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
    parser.add_argument("--min-transcript-chars", type=float,
                        default=DEFAULT_MIN_TRANSCRIPT_CHARS,
                        help="No-transcript threshold: a gold window below this many "
                             "non-whitespace transcript chars routes to visual")
    parser.add_argument("--output", type=pathlib.Path, default=None)
    parser.add_argument("--json", dest="output_json", action="store_true")
    args = parser.parse_args(argv)

    manifest = eval_clip_finder.load_manifest(pathlib.Path(args.manifest))
    anchors = eval_clip_finder.read_anchor_vod_ids(pathlib.Path(args.anchor_csv))

    (raw_by_vod, pool_by_vod, segments_by_vod, response_by_vod,
     chunk_duration_by_vod, gaps) = build_inputs(anchors, pathlib.Path(args.predictions_dir))
    if gaps:
        # No silent truncation of the analysis: the artifact-less VODs are logged
        # here too (they also surface in the result as missing_prediction_vods).
        print(
            f"hard_miss_decomposition: {len(gaps)}/{len(anchors)} anchor VODs have no "
            f"replayable artifacts — coverage gap: {', '.join(gaps)}",
            file=sys.stderr)

    result = run(
        manifest, raw_by_vod, pool_by_vod,
        segments_by_vod=segments_by_vod, response_by_vod=response_by_vod,
        chunk_duration_by_vod=chunk_duration_by_vod,
        anchor_vod_ids=anchors, headline_min_views=args.headline_min_views,
        min_transcript_chars=args.min_transcript_chars)
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
